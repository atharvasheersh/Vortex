#include "vortex/storage.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vortex {
MappedFile::MappedFile(const std::filesystem::path &path) {
#ifdef _WIN32
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot open snapshot");
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0 ||
        static_cast<unsigned long long>(size.QuadPart) > std::numeric_limits<std::size_t>::max()) {
        CloseHandle(file);
        throw std::runtime_error("invalid snapshot length");
    }
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        CloseHandle(file);
        throw std::runtime_error("cannot map snapshot");
    }
    auto view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(mapping);
        CloseHandle(file);
        throw std::runtime_error("cannot map view");
    }
    file_ = file;
    mapping_ = mapping;
    data_ = static_cast<const std::byte *>(view);
    size_ = static_cast<std::size_t>(size.QuadPart);
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw std::runtime_error("cannot open snapshot");
    struct stat info{};
    if (fstat(fd_, &info) != 0 || info.st_size <= 0 ||
        static_cast<std::uintmax_t>(info.st_size) > std::numeric_limits<std::size_t>::max()) {
        close(fd_);
        throw std::runtime_error("invalid snapshot length");
    }
    size_ = static_cast<std::size_t>(info.st_size);
    void *view = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (view == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("cannot map snapshot");
    }
    data_ = static_cast<const std::byte *>(view);
#endif
}
MappedFile::~MappedFile() {
#ifdef _WIN32
    if (data_)
        UnmapViewOfFile(data_);
    if (mapping_)
        CloseHandle(mapping_);
    if (file_)
        CloseHandle(file_);
#else
    if (data_)
        munmap(const_cast<std::byte *>(data_), size_);
    if (fd_ >= 0)
        close(fd_);
#endif
}
BlockCache::BlockCache(std::size_t budget, std::size_t block_bytes)
    : budget_(budget), block_bytes_(block_bytes) {
    if (block_bytes == 0 || block_bytes > (64u << 20))
        throw std::invalid_argument("block size must be 1..64MiB");
}
void BlockCache::read(const MappedFile &file, std::uint64_t offset, void *destination,
                      std::size_t bytes) const {
    if (offset > file.size() || bytes > file.size() - offset)
        throw std::runtime_error("read outside snapshot");
    auto *output = static_cast<std::byte *>(destination);
    std::lock_guard lock(mutex_);
    while (bytes) {
        auto block = offset / block_bytes_;
        auto start = static_cast<std::size_t>(block * block_bytes_);
        auto length = std::min(block_bytes_, file.size() - start);
        auto within = static_cast<std::size_t>(offset - start);
        auto take = std::min(bytes, length - within);
        auto found = entries_.find(block);
        if (found != entries_.end()) {
            ++stats_.hits;
            lru_.splice(lru_.begin(), lru_, found->second.position);
            std::memcpy(output, found->second.bytes.data() + within, take);
        } else {
            ++stats_.misses;
            auto begin = std::chrono::steady_clock::now();
            if (length > budget_) {
                std::memcpy(output, file.data() + offset, take);
                stats_.mapped_bytes_fetched += take;
            } else {
                while (stats_.resident_bytes > budget_ - length) {
                    auto id = lru_.back();
                    stats_.resident_bytes -= entries_.at(id).bytes.size();
                    entries_.erase(id);
                    lru_.pop_back();
                    ++stats_.evictions;
                }
                Entry entry;
                entry.bytes.resize(length);
                std::memcpy(entry.bytes.data(), file.data() + start, length);
                lru_.push_front(block);
                entry.position = lru_.begin();
                try {
                    found = entries_.emplace(block, std::move(entry)).first;
                } catch (...) {
                    lru_.pop_front();
                    throw;
                }
                stats_.resident_bytes += length;
                stats_.mapped_bytes_fetched += length;
                std::memcpy(output, found->second.bytes.data() + within, take);
            }
            stats_.miss_nanoseconds += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin)
                    .count());
        }
        stats_.bytes_served += take;
        offset += take;
        output += take;
        bytes -= take;
    }
}
CacheStats BlockCache::stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}
void BlockCache::clear() const {
    std::lock_guard lock(mutex_);
    entries_.clear();
    lru_.clear();
    stats_ = {};
}
void replace_file(const std::filesystem::path &source, const std::filesystem::path &destination) {
#ifdef _WIN32
    if (!MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                "cannot publish snapshot");
#else
    std::filesystem::rename(source, destination);
#endif
}
} // namespace vortex
