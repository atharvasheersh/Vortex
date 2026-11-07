#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vortex {
class MappedFile {
  public:
    explicit MappedFile(const std::filesystem::path &path);
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;
    const std::byte *data() const { return data_; }
    std::size_t size() const { return size_; }

  private:
    const std::byte *data_ = nullptr;
    std::size_t size_ = 0;
    void *file_ = nullptr;
    void *mapping_ = nullptr;
    int fd_ = -1;
};
struct CacheStats {
    std::uint64_t hits = 0, misses = 0, evictions = 0;
    std::uint64_t mapped_bytes_fetched = 0, bytes_served = 0;
    std::uint64_t miss_nanoseconds = 0;
    std::size_t resident_bytes = 0;
};
class BlockCache {
  public:
    BlockCache(std::size_t budget, std::size_t block_bytes);
    void read(const MappedFile &file, std::uint64_t offset, void *destination, std::size_t bytes) const;
    CacheStats stats() const;
    void clear() const;

  private:
    struct Entry {
        std::vector<std::byte> bytes;
        std::list<std::uint64_t>::iterator position;
    };
    std::size_t budget_, block_bytes_;
    mutable std::mutex mutex_;
    mutable CacheStats stats_;
    mutable std::list<std::uint64_t> lru_;
    mutable std::unordered_map<std::uint64_t, Entry> entries_;
};
void replace_file(const std::filesystem::path &source, const std::filesystem::path &destination);
} // namespace vortex
