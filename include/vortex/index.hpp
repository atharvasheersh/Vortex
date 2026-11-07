#pragma once
#include "vortex/distance.hpp"
#include "vortex/storage.hpp"
#include "vortex/thread_pool.hpp"
#include <filesystem>
#include <memory>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

namespace vortex {
struct Config {
    std::size_t dimension;
    Metric metric = Metric::L2;
    std::size_t cache_bytes = 8 * 1024 * 1024;
    std::size_t block_bytes = 4096;
    std::size_t threads = 1;
    Kernel kernel = Kernel::Auto;
};
struct Hit {
    std::uint64_t id;
    float score;
    bool operator==(const Hit &) const = default;
};
struct SearchResult {
    std::vector<Hit> hits;
    std::size_t scanned = 0;
    double milliseconds = 0;
};
class Index {
  public:
    explicit Index(Config config);
    static std::unique_ptr<Index> open(const std::filesystem::path &path, std::size_t cache_bytes = 8388608,
                                       std::size_t threads = 1, Kernel kernel = Kernel::Auto,
                                       std::size_t block_bytes = 4096);
    void add(std::uint64_t id, std::span<const float> vector);
    void build_ivf(std::size_t nlist, std::size_t iterations = 12, std::uint64_t seed = 42,
                   std::size_t training_limit = 10000);
    void set_dtype(DType dtype);
    void save(const std::filesystem::path &path) const;
    SearchResult search(std::span<const float> query, std::size_t k = 10, std::size_t nprobe = 0,
                        std::size_t rerank_n = 0) const;
    std::vector<SearchResult> search_batch(std::span<const float> queries, std::size_t count,
                                           std::size_t k = 10, std::size_t nprobe = 0,
                                           std::size_t rerank_n = 0) const;
    CacheStats cache_stats() const { return cache_.stats(); }
    void clear_cache() const { cache_.clear(); }
    std::size_t size() const;
    std::size_t dimension() const { return config_.dimension; }
    std::size_t nlist() const;
    DType dtype() const;
    Metric metric() const { return config_.metric; }
    bool mapped() const { return static_cast<bool>(mapped_); }

  private:
    SearchResult search_locked(std::span<const float>, std::size_t, std::size_t, std::size_t) const;
    const float *vector_at(std::size_t row, bool original, std::vector<float> &scratch) const;
    void ensure_mutable() const;
    Config config_;
    mutable std::shared_mutex mutex_;
    mutable ThreadPool pool_;
    mutable BlockCache cache_;
    DType dtype_ = DType::FP32;
    std::vector<std::uint64_t> ids_;
    std::unordered_set<std::uint64_t> unique_ids_;
    std::vector<float> vectors_;
    std::vector<std::byte> encoded_;
    std::vector<float> centroids_;
    std::vector<std::vector<std::uint64_t>> lists_;
    bool stale_ = false;
    std::unique_ptr<MappedFile> mapped_;
    std::uint64_t original_offset_ = 0, encoded_offset_ = 0;
};
} // namespace vortex
