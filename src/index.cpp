#include "vortex/index.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>

namespace vortex {
namespace {
constexpr std::uint64_t header_size = 128;
std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
        throw std::runtime_error("size overflow");
    return a * b;
}
std::uint64_t add_size(std::uint64_t a, std::uint64_t b) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b)
        throw std::runtime_error("offset overflow");
    return a + b;
}
std::uint64_t align8(std::uint64_t n) { return add_size(n, 7) & ~std::uint64_t{7}; }
std::size_t stride(DType type, std::size_t dimension) {
    if (type == DType::FP16)
        return dimension * 2;
    if (type == DType::INT8)
        return dimension + 4;
    return dimension * 4;
}
template <class T> T load(const std::byte *data) {
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}
template <class T> void put(std::byte *data, T value) { std::memcpy(data, &value, sizeof(T)); }
std::uint64_t hash_header(const std::byte *p) {
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < 96; ++i) {
        hash ^= std::to_integer<unsigned char>(p[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}
struct Layout {
    std::uint64_t ids = header_size, original, encoded, centroids, offsets, postings, total;
    Layout(std::uint64_t n, std::uint64_t d, DType type, std::uint64_t clusters) {
        original = add_size(ids, multiply(n, 8));
        encoded = add_size(original, multiply(multiply(n, d), 4));
        auto encoded_bytes = type == DType::FP32 ? 0 : multiply(n, stride(type, static_cast<std::size_t>(d)));
        centroids = align8(add_size(encoded, encoded_bytes));
        offsets = add_size(centroids, multiply(multiply(clusters, d), 4));
        postings = add_size(offsets, multiply(add_size(clusters, 1), 8));
        total = add_size(postings, clusters ? multiply(n, 8) : 0);
        if (total > std::numeric_limits<std::size_t>::max() ||
            total > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("snapshot too large for platform");
    }
};
struct Candidate {
    float distance;
    std::uint64_t id;
    std::size_t row;
};
bool better(const Candidate &a, const Candidate &b) {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}
using Heap = std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)>;
void retain(Heap &heap, Candidate candidate, std::size_t limit) {
    if (heap.size() < limit)
        heap.push(candidate);
    else if (better(candidate, heap.top())) {
        heap.pop();
        heap.push(candidate);
    }
}
void require_little_endian() {
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("little-endian platform required");
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
}
} // namespace

Index::Index(Config config)
    : config_(config), pool_(config.threads), cache_(config.cache_bytes, config.block_bytes) {
    require_little_endian();
    if (config.dimension == 0 || config.dimension > (1u << 20))
        throw std::invalid_argument("dimension must be 1..1048576");
    if (static_cast<unsigned>(config.metric) > 2)
        throw std::invalid_argument("unknown metric");
}
std::size_t Index::size() const {
    std::shared_lock lock(mutex_);
    return ids_.size();
}
std::size_t Index::nlist() const {
    std::shared_lock lock(mutex_);
    return stale_ ? 0 : lists_.size();
}
DType Index::dtype() const {
    std::shared_lock lock(mutex_);
    return dtype_;
}
void Index::ensure_mutable() const {
    if (mapped_)
        throw std::logic_error("opened snapshots are read-only; use a new builder");
}
void Index::add(std::uint64_t id, std::span<const float> vector) {
    auto prepared = prepare(vector, config_.dimension, config_.metric);
    std::unique_lock lock(mutex_);
    ensure_mutable();
    if (dtype_ != DType::FP32)
        throw std::logic_error("reset dtype to FP32 before inserting");
    if (unique_ids_.contains(id))
        throw std::invalid_argument("duplicate vector ID");
    // Geometric growth avoids a full matrix copy on every append.
    if (vectors_.capacity() - vectors_.size() < prepared.size())
        vectors_.reserve(std::max(add_size(vectors_.size(), prepared.size()),
                                  multiply(std::max<std::size_t>(1, vectors_.capacity()), 2)));
    if (ids_.size() == ids_.capacity())
        ids_.reserve(std::max<std::size_t>(1, ids_.capacity() * 2));
    unique_ids_.insert(id);
    vectors_.insert(vectors_.end(), prepared.begin(), prepared.end());
    ids_.push_back(id);
    if (!lists_.empty())
        stale_ = true;
}
const float *Index::vector_at(std::size_t row, bool original, std::vector<float> &scratch) const {
    auto d = config_.dimension;
    if (!mapped_ && (original || dtype_ == DType::FP32))
        return vectors_.data() + row * d;
    scratch.resize(d);
    if (original || dtype_ == DType::FP32) {
        cache_.read(*mapped_, original_offset_ + row * d * 4, scratch.data(), d * 4);
    } else {
        std::vector<std::byte> bytes(stride(dtype_, d));
        if (mapped_)
            cache_.read(*mapped_, encoded_offset_ + row * bytes.size(), bytes.data(), bytes.size());
        else
            std::memcpy(bytes.data(), encoded_.data() + row * bytes.size(), bytes.size());
        if (dtype_ == DType::FP16) {
            for (std::size_t j = 0; j < d; ++j)
                scratch[j] = from_half(load<std::uint16_t>(bytes.data() + j * 2));
        } else {
            auto scale = load<float>(bytes.data());
            if (!std::isfinite(scale) || scale <= 0)
                throw std::runtime_error("invalid INT8 scale");
            for (std::size_t j = 0; j < d; ++j)
                scratch[j] = scale * load<std::int8_t>(bytes.data() + 4 + j);
        }
    }
    for (float v : scratch)
        if (!std::isfinite(v))
            throw std::runtime_error("non-finite snapshot vector");
    return scratch.data();
}
void Index::set_dtype(DType type) {
    std::unique_lock lock(mutex_);
    ensure_mutable();
    if (static_cast<unsigned>(type) > 2)
        throw std::invalid_argument("unknown representation");
    std::vector<std::byte> encoded;
    if (type != DType::FP32) {
        auto d = config_.dimension;
        auto bytes = stride(type, d);
        encoded.resize(multiply(ids_.size(), bytes));
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            auto *out = encoded.data() + i * bytes;
            auto *v = vectors_.data() + i * d;
            if (type == DType::FP16) {
                for (std::size_t j = 0; j < d; ++j)
                    put(out + j * 2, to_half(v[j]));
            } else {
                float maximum = 0;
                for (std::size_t j = 0; j < d; ++j)
                    maximum = std::max(maximum, std::abs(v[j]));
                float scale =
                    maximum == 0 ? 1.0f : std::max(maximum / 127.0f, std::numeric_limits<float>::min());
                put(out, scale);
                for (std::size_t j = 0; j < d; ++j)
                    put(out + 4 + j,
                        static_cast<std::int8_t>(std::clamp(std::round(v[j] / scale), -127.0f, 127.0f)));
            }
        }
    }
    encoded_ = std::move(encoded);
    dtype_ = type;
}
void Index::build_ivf(std::size_t clusters, std::size_t iterations, std::uint64_t seed,
                      std::size_t training_limit) {
    std::unique_lock lock(mutex_);
    ensure_mutable();
    auto n = ids_.size(), d = config_.dimension;
    if (clusters == 0 || clusters > n || iterations == 0 || training_limit < clusters)
        throw std::invalid_argument("require 1 <= nlist <= count, iterations > 0, training_limit >= nlist");
    auto samples = std::min(n, training_limit);
    std::vector<std::size_t> rows(n);
    std::iota(rows.begin(), rows.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(rows.begin(), rows.end(), rng);
    std::vector<float> centers(clusters * d);
    for (std::size_t c = 0; c < clusters; ++c)
        std::copy_n(vectors_.data() + rows[c] * d, d, centers.data() + c * d);
    auto nearest = [&](const float *vector) {
        std::size_t best = 0;
        float best_score = std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < clusters; ++c) {
            auto score = distance(vector, centers.data() + c * d, d, Metric::L2, config_.kernel);
            if (score < best_score) {
                best_score = score;
                best = c;
            }
        }
        return best;
    };
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        std::vector<double> sums(clusters * d, 0);
        std::vector<std::size_t> counts(clusters, 0);
        for (std::size_t s = 0; s < samples; ++s) {
            auto *v = vectors_.data() + rows[s] * d;
            auto c = nearest(v);
            ++counts[c];
            for (std::size_t j = 0; j < d; ++j)
                sums[c * d + j] += v[j];
        }
        for (std::size_t c = 0; c < clusters; ++c) {
            if (counts[c] == 0) {
                std::copy_n(vectors_.data() + rows[(iteration + c) % samples] * d, d, centers.data() + c * d);
            } else {
                for (std::size_t j = 0; j < d; ++j)
                    centers[c * d + j] = static_cast<float>(sums[c * d + j] / static_cast<double>(counts[c]));
            }
        }
    }
    std::vector<std::vector<std::uint64_t>> lists(clusters);
    for (std::size_t row = 0; row < n; ++row)
        lists[nearest(vectors_.data() + row * d)].push_back(row);
    centroids_ = std::move(centers);
    lists_ = std::move(lists);
    stale_ = false;
}
SearchResult Index::search_locked(std::span<const float> query, std::size_t k, std::size_t nprobe,
                                  std::size_t rerank_n) const {
    auto start = std::chrono::steady_clock::now();
    auto q = prepare(query, config_.dimension, config_.metric);
    if (k == 0)
        throw std::invalid_argument("k must be positive");
    if (rerank_n && rerank_n < k)
        throw std::invalid_argument("rerank_n must be zero or at least k");
    if (nprobe && (stale_ || lists_.empty()))
        throw std::logic_error("IVF missing or stale; rebuild first");
    if (nprobe > lists_.size())
        throw std::invalid_argument("nprobe exceeds nlist");
    SearchResult result;
    if (ids_.empty())
        return result;
    auto limit = std::min(ids_.size(), rerank_n ? rerank_n : k);
    Heap heap(&better);
    std::vector<float> scratch;
    auto scan = [&](std::size_t row) {
        auto *v = vector_at(row, false, scratch);
        auto score = distance(q.data(), v, q.size(), config_.metric, config_.kernel);
        retain(heap, {score, ids_[row], row}, limit);
        ++result.scanned;
    };
    if (nprobe == 0) {
        for (std::size_t row = 0; row < ids_.size(); ++row)
            scan(row);
    } else {
        std::vector<Candidate> clusters;
        for (std::size_t c = 0; c < lists_.size(); ++c)
            clusters.push_back({distance(q.data(), centroids_.data() + c * q.size(), q.size(), config_.metric,
                                         config_.kernel),
                                c, c});
        std::partial_sort(clusters.begin(), clusters.begin() + static_cast<std::ptrdiff_t>(nprobe),
                          clusters.end(), better);
        for (std::size_t i = 0; i < nprobe; ++i)
            for (auto row : lists_[clusters[i].row])
                scan(static_cast<std::size_t>(row));
    }
    std::vector<Candidate> candidates;
    while (!heap.empty()) {
        candidates.push_back(heap.top());
        heap.pop();
    }
    if (rerank_n)
        for (auto &candidate : candidates)
            candidate.distance = distance(q.data(), vector_at(candidate.row, true, scratch), q.size(),
                                          config_.metric, config_.kernel);
    std::sort(candidates.begin(), candidates.end(), better);
    for (std::size_t i = 0; i < std::min(k, candidates.size()); ++i)
        result.hits.push_back({candidates[i].id, config_.metric == Metric::L2 ? candidates[i].distance
                                                                              : -candidates[i].distance});
    result.milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}
SearchResult Index::search(std::span<const float> q, std::size_t k, std::size_t nprobe,
                           std::size_t rerank_n) const {
    std::shared_lock lock(mutex_);
    return search_locked(q, k, nprobe, rerank_n);
}
std::vector<SearchResult> Index::search_batch(std::span<const float> queries, std::size_t count,
                                              std::size_t k, std::size_t nprobe, std::size_t rerank_n) const {
    if (multiply(count, config_.dimension) != queries.size())
        throw std::invalid_argument("batch shape mismatch");
    // A shared lock prevents mutation until all submitted work and all failure paths finish.
    std::shared_lock lock(mutex_);
    std::vector<SearchResult> results(count);
    std::vector<std::future<void>> futures;
    std::exception_ptr failure;
    try {
        auto tasks = std::min(count, config_.threads);
        futures.reserve(tasks);
        for (std::size_t worker = 0; worker < tasks; ++worker)
            futures.push_back(pool_.submit([&, worker, tasks] {
                for (std::size_t i = worker; i < count; i += tasks)
                    results[i] = search_locked(queries.subspan(i * config_.dimension, config_.dimension), k,
                                               nprobe, rerank_n);
            }));
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto &future : futures) {
        try {
            future.get();
        } catch (...) {
            if (!failure)
                failure = std::current_exception();
        }
    }
    if (failure)
        std::rethrow_exception(failure);
    return results;
}
void Index::save(const std::filesystem::path &path) const {
    std::shared_lock lock(mutex_);
    if (stale_)
        throw std::logic_error("cannot save stale IVF; rebuild first");
    if (path.empty())
        throw std::invalid_argument("snapshot path required");
    auto parent = path.parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    auto temporary = path;
    temporary += ".tmp." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "." +
                 std::to_string(std::random_device{}());
    try {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        if (mapped_) {
            file.write(reinterpret_cast<const char *>(mapped_->data()),
                       static_cast<std::streamsize>(mapped_->size()));
        } else {
            Layout layout(ids_.size(), config_.dimension, dtype_, lists_.size());
            std::array<std::byte, header_size> header{};
            std::memcpy(header.data(), "VORTEX01", 8);
            put<std::uint32_t>(header.data() + 8, 1);
            put<std::uint32_t>(header.data() + 12, static_cast<std::uint32_t>(config_.dimension));
            put(header.data() + 16, static_cast<std::uint32_t>(config_.metric));
            put(header.data() + 20, static_cast<std::uint32_t>(dtype_));
            put(header.data() + 24, static_cast<std::uint64_t>(ids_.size()));
            put(header.data() + 32, static_cast<std::uint64_t>(lists_.size()));
            std::array<std::uint64_t, 7> positions{layout.ids,       layout.original, layout.encoded,
                                                   layout.centroids, layout.offsets,  layout.postings,
                                                   layout.total};
            for (std::size_t i = 0; i < positions.size(); ++i)
                put(header.data() + 40 + i * 8, positions[i]);
            put(header.data() + 96, hash_header(header.data()));
            file.write(reinterpret_cast<const char *>(header.data()), header.size());
            auto write = [&](const void *data, std::size_t bytes) {
                if (bytes)
                    file.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
            };
            write(ids_.data(), ids_.size() * 8);
            write(vectors_.data(), vectors_.size() * 4);
            write(encoded_.data(), encoded_.size());
            auto padding = layout.centroids - (layout.encoded + encoded_.size());
            const std::array<char, 8> zeros{};
            write(zeros.data(), static_cast<std::size_t>(padding));
            write(centroids_.data(), centroids_.size() * 4);
            std::uint64_t offset = 0;
            write(&offset, 8);
            for (const auto &list : lists_) {
                offset += list.size();
                write(&offset, 8);
            }
            for (const auto &list : lists_)
                write(list.data(), list.size() * 8);
        }
        file.flush();
        file.close();
        replace_file(temporary, path);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(temporary, ec);
        throw;
    }
}
std::unique_ptr<Index> Index::open(const std::filesystem::path &path, std::size_t cache_bytes,
                                   std::size_t threads, Kernel kernel, std::size_t block_bytes) {
    require_little_endian();
    auto mapped = std::make_unique<MappedFile>(path);
    auto *p = mapped->data();
    if (mapped->size() < header_size || std::memcmp(p, "VORTEX01", 8) || load<std::uint32_t>(p + 8) != 1)
        throw std::runtime_error("invalid snapshot magic, length, or version");
    if (load<std::uint64_t>(p + 96) != hash_header(p))
        throw std::runtime_error("header checksum mismatch");
    for (std::size_t i = 104; i < header_size; ++i)
        if (p[i] != std::byte{0})
            throw std::runtime_error("reserved header fields must be zero");
    auto d = load<std::uint32_t>(p + 12), metric = load<std::uint32_t>(p + 16),
         type = load<std::uint32_t>(p + 20);
    auto n = load<std::uint64_t>(p + 24), clusters = load<std::uint64_t>(p + 32);
    if (d == 0 || d > (1u << 20) || metric > 2 || type > 2 || clusters > n)
        throw std::runtime_error("invalid snapshot configuration");
    Layout layout(n, d, static_cast<DType>(type), clusters);
    std::array<std::uint64_t, 7> positions{layout.ids,     layout.original, layout.encoded, layout.centroids,
                                           layout.offsets, layout.postings, layout.total};
    for (std::size_t i = 0; i < positions.size(); ++i)
        if (load<std::uint64_t>(p + 40 + i * 8) != positions[i])
            throw std::runtime_error("invalid snapshot section offsets");
    if (layout.total != mapped->size())
        throw std::runtime_error("snapshot size mismatch");
    auto index = std::make_unique<Index>(
        Config{d, static_cast<Metric>(metric), cache_bytes, block_bytes, threads, kernel});
    index->dtype_ = static_cast<DType>(type);
    index->ids_.resize(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < n; ++i) {
        auto id = load<std::uint64_t>(p + layout.ids + i * 8);
        if (!index->unique_ids_.insert(id).second)
            throw std::runtime_error("duplicate snapshot ID");
        index->ids_[i] = id;
    }
    index->centroids_.resize(static_cast<std::size_t>(clusters * d));
    for (std::size_t i = 0; i < index->centroids_.size(); ++i) {
        auto v = load<float>(p + layout.centroids + i * 4);
        if (!std::isfinite(v))
            throw std::runtime_error("invalid centroid");
        index->centroids_[i] = v;
    }
    index->lists_.resize(static_cast<std::size_t>(clusters));
    std::vector<bool> seen(static_cast<std::size_t>(clusters ? n : 0), false);
    std::uint64_t last = 0;
    if (load<std::uint64_t>(p + layout.offsets) != 0)
        throw std::runtime_error("invalid first list offset");
    for (std::size_t c = 0; c < clusters; ++c) {
        auto end = load<std::uint64_t>(p + layout.offsets + (c + 1) * 8);
        if (end < last || end > n)
            throw std::runtime_error("invalid IVF offsets");
        auto &list = index->lists_[c];
        list.reserve(static_cast<std::size_t>(end - last));
        for (auto i = last; i < end; ++i) {
            auto row = load<std::uint64_t>(p + layout.postings + i * 8);
            if (row >= n || seen[static_cast<std::size_t>(row)])
                throw std::runtime_error("invalid or duplicate IVF row");
            seen[static_cast<std::size_t>(row)] = true;
            list.push_back(row);
        }
        last = end;
    }
    if (clusters && last != n)
        throw std::runtime_error("IVF coverage mismatch");
    index->original_offset_ = layout.original;
    index->encoded_offset_ = layout.encoded;
    index->mapped_ = std::move(mapped);
    return index;
}
} // namespace vortex
