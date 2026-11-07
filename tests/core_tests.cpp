#include "vortex/index.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
int checks = 0;
void require(bool condition, const char *message) {
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> void rejects(F action) {
    bool caught = false;
    try {
        action();
    } catch (const std::exception &) {
        caught = true;
    }
    require(caught, "expected rejection");
}
bool close(float a, float b) { return std::abs(a - b) <= 2e-5f * std::max(1.0f, std::abs(a)); }
} // namespace
int main() {
    auto directory =
        std::filesystem::temp_directory_path() / ("vortex-test-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(directory);
    try {
        using namespace vortex;
        std::mt19937 rng(9);
        std::uniform_real_distribution<float> random(-2, 2);
        for (auto d : {1u, 7u, 8u, 9u, 31u, 128u, 769u}) {
            std::vector<float> a(d), b(d);
            for (auto &v : a)
                v = random(rng);
            for (auto &v : b)
                v = random(rng);
            for (auto metric : {Metric::L2, Metric::InnerProduct, Metric::Cosine})
                require(close(distance(a.data(), b.data(), d, metric, Kernel::Scalar),
                              distance(a.data(), b.data(), d, metric, Kernel::Auto)),
                        "kernel mismatch");
        }
        for (unsigned bits = 0; bits < 65536; ++bits) {
            if (((bits >> 10) & 31) != 31)
                require(to_half(from_half(static_cast<std::uint16_t>(bits))) == bits, "FP16 round trip");
        }
        rejects([] { (void)to_half(70000.0f); });
        rejects([] { Index invalid({0}); });
        rejects([] { Index invalid({3, Metric::L2, 0, 0}); });
        Index ties({2});
        std::vector<float> v{1, 0};
        require(ties.search(v).hits.empty(), "empty exact search");
        ties.add(9, v);
        ties.add(2, v);
        ties.add(7, v);
        auto h = ties.search(v, 10).hits;
        require(h.size() == 3 && h[0].id == 2 && h[1].id == 7 && h[2].id == 9, "stable ties");
        rejects([&] { ties.add(2, v); });
        rejects([&] { ties.search(v, 0); });
        rejects([&] { ties.search(v, 10, 1); });
        rejects([&] { ties.search(v, 10, 0, 3); });
        std::vector<float> bad{std::numeric_limits<float>::quiet_NaN(), 0};
        rejects([&] { ties.add(1, bad); });
        Index cosine({2, Metric::Cosine});
        std::vector<float> zero{0, 0};
        rejects([&] { cosine.add(1, zero); });
        cosine.add(1, v);
        require(close(cosine.search(v).hits[0].score, 1), "cosine score");
        auto file = directory / "index.vtx";
        constexpr std::size_t n = 257, d = 17;
        Index index({d, Metric::L2, 4096, 128, 4});
        std::vector<float> matrix(n * d), queries(32 * d);
        for (auto &x : matrix)
            x = random(rng);
        for (auto &x : queries)
            x = random(rng);
        for (std::size_t i = 0; i < n; ++i)
            index.add(1000 + i, std::span(matrix).subspan(i * d, d));
        auto exact = index.search_batch(queries, 32, 10);
        index.build_ivf(13, 6, 42, n);
        for (std::size_t q = 0; q < 32; ++q)
            require(index.search(std::span(queries).subspan(q * d, d), 10, 13).hits == exact[q].hits,
                    "all-list exact agreement");
        index.save(file);
        {
            auto mapped = Index::open(file, 256, 4, Kernel::Auto, 128);
            require(mapped->mapped() && mapped->size() == n && mapped->nlist() == 13, "mapped metadata");
            auto results = mapped->search_batch(queries, 32, 10, 13);
            for (std::size_t q = 0; q < 32; ++q)
                require(results[q].hits == exact[q].hits, "reopen exact agreement");
            auto stats = mapped->cache_stats();
            require(stats.resident_bytes <= 256 && stats.misses > 0 && stats.evictions > 0, "cache bound");
            rejects([&] { mapped->add(99999, std::span(matrix).first(d)); });
            std::vector<std::future<void>> futures;
            for (int t = 0; t < 8; ++t)
                futures.push_back(std::async(std::launch::async, [&] {
                    auto out = mapped->search_batch(queries, 32);
                    if (out[0].hits != exact[0].hits)
                        throw std::runtime_error("concurrent mismatch");
                }));
            for (auto &future : futures)
                future.get();
            mapped->clear_cache();
            require(mapped->cache_stats().hits == 0, "cache reset");
            auto malformed = queries;
            malformed[0] = std::numeric_limits<float>::infinity();
            rejects([&] { mapped->search_batch(malformed, 32); });
            require(mapped->search_batch(queries, 32)[0].hits == exact[0].hits, "pool survives task error");
            // Windows may reject replacement while a view is mapped; preserve the old snapshot.
#ifdef _WIN32
            rejects([&] { index.save(file); });
#else
            index.save(file);
#endif
            require(mapped->search(std::span(queries).first(d)).hits == exact[0].hits,
                    "mapping survives replace");
        }
        for (auto type : {DType::FP16, DType::INT8}) {
            index.set_dtype(type);
            index.save(file);
            auto mapped = Index::open(file, 0);
            auto results = mapped->search_batch(queries, 32, 10, 13, n);
            for (std::size_t q = 0; q < 32; ++q)
                require(results[q].hits == exact[q].hits, "full rerank exact agreement");
            require(mapped->cache_stats().resident_bytes == 0, "zero cache budget");
            rejects([&] { index.add(9999, std::span(matrix).first(d)); });
        }
        index.set_dtype(DType::FP32);
        index.add(9999, std::span(matrix).first(d));
        rejects([&] { index.search(std::span(queries).first(d), 10, 1); });
        rejects([&] { index.save(file); });
        require(!index.search(std::span(queries).first(d)).hits.empty(), "flat fallback for stale IVF");
        index.build_ivf(13, 2);
        index.save(file);
        {
            std::fstream corrupt(file, std::ios::in | std::ios::out | std::ios::binary);
            corrupt.seekp(12);
            char invalid = 0;
            corrupt.write(&invalid, 1);
        }
        rejects([&] { Index::open(file); });
        {
            std::ofstream tiny(file, std::ios::binary | std::ios::trunc);
            tiny << "bad";
        }
        rejects([&] { Index::open(file); });
        std::filesystem::remove_all(directory);
        std::cout << checks << " assertions passed; AVX2 available=" << avx2_available() << '\n';
    } catch (const std::exception &error) {
        std::cerr << "FAILED after " << checks << " assertions: " << error.what() << '\n';
        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
        return 1;
    }
}
