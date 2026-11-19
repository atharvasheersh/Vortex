#include "vortex/index.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <queue>
#include <random>

namespace {
volatile double sink = 0;
template <class F> void measure(const char *name, F action) {
    auto begin = std::chrono::steady_clock::now();
    action();
    std::cout << name << ','
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count()
              << '\n';
}
}
int main(int argc, char **argv) {
    try {
        using namespace vortex;
        const std::size_t n = argc > 1 ? std::stoull(argv[1]) : 100000;
        constexpr std::size_t d = 64, nq = 12;
        if (n < 32 || n > 1000000) throw std::invalid_argument("rows must be 32..1000000");
        auto path = std::filesystem::temp_directory_path() /
                    ("vortex-profile-" + std::to_string(std::random_device{}()) + ".vtx");
        std::mt19937 rng(42);
        std::normal_distribution<float> normal;
        std::vector<float> data(n * d), queries(nq * d);
        for (auto &v : data) v = normal(rng);
        for (auto &v : queries) v = normal(rng);
        Index index({d, Metric::L2, 8388608, 4096, 4});
        for (std::size_t row = 0; row < n; ++row) index.add(row, std::span(data).subspan(row*d,d));
        index.build_ivf(32, 4, 42, std::min<std::size_t>(n, 4096));
        index.save(path);
        std::cout << "scope,milliseconds\n";
        auto run = [&](const Index &collection, std::size_t probes) {
            for (std::size_t q = 0; q < nq; ++q)
                sink = collection.search(std::span(queries).subspan(q*d,d),10,probes).hits[0].score;
        };
        measure("resident_exact_12_queries", [&] { run(index,0); });
        measure("resident_ivf24_12_queries", [&] { run(index,24); });
        for (auto type : {DType::FP16,DType::INT8}) {
            index.set_dtype(type);
            measure(type == DType::FP16 ? "resident_fp16_exact_12_queries" : "resident_int8_exact_12_queries",
                    [&] { run(index,0); });
        }
        index.set_dtype(DType::FP32);
        {
            auto mapped = Index::open(path,8388608,4);
            run(*mapped,0);
            mapped->clear_cache();
            measure("mapped_exact_12_queries", [&] { run(*mapped,0); });
            measure("mapped_ivf24_12_queries", [&] { run(*mapped,24); });
            measure("mapped_batch12_4_workers", [&] { sink = mapped->search_batch(queries,nq)[0].hits[0].score; });
        }
        for (auto kernel : {Kernel::Scalar,Kernel::Auto})
            measure(kernel == Kernel::Scalar ? "distance_scalar_one_pass" : "distance_auto_one_pass", [&] {
                double total = 0;
                for (std::size_t row = 0; row < n; ++row)
                    total += distance(queries.data(),data.data()+row*d,d,Metric::L2,kernel);
                sink = total;
            });
        measure("finite_validation_one_pass", [&] {
            std::size_t valid=0;
            for (auto v : data) valid += std::isfinite(v);
            sink = static_cast<double>(valid);
        });
        measure("topk_heap_one_pass", [&] {
            std::priority_queue<float> heap;
            for (std::size_t row=0; row<n; ++row) {
                const auto value=data[row*d];
                if (heap.size()<10) heap.push(value);
                else if (value<heap.top()) { heap.pop(); heap.push(value); }
            }
            sink=heap.top();
        });
        {
            MappedFile file(path);
            BlockCache cache(8388608,4096);
            std::vector<float> buffer(256*d);
            measure("cache_one_read_per_vector", [&] {
                for (std::size_t row=0;row<n;++row) {
                    cache.read(file,128+n*8+row*d*4,buffer.data(),d*4);
                    sink=buffer[0];
                }
            });
            cache.clear();
            measure("cache_one_read_per_256_vectors", [&] {
                for (std::size_t row=0;row<n;row+=256) {
                    cache.read(file,128+n*8+row*d*4,buffer.data(),std::min<std::size_t>(256,n-row)*d*4);
                    sink=buffer[0];
                }
            });
        }
        std::filesystem::remove(path);
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
