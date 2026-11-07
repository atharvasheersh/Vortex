#include "vortex/index.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

int main(int argc, char **argv) {
    try {
        std::size_t n = argc > 1 ? std::stoull(argv[1]) : 10000;
        std::size_t d = argc > 2 ? std::stoull(argv[2]) : 128;
        std::size_t nq = argc > 3 ? std::stoull(argv[3]) : 100;
        if (!n || !d || !nq || d > 1048576 || n > 10000000 || nq > 1000000)
            throw std::invalid_argument("invalid benchmark sizes");
        std::mt19937 rng(42);
        std::normal_distribution<float> normal;
        std::vector<float> data(n * d), queries(nq * d);
        for (auto &v : data)
            v = normal(rng);
        for (auto &v : queries)
            v = normal(rng);
        std::cout << "kernel,vectors,dimension,queries,p50_ms,p95_ms,p99_ms,qps,avx2\n";
        for (auto kernel : {vortex::Kernel::Scalar, vortex::Kernel::Auto}) {
            vortex::Index index({d, vortex::Metric::L2, 0, 4096, 1, kernel});
            for (std::size_t i = 0; i < n; ++i)
                index.add(i, std::span(data).subspan(i * d, d));
            for (std::size_t i = 0; i < std::min<std::size_t>(5, nq); ++i)
                (void)index.search(std::span(queries).subspan(i * d, d));
            std::vector<double> timings;
            auto begin = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < nq; ++i)
                timings.push_back(index.search(std::span(queries).subspan(i * d, d)).milliseconds);
            auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            std::sort(timings.begin(), timings.end());
            auto p = [&](double q) {
                return timings[static_cast<std::size_t>(q * static_cast<double>(nq - 1))];
            };
            std::cout << (kernel == vortex::Kernel::Scalar ? "scalar" : "auto") << ',' << n << ',' << d << ','
                      << nq << ',' << p(.50) << ',' << p(.95) << ',' << p(.99) << ','
                      << static_cast<double>(nq) / seconds << ',' << vortex::avx2_available() << '\n';
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
