#include "vortex/index.hpp"
#include <array>
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace vortex;
namespace {
Metric metric_from(const std::string &metric) {
    if (metric == "l2")
        return Metric::L2;
    if (metric == "ip")
        return Metric::InnerProduct;
    if (metric == "cosine")
        return Metric::Cosine;
    throw std::invalid_argument("metric must be l2, ip, or cosine");
}
Kernel kernel_from(const std::string &kernel) {
    if (kernel == "auto")
        return Kernel::Auto;
    if (kernel == "scalar")
        return Kernel::Scalar;
    throw std::invalid_argument("kernel must be auto or scalar");
}
using Array = py::array_t<float, py::array::c_style>;
std::filesystem::path utf8_path(const std::string &text) {
    std::u8string encoded(text.size(), u8'\0');
    if (!text.empty())
        std::memcpy(encoded.data(), text.data(), text.size());
    return std::filesystem::path(encoded);
}
std::vector<float> copy_array(const Array &array, int ndim, std::size_t dimension) {
    if (array.ndim() != ndim || array.shape(ndim - 1) != static_cast<py::ssize_t>(dimension))
        throw std::invalid_argument("expected contiguous float32 array with matching last dimension");
    std::vector<float> copy(static_cast<std::size_t>(array.size()));
    if (!copy.empty())
        std::memcpy(copy.data(), array.data(), copy.size() * sizeof(float));
    return copy;
}
py::dict result_dict(const SearchResult &result) {
    py::list hits;
    for (const auto &hit : result.hits)
        hits.append(py::make_tuple(hit.id, hit.score));
    py::dict out;
    out["hits"] = hits;
    out["scanned"] = result.scanned;
    out["milliseconds"] = result.milliseconds;
    return out;
}
} // namespace
PYBIND11_MODULE(_vortex, m) {
    m.doc() = "Native vector retrieval engine";
    m.attr("__version__") = "0.1.0";
    m.def("avx2_available", &avx2_available);
    py::class_<Index>(m, "Index")
        .def(py::init([](std::size_t dimension, const std::string &metric, std::size_t cache_bytes,
                         std::size_t threads, const std::string &kernel, std::size_t block_bytes) {
                 return std::make_unique<Index>(Config{dimension, metric_from(metric), cache_bytes,
                                                       block_bytes, threads, kernel_from(kernel)});
             }),
             py::arg("dimension"), py::arg("metric") = "l2", py::arg("cache_bytes") = 8388608,
             py::arg("threads") = 1, py::arg("kernel") = "auto", py::arg("block_bytes") = 4096)
        .def_static(
            "open",
            [](const std::string &path, std::size_t cache_bytes, std::size_t threads,
               const std::string &kernel, std::size_t block_bytes) {
                auto selected = kernel_from(kernel);
                py::gil_scoped_release release;
                return Index::open(utf8_path(path), cache_bytes, threads, selected, block_bytes);
            },
            py::arg("path"), py::arg("cache_bytes") = 8388608, py::arg("threads") = 1,
            py::arg("kernel") = "auto", py::arg("block_bytes") = 4096)
        .def(
            "add",
            [](Index &index, std::uint64_t id, const Array &vector) {
                auto copy = copy_array(vector, 1, index.dimension());
                py::gil_scoped_release release;
                index.add(id, copy);
            },
            py::arg("id"), py::arg("vector").noconvert())
        .def("build_ivf", &Index::build_ivf, py::arg("nlist"), py::arg("iterations") = 12,
             py::arg("seed") = 42, py::arg("training_limit") = 10000,
             py::call_guard<py::gil_scoped_release>())
        .def("set_dtype",
             [](Index &index, const std::string &name) {
                 DType dtype;
                 if (name == "fp32")
                     dtype = DType::FP32;
                 else if (name == "fp16")
                     dtype = DType::FP16;
                 else if (name == "int8")
                     dtype = DType::INT8;
                 else
                     throw std::invalid_argument("dtype must be fp32, fp16, or int8");
                 py::gil_scoped_release release;
                 index.set_dtype(dtype);
             })
        .def("save",
             [](const Index &index, const std::string &path) {
                 py::gil_scoped_release release;
                 index.save(utf8_path(path));
             })
        .def(
            "search",
            [](const Index &index, const Array &query, std::size_t k, std::size_t nprobe,
               std::size_t rerank_n) {
                auto copy = copy_array(query, 1, index.dimension());
                SearchResult result;
                {
                    py::gil_scoped_release release;
                    result = index.search(copy, k, nprobe, rerank_n);
                }
                return result_dict(result);
            },
            py::arg("query").noconvert(), py::arg("k") = 10, py::arg("nprobe") = 0, py::arg("rerank_n") = 0)
        .def(
            "search_batch",
            [](const Index &index, const Array &queries, std::size_t k, std::size_t nprobe,
               std::size_t rerank_n) {
                auto copy = copy_array(queries, 2, index.dimension());
                auto count = static_cast<std::size_t>(queries.shape(0));
                std::vector<SearchResult> results;
                {
                    py::gil_scoped_release release;
                    results = index.search_batch(copy, count, k, nprobe, rerank_n);
                }
                py::list out;
                for (const auto &result : results)
                    out.append(result_dict(result));
                return out;
            },
            py::arg("queries").noconvert(), py::arg("k") = 10, py::arg("nprobe") = 0, py::arg("rerank_n") = 0)
        .def("cache_stats",
             [](const Index &index) {
                 auto stats = index.cache_stats();
                 py::dict out;
                 out["hits"] = stats.hits;
                 out["misses"] = stats.misses;
                 out["evictions"] = stats.evictions;
                 out["resident_bytes"] = stats.resident_bytes;
                 out["bytes_served"] = stats.bytes_served;
                 out["mapped_bytes_fetched"] = stats.mapped_bytes_fetched;
                 out["miss_nanoseconds"] = stats.miss_nanoseconds;
                 return out;
             })
        .def("clear_cache", &Index::clear_cache)
        .def_property_readonly("dimension", &Index::dimension)
        .def_property_readonly("nlist", &Index::nlist)
        .def_property_readonly("mapped", &Index::mapped)
        .def_property_readonly("dtype",
                               [](const Index &index) {
                                   return std::array<const char *, 3>{"fp32", "fp16", "int8"}.at(
                                       static_cast<std::size_t>(index.dtype()));
                               })
        .def("__len__", &Index::size);
}
