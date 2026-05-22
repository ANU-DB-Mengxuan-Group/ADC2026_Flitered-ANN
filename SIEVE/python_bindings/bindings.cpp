// SIEVE bindings - stripped of FAISS/ACORN/CAPS dependencies
// Only keeps: HierarchicalIndex (core SIEVE), OraclePartition, PreFilter, HNSWBase
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "partitioned_hnsw.h"
#include "hnswalg.h"
#include "filters.h"
#include "space_l2.h"
#include <limits>
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <utility>
#include <stdint.h>
#include <assert.h>
#include <chrono>
#include <immintrin.h>

#include "roaring.hh"
#include "roaring.c"
#include "xxhash.h"

namespace py = pybind11;
using namespace pybind11::literals;

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read(int d, const float* x) {
    assert(0 <= d && d < 4);
    ALIGNED(16) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps(buf);
}

float Faiss_fvec_L2sqr(const float* x, const float* y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(a_m_b1, a_m_b1));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

template <typename T>
void read_dataset_and_filters(
    const std::string& filename,
    const std::string& filter_filename,
    size_t dataset_size,
    size_t dim,
    size_t num_threads,
    bool is_range,
    T*& data_out,
    hnswlib::DatasetFilters*& filters_out
) {
    // setup filters
    filters_out = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads, is_range);
    filters_out->transpose_inplace();
    filters_out->make_bvs();

    // setup data
    std::ifstream reader(filename);
    assert(reader.is_open());
    size_t num_points;
    size_t d;
    reader.read((char*)(&num_points), sizeof(unsigned int));
    reader.read((char*)(&d), sizeof(unsigned int));

    data_out = new T[dim * dataset_size];
    reader.read((char*)data_out, sizeof(T) * dim * dataset_size);
}

// ============================================================
// Core SIEVE: HierarchicalIndex (Partitioned HNSW)
// ============================================================
template <typename T, typename SpaceType, typename AlgType>
class HierarchicalIndexBase {
 public:
    T* _data;
    AlgType* alg;
    hnswlib::DatasetFilters* dataset_filters;
    size_t _dataset_size;

    T* _new_data = nullptr;
    hnswlib::DatasetFilters* new_dataset_filters = nullptr;

    bool _is_range = false;

    HierarchicalIndexBase(
        std::string filename,
        std::string filter_filename,
        const std::vector<hnswlib::QueryFilter>& historical_workload,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t index_vector_budget,
        size_t bitvector_cutoff,
        size_t historical_workload_window_size,
        bool enable_heterogeneous_indexing,
        bool enable_heterogeneous_search,
        size_t num_threads,
        float query_correlation_constant = 0.5,
        float ef_search_scaling_constant = 3,
        bool enable_multipartition_search = false,
        bool is_range = false
    ) : _dataset_size(dataset_size), _is_range(is_range) {
        hnswlib::PartitionedIndexParams index_params{
            dataset_size,
            dim,
            M,
            ef_construction,
            index_vector_budget,
            bitvector_cutoff,
            historical_workload_window_size,
            enable_heterogeneous_indexing,
            enable_heterogeneous_search,
            query_correlation_constant,
            num_threads,  // must match struct field order
            ef_search_scaling_constant,
            enable_multipartition_search
        };

        auto start = std::chrono::high_resolution_clock::now();
        read_dataset_and_filters<T>(
            filename, filter_filename, dataset_size, dim, num_threads, is_range, _data, dataset_filters
        );
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        if constexpr (std::is_same<T, uint8_t>::value) {
            SpaceType* space = new SpaceType(dim / 4);
            alg = new AlgType(_data, space, dataset_filters, index_params, historical_workload);
        } else {
            SpaceType* space = new SpaceType(dim);
            alg = new AlgType(_data, space, dataset_filters, index_params, historical_workload);
        }
    }

    void update_index(const std::vector<hnswlib::QueryFilter>& historical_workload) {
        alg->updateIndexWorkload(historical_workload);
        alg->fitIndex();
    }

    py::object batch_filter_search(
        py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
        const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
        uint64_t knn, size_t ef_search, uint64_t num_threads
    ) {
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> times(num_queries);
        py::array_t<size_t> cardinalities(num_queries);

        auto start = std::chrono::high_resolution_clock::now();

        alg->setEf(ef_search);

        std::vector<hnswlib::Predicate> predicate_arr;
        predicate_arr.reserve(filters.size());
        for (const auto& filter : filters) {
            predicate_arr.emplace_back(dataset_filters, filter);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        start = std::chrono::high_resolution_clock::now();
        hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto results = alg->searchKnn(queries.data(i), knn, predicate_arr[i]);
            auto t1 = std::chrono::high_resolution_clock::now();

            for (size_t j = 0; j < knn; ++j) {
                if (!results.empty()) {
                    ids.mutable_data(i)[j] = results.top().second;
                    results.pop();
                } else {
                    ids.mutable_data(i)[j] = 0;
                }
            }
            times.mutable_at(i) = std::chrono::duration<float>(t1 - t0).count();
            cardinalities.mutable_at(i) = predicate_arr[i].cardinality();
        });
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
        alg->printTally();

        return py::make_tuple(ids, times, cardinalities);
    }
};

using HierarchicalIndexFloat = HierarchicalIndexBase<float, hnswlib::L2Space, hnswlib::PartitionedHNSW<float, float>>;

// ============================================================
// Oracle Partition (unlimited budget baseline)
// ============================================================
class OraclePartitionFloat : public HierarchicalIndexBase<float, hnswlib::L2Space, hnswlib::PartitionedHNSW<float, float>> {
public:
    OraclePartitionFloat(
        std::string filename,
        std::string filter_filename,
        const std::vector<hnswlib::QueryFilter>& historical_workload,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t bitvector_cutoff,
        bool enable_heterogeneous_indexing,
        size_t num_threads,
        bool is_range = false
    ) : HierarchicalIndexBase<float, hnswlib::L2Space, hnswlib::PartitionedHNSW<float, float>>(
            filename,
            filter_filename,
            historical_workload,
            dataset_size,
            dim,
            M,
            ef_construction,
            std::numeric_limits<int>::max(),
            bitvector_cutoff,
            std::numeric_limits<int>::max(),
            enable_heterogeneous_indexing,
            false,
            num_threads,
            0.5f,
            3.0f,
            false,
            is_range
        )
    {}
};

// ============================================================
// PreFilter baseline (brute-force on filtered subset)
// ============================================================
class PreFilterFloat {
 public:
    float* _data;
    hnswlib::DatasetFilters* dataset_filters;
    hnswlib::SpaceInterface<float>* _space;
    hnswlib::HierarchicalNSW<float>* _hnsw;
    size_t _dataset_size;
    size_t _dim;
    bool _is_range;

    PreFilterFloat(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t num_threads,
        bool is_range = false
    ) : _dataset_size(dataset_size), _dim(dim), _is_range(is_range) {
        auto start = std::chrono::high_resolution_clock::now();
        read_dataset_and_filters<float>(
            filename, filter_filename, dataset_size, dim, num_threads, is_range, _data, dataset_filters
        );
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        _space = new hnswlib::L2Space(dim);
        _hnsw = new hnswlib::HierarchicalNSW<float>(_space, 10, 16, 40);
    }

    py::object batch_filter_search(
        py::array_t<float, py::array::c_style | py::array::forcecast>& queries,
        const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
        uint64_t knn, uint64_t num_threads
    ) {
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> times(num_queries);
        py::array_t<size_t> cardinalities(num_queries);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<hnswlib::Predicate> predicate_arr;
        for (size_t i = 0; i < filters.size(); i++) {
            predicate_arr.push_back(hnswlib::Predicate(dataset_filters, filters[i]));
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        start = std::chrono::high_resolution_clock::now();
        hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring& deref = *predicate_arr[i]._bitvector;
            std::priority_queue<std::pair<float, size_t>> max_priority_queue;
            for (roaring::Roaring::const_iterator j = deref.begin(); j != deref.end(); j++) {
                auto dist = Faiss_fvec_L2sqr(queries.data(i), (_data + _dim * *j), _dim);
                if (max_priority_queue.size() < knn) {
                    max_priority_queue.push(std::make_pair(dist, *j));
                } else if (dist < max_priority_queue.top().first) {
                    max_priority_queue.pop();
                    max_priority_queue.push(std::make_pair(dist, *j));
                }
            }
            for (size_t j = 0; j < knn; j++) {
                if (!max_priority_queue.empty()) {
                    ids.mutable_data(i)[j] = max_priority_queue.top().second;
                    max_priority_queue.pop();
                } else {
                    ids.mutable_data(i)[j] = 0;
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            times.mutable_at(i) = std::chrono::duration<double>(t1 - t0).count();
            cardinalities.mutable_at(i) = predicate_arr[i].cardinality();
        });
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        return py::make_tuple(ids, times, cardinalities);
    }
};

// ============================================================
// HNSW + bitmap filter baseline
// ============================================================
class HNSWBaseFloat {
 public:
    float* _data;
    hnswlib::DatasetFilters* dataset_filters;
    hnswlib::SpaceInterface<float>* _space;
    hnswlib::HierarchicalNSW<float>* _hnsw;
    size_t _dim;
    bool _is_range;

    HNSWBaseFloat(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t num_threads,
        bool is_range = false
    ) : _dim(dim), _is_range(is_range) {
        auto start = std::chrono::high_resolution_clock::now();
        read_dataset_and_filters<float>(
            filename, filter_filename, dataset_size, dim, num_threads, is_range, _data, dataset_filters
        );
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        start = std::chrono::high_resolution_clock::now();
        _space = new hnswlib::L2Space(dim);
        _hnsw = new hnswlib::HierarchicalNSW<float>(_space, dataset_size, M, ef_construction);
        hnswlib::ParallelFor(0, dataset_size, num_threads, [&](size_t row, size_t threadId) {
            _hnsw->addPoint((void*)(_data + dim * row), row);
        });
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to build index: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    }

    py::object batch_filter_search(
     py::array_t<float, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> times(num_queries);
        py::array_t<size_t> cardinalities(num_queries);

        auto start = std::chrono::high_resolution_clock::now();
        _hnsw->setEf(ef_search);
        std::vector<hnswlib::Predicate> predicate_arr;
        for (size_t i = 0; i < filters.size(); i++) {
            predicate_arr.push_back(hnswlib::Predicate(dataset_filters, filters[i]));
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        start = std::chrono::high_resolution_clock::now();
        hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
            hnswlib::BitMapFilter QueryBitset(predicate_arr[i]._bitvector);
            auto t0 = std::chrono::high_resolution_clock::now();
            auto results = _hnsw->searchKnn(queries.data(i), knn, &QueryBitset);
            auto t1 = std::chrono::high_resolution_clock::now();
            for (size_t j = 0; j < knn; j++) {
                if (!results.empty()) {
                    ids.mutable_data(i)[j] = results.top().second;
                    results.pop();
                } else {
                    ids.mutable_data(i)[j] = 0;
                }
            }
            times.mutable_at(i) = std::chrono::duration<double>(t1 - t0).count();
            cardinalities.mutable_at(i) = predicate_arr[i].cardinality();
        });
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        return py::make_tuple(ids, times, cardinalities);
    }
};


// ============================================================
// Python module
// ============================================================
PYBIND11_MODULE(hnswlib, m) {
    py::class_<hnswlib::QueryFilter>(m, "QueryFilter")
        .def(py::init<std::unordered_set<int32_t>, bool>(),
             py::arg("filters"), py::arg("is_and"))
        .def(py::init<std::vector<float>, bool>(),
             py::arg("filters"), py::arg("is_and"));

    py::class_<HierarchicalIndexFloat>(m, "HierarchicalIndexFloat")
        .def(py::init<
                std::string, std::string, std::vector<hnswlib::QueryFilter>&,
                size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                bool, bool, size_t, float, float, bool, bool>(),
             py::arg("filename"), py::arg("filter_filename"),
             py::arg("historical_workload"), py::arg("dataset_size"),
             py::arg("dim"), py::arg("M"), py::arg("ef_construction"),
             py::arg("index_vector_budget"), py::arg("bitvector_cutoff"),
             py::arg("historical_workload_window_size"),
             py::arg("enable_heterogeneous_indexing"),
             py::arg("enable_heterogeneous_search"),
             py::arg("num_threads"),
             py::arg("query_correlation_constant") = 0.5f,
             py::arg("ef_search_scaling_constant") = 3.0f,
             py::arg("enable_multipartition_search") = false,
             py::arg("is_range") = false)
        .def("update_index", &HierarchicalIndexFloat::update_index)
        .def("batch_filter_search", &HierarchicalIndexFloat::batch_filter_search);

    py::class_<OraclePartitionFloat>(m, "OraclePartitionFloat")
        .def(py::init<
                std::string, std::string, std::vector<hnswlib::QueryFilter>&,
                size_t, size_t, size_t, size_t, size_t, bool, size_t, bool>(),
             py::arg("filename"), py::arg("filter_filename"),
             py::arg("historical_workload"), py::arg("dataset_size"),
             py::arg("dim"), py::arg("M"), py::arg("ef_construction"),
             py::arg("bitvector_cutoff"),
             py::arg("enable_heterogeneous_indexing"),
             py::arg("num_threads"), py::arg("is_range") = false)
        .def("update_index", &OraclePartitionFloat::update_index)
        .def("batch_filter_search", &OraclePartitionFloat::batch_filter_search);

    py::class_<PreFilterFloat>(m, "PreFilterFloat")
        .def(py::init<
                std::string, std::string, size_t, size_t, size_t, bool>(),
             py::arg("filename"), py::arg("filter_filename"),
             py::arg("dataset_size"), py::arg("dim"),
             py::arg("num_threads"), py::arg("is_range") = false)
        .def("batch_filter_search", &PreFilterFloat::batch_filter_search);

    py::class_<HNSWBaseFloat>(m, "HNSWBaseFloat")
        .def(py::init<
                std::string, std::string, size_t, size_t, size_t, size_t, size_t, bool>(),
             py::arg("filename"), py::arg("filter_filename"),
             py::arg("dataset_size"), py::arg("dim"),
             py::arg("M"), py::arg("ef_construction"),
             py::arg("num_threads"), py::arg("is_range") = false)
        .def("batch_filter_search", &HNSWBaseFloat::batch_filter_search);
}
