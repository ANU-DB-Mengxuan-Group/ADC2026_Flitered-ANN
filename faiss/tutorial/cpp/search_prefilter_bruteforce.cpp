/**
 * Pre-filter baseline: Filter first, then brute-force search
 *
 * For each query:
 * 1. Find all base vectors matching label condition
 * 2. Brute-force search on matching vectors
 * 3. Return top-k
 */

#include <omp.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>
#include <queue>

#include <sys/stat.h>
#include <sys/time.h>

#include <faiss/IndexFlat.h>

using idx_t = faiss::idx_t;

const unsigned int nthreads = 16;

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

std::vector<std::set<int>> load_multi_label(const std::string& input_path, int N) {
    std::vector<std::set<int>> metadata;
    std::ifstream file(input_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << input_path << std::endl;
        return metadata;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::set<int> s;
        std::istringstream iss(line);
        for (std::string token; std::getline(iss, token, ',');) {
            s.insert(std::stoi(token));
        }
        metadata.push_back(s);
    }
    assert(metadata.size() == N);
    return metadata;
}

bool file_exist(const std::string& path) {
    if (FILE* file = fopen(path.c_str(), "r")) {
        fclose(file);
        return true;
    }
    return false;
}

// Check if base vector matches query label condition
bool matches_filter(const std::set<int>& base_labels,
                   const std::set<int>& query_labels,
                   const std::string& scenario) {
    if (scenario == "and") {
        // base must contain ALL query labels
        return std::includes(base_labels.begin(), base_labels.end(),
                           query_labels.begin(), query_labels.end());
    } else if (scenario == "or") {
        // base must contain AT LEAST ONE query label
        for (int lbl : query_labels) {
            if (base_labels.count(lbl)) return true;
        }
        return false;
    } else if (scenario == "equal") {
        // base labels must EXACTLY match query labels
        return base_labels == query_labels;
    }
    return false;
}

double compute_recall(
        const std::vector<idx_t>& gt,
        const std::vector<idx_t>& res,
        int num_queries,
        int k) {
    int correct = 0;
    for (int i = 0; i < num_queries; i++) {
        std::set<idx_t> gt_set(gt.begin() + i * k, gt.begin() + (i + 1) * k);
        std::set<idx_t> res_set(res.begin() + i * k, res.begin() + (i + 1) * k);
        std::vector<idx_t> intersection;
        std::set_intersection(gt_set.begin(), gt_set.end(),
                            res_set.begin(), res_set.end(),
                            std::back_inserter(intersection));
        correct += intersection.size();
    }
    return static_cast<double>(correct) / (num_queries * k);
}

void write_results(
        const std::string& output_csv,
        double qps,
        double recall,
        double avg_candidates,
        double filter_time,
        double search_time) {
    std::ofstream f(output_csv);
    f << "Method,QPS,Recall,AvgCandidates,FilterTime,SearchTime" << std::endl;
    f << "prefilter_bruteforce," << qps << "," << recall << ","
      << avg_candidates << "," << filter_time << "," << search_time << std::endl;
    f.close();
}

int main(int argc, char* argv[]) {
    std::cout << "==================== Pre-filter Brute-force Baseline ====================" << std::endl;
    omp_set_num_threads(nthreads);
    double t0 = elapsed();

    if (argc != 10) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <scenario> <output_path> "
                  << "<base_file> <base_label> <query_file> <query_label> <gt_path> <k>"
                  << std::endl;
        return 1;
    }

    std::string dataset = argv[1];
    std::string scenario = argv[2];
    std::string output_path = argv[3];
    std::string base_file = argv[4];
    std::string base_label_file = argv[5];
    std::string query_file = argv[6];
    std::string query_label_file = argv[7];
    std::string gt_path = argv[8];
    int k = atoi(argv[9]);

    // Load base vectors
    printf("[%.3f s] Loading base vectors\n", elapsed() - t0);
    size_t nb, d;
    float* xb = fvecs_read(base_file.c_str(), &d, &nb);
    printf("[%.3f s] Loaded %ld base vectors, dim=%ld\n", elapsed() - t0, nb, d);

    // Load base labels
    printf("[%.3f s] Loading base labels\n", elapsed() - t0);
    auto base_labels = load_multi_label(base_label_file, nb);
    printf("[%.3f s] Loaded %ld base labels\n", elapsed() - t0, base_labels.size());

    // Load query vectors
    printf("[%.3f s] Loading query vectors\n", elapsed() - t0);
    size_t nq, d2;
    float* xq = fvecs_read(query_file.c_str(), &d2, &nq);
    assert(d == d2);
    printf("[%.3f s] Loaded %ld query vectors\n", elapsed() - t0, nq);

    // Load query labels
    printf("[%.3f s] Loading query labels\n", elapsed() - t0);
    auto query_labels = load_multi_label(query_label_file, nq);
    printf("[%.3f s] Loaded %ld query labels\n", elapsed() - t0, query_labels.size());

    // Load ground truth
    printf("[%.3f s] Loading ground truth\n", elapsed() - t0);
    std::vector<idx_t> gt;
    std::ifstream gt_file(gt_path);
    if (gt_file.is_open()) {
        std::string line;
        while (std::getline(gt_file, line)) {
            std::istringstream iss(line);
            idx_t val;
            while (iss >> val) {
                gt.push_back(val);
            }
        }
        gt_file.close();
    } else {
        std::cerr << "Unable to open gt file: " << gt_path << std::endl;
        return 1;
    }
    printf("[%.3f s] Loaded ground truth\n", elapsed() - t0);

    // Results storage
    std::vector<idx_t> results(nq * k, -1);
    std::vector<size_t> candidate_counts(nq, 0);

    double filter_time = 0;
    double search_time = 0;

    printf("[%.3f s] Starting pre-filter + brute-force search\n", elapsed() - t0);
    printf("Scenario: %s, k=%d\n", scenario.c_str(), k);

    double total_start = elapsed();

    // Process each query
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:filter_time, search_time)
    for (size_t q = 0; q < nq; q++) {
        double q_filter_start = elapsed();

        // Step 1: Find all matching base IDs (pre-filter)
        std::vector<idx_t> matching_ids;
        matching_ids.reserve(nb / 10);  // estimate

        for (size_t i = 0; i < nb; i++) {
            if (matches_filter(base_labels[i], query_labels[q], scenario)) {
                matching_ids.push_back(i);
            }
        }

        double q_filter_end = elapsed();
        filter_time += (q_filter_end - q_filter_start);
        candidate_counts[q] = matching_ids.size();

        if (matching_ids.empty()) {
            continue;  // No matching vectors
        }

        double q_search_start = elapsed();

        // Step 2: Brute-force search on matching vectors
        float* query_vec = xq + q * d;

        // Use priority queue to find top-k (min-heap by distance)
        // We want k smallest distances, so use max-heap and pop when full
        std::priority_queue<std::pair<float, idx_t>> topk_heap;

        for (idx_t base_id : matching_ids) {
            float* base_vec = xb + base_id * d;

            // Compute L2 distance
            float dist = 0;
            for (size_t j = 0; j < d; j++) {
                float diff = query_vec[j] - base_vec[j];
                dist += diff * diff;
            }

            if (topk_heap.size() < k) {
                topk_heap.push({dist, base_id});
            } else if (dist < topk_heap.top().first) {
                topk_heap.pop();
                topk_heap.push({dist, base_id});
            }
        }

        // Extract results (in reverse order since it's a max-heap)
        std::vector<idx_t> query_results;
        while (!topk_heap.empty()) {
            query_results.push_back(topk_heap.top().second);
            topk_heap.pop();
        }
        std::reverse(query_results.begin(), query_results.end());

        // Copy to results
        for (size_t i = 0; i < query_results.size() && i < k; i++) {
            results[q * k + i] = query_results[i];
        }

        double q_search_end = elapsed();
        search_time += (q_search_end - q_search_start);
    }

    double total_end = elapsed();
    double total_time = total_end - total_start;

    // Compute statistics
    double avg_candidates = 0;
    for (size_t c : candidate_counts) avg_candidates += c;
    avg_candidates /= nq;

    double qps = nq / total_time;
    double recall = compute_recall(gt, results, nq, k);

    printf("\n==================== Results ====================\n");
    printf("Total queries: %ld\n", nq);
    printf("Total time: %.3f s\n", total_time);
    printf("  - Filter time: %.3f s\n", filter_time / nthreads);  // approximate
    printf("  - Search time: %.3f s\n", search_time / nthreads);
    printf("QPS: %.2f\n", qps);
    printf("Recall@%d: %.4f\n", k, recall);
    printf("Avg candidates per query: %.1f\n", avg_candidates);
    printf("=================================================\n");

    // Write results
    std::string output_csv = output_path + "/prefilter_bruteforce_result.csv";
    write_results(output_csv, qps, recall, avg_candidates,
                  filter_time / nthreads, search_time / nthreads);
    printf("Results written to: %s\n", output_csv.c_str());

    // Save per-query result IDs to binary file (nq uint32, k uint32, then nq*k uint32)
    {
        std::string result_bin = output_path + "/prefilter_bruteforce_idx_uint32.bin";
        std::ofstream fout(result_bin, std::ios::binary);
        uint32_t nq32 = static_cast<uint32_t>(nq);
        uint32_t k32 = static_cast<uint32_t>(k);
        fout.write(reinterpret_cast<const char*>(&nq32), sizeof(uint32_t));
        fout.write(reinterpret_cast<const char*>(&k32), sizeof(uint32_t));
        for (size_t q = 0; q < nq; ++q) {
            for (int j = 0; j < k; ++j) {
                uint32_t id = static_cast<uint32_t>(
                    std::max(static_cast<idx_t>(0), results[q * k + j]));
                fout.write(reinterpret_cast<const char*>(&id), sizeof(uint32_t));
            }
        }
        fout.close();
        printf("Per-query results saved to: %s\n", result_bin.c_str());
    }

    delete[] xb;
    delete[] xq;

    printf("[%.3f s] ----- DONE -----\n", elapsed() - t0);
    return 0;
}
