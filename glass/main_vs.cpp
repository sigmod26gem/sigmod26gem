#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include "./glass/simd/distance.h" // Adjust path as needed
#include "./solution2.cpp"

constexpr int VECTOR_DIM = 128;
constexpr int BASE_VECTOR_SET_MIN = 36;
constexpr int BASE_VECTOR_SET_MAX = 48;
constexpr int NUM_BASE_SETS = 1000;
constexpr int NUM_QUERY_SETS = 10;
constexpr int QUERY_VECTOR_COUNT = 32;
constexpr int K = 5;

class GroundTruth {
public:
    void build(int d, const std::vector<float>& base, const std::vector<int>& vec_num) {
        dimension = d;
        base_vectors = base;
        base_vec_num = vec_num;
    }

    void search(const std::vector<float>& query, int num_vec, int k, std::vector<int>& res) const {
        res.clear();
        std::vector<std::pair<float, int>> distances;

        // Calculate Chamfer distance between query set and each base set
        int base_offset = 0;
        for (size_t i = 0; i < base_vec_num.size(); ++i) {
            int base_num_vec = base_vec_num[i];
            float chamfer_dist = glass::L2SqrVC(query.data(), QUERY_VECTOR_COUNT, base_vectors.data() + base_offset, base_num_vec, dimension);
            distances.push_back({chamfer_dist, static_cast<int>(i)});
            base_offset += base_num_vec * dimension;
        }

        // Sort distances to find top-k nearest neighbors
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        for (int i = 0; i < k; ++i) {
            res.push_back(distances[i].second);
        }
    }

private:
    int dimension;
    std::vector<float> base_vectors;
    std::vector<int> base_vec_num;
};

// class Solution {
// public:
//     void build(int d, const std::vector<float>& base, const std::vector<int>& vec_num) {
//         dimension = d;
//         base_vectors = base;
//         base_vec_num = vec_num;
//         // Add any necessary pre-computation or indexing for the optimized search
//     }

//     void search(const std::vector<float>& query, int num_vec, int k, std::vector<int>& res) const {
//         res.clear();
//         std::vector<std::pair<float, int>> distances;

//         // Calculate Chamfer distance between query set and each base set
//         int base_offset = 0;
//         for (size_t i = 0; i < base_vec_num.size(); ++i) {
//             int base_num_vec = base_vec_num[i];
//             float chamfer_dist = glass::L2SqrVC(query.data(), QUERY_VECTOR_COUNT, base_vectors.data() + base_offset, base_num_vec, dimension);
//             distances.push_back({chamfer_dist, static_cast<int>(i)});
//             base_offset += base_num_vec * dimension;
//         }

//         // Sort distances to find top-k nearest neighbors
//         std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
//         for (int i = 0; i < k; ++i) {
//             res.push_back(distances[i].second);
//         }
//     }

// private:
//     int dimension;
//     std::vector<float> base_vectors;
//     std::vector<int> base_vec_num;
// };

void generate_vector_sets(std::vector<float>& base, std::vector<int>& base_vec_num,
                          std::vector<float>& query, int num_base_sets, int num_query_sets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    std::uniform_int_distribution<int> vec_count_dis(BASE_VECTOR_SET_MIN, BASE_VECTOR_SET_MAX);

    // Generate base vector sets
    for (int i = 0; i < num_base_sets; ++i) {
        int num_vectors = vec_count_dis(gen);
        base_vec_num.push_back(num_vectors);
        for (int j = 0; j < num_vectors; ++j) {
            for (int d = 0; d < VECTOR_DIM; ++d) {
                base.push_back(dis(gen));
            }
        }
    }

    // Generate query vector sets
    for (int i = 0; i < num_query_sets; ++i) {
        for (int j = 0; j < QUERY_VECTOR_COUNT; ++j) {
            for (int d = 0; d < VECTOR_DIM; ++d) {
                query.push_back(dis(gen));
            }
        }
    }
}

double calculate_recall(const std::vector<int>& result, const std::vector<int>& ground_truth) {
    std::unordered_set<int> ground_truth_set(ground_truth.begin(), ground_truth.end());
    int correct_matches = 0;
    for (int idx : result) {
        if (ground_truth_set.count(idx)) {
            correct_matches++;
        }
    }
    return static_cast<double>(correct_matches) / ground_truth.size();
}

int main() {
    std::vector<float> base;
    std::vector<int> base_vec_num;
    std::vector<float> query;

    // Generate dataset
    generate_vector_sets(base, base_vec_num, query, NUM_BASE_SETS, NUM_QUERY_SETS);

    GroundTruth ground_truth;
    ground_truth.build(VECTOR_DIM, base, base_vec_num);
    std::cout<< "Generate Groundtruth and Dataset" <<std::endl;
    Solution solution;
    solution.build(VECTOR_DIM, base, base_vec_num);

    double total_recall = 0.0;

    // for (int i = 0; i < NUM_QUERY_SETS; ++i) {
    //     std::vector<int> ground_truth_indices, solution_indices;
    //     std::vector<float> query_set(query.begin() + i * QUERY_VECTOR_COUNT * VECTOR_DIM,
    //                                  query.begin() + (i + 1) * QUERY_VECTOR_COUNT * VECTOR_DIM);

    //     // Search with GroundTruth
    //     ground_truth.search(query_set, QUERY_VECTOR_COUNT, K, ground_truth_indices);

    //     // Search with Solution
    //     solution.search(query_set, QUERY_VECTOR_COUNT, K, solution_indices);

    //     // Calculate recall for this query set
    //     double recall = calculate_recall(solution_indices, ground_truth_indices);
    //     total_recall += recall;

    //     std::cout << "Recall for query set " << i << ": " << recall << std::endl;
    // }

    // Calculate average recall
    // double average_recall = total_recall / NUM_QUERY_SETS;
    // std::cout << "Average Recall: " << average_recall << std::endl;

    return 0;
}
