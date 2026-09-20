#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <omp.h>
#include<complex>
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/space_l2.h"
#include "../../hnswlib/vectorset.h"
#include "../../cnpy/cnpy.h"

constexpr int VECTOR_DIM = 128;
constexpr int BASE_VECTOR_SET_MIN = 36;
constexpr int BASE_VECTOR_SET_MAX = 48;
constexpr int NUM_BASE_SETS = 10000;
constexpr int NUM_QUERY_SETS = 10;
constexpr int QUERY_VECTOR_COUNT = 32;
constexpr int K = 10;

class GroundTruth {
public:
    void build(int d, const std::vector<vectorset>& base) {
        dimension = d;
        base_vectors = base;
    }

    void search(const vectorset query, int k, std::vector<std::pair<int, float>>& res) const {
        res.clear();
        std::vector<std::pair<float, int>> distances;

        // Calculate Chamfer distance between query set and each base set
        //std::cout<<"Calc Dis"<<std::endl;
        int base_offset = 0;
        for (size_t i = 0; i < base_vectors.size(); ++i) {
            float chamfer_dist = hnswlib::L2SqrVecSet(&query, &base_vectors[i], 0);
            distances.push_back({chamfer_dist, static_cast<int>(i)});
        }
        //std::cout<<"return ans"<<std::endl;
        // Sort distances to find top-k nearest neighbors
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        for (int i = 0; i < k; ++i) {
            res.push_back(std::make_pair(distances[i].second, distances[i].first));
        }
        
    }

private:
    int dimension;
    std::vector<vectorset> base_vectors;
};

class Solution {
public:
    void build(int d, const std::vector<vectorset>& base) {
        double time = omp_get_wtime();
        dimension = d;
        base_vectors = base;
        space_ptr = new hnswlib::L2VSSpace(dimension);
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(space_ptr, NUM_BASE_SETS, 16, 32);
        #pragma omp parallel for schedule(dynamic)
        for(hnswlib::labeltype i = 0; i < base_vectors.size(); i++){
            if (i % 100 == 0) {
                std::cout << i << std::endl;
            }
            alg_hnsw->addPoint(&base_vectors[i], i);            
        }

        std::cout << "Build time: " << omp_get_wtime() - time << "sec"<<std::endl;
        // Add any necessary pre-computation or indexing for the optimized search
    }

    void search(const vectorset query, int k, std::vector<std::pair<int, float>>& res) const {
        res.clear();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(&query, k);
        for(int i = 0; i < k; i++){
            res.push_back(std::make_pair(result.top().second, result.top().first));
            result.pop();
        }
    }

private:
    int dimension;
    std::vector<vectorset> base_vectors;
    hnswlib::L2VSSpace* space_ptr;
    hnswlib::HierarchicalNSW<float>* alg_hnsw;
};



void generate_vector_sets(std::vector<float>& base_data, std::vector<vectorset>& base,
                          std::vector<float>& query_data, std::vector<vectorset>& query, 
                          int num_base_sets, int num_query_sets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    std::uniform_int_distribution<int> vec_count_dis(BASE_VECTOR_SET_MIN, BASE_VECTOR_SET_MAX);

    // Step 1: Generate base vector data
    std::vector<int> base_vec_counts;  // Track the number of vectors for each base set
    for (int i = 0; i < num_base_sets; ++i) {
        int num_vectors = vec_count_dis(gen);
        // int num_vectors = i + 1;
        base_vec_counts.push_back(num_vectors);
        
        // Fill base_data with random values for each vector
        for (int j = 0; j < num_vectors; ++j) {
            for (int d = 0; d < VECTOR_DIM; ++d) {
                base_data.push_back(dis(gen));
            }
        }
    }

    // Step 2: Create vectorset objects for the base vector sets
    int offset = 0;
    for (int i = 0; i < num_base_sets; ++i) {
        base.push_back(vectorset(base_data.data() + offset, VECTOR_DIM, base_vec_counts[i]));
        offset += base_vec_counts[i] * VECTOR_DIM;
    }

    // Step 3: Generate query vector data
    for (int i = 0; i < num_query_sets; ++i) {
        // Fill query_data with random values for each vector
        for (int j = 0; j < QUERY_VECTOR_COUNT; ++j) {
            for (int d = 0; d < VECTOR_DIM; ++d) {
                query_data.push_back(dis(gen));
            }
        }
    }

    // Step 4: Create vectorset objects for the query vector sets
    offset = 0;
    for (int i = 0; i < num_query_sets; ++i) {
        query.push_back(vectorset(query_data.data() + offset, VECTOR_DIM, QUERY_VECTOR_COUNT));
        offset += QUERY_VECTOR_COUNT * VECTOR_DIM;
    }

    // Debug output to confirm correct data range
    std::cout << "Sample value from base_data: " << base_data[16] << std::endl;
}

double calculate_recall(const std::vector<std::pair<int, float>>& solution_indices,
                        const std::vector<std::pair<int, float>>& ground_truth_indices) {
    std::unordered_set<int> solution_set;
    std::unordered_set<int> ground_truth_set;
    for (const auto& pair : solution_indices) {
        solution_set.insert(pair.first);
    }
    for (const auto& pair : ground_truth_indices) {
        ground_truth_set.insert(pair.first);
    }
    int intersection_count = 0;
    for (const int& index : solution_set) {
        if (ground_truth_set.find(index) != ground_truth_set.end()) {
            intersection_count++;
        }
    }

    double recall = static_cast<double>(intersection_count) / ground_truth_set.size();
    return recall;
}

int main() {
    std::vector<float> base_data;
    std::vector<int> base_vec_num;
    std::vector<float> query_data;
    std::vector<vectorset> base;
    std::vector<vectorset> query;
    // Generate dataset
    generate_vector_sets(base_data, base, query_data, query, NUM_BASE_SETS, NUM_QUERY_SETS);
    

    GroundTruth ground_truth;
    ground_truth.build(VECTOR_DIM, base);
    std::cout<< "Generate Groundtruth and Dataset" <<std::endl;
    Solution solution;
    solution.build(VECTOR_DIM, base);
    double total_recall = 0.0;

    std::cout<<"Processing Queries"<<std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_QUERY_SETS; ++i) {
        std::vector<std::pair<int, float>> ground_truth_indices, solution_indices;
        // Search with GroundTruth
        ground_truth.search(query[i], K, ground_truth_indices);
        std::cout<<"BruteForce Result: ";
        for(int j = 0 ; j < ground_truth_indices.size(); j++)
            std::cout<<ground_truth_indices[j].first<<":"<<ground_truth_indices[j].second<<std::endl;
        std::cout<<std::endl;
        // Search with Solution
        solution.search(query[i], K, solution_indices);
        std::cout<<"HNSW Result: ";
        for(int j = 0 ; j < solution_indices.size(); j++)
            std::cout<<solution_indices[j].first<<":"<<solution_indices[j].second<<std::endl;
        std::cout<<std::endl<<std::endl;
        // Calculate recall for this query set
        double recall = calculate_recall(solution_indices, ground_truth_indices);
        total_recall += recall;
        std::cout << "Recall for query set " << i << ": " << recall << std::endl;
    }

    // Calculate average recall
    double average_recall = total_recall / NUM_QUERY_SETS;
    std::cout << "Average Recall: " << average_recall << std::endl;

    return 0;
}