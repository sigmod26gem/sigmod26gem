#include <iostream>
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <omp.h>
#include <complex>
#include <cblas.h>
#include <chrono>  
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/space_l2.h"
#include "../../hnswlib/vectorset.h"
#include "../../cnpy/cnpy.h"
// #include <experimental/filesystem>
#define TEST_MSMARCO 0
#define TEST_LOTTE 1
#define TEST_OKVQA 2
#define TEST_EVQA 3

int dataset = TEST_MSMARCO; // 0: MSMARCO, 1: LOTTE, 2: OKVQA, 3: EVQA
std::string dataset_path = "../../gem_data/";

constexpr int VECTOR_DIM = 128;

constexpr int MSMACRO_TEST_NUMBER = 354;
constexpr int LOTTE_TEST_NUMBER = 98;
constexpr int OKVQA_TEST_NUMBER = 5;
constexpr int EVQA_TEST_NUMBER = 3;

constexpr int NUM_BASE_SETS_MS = 25000 * MSMACRO_TEST_NUMBER;
constexpr int NUM_QUERT_MS = 6980;
constexpr int NUM_CLUSTER_MS = 262144;
constexpr int NUM_GRAPH_CLUSTER_MS = 40960;

constexpr long long NUM_BASE_VECTOR_LOTTE = 339419977;
constexpr int NUM_BASE_SETS_LOTTE = 2428853;
constexpr int NUM_QUERT_LOTTE = 2930;
constexpr int NUM_CLUSTER_LOTTE = 262144;
constexpr int NUM_GRAPH_CLUSTER_LOTTE = 10240;

constexpr long long NUM_BASE_VECTOR_OKVQA = 14119353;
constexpr int NUM_BASE_SETS_OKVQA = 114809;
constexpr int NUM_QUERT_OKVQA = 5046;
constexpr int NUM_CLUSTER_OKVQA = 32768;
constexpr int NUM_GRAPH_CLUSTER_OKVQA = 1024;

constexpr long long NUM_BASE_VECTOR_EVQA = 9745953;
constexpr int NUM_BASE_SETS_EVQA = 51472;
constexpr int NUM_QUERT_EVQA = 3750;
constexpr int NUM_CLUSTER_EVQA = 32768;
constexpr int NUM_GRAPH_CLUSTER_EVQA = 1024;
 
constexpr int CPU_num = 1;
constexpr int M_index = 24;
constexpr int EF_index = 80;


int NUM_BASE_SETS = 25000 * MSMACRO_TEST_NUMBER;
int NUM_QUERY_SETS = 6980;
int NUM_CLUSTER = 262144;
int NUM_GRAPH_CLUSTER = 40960;
int QUERY_VECTOR_COUNT = 32;

// 320
constexpr int NPROB = 4;
constexpr int K = 100;
int rerankK = 512;

// 5
// std::vector<int>eflist = {10, 32, 64, 100, 200, 400, 800, 1000, 2000, 4000, 6000, 10000, 15000, 20000, 40000};
// std::vector<int>eflist = {2000};
// std::vector<int>eflist = {2000, 4000, 8000, 10000, 15000, 20000, 30000};
// 10
// std::vector<int>eflist = {1000};
// std::vector<int>eflist = {10, 32, 64, 100, 200, 400, 800, 2000, 4000, 8000, 10000, 15000, 20000, 30000};
// std::vector<int>eflist = {100, 256, 512, 1000, 2000, 4000, 80000, 16000, 24000, 32000};
std::vector<int>eflist = {4000};
// 100
// std::vector<int>eflist = {10, 32, 100, 200, 400, 800, 1000, 2000, 4000, 8000, 16000, 24000, 40000, 50000};
// std::vector<int>eflist = {80000, 120000, 150000, 200000};
//std::vector<int>eflist = {4000};

void get_unique_top_k_indices_col(const std::vector<float>& matrix, int rows, int cols, int topk, std::unordered_set<int>& unique_indices) {
    // std::cout << matrix.size() << std::endl;
    std::vector<int> all_scores(rows * topk);
    // #pragma omp parallel for
    // std::cout << matrix.size() << std::endl;
    // std::cout << rows << std::endl;
    // std::cout << cols << std::endl;
    // std::cout << topk << std::endl;
    // std::cout << all_scores.size() << std::endl;
    #pragma omp parallel for
    for (int row = 0; row < rows; ++row) {
        // std::cout << row << std::endl;
        std::vector<std::pair<float, int>> scores(cols);
        // 提取该列数据
        for (int col = 0; col < cols; ++col) {
            // std::cout << row << " " << col << std::endl;
            scores[col] = {matrix[row * cols + col], col};  
        }

        // 仅排序前 K 个最大元素
        std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                              return a.first > b.first; // 降序
                          });

        // 直接存入集合去重
        for (int i = 0; i < topk; ++i) {
            // std::cout << row << " " << row * topk + i << " " << scores[i].first << " " << scores[i].second << std::endl;
            all_scores[row * topk + i] = scores[i].second;
        }
    }
    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < topk; ++i) {
            // std::cout << row << " " << i << std::endl;
            unique_indices.insert(all_scores[row * topk + i]);
        }
    }
    // for(int t: unique_indices) {
    //     std::cout<< t << " ";
    // }
    // std::cout << std::endl;
    return;
}


class Solution {
public:
    void build_fine_cluster(int d, const std::vector<vectorset>& base, const std::vector<std::vector<hnswlib::labeltype>>& cluster_set, const std::vector<int>& temp, const std::vector<float>& cluster_distance) {
        double time = omp_get_wtime();
        alg_hnsw_list.resize(1);
        dimension = d;
        base_vectors = base;
        space_ptr = new hnswlib::L2VSSpace(dimension);
        temp_cluster_id = temp;
        std::cout << "init alg" << std::endl;
        cluster_entries.resize(NUM_GRAPH_CLUSTER);
        for (int i = 0; i < cluster_set.size(); i++) {
            cluster_entries[i] = -1;
        }
        alg_hnsw_list[0] = new hnswlib::HierarchicalNSW<float>(space_ptr, base.size() + 1, M_index, EF_index);
        for(int tmpi = 0; tmpi < temp_cluster_id.size(); tmpi++) {
            int i = temp_cluster_id[tmpi];
            double cur_time = omp_get_wtime();
            std::cout << "cluster build begin: " + std::to_string(i) + " " + std::to_string(cluster_set[i].size()) << std::endl;
            alg_hnsw_list[0]->entry_map.clear();
            cluster_entries[i] = cluster_set[i][0];
            alg_hnsw_list[0]->addClusterPointEntry(&base_vectors[cluster_set[i][0]], cluster_distance.data(), cluster_set[i][0], cluster_set[i][0]);
            if (cluster_set[i].size() <= 1) {
                continue;
            }
            #pragma omp parallel for schedule(dynamic)
            for (int j = 1; j < cluster_set[i].size(); j++) {
                if (j % 1000 == 0) {
                    // #pragma omp critical
                    std::cout << std::to_string(i) + " " + std::to_string(j) + " " + std::to_string(cluster_set[i][j]) << std::endl;
                }
                alg_hnsw_list[0]->addClusterPointEntry(&base_vectors[cluster_set[i][j]], cluster_distance.data(), cluster_set[i][j], cluster_set[i][0]);
            }
            std::cout << "cluster build finish: " + std::to_string(i) + " " + std::to_string(omp_get_wtime() - cur_time) << std::endl;   
        }
        std::cout << "Build time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

    double search_with_fine_cluster(const vectorset& query, std::vector<float>& query_cluster_scores, std::vector<float>& col_query_cluster_scores, std::vector<float>& center_data, std::vector<float>& graph_center_data, int k, int ef, std::vector<std::pair<int, float>>& res) {
        res.clear();
        res.resize(rerankK);

        alg_hnsw_list[0]->search_set.assign(alg_hnsw_list[0]->search_set.size(), 0);
        volatile int temp = 6;

        double start_time = omp_get_wtime();
        hnswlib::fast_dot_product_blas((&query)->vecnum, 128, NUM_GRAPH_CLUSTER, (&query)->data, graph_center_data.data(), query_cluster_scores.data()); 
        std::unordered_set<int> unique_indices;
        get_unique_top_k_indices_col(query_cluster_scores, (&query)->vecnum, NUM_GRAPH_CLUSTER, NPROB, unique_indices); 
        hnswlib::fast_dot_product_blas(NUM_CLUSTER, 128, (&query)->vecnum, center_data.data(), (&query)->data, col_query_cluster_scores.data()); 
        vectorset query_cluster = vectorset(col_query_cluster_scores.data(), nullptr, NUM_CLUSTER, (&query)->vecnum);

        std::vector<hnswlib::labeltype> entry_points;
        
        for (const int idx : unique_indices) {
            if (cluster_entries[idx] != -1) {
                entry_points.push_back(cluster_entries[idx]);
                for (const int j: self_cluster_set[idx]) {
                    alg_hnsw_list[0]->search_set[alg_hnsw_list[0]->label_lookup_[j]] = true;
                }
            }
        }
        double cluster_time = omp_get_wtime();
        alg_hnsw_list[0]->setEf(ef);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_list[0]->searchKnnClusterEntries(&query_cluster, ef, entry_points);
        std::vector<std::pair<float, hnswlib::labeltype>> merge_result;
        while(result.size() > 0) {
            merge_result.push_back(result.top());
            result.pop();
        }

        double search_time = omp_get_wtime();
        int numrerank = std::min((int)merge_result.size(), rerankK);
        std::partial_sort(merge_result.begin(), merge_result.begin() + numrerank, merge_result.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first < b.first;  // 按 float 排序，越小越靠前
                      });

        res.resize(numrerank);
        if (CPU_num == 1) {
            // using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_mat(query.data, query.vecnum, VECTOR_DIM);
            for (int i = 0; i < numrerank; i++){
                // for (const hnswlib::labeltype ind : search_result){
                hnswlib::labeltype ind = merge_result[i].second;
                Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_mat(base_vectors[ind].data, base_vectors[ind].vecnum, VECTOR_DIM);
                Eigen::MatrixXf C = A_mat * B_mat.transpose();
                res[i] = std::make_pair(ind, 1 - C.rowwise().maxCoeff().sum() / query.vecnum);
            }            
        } else {
            for (int i = 0; i < numrerank; i++){
                // for (const hnswlib::labeltype ind : search_result){
                hnswlib::labeltype ind = merge_result[i].second;
                res[i] = std::make_pair(ind, hnswlib::L2SqrVecCF(&query, &base_vectors[ind], 0));
            }
        }

        std::sort(res.begin(), res.end(), 
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second < b.second;
            });
        if (res.size() > k) {
            res.resize(k);
        }

        double end_time = omp_get_wtime();
        return end_time - start_time;
    }

    void save_fine_cluster(const std::string &location) {
        double time = omp_get_wtime();
        std::string locai = location + std::to_string(0) + ".bin";
        alg_hnsw_list[0]->saveIndex(locai);
        std::cout << "save time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

    void load_fine_cluster(const std::string &location, int d, const std::vector<vectorset>& base, const std::vector<std::vector<hnswlib::labeltype>>& cluster_set, const std::vector<int>& temp) {
        double time = omp_get_wtime();
        self_cluster_set = cluster_set;
        dimension = d;
        base_vectors = base;
        space_ptr = new hnswlib::L2VSSpace(d);
        temp_cluster_id = temp;
        alg_hnsw_list.resize(1);
        cluster_entries.resize(NUM_GRAPH_CLUSTER);
        for (int i = 0; i < cluster_set.size(); i++) {
            // cluster_entries[i] = cluster_set[i][0];
            cluster_entries[i] = -1;
        }   

        alg_hnsw_list[0] = new hnswlib::HierarchicalNSW<float>(space_ptr, base.size() + 1, M_index, EF_index);
        alg_hnsw_list[0]->loadIndex(location + std::to_string(0) + ".bin", space_ptr);

        for(int tmpi = 0; tmpi < temp_cluster_id.size(); tmpi++) {
            int i = temp_cluster_id[tmpi];
            cluster_entries[i] = cluster_set[i][0];
            for (int j = 0; j < cluster_set[i].size(); j++) {
                alg_hnsw_list[0]->loadDataAddress(&base_vectors[cluster_set[i][j]], cluster_set[i][j]);
            }
            if (i % 1000 == 0) {
                std::cout << i << " ";
            }
        }
        std::cout << std::endl;
        alg_hnsw_list[0]->search_set.resize(alg_hnsw_list[0]->max_elements_);
        std::cout << "load time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

    void repair_fine_graph_structure(const std::vector<std::vector<hnswlib::labeltype>>& cluster_set) {
        double time = omp_get_wtime();
        for (int i = 0; i < cluster_set.size(); i++) {
            alg_hnsw_list[0]->entry_map.clear();
            // std::cout << "cluster: " << i << std::endl;
            if (i % 10000 == 0) {
                std::cout << i << std::endl;
            }
            for (int d: cluster_set[i]) {
                alg_hnsw_list[0]->entry_map[alg_hnsw_list[0]->label_lookup_[d]] = i;
            }
            // std::cout << solution.alg_hnsw_list[0]->entry_map.size() << " " << cluster_set[i].size() << std::endl;
            std::vector<std::pair<hnswlib::tableint, int>>search_result = alg_hnsw_list[0]->searchNodesForFix(cluster_set[i][0], i);
            int l = 0;
            for (int j = 1; j < cluster_set[i].size(); j++) {
                int d = alg_hnsw_list[0]->label_lookup_[cluster_set[i][j]];
                bool connect_success = false;
                if (alg_hnsw_list[0]->entry_map[d] != i + 1) {
                    while(l < search_result.size()) {
                        if (alg_hnsw_list[0]->canAddEdgeinter(search_result[l].first)) {
                            alg_hnsw_list[0]->mutuallyConnectTwoInterElement(search_result[l].first, d);
                            if (alg_hnsw_list[0]->canAddEdgeinter(d)) {
                                alg_hnsw_list[0]->mutuallyConnectTwoInterElement(d, search_result[l].first);
                            }
                            connect_success = true;
                            break;
                        } else {
                            l++;
                        }
                    }
                    std::vector<std::pair<hnswlib::tableint, int>> newsearch_result = alg_hnsw_list[0]->searchNodesForFix(cluster_set[i][j], i);
                    for (int k = 0; k < newsearch_result.size(); k++) {
                        search_result.push_back(newsearch_result[k]);
                    }
                }
            }
            // std::cout << "cluster id: " << i << " doc id: " << corresponding_doc_id[tmpi] << " cluster size: " << cluster_set[i].size() << " searched size: " << search_result.size() << " hop distance: " << reach_hop << std::endl;
            // std::cout <<search_result.size() << " " << can_reach << std::endl;
            // std::cout << alg_hnsw_list[0]->searchNodes(cluster_set[i][0], i).size() << std::endl;
        }
        std::cout << "load time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

public:
    int dimension;
    std::vector<vectorset> base_vectors;
    hnswlib::L2VSSpace* space_ptr;
    // hnswlib::HierarchicalNSW<float>* alg_hnsw;
    std::vector<int> temp_cluster_id;
    std::vector<int> cluster_entries;
    std::vector<std::vector<hnswlib::labeltype>> self_cluster_set;
    std::vector<hnswlib::HierarchicalNSW<float>*> alg_hnsw_list;
};


float half_to_float(uint16_t h) {
    // 参考 IEEE 754 半精度转换公式
    int s = (h >> 15) & 0x1;                   // 符号位
    int e = (h >> 10) & 0x1F;                  // 指数部分
    int f = h & 0x3FF;                         // 尾数部分
    if (e == 0) {                              // 次正规数
        return (s ? -1 : 1) * std::ldexp(f, -24);
    } else if (e == 31) {                      // 特殊值（NaN 或 Infinity）
        return (s ? -1 : 1) * (f ? NAN : INFINITY);
    } else {                                   // 规范化数
        return (s ? -1 : 1) * std::ldexp(f + 1024, e - 15 - 10);
    }
}


void load_from_msmarco(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query,
                       std::vector<int>& base_data_codes, std::vector<float>& center_data,
                       std::vector<float>& graph_center_data,
                       std::vector<std::vector<hnswlib::labeltype>>& cluster_set, 
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0; 
    long long code_offset = 0;
    long long all_codes = 0;  
    std::string cembfile_name = dataset_path + "cdata/centroids.npy";
    std::string qembfile_name = dataset_path + "qdata/qembs.npy";
    std::string qrelfile_name = dataset_path + "qdata/qrels.tsv"; 

    std::string cdocsfile_name = dataset_path + "cdata/coarse_cluster_info.txt"; 
    std::string gcembfile_name = dataset_path + "cdata/coarse_centroids.npy"; 


    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = dataset_path + "docdata/encoding" + std::to_string(i) + "_float16.npy";
        std::string codesfile_name = dataset_path + "docdata/doc_codes_" + std::to_string(i) + ".npy";
        std::string lensfile_name = dataset_path + "docdata/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray codes_npy = cnpy::npy_load(codesfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];

        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];

        int32_t* codes_data = codes_npy.data<int32_t>();
        size_t num_codes = codes_npy.shape[0];
        // std::cout << codes_npy.word_size << std::endl;
        // std::cout << sizeof(int32_t) << std::endl;
        // std::cout << sizeof(int16_t) << std::endl;
        // std::cout << sizeof(int) << std::endl;
        std::cout << num_codes << std::endl;
        std::cout << all_codes << std::endl;
        std::cout << "Processing file " << i << std::endl;
        
        for (long long i = 0; i < num_elements; ++i) {
            base_data[all_elements + i] = (static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        // std::cout << "?" << std::endl;
        for (long long i = 0; i < num_codes; ++i) {
            base_data_codes[all_codes + i] = static_cast<int>(codes_data[i]);
            if (i < 10) {
                // std::cout << i << " " << all_codes + i << std::endl;
                std::cout << codes_data[i] << " ";
                std::cout << base_data_codes[all_codes + i] << " ";
            }
        }
        std::cout << std::endl;

        all_elements += num_elements;
        all_codes += num_codes;
        
        for (int i = 0; i < doc_num; ++i) {
            base.push_back(vectorset(base_data.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
            code_offset += lens_data[i].real();
        }
    }

    cnpy::NpyArray cembs_npy = cnpy::npy_load(cembfile_name);
    uint16_t* raw_cembs_data = cembs_npy.data<uint16_t>();
    size_t num_cembs_elements = cembs_npy.shape[0] * cembs_npy.shape[1];
    for (size_t i = 0; i < num_cembs_elements; ++i) {
        center_data[i] = (static_cast<float>(half_to_float(raw_cembs_data[i])));
    }

    cnpy::NpyArray gcembs_npy = cnpy::npy_load(gcembfile_name);
    float* raw_gcembs_data = gcembs_npy.data<float>();
    size_t num_gcembs_elements = gcembs_npy.shape[0] * gcembs_npy.shape[1];
    for (size_t i = 0; i < num_gcembs_elements; ++i) {
        graph_center_data[i] = (static_cast<float>((raw_gcembs_data[i])));
    }
    // graph_center_data = center_data;
    // std::cout << num_cembs_elements << std::endl;

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = NUM_QUERY_SETS * qembs_npy.shape[1] * qembs_npy.shape[2];
    size_t q_num = NUM_QUERY_SETS;

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }
    
    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, QUERY_VECTOR_COUNT));
        q_offset += QUERY_VECTOR_COUNT * VECTOR_DIM;
    }
    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            qrels[num1].push_back(num2);
        }
    }
    file.close();

    std::ifstream codcsfile(cdocsfile_name);
    std::string cdocs_line;
    int lineid = 0;
    while (std::getline(codcsfile, cdocs_line)) { // 逐行读取
        std::istringstream iss(cdocs_line);  // 创建字符串流
        hnswlib::labeltype num1;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        while (iss >> num1) {
            cluster_set[lineid].push_back(num1);
        }
        if (lineid % 100 == 0) {
            std::cout << lineid << " " << cluster_set[lineid].size() << std::endl;
        }
        lineid++;
    }
    file.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}

void load_from_okvqa(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query,
                       std::vector<int>& base_data_codes, std::vector<float>& center_data,
                       std::vector<float>& graph_center_data,
                       std::vector<std::vector<hnswlib::labeltype>>& cluster_set, 
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0; 
    long long code_offset = 0;
    long long all_codes = 0;  

    std::string cembfile_name = dataset_path + "cdata/centroids.npy";
    std::string qembfile_name = dataset_path + "qdata/filterd_query.npy";
    std::string qrelfile_name = dataset_path + "qdata/qrels.tsv"; 
    std::string qlensfile_name = dataset_path + "qdata/filterd_query_len.npy";

    std::string cdocsfile_name = dataset_path + "cdata/coarse_cluster_info.txt"; 
    std::string gcembfile_name = dataset_path + "cdata/coarse_centroids.npy"; 

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = dataset_path + "docdata/encoding" + std::to_string(i) + "_float16.npy";
        std::string codesfile_name = dataset_path + "docdata/" + std::to_string(i) + ".codes.npy";
        std::string lensfile_name = dataset_path + "docdata/doclens" + std::to_string(i) + ".npy";

        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray codes_npy = cnpy::npy_load(codesfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];

        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];

        int32_t* codes_data = codes_npy.data<int32_t>();
        size_t num_codes = codes_npy.shape[0];
        // std::cout << codes_npy.word_size << std::endl;
        // std::cout << sizeof(int32_t) << std::endl;
        // std::cout << sizeof(int16_t) << std::endl;
        // std::cout << sizeof(int) << std::endl;
        std::cout << num_codes << std::endl;
        std::cout << all_codes << std::endl;
        std::cout << "Processing file " << i << std::endl;
        
        for (long long i = 0; i < num_elements; ++i) {
            base_data[all_elements + i] = (static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        // std::cout << "?" << std::endl;
        for (long long i = 0; i < num_codes; ++i) {
            base_data_codes[all_codes + i] = static_cast<int>(codes_data[i]);
            if (i < 10) {
                // std::cout << i << " " << all_codes + i << std::endl;
                std::cout << codes_data[i] << " ";
                std::cout << base_data_codes[all_codes + i] << " ";
            }
        }
        std::cout << std::endl;

        all_elements += num_elements;
        all_codes += num_codes;
        
        for (int i = 0; i < doc_num; ++i) {
            base.push_back(vectorset(base_data.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
            code_offset += lens_data[i].real();
        }
    }

    cnpy::NpyArray cembs_npy = cnpy::npy_load(cembfile_name);
    uint16_t* raw_cembs_data = cembs_npy.data<uint16_t>();
    size_t num_cembs_elements = cembs_npy.shape[0] * cembs_npy.shape[1];
    for (size_t i = 0; i < num_cembs_elements; ++i) {
        center_data[i] = (static_cast<float>(half_to_float(raw_cembs_data[i])));
    }

    cnpy::NpyArray gcembs_npy = cnpy::npy_load(gcembfile_name);
    float* raw_gcembs_data = gcembs_npy.data<float>();
    size_t num_gcembs_elements = gcembs_npy.shape[0] * gcembs_npy.shape[1];
    for (size_t i = 0; i < num_gcembs_elements; ++i) {
        graph_center_data[i] = (static_cast<float>((raw_gcembs_data[i])));
    }
    // graph_center_data = center_data;
    // std::cout << num_cembs_elements << std::endl;

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
    cnpy::NpyArray qlens_npy = cnpy::npy_load(qlensfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1];
    size_t q_num = NUM_QUERT_OKVQA;

    std::complex<int>* qlens_data = qlens_npy.data<std::complex<int>>();

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }

    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, qlens_data[i].real()));
        q_offset += qlens_data[i].real() * VECTOR_DIM;
    }

    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            if (num1 < 0 || num1 >= q_num) {
                continue;
                // std::cerr << "?" << line << std::endl;
            } else {
                // std::cout << num1 << " " << num2 << std::endl;
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();

    std::ifstream codcsfile(cdocsfile_name);
    std::string cdocs_line;
    int lineid = 0;
    while (std::getline(codcsfile, cdocs_line)) { // 逐行读取
        std::istringstream iss(cdocs_line);  // 创建字符串流
        hnswlib::labeltype num1;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        while (iss >> num1) {
            cluster_set[lineid].push_back(num1);
        }
        if (lineid % 100 == 0) {
            std::cout << lineid << " " << cluster_set[lineid][cluster_set[lineid].size()-1] << std::endl;
        }
        lineid++;
    }
    file.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}


void load_from_evqa(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query,
                       std::vector<int>& base_data_codes, std::vector<float>& center_data,
                       std::vector<float>& graph_center_data,
                       std::vector<std::vector<hnswlib::labeltype>>& cluster_set, 
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0; 
    long long code_offset = 0;
    long long all_codes = 0;  

    std::string cembfile_name = dataset_path + "cdata/centroids.npy";
    std::string qembfile_name = dataset_path + "qdata/filterd_query.npy";
    std::string qrelfile_name = dataset_path + "qdata/qrels.tsv"; 
    std::string qlensfile_name = dataset_path + "qdata/filterd_query_len.npy";
    
    std::string cdocsfile_name = dataset_path + "cdata/coarse_cluster_info.txt"; 
    std::string gcembfile_name = dataset_path + "cdata/coarse_centroids.npy"; 

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = dataset_path + "docdata/encoding" + std::to_string(i) + "_float16.npy";
        std::string codesfile_name = dataset_path + "docdata/" + std::to_string(i) + ".codes.npy";
        std::string lensfile_name = dataset_path + "docdata/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray codes_npy = cnpy::npy_load(codesfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];

        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];

        int32_t* codes_data = codes_npy.data<int32_t>();
        size_t num_codes = codes_npy.shape[0];
        // std::cout << codes_npy.word_size << std::endl;
        // std::cout << sizeof(int32_t) << std::endl;
        // std::cout << sizeof(int16_t) << std::endl;
        // std::cout << sizeof(int) << std::endl;
        std::cout << num_codes << std::endl;
        std::cout << all_codes << std::endl;
        std::cout << "Processing file " << i << std::endl;
        
        for (long long i = 0; i < num_elements; ++i) {
            base_data[all_elements + i] = (static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        // std::cout << "?" << std::endl;
        for (long long i = 0; i < num_codes; ++i) {
            base_data_codes[all_codes + i] = static_cast<int>(codes_data[i]);
            if (i < 10) {
                // std::cout << i << " " << all_codes + i << std::endl;
                std::cout << codes_data[i] << " ";
                std::cout << base_data_codes[all_codes + i] << " ";
            }
        }
        std::cout << std::endl;

        all_elements += num_elements;
        all_codes += num_codes;
        
        for (int i = 0; i < doc_num; ++i) {
            base.push_back(vectorset(base_data.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
            code_offset += lens_data[i].real();
        }
    }

    cnpy::NpyArray cembs_npy = cnpy::npy_load(cembfile_name);
    uint16_t* raw_cembs_data = cembs_npy.data<uint16_t>();
    size_t num_cembs_elements = cembs_npy.shape[0] * cembs_npy.shape[1];
    for (size_t i = 0; i < num_cembs_elements; ++i) {
        center_data[i] = (static_cast<float>(half_to_float(raw_cembs_data[i])));
    }

    cnpy::NpyArray gcembs_npy = cnpy::npy_load(gcembfile_name);
    float* raw_gcembs_data = gcembs_npy.data<float>();
    size_t num_gcembs_elements = gcembs_npy.shape[0] * gcembs_npy.shape[1];
    for (size_t i = 0; i < num_gcembs_elements; ++i) {
        graph_center_data[i] = (static_cast<float>((raw_gcembs_data[i])));
    }
    // graph_center_data = center_data;
    // std::cout << num_cembs_elements << std::endl;

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
    cnpy::NpyArray qlens_npy = cnpy::npy_load(qlensfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1];
    size_t q_num = NUM_QUERT_EVQA;

    std::complex<int>* qlens_data = qlens_npy.data<std::complex<int>>();

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }

    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, qlens_data[i].real()));
        q_offset += qlens_data[i].real() * VECTOR_DIM;
    }

    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            if (num1 < 0 || num1 >= q_num) {
                continue;
                // std::cerr << "?" << line << std::endl;
            } else {
                // std::cout << num1 << " " << num2 << std::endl;
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();

    std::ifstream codcsfile(cdocsfile_name);
    std::string cdocs_line;
    int lineid = 0;
    while (std::getline(codcsfile, cdocs_line)) { // 逐行读取
        std::istringstream iss(cdocs_line);  // 创建字符串流
        hnswlib::labeltype num1;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        while (iss >> num1) {
            cluster_set[lineid].push_back(num1);
        }
        if (lineid % 100 == 0) {
            std::cout << lineid << " " << cluster_set[lineid][cluster_set[lineid].size()-1] << std::endl;
        }
        lineid++;
    }
    file.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}

void load_from_lotte(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query,
                       std::vector<int>& base_data_codes, std::vector<float>& center_data,
                       std::vector<float>& graph_center_data,
                       std::vector<std::vector<hnswlib::labeltype>>& cluster_set, 
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0; 
    long long code_offset = 0;
    long long all_codes = 0;  

    std::string cembfile_name = dataset_path + "cdata/centroids.npy";
    std::string qembfile_name = dataset_path + "qdata/lotte_pooled_dev_query.npy";
    std::string qrelfile_name = dataset_path + "qdata/qas.search.tsv"; 

    std::string cdocsfile_name = dataset_path + "cdata/coarse_cluster_info.txt"; 
    std::string gcembfile_name = dataset_path + "cdata/coarse_centroids.npy"; 

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = dataset_path + "docdata/encoding" + std::to_string(i) + "_float16.npy";
        std::string codesfile_name = dataset_path + "docdata/doc_codes_" + std::to_string(i) + ".npy";
        std::string lensfile_name = dataset_path + "docdata/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray codes_npy = cnpy::npy_load(codesfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];

        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];

        int32_t* codes_data = codes_npy.data<int32_t>();
        size_t num_codes = codes_npy.shape[0];
        // std::cout << codes_npy.word_size << std::endl;
        // std::cout << sizeof(int32_t) << std::endl;
        // std::cout << sizeof(int16_t) << std::endl;
        // std::cout << sizeof(int) << std::endl;
        std::cout << num_codes << std::endl;
        std::cout << all_codes << std::endl;
        std::cout << "Processing file " << i << std::endl;
        
        for (long long i = 0; i < num_elements; ++i) {
            base_data[all_elements + i] = (static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        // std::cout << "?" << std::endl;
        for (long long i = 0; i < num_codes; ++i) {
            base_data_codes[all_codes + i] = static_cast<int>(codes_data[i]);
            if (i < 10) {
                // std::cout << i << " " << all_codes + i << std::endl;
                std::cout << codes_data[i] << " ";
                std::cout << base_data_codes[all_codes + i] << " ";
            }
        }
        std::cout << std::endl;

        all_elements += num_elements;
        all_codes += num_codes;
        
        for (int i = 0; i < doc_num; ++i) {
            base.push_back(vectorset(base_data.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
            code_offset += lens_data[i].real();
        }
    }

    cnpy::NpyArray cembs_npy = cnpy::npy_load(cembfile_name);
    uint16_t* raw_cembs_data = cembs_npy.data<uint16_t>();
    size_t num_cembs_elements = cembs_npy.shape[0] * cembs_npy.shape[1];
    for (size_t i = 0; i < num_cembs_elements; ++i) {
        center_data[i] = (static_cast<float>(half_to_float(raw_cembs_data[i])));
    }

    cnpy::NpyArray gcembs_npy = cnpy::npy_load(gcembfile_name);
    float* raw_gcembs_data = gcembs_npy.data<float>();
    size_t num_gcembs_elements = gcembs_npy.shape[0] * gcembs_npy.shape[1];
    for (size_t i = 0; i < num_gcembs_elements; ++i) {
        graph_center_data[i] = (static_cast<float>((raw_gcembs_data[i])));
    }
    // graph_center_data = center_data;
    // std::cout << num_cembs_elements << std::endl;

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = NUM_QUERT_LOTTE * qembs_npy.shape[1] * qembs_npy.shape[2];
    size_t q_num = NUM_QUERT_LOTTE;

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }
    
    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, QUERY_VECTOR_COUNT));
        q_offset += QUERY_VECTOR_COUNT * VECTOR_DIM;
    }
    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            if (num1 < 0 || num1 >= q_num) {
                continue;
                // std::cerr << "?" << line << std::endl;
            } else {
                // std::cout << num1 << " " << num2 << std::endl;
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();

    std::ifstream codcsfile(cdocsfile_name);
    std::string cdocs_line;
    int lineid = 0;
    while (std::getline(codcsfile, cdocs_line)) { // 逐行读取
        std::istringstream iss(cdocs_line);  // 创建字符串流
        hnswlib::labeltype num1;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        while (iss >> num1) {
            cluster_set[lineid].push_back(num1);
        }
        if (lineid % 100 == 0) {
            std::cout << lineid << " " << cluster_set[lineid][cluster_set[lineid].size()-1] << std::endl;
        }
        lineid++;
    }
    file.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}

void load_msmarco_train_query(std::vector<float>& query_data, std::vector<vectorset>& query, 
                      std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0;   
    std::string qembfile_name = "/home/zhoujin/data/train_query.npy";
    std::string qrelfile_name = "/home/zhoujin/data/qrels.train.select.reorder.tsv";    

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1] * qembs_npy.shape[2];
    size_t q_num = qembs_npy.shape[0];

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }
    std::cout << num_qembs_elements << std::endl;
    
    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, VECTOR_DIM, QUERY_VECTOR_COUNT));
        q_offset += QUERY_VECTOR_COUNT * VECTOR_DIM;
    }
    std::cout << query.size() << std::endl;
    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            qrels[num1].push_back(num2);
        }
    }
    file.close();
    std::cout << "load train query finish! query count: " << query.size() << " " << qrels.size() << std::endl;
}

void load_msmarco_train_addedge(std::vector<std::pair<int, int>>& edge_pair) {
    std::string edgefile_name = "/home/zhoujin/project/forremove/VecSetSearch/hnswlib/examples/msmarco_add_edge_100k_top5.txt";    
    std::ifstream file(edgefile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            edge_pair.push_back(std::make_pair(num1, num2));
        }
    }
    file.close();
    std::cout << "load train edge finish! edge count: " << edge_pair.size() << std::endl;
}


double calculate_recall_for_datasetlabel(const std::vector<std::pair<int, float>>& solution_indices,
                        const std::vector<int>& ground_truth_indices) {
    std::unordered_set<int> solution_set;
    std::unordered_set<int> ground_truth_set;
    for (const auto& pair : solution_indices) {
        solution_set.insert(pair.first);
        // if (solution_set.size() >= K) {
        //     break;
        // }
    }
    for (const auto& pid : ground_truth_indices) {
        ground_truth_set.insert(pid);
    }
    int intersection_count = 0;
    for (const int& index : solution_set) {
        if (ground_truth_set.find(index) != ground_truth_set.end()) {
            intersection_count++;
        }
    }

    double recall = static_cast<double>(intersection_count) / ground_truth_set.size();
    // double recall = static_cast<double>(intersection_count > 0);
    return recall;
}

double calculate_hitrate_for_datasetlabel(const std::vector<std::pair<int, float>>& solution_indices,
                        const std::vector<int>& ground_truth_indices) {
    std::unordered_set<int> solution_set;
    std::unordered_set<int> ground_truth_set;
    for (const auto& pair : solution_indices) {
        solution_set.insert(pair.first);
    }
    for (const auto& pid : ground_truth_indices) {
        ground_truth_set.insert(pid);
    }
    int intersection_count = 0;
    for (const int& index : solution_set) {
        if (ground_truth_set.find(index) != ground_truth_set.end()) {
            intersection_count++;
        }
    }
    // double hitrate = static_cast<double>(intersection_count) / ground_truth_set.size();
    double hitrate = static_cast<double>(intersection_count > 0);
    return hitrate;
}

double calculate_mrr_for_datasetlabel(const std::vector<std::pair<int, float>>& solution_indices,
    const std::vector<int>& ground_truth_indices) {
    std::unordered_set<int> ground_truth_set;
    for (const auto& pid : ground_truth_indices) {
        ground_truth_set.insert(pid);
    }
    for (size_t rank = 0; rank < K; ++rank) {
        int pid = solution_indices[rank].first;
        if (ground_truth_set.find(pid) != ground_truth_set.end()) {
            // MRR: Reciprocal of (1-based) rank
            return 1.0 / (rank + 1);
        }
    }
    // No relevant document found
    return 0.0;
}

int main() {
    omp_set_nested(1);

    std::vector<float> base_data;
    std::vector<int> base_vec_num;
    std::vector<float> query_data;
    std::vector<float> train_query_data;
    std::vector<vectorset> base;
    std::vector<vectorset> query;
    std::vector<vectorset> train_query;
    std::vector<int> base_data_codes;
    std::vector<float> center_data;
    std::vector<float> graph_center_data;
    std::vector<float> query_cluster_scores;
    std::vector<float> test_query_cluster_scores;
    std::vector<std::vector<hnswlib::labeltype>> cluster_set;
    std::vector<std::vector<int>> qrels;
    std::vector<std::vector<int>> train_qrels;

    train_query_data.resize((long long) 808731 * 128 * 32);
    load_msmarco_train_query(train_query_data, train_query, train_qrels);

    bool rebuild = false;
    bool save_result = false;

    std::string index_file, save_result_file;

    std::vector<int> temp_cluster_id(NUM_GRAPH_CLUSTER);
    for (int i = 0;i < NUM_GRAPH_CLUSTER; i++) {
        temp_cluster_id[i] = i;
    }
    std::cout << temp_cluster_id.size() << std::endl;

    if (dataset == 0) {
        NUM_BASE_SETS = NUM_BASE_SETS_MS;
        NUM_QUERY_SETS = NUM_QUERT_MS;
        NUM_CLUSTER = NUM_CLUSTER_MS;
        NUM_GRAPH_CLUSTER = NUM_GRAPH_CLUSTER_MS;
        QUERY_VECTOR_COUNT = 32;
        dataset_path = dataset_path + "msmarco/";
        index_file = "../../example_index/msmarcoIndex" + std::to_string(NUM_GRAPH_CLUSTER_MS) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "/";
        save_result_file = "../../example_index/msmarco_results_" + std::to_string(NUM_GRAPH_CLUSTER_MS) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
        std::cout << index_file << std::endl;
        // test on all msmacro dataset
        base_data.resize((long long) 25000 * MSMACRO_TEST_NUMBER * 128 * 80);
        query_data.resize((long long) NUM_QUERY_SETS * 128 * QUERY_VECTOR_COUNT + 1);
        base_data_codes.resize((long long) 25000 * MSMACRO_TEST_NUMBER * 80);
        center_data.resize((long long) NUM_CLUSTER * 128);
        graph_center_data.resize((long long) NUM_GRAPH_CLUSTER * 128);
        // cluster_set.resize(NUM_GRAPH_CLUSTER);
        cluster_set.resize(NUM_GRAPH_CLUSTER);
        // std::cout<< (long long) 25000 * MSMACRO_TEST_NUMBER * 80 << std::endl;
        load_from_msmarco(base_data, base, query_data, query, base_data_codes, center_data, graph_center_data, cluster_set, MSMACRO_TEST_NUMBER, qrels);
    }
    else if (dataset == 1) {
        NUM_BASE_SETS = NUM_BASE_SETS_LOTTE;
        NUM_QUERY_SETS = NUM_QUERT_LOTTE;
        NUM_CLUSTER = NUM_CLUSTER_LOTTE;
        NUM_GRAPH_CLUSTER = NUM_GRAPH_CLUSTER_LOTTE;
        QUERY_VECTOR_COUNT = 32;
        dataset_path = dataset_path + "lotte/";
        base_data.resize((long long) NUM_BASE_VECTOR_LOTTE * 128);
        query_data.resize((long long) NUM_QUERT_LOTTE * 128 * QUERY_VECTOR_COUNT);
        base_data_codes.resize((long long) NUM_BASE_VECTOR_LOTTE);
        center_data.resize((long long) NUM_CLUSTER * 128);
        graph_center_data.resize((long long) NUM_GRAPH_CLUSTER * 128);
        cluster_set.resize(NUM_GRAPH_CLUSTER);
        load_from_lotte(base_data, base, query_data, query, base_data_codes, center_data, graph_center_data, cluster_set, LOTTE_TEST_NUMBER, qrels);
        index_file = "../../example_index/lotteIndex" + std::to_string(NUM_GRAPH_CLUSTER_LOTTE) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "/";
        save_result_file = "../../example_index/lotte_results_" + std::to_string(NUM_GRAPH_CLUSTER_LOTTE) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
    } else if (dataset == 2) {
        // OKVQA
        NUM_BASE_SETS = NUM_BASE_SETS_OKVQA;
        NUM_QUERY_SETS = NUM_QUERT_OKVQA;
        NUM_CLUSTER = NUM_CLUSTER_OKVQA;
        NUM_GRAPH_CLUSTER = NUM_GRAPH_CLUSTER_OKVQA;
        QUERY_VECTOR_COUNT = 320;
        dataset_path = dataset_path + "okvqa/";
        base_data.resize((long long) NUM_BASE_VECTOR_OKVQA * 128);
        query_data.resize((long long) NUM_QUERT_OKVQA * 128 * QUERY_VECTOR_COUNT);
        base_data_codes.resize((long long) NUM_BASE_VECTOR_OKVQA);
        center_data.resize((long long) NUM_CLUSTER * 128);
        graph_center_data.resize((long long) NUM_GRAPH_CLUSTER * 128);
        cluster_set.resize(NUM_GRAPH_CLUSTER);
        load_from_okvqa(base_data, base, query_data, query, base_data_codes, center_data, graph_center_data, cluster_set, OKVQA_TEST_NUMBER, qrels);
        index_file = "../../example_index/okvqaIndex" + std::to_string(NUM_GRAPH_CLUSTER_OKVQA) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "/";
        save_result_file = "../../example_index/okvqa_results_" + std::to_string(NUM_GRAPH_CLUSTER_OKVQA) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
    } else if (dataset == 3) {
        // OKVQA
        NUM_BASE_SETS = NUM_BASE_SETS_EVQA;
        NUM_QUERY_SETS = NUM_QUERT_EVQA;
        NUM_CLUSTER = NUM_CLUSTER_EVQA;
        NUM_GRAPH_CLUSTER = NUM_GRAPH_CLUSTER_EVQA;
        QUERY_VECTOR_COUNT = 320;
        dataset_path = dataset_path + "evqa/"; 
        base_data.resize((long long) NUM_BASE_VECTOR_EVQA * 128);
        query_data.resize((long long) NUM_QUERT_EVQA * 128 * QUERY_VECTOR_COUNT);
        base_data_codes.resize((long long) NUM_BASE_VECTOR_EVQA);
        center_data.resize((long long) NUM_CLUSTER * 128);
        graph_center_data.resize((long long) NUM_GRAPH_CLUSTER * 128);
        cluster_set.resize(NUM_GRAPH_CLUSTER);
        load_from_evqa(base_data, base, query_data, query, base_data_codes, center_data, graph_center_data, cluster_set, EVQA_TEST_NUMBER, qrels);
        index_file = "../../example_index/evqaIndex" + std::to_string(NUM_GRAPH_CLUSTER_EVQA) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "/";
        save_result_file = "../../example_index/evqa_results_" + std::to_string(NUM_GRAPH_CLUSTER_EVQA) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
    }
    std::remove(save_result_file.c_str());
    std::vector<float> col_query_cluster_scores(NUM_CLUSTER * QUERY_VECTOR_COUNT);
    test_query_cluster_scores.resize(NUM_CLUSTER * QUERY_VECTOR_COUNT);
    col_query_cluster_scores.resize(NUM_CLUSTER * QUERY_VECTOR_COUNT);

    Solution solution;
    if (rebuild) {
        std::vector<float>cluster_distance((long long) NUM_CLUSTER * NUM_CLUSTER);
        hnswlib::fast_dot_product_blas(NUM_CLUSTER, 128, NUM_CLUSTER, center_data.data(), center_data.data(), cluster_distance.data()); 
        solution.build_fine_cluster(VECTOR_DIM, base, cluster_set, temp_cluster_id, cluster_distance);
        solution.save_fine_cluster(index_file);
    } else {
        solution.load_fine_cluster(index_file, VECTOR_DIM, base, cluster_set, temp_cluster_id);
        solution.repair_fine_graph_structure(cluster_set);
        std::vector<std::pair<int, int>> edge_pair;
        // load_msmarco_train_addedge(edge_pair);
        // int connect_count = 0;
        // for (size_t i = 0; i < edge_pair.size(); ++i) {
        //     if (solution.alg_hnsw_list[0]->canAddEdgeinter(edge_pair[i].first)) {
        //         solution.alg_hnsw_list[0]->mutuallyConnectTwoInterElement(edge_pair[i].first, edge_pair[i].second);
        //         if (solution.alg_hnsw_list[0]->canAddEdgeinter(edge_pair[i].second)) {
        //             solution.alg_hnsw_list[0]->mutuallyConnectTwoInterElement(edge_pair[i].second, edge_pair[i].first);
        //         }
        //         connect_count += 1;
        //     }
        // }
    }


    std::vector<int> reranklist = {128, 256, 378, 512};
    for (int r: reranklist) {
        rerankK = r;
        for (int tmpef: eflist) {
            double total_dataset_hnsw_recall = 0.0;
            double total_query_time = 0.0;
            double total_dataset_hnsw_mrr = 0.0;
            double total_dataset_hitrate = 0.0;

            std::cout<<"Processing Queries HNSW"<<std::endl;
        
            for (int i = 0; i < NUM_QUERY_SETS; ++i) {
                std::vector<std::pair<int, float>> solution_indices;
                double query_time = solution.search_with_fine_cluster(query[i], test_query_cluster_scores, col_query_cluster_scores, center_data, graph_center_data, K, tmpef, solution_indices);     
                total_query_time += query_time;
                double dataset_hnsw_recall = calculate_recall_for_datasetlabel(solution_indices, qrels[i]);
                double dataset_hnsw_mrr = calculate_mrr_for_datasetlabel(solution_indices, qrels[i]);
                double dataset_hitrate = calculate_hitrate_for_datasetlabel(solution_indices, qrels[i]);

                total_dataset_hnsw_recall += dataset_hnsw_recall;
                total_dataset_hnsw_mrr += dataset_hnsw_mrr;
                total_dataset_hitrate += dataset_hitrate;
                std::cout << "Recall for query set " << i << ": " << dataset_hnsw_recall << " " << dataset_hnsw_mrr << " | " << query_time << std::endl;
            }
            std::cout << "rerankK: " << rerankK << " ef: " << tmpef << std::endl;
            std::cout << "Average our method recall v.s. dataset label: " << total_dataset_hnsw_recall / NUM_QUERY_SETS << std::endl;
            std::cout << "Average our method mrr v.s. dataset label: " << total_dataset_hnsw_mrr/ NUM_QUERY_SETS << std::endl;
            std::cout << "Average our method hitrate v.s. dataset label: " << total_dataset_hitrate/ NUM_QUERY_SETS << std::endl;
            std::cout << "Average query time: " << total_query_time/NUM_QUERY_SETS << " seconds" << std::endl;
        }
    }
    return 0;

}
