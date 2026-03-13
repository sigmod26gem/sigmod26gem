#include "solution.cpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

template<class type>
void load_vecs_data(const char* filename, vector<type> &results, unsigned &num, unsigned &dim) {
    ifstream in(filename, ios::binary);
    in.read((char *)&dim, sizeof(int));

    in.seekg(0, ios::end);
    ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (sizeof(int) + dim * sizeof(type)));
    results.reserve(num * dim);
    results.resize(num * dim);
    in.seekg(0, ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(sizeof(int), ios::cur);
        in.read((char*)(results.data() + i * dim), dim * sizeof(type));
    }
    in.close();
}

float compute_recall(const vector<int>& search_result, const vector<int>& true_result) {
    int relevant = 0;
    for (int idx : true_result) {
        if (find(search_result.begin(), search_result.end(), idx) != search_result.end()) {
            relevant++;
        }
    }
    return static_cast<float>(relevant) / true_result.size();
}

int main() {
    unsigned k = 10;
    unsigned nb, nq, dim, truthk;
    vector<float> base, queries;
    vector<int> truth;

    load_vecs_data("/root/sift/sift_base.fvecs", base, nb, dim);
    load_vecs_data("/root/sift/sift_query.fvecs", queries, nq, dim);
    load_vecs_data("/root/sift/sift_groundtruth.ivecs", truth, nq, truthk);

    Solution solution;

    auto build_start = high_resolution_clock::now();
    solution.build(dim, base);
    double build_time = duration_cast<milliseconds>(high_resolution_clock::now() - build_start).count();
    std::cout << "Build Time: " << build_time / 1000 << endl;

    auto search_start = high_resolution_clock::now();
    vector<int> search_result(nq * k);
    solution.search(queries, search_result.data());
    float total_recall = 0.0;
    for (int i = 0; i < nq; ++i) {
        vector<int> true_result = vector<int>(truth.begin() + i * truthk, truth.begin() + i * truthk + k);

        float recall = compute_recall(vector<int>(search_result.begin() + i * k, search_result.begin() + (i + 1) * k), true_result);
        total_recall += recall;
    }
    double search_time = duration_cast<milliseconds>(high_resolution_clock::now() - search_start).count();
    std::cout << "Search Time: " << search_time / 1000 << endl;

    cout << "Average Recall: " << total_recall / nq << endl;

    return 0;
}
