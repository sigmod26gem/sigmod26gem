#pragma once 
#include <vector>
#include <chrono>
#include <iostream>
#include <utility>
#include <algorithm>

#include "./bindings.cpp"

const int NUM_THREADS = 16;

class Solution {
public:
    Solution() = default;

    ~Solution()
    {
        delete m_graph;
        delete m_searcher;
    }

    void build(int d, const std::vector<float> &base, const std::vector<int> &vec_num){
        R = 64;
        L = 300;
        EF = 150;
        dim = d;
        int rows = base.size() / d;
        int n = vec_num.size();
        std::cout<<"building start"<<std::endl;
        std::unique_ptr<glass::FP32VCQuantizer<glass::Metric::L2, 128>> buildQuant(new glass::FP32VCQuantizer<glass::Metric::L2, 128>(d));
        std::cout<<"start training"<<std::endl;
        buildQuant->train(base.data(), vec_num.data(), vec_num.size());
        std::cout<<"Build Quant finished" <<std::endl;
        std::unique_ptr<Index> hnsw_index(new Index("HNSW", d, "L2", R, L));
        m_graph = new Graph(hnsw_index->build((float *)buildQuant->get_data(0), rows));
        std::cout<<"Build Graph finished"<<std::endl;
        m_searcher = new SearcherVS(*m_graph, base.data(), rows, d, "L2", )
    }

    void search(const std::vector<float> & query, const int num_vec, const int k, int* res){

    }
private:
    Graph *m_graph;
    Searcher *m_searcher;
    int R;
    int L;
    int EF;
    int LEVEL;
    int dim;
};