#include <vector>
#include <chrono>
#include <iostream>
#include <utility>
#include <algorithm>

#include "rnndescent/rnndescent.h"

#include "bindings.cpp"

const int NUM_THREADS = 16;

class Solution {
public:
    Solution() = default;

    ~Solution()
    {
        delete m_graph;
        delete m_searcher;
    }

    void build(int d, const std::vector<float> &base)
    {
        dim = d;
        int rows = base.size() / d;
        rnndescent::rnn_para para;
        if (d <= 512) {
            para.K0 = 72;
            para.S = 132;
            para.R = para.K0 * 3;
            para.T1 = 3;
            para.T2 = 8;
            R = 64;
            L = 400;
            EF = 350;
            LEVEL = 11;
        }
        else if(d == 1024) {
            para.K0 = 40;
            para.S = 152;
            para.R = para.K0 * 4;
            para.T1 = 3;
            para.T2 = 8;
            R = 40;
            L = 200;
            EF = 105;
            LEVEL = 12;
        }
        else {
            para.K0 = 64;
            para.S = 208;
            para.R = para.K0 * 5;
            para.T1 = 3;
            para.T2 = 8;
            R = 64;
            L = 300;
            EF = 150;
            LEVEL = 13;
        }

        rnndescent::Matrix<uint8_t> base_data;
    
        if (d <= 512) {
            const int d_reduced = 384;
            std::unique_ptr<glass::SQ8RPQuantizer<glass::Metric::L2, d_reduced>> buildQuant(new glass::SQ8RPQuantizer<glass::Metric::L2, d_reduced>(d));
            buildQuant->train(base.data(), rows);
            
            base_data.load((uint8_t *)buildQuant->get_data(0), rows, d_reduced);

            std::unique_ptr<IndexSQ8> hnsw_index(new IndexSQ8("HNSWSQ8", d_reduced, "L2SQ8", R, L));
            m_graph = new Graph(hnsw_index->build((uint8_t *)buildQuant->get_data(0), rows));
        }
        else if (d == 1024) {
            const int d_reduced = 256;
            std::unique_ptr<glass::SQ8RPQuantizer<glass::Metric::L2, d_reduced>> buildQuant(new glass::SQ8RPQuantizer<glass::Metric::L2, d_reduced>(d));
            buildQuant->train(base.data(), rows);
            
            base_data.load((uint8_t *)buildQuant->get_data(0), rows, d_reduced);

            std::unique_ptr<IndexSQ8> hnsw_index(new IndexSQ8("HNSWSQ8", d_reduced, "L2SQ8", R, L));
            m_graph = new Graph(hnsw_index->build((uint8_t *)buildQuant->get_data(0), rows));
        }
        else {
            const int d_reduced = 384;
            std::unique_ptr<glass::SQ8RPQuantizer<glass::Metric::L2, d_reduced>> buildQuant(new glass::SQ8RPQuantizer<glass::Metric::L2, d_reduced>(d));
            buildQuant->train(base.data(), rows);
            
            base_data.load((uint8_t *)buildQuant->get_data(0), rows, d_reduced);

            std::unique_ptr<IndexSQ8> hnsw_index(new IndexSQ8("HNSWSQ8", d_reduced, "L2SQ8", R, L));
            m_graph = new Graph(hnsw_index->build((uint8_t *)buildQuant->get_data(0), rows));
        }

        rnndescent::MatrixOracleSQ8<uint8_t, rnndescent::metric::l2sqr> oracle(base_data);
        std::unique_ptr<rnndescent::RNNDescent> rnnd_index(new rnndescent::RNNDescent(oracle, para));

        auto start = std::chrono::high_resolution_clock::now();
        rnnd_index->build(oracle.size(), true);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "RNND build time in milliseconds: "
            << 1.0 * std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000
            << " s" << std::endl;

        m_graph->graph.data = (int *)glass::alloc2M((size_t)rnnd_index->ntotal * rnnd_index->K0 * sizeof(int));
        m_graph->graph.N = rnnd_index->ntotal;
        m_graph->graph.K = rnnd_index->K0;
#pragma omp parallel for
        for (int u = 0; u < rnnd_index->ntotal; ++u) {
            auto &pool = rnnd_index->graph[u].pool;
            int K = std::min(rnnd_index->K0, (int)pool.size());
            for (int m = 0; m < K; ++m) {
                m_graph->graph.data[u * rnnd_index->K0 + m] = pool[m].id;
            }
        }

        m_searcher = new Searcher(*m_graph, base.data(), rows, d, "L2", LEVEL);
        m_searcher->optimize(NUM_THREADS);
        m_searcher->set_ef(EF);
    }

    void search(const std::vector<float> &query, int *res)
    {
        m_searcher->batch_search(query, 10, res, NUM_THREADS);
    }

private:
    Graph *m_graph;
    Searcher *m_searcher;
    int R;
    int L;
    int EF;
    int LEVEL;
};

