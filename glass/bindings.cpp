#include <thread>

#include <omp.h>

#include "glass/builder.h"
#include "glass/hnsw/hnsw.h"
#include "glass/nsg/nsg.h"
#include "glass/searcher.h"

// void set_num_threads(int num_threads)
// {
//     omp_set_num_threads(num_threads);
// }

struct Graph {
    glass::Graph<int> graph;

    Graph() = default;

    explicit Graph(const Graph &rhs) : graph(rhs.graph)
    {}

    explicit Graph(const std::string &filename)
    {
        graph.load(filename);
    }

    explicit Graph(const glass::Graph<int> &graph) : graph(graph)
    {}

    void save(const std::string &filename)
    {
        graph.save(filename);
    }

    void load(const std::string &filename)
    {
        graph.load(filename);
    }
};

struct Index {
    std::unique_ptr<glass::Builder> index = nullptr;

    Index() = default;

    Index(const std::string &index_type, int dim, const std::string &metric, int R = 32, int L = 200)
    {
        if (index_type == "NSG") {
            index = std::unique_ptr<glass::Builder>((glass::Builder *)new glass::NSG(dim, metric, R, L));
        } else if (index_type == "HNSW") {
            index = std::unique_ptr<glass::Builder>((glass::Builder *)new glass::HNSW(dim, metric, R, L));
        } else {
            printf("Index type [%s] not supported\n", index_type.c_str());
        }
    }

    Graph build(const float *data, int rows)
    {
        index->Build((float *)data, rows);
        return Graph(index->GetGraph());
    }
};

struct IndexVS {
    std::unique_ptr<glass::Builder> index = nullptr;

    IndexVS() = default;

    IndexVS(const std::string &index_type, int dim, const std::string &metric, int R = 32, int L = 200)
    {
        if (index_type == "NSG") {
            index = std::unique_ptr<glass::Builder>((glass::Builder *)new glass::NSG(dim, metric, R, L));
        } else if (index_type == "HNSW") {
            index = std::unique_ptr<glass::Builder>((glass::Builder *)new glass::HNSW(dim, metric, R, L));
        } else {
            printf("Index type [%s] not supported\n", index_type.c_str());
        }
    }

    Graph build(const float *data, int rows)
    {
        index->Build((float *)data, rows);
        return Graph(index->GetGraph());
    }
};

struct IndexSQ8 {
    std::unique_ptr<glass::HNSWSQ8> index = nullptr;

    IndexSQ8() = default;

    IndexSQ8(const std::string &index_type, int dim, const std::string &metric, int R = 32, int L = 200)
    {
        if (index_type == "HNSWSQ8") {
            index = std::unique_ptr<glass::HNSWSQ8>(new glass::HNSWSQ8(dim, metric, R, L));
        } else {
            printf("Index type [%s] not supported\n", index_type.c_str());
        }
    }

    Graph build(const uint8_t *data, int rows)
    {
        index->Build((uint8_t *)data, rows);
        return Graph(index->GetGraph());
    }
};

struct Searcher {

    int dim;

    std::unique_ptr<glass::SearcherBase> searcher;

    Searcher() = default;

    Searcher(const Graph &graph, const float *data, int rows, int features, const std::string &metric, int level)
        : dim(features), searcher(std::unique_ptr<glass::SearcherBase>(glass::create_searcher(graph.graph, metric, level)))
    {
        searcher->SetData(data, rows, features);
    }

    void search(const float *query, int k, int *ids)
    {
        searcher->Search(query, k, ids);
    }

    void batch_search(const std::vector<float> &query, int k, int *ids, int num_threads = 0) {
        int nq = query.size() / dim;
        if (num_threads != 0) {
            omp_set_num_threads(num_threads);
        }
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nq; ++i) {
            searcher->Search(query.data() + i * dim, k, ids + i * k);
        }
    }

    void set_ef(int ef)
    {
        searcher->SetEf(ef);
    }

    void optimize(int num_threads = 0)
    {
        searcher->Optimize(num_threads);
    }
};


struct SearcherVS {

    int dim;
    int base_num;
    int* p_num;
    std::unique_ptr<glass::SearcherVSBase> searcher;

    SearcherVS() = default;

    SearcherVS(const Graph &graph, const float *data, int setnum, int features, const std::string &metric, int level = 0)
        : dim(features), base_num(setnum), searcher(std::unique_ptr<glass::SearcherVSBase>(glass::create_vssearcher(graph.graph, metric, level)))
    {
        searcher->SetData(data, base_num, features, p_num);
    }

    void search(const float *query, int k, int *ids, int q_num)
    {
        searcher->Search(query, k, ids, q_num);
    }

    void batch_search(const std::vector<float> &query, int k, int *ids, int q_num, int num_threads = 0) {
        int nq = query.size() / dim;
        if (num_threads != 0) {
            omp_set_num_threads(num_threads);
        }
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nq; ++i) {
            searcher->Search(query.data() + i * dim, k, ids + i * k, q_num);
        }
    }

    void set_ef(int ef)
    {
        searcher->SetEf(ef);
    }

    void optimize(int num_threads = 0)
    {
        searcher->Optimize(num_threads);
    }
};

struct ParaSearcher {

    std::unique_ptr<glass::ParaSearcherBase> searcher;

    ParaSearcher() = default;

    ParaSearcher(const Graph &graph, const float *data, int rows, int features, const std::string &metric, int level)
        : searcher(std::unique_ptr<glass::ParaSearcherBase>(glass::create_parasearcher(graph.graph, metric, level)))
    {
        searcher->SetData(data, rows, features);
    }

    void search(const float *query, int k, int *ids, int num_threads = 0)
    {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        searcher->Search(query, k, ids, num_threads);
    }

    //   py::object batch_search(py::object query, int k, int num_threads = 0) {
    //     py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    //     auto buffer = items.request();
    //     int *ids;
    //     size_t nq, dim;
    //     {
    //       py::gil_scoped_release l;
    //       get_input_array_shapes(buffer, &nq, &dim);
    //       ids = new int[nq * k];
    //       if (num_threads != 0) {
    //         omp_set_num_threads(num_threads);
    //       }
    // #pragma omp parallel for schedule(dynamic)
    //       for (int i = 0; i < nq; ++i) {
    //         searcher->Search(items.data(i), k, ids + i * k);
    //       }
    //     }
    //     py::capsule free_when_done(ids, [](void *f) { delete[] f; });
    //     return py::array_t<int>({nq * k}, {sizeof(int)}, ids, free_when_done);
    //   }

    void set_ef(int ef_master, int ef_local)
    {
        searcher->SetEf(ef_master, ef_local);
    }

    void set_subsearch_iterations(int subsearch_iterations)
    {
        searcher->SetSubsearchIterations(subsearch_iterations);
    }

    void optimize(int num_threads = 0)
    {
        searcher->Optimize(num_threads);
    }
};