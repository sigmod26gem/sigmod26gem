// Run the upstream search expression and traversal on the same input and graph.
// The example's main is never executed; the test harness supplies configuration.
#define main gem_upstream_example_main
#include "reference/gem_example.cpp"
#undef main
#include "config.h"
#include "gem/io.h"
#include <iomanip>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        const auto c = gem::app::read_config(argv[1]);
        if (c.workers != 1 || c.search.nprobe != NPROB)
            throw std::runtime_error("reference requires workers=1 and nprobe=4");
        omp_set_num_threads(c.inner_threads);
        openblas_set_num_threads(c.inner_threads);
        Eigen::setNbThreads(1);
        auto data = gem::load_corpus(c.data);
        if (data.documents.dimension != VECTOR_DIM) throw std::runtime_error("reference requires dimension=128");
        auto queries = gem::load_queries(c.query_vectors, c.query_lengths);
        NUM_GRAPH_CLUSTER = data.clusters.size();
        NUM_CLUSTER = data.fine_centroids.size() / VECTOR_DIM;
        rerankK = c.search.rerank_k;
        std::vector<vectorset> docs;
        for (std::size_t id = 0; id < data.documents.size(); ++id) {
            const auto offset = data.documents.offsets[id];
            const auto view = data.documents.at(id);
            docs.emplace_back(data.documents.values.data() + offset * VECTOR_DIM,
                              data.codes.data() + offset, VECTOR_DIM, view.count);
        }
        std::vector<int> cluster_ids(NUM_GRAPH_CLUSTER);
        std::iota(cluster_ids.begin(), cluster_ids.end(), 0);
        Solution solution;
        // load_fine_cluster expects a directory containing 0.bin.
        if (std::filesystem::path(c.index).filename() != "0.bin") throw std::runtime_error("reference requires graph filename 0.bin");
        solution.load_fine_cluster(std::filesystem::path(c.index).parent_path().string() + "/",
                                   VECTOR_DIM, docs, data.clusters, cluster_ids);
        if (solution.alg_hnsw_list[0]->cur_element_count != docs.size())
            throw std::runtime_error("reference graph and corpus document counts differ");
        const auto count = c.limit ? std::min(c.limit, queries.size()) : queries.size();
        std::vector<std::vector<std::pair<int, float>>> all(count);
        std::vector<float> route, fine;
        for (std::size_t repeat = 0; repeat < c.warmup + c.repeats; ++repeat) {
            double total = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const auto q = queries.at(i);
                route.resize(NUM_GRAPH_CLUSTER * q.count);
                fine.resize(NUM_CLUSTER * q.count);
                vectorset view(const_cast<float*>(q.data), VECTOR_DIM, q.count);
                total += solution.search_with_fine_cluster(view, route, fine, data.fine_centroids,
                          data.graph_centroids, c.search.k, c.search.ef, all[i]);
            }
            if (repeat >= c.warmup)
                std::cout << std::setprecision(9) << "{\"reference\":true,\"repeat\":" << repeat - c.warmup
                          << ",\"queries\":" << count << ",\"qps\":" << count / total
                          << ",\"mean_ms\":" << total * 1000 / count << "}\n";
        }
        std::ofstream out(c.output);
        if (!out) throw std::runtime_error("reference needs benchmark.output");
        out << std::setprecision(9);
        for (std::size_t q = 0; q < count; ++q)
            for (std::size_t rank = 0; rank < all[q].size(); ++rank)
                out << q << '\t' << rank << '\t' << all[q][rank].first << '\t' << all[q][rank].second << '\n';
        for (auto* graph : solution.alg_hnsw_list) delete graph;
        delete solution.space_ptr;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
