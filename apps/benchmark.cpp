#include "benchmark.h"
#include <atomic>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_set>

namespace gem::app {
using Clock = std::chrono::steady_clock;
void benchmark(const Index& index, const MultiVectors& queries, const RunConfig& c) {
    const auto count = c.limit ? std::min(c.limit, queries.size()) : queries.size();
    std::vector<QueryWorkspace> workspaces(c.workers);
    std::vector<std::vector<Result>> results(count);
    std::vector<double> elapsed(count);
    std::vector<std::unordered_set<std::size_t>> qrels(count);
    if (!c.qrels.empty()) {
        std::ifstream in(c.qrels);
        if (!in) throw std::runtime_error("cannot open qrels: " + c.qrels);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream row(line);
            std::size_t q, d;
            std::string extra;
            if (!(row >> q >> d) || (row >> extra)) throw std::runtime_error("qrels require two columns: query_id document_id");
            if (q < count) qrels[q].insert(d);
        }
    }
    for (std::size_t repeat = 0; repeat < c.warmup + c.repeats; ++repeat) {
        std::atomic<std::size_t> next{0};
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        std::mutex error_mutex;
        std::vector<std::thread> workers;
        struct JoinWorkers {
            std::vector<std::thread>& threads;
            ~JoinWorkers() { for (auto& thread : threads) if (thread.joinable()) thread.join(); }
        } joiner{workers};
        const auto start = Clock::now();
        for (std::size_t worker = 0; worker < c.workers; ++worker) workers.emplace_back([&, worker] {
            try {
                for (;;) {
                    const auto q = next.fetch_add(1);
                    if (q >= count || failed.load()) break;
                    const auto begin = Clock::now();
                    index.search(queries.at(q), c.search, workspaces[worker], results[q]);
                    elapsed[q] = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
                }
            } catch (...) {
                std::lock_guard<std::mutex> guard(error_mutex);
                if (!error) error = std::current_exception();
                failed.store(true);
            }
        });
        for (auto& worker : workers) worker.join();
        if (error) std::rethrow_exception(error);
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        if (repeat < c.warmup) continue;
        double recall = 0;
        std::size_t labeled = 0;
        for (std::size_t q = 0; q < count; ++q) {
            if (qrels[q].empty()) continue;
            std::size_t hits = 0;
            std::unordered_set<std::size_t> returned;
            for (const auto& result : results[q])
                if (returned.insert(result.id).second) hits += qrels[q].count(result.id);
            recall += double(hits) / qrels[q].size();
            ++labeled;
        }
        std::cout << std::setprecision(9) << "{\"repeat\":" << repeat - c.warmup
                  << ",\"queries\":" << count << ",\"workers\":" << c.workers
                  << ",\"nprobe\":" << c.search.nprobe << ",\"ef\":" << c.search.ef
                  << ",\"rerank_k\":" << c.search.rerank_k << ",\"k\":" << c.search.k
                  << ",\"qps\":" << count / seconds
                  << ",\"mean_ms\":" << std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / count
                  << ",\"recall\":" << (labeled ? std::to_string(recall / labeled) : "null") << "}\n";
    }
    if (!c.output.empty()) {
        std::ofstream out(c.output);
        if (!out) throw std::runtime_error("cannot write results: " + c.output);
        out << std::setprecision(9);
        for (std::size_t q = 0; q < count; ++q)
            for (std::size_t rank = 0; rank < results[q].size(); ++rank)
                out << q << '\t' << rank << '\t' << results[q][rank].id << '\t' << results[q][rank].distance << '\n';
    }
}
}  // namespace gem::app
