#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

namespace hnswlib {
template<class Distance>
struct CompareDistance {
    bool operator()(const std::pair<Distance, unsigned int>& a,
                    const std::pair<Distance, unsigned int>& b) const noexcept {
        return a.first < b.first;
    }
};

template<class Value, class Compare = std::less<Value>>
class ReusableHeap : public std::priority_queue<Value, std::vector<Value>, Compare> {
 public:
    void clear() { this->c.clear(); }
    std::size_t capacity() const { return this->c.capacity(); }
};

template<class Distance>
struct EntrySearchScratch {
    using Heap = ReusableHeap<std::pair<Distance, unsigned int>, CompareDistance<Distance>>;
    Heap top;
    std::vector<Heap> frontiers;
    std::vector<bool> stopped;
    std::vector<std::uint16_t> visited;
    std::uint16_t generation = 0;
    ReusableHeap<std::pair<Distance, std::size_t>> results;

    void reset(std::size_t nodes, std::size_t entries) {
        visited.resize(nodes, 0);
        if (++generation == 0) {
            std::fill(visited.begin(), visited.end(), 0);
            generation = 1;
        }
        if (frontiers.size() < entries) frontiers.resize(entries);
        stopped.assign(entries, false);
        top.clear();
        results.clear();
        for (auto& frontier : frontiers) frontier.clear();
    }
};
}  // namespace hnswlib
