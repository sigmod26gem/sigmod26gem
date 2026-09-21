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
    const std::vector<Value>& values() const { return this->c; }
};

class TouchedBitset {
 public:
    void reset(std::size_t bits) {
        // Clear before shrinking so that every recorded word remains addressable.
        for (auto word : touched_) words_[word] = 0;
        touched_.clear();
        words_.resize(bits / 64 + (bits % 64 != 0), 0);
    }

    bool contains(std::size_t id) const {
        return (words_[id / 64] & (std::uint64_t{1} << (id % 64))) != 0;
    }

    // Returns true if the bit was already set. IDs must be below the reset size.
    bool test_and_set(std::size_t id) {
        auto& word = words_[id / 64];
        const auto mask = std::uint64_t{1} << (id % 64);
        if (word & mask) return true;
        if (!word) touched_.push_back(id / 64);
        word |= mask;
        return false;
    }

    const std::uint64_t* data() const { return words_.data(); }

 private:
    std::vector<std::uint64_t> words_;
    std::vector<std::size_t> touched_;
};

template<class Distance>
struct EntrySearchScratch {
    using Heap = ReusableHeap<std::pair<Distance, unsigned int>, CompareDistance<Distance>>;
    Heap top;
    std::vector<Heap> frontiers;
    std::vector<bool> stopped;
    TouchedBitset visited;

    void reset(std::size_t nodes, std::size_t entries) {
        visited.reset(nodes);
        if (frontiers.size() < entries) frontiers.resize(entries);
        stopped.assign(entries, false);
        top.clear();
        for (auto& frontier : frontiers) frontier.clear();
    }
};
}  // namespace hnswlib
