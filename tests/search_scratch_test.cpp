#include "graph/search_scratch.h"
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void visited() {
    hnswlib::TouchedBitset bits;
    std::mt19937 random(42);
    // Exercise boundaries, shrinking, regrowth and repeated reset.
    for (std::size_t size : {0, 1, 63, 64, 65, 4097, 0, 3, 130, 64, 4097}) {
        for (int repeat = 0; repeat < 8; ++repeat) {
            bits.reset(size);
            std::vector<bool> expected(size, false);
            for (std::size_t id = 0; id < size; ++id)
                check(!bits.contains(id), "reset retained a visited bit");
            if (!size) continue;
            for (std::size_t i = 0; i < size * 3; ++i) {
                const auto id = random() % size;
                check(bits.test_and_set(id) == expected[id], "visited membership mismatch");
                expected[id] = true;
                check(bits.test_and_set(id), "repeated visit changed membership");
            }
            for (std::size_t id = 0; id < size; ++id)
                check(bits.contains(id) == expected[id], "neighboring bit was overwritten");
        }
    }
    bits.reset(130);
    for (std::size_t id : {0, 63, 64, 65, 127, 128, 129}) bits.test_and_set(id);
    check(bits.data()[0] == (std::uint64_t{1} | (std::uint64_t{1} << 63)), "first word boundary");
    check(bits.data()[1] == (std::uint64_t{3} | (std::uint64_t{1} << 63)), "second word boundary");
    check(bits.data()[2] == 3, "last word boundary");
    try {
        bits.test_and_set(17);
        throw std::runtime_error("simulated query failure");
    } catch (const std::runtime_error&) {}
    bits.reset(130);
    for (std::size_t id = 0; id < 130; ++id)
        check(!bits.contains(id), "failed query retained visited bits");
}

void candidates() {
    using Internal = std::pair<float, unsigned int>;
    using External = std::pair<float, std::size_t>;
    std::mt19937 random(7);
    for (unsigned int size : {0, 1, 16, 512, 4000}) {
        hnswlib::ReusableHeap<Internal, hnswlib::CompareDistance<float>> top;
        std::vector<std::size_t> labels(size);
        std::iota(labels.begin(), labels.end(), std::size_t{1000000});
        std::shuffle(labels.begin(), labels.end(), random);
        for (unsigned int id = 0; id < size; ++id)
            top.emplace(static_cast<float>(random() % 17) - 8, id);
        auto old_top = top;
        std::priority_queue<External> old_results;
        while (!old_top.empty()) {
            const auto item = old_top.top();
            old_results.emplace(item.first, labels[item.second]);
            old_top.pop();
        }
        std::vector<External> expected, actual;
        while (!old_results.empty()) {
            expected.push_back(old_results.top());
            old_results.pop();
        }
        for (const auto& item : top.values()) actual.emplace_back(item.first, labels[item.second]);
        const auto capacity = top.capacity();
        top.clear();
        check(top.empty() && top.capacity() == capacity, "heap lost reusable storage");
        std::sort(actual.begin(), actual.end(), std::greater<>());
        check(actual == expected, "candidate order differs from the external-label heap");
    }
}
}  // namespace

int main() {
    try {
        visited();
        candidates();
        std::cout << "visited bitset and candidate ordering passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
