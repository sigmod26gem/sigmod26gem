#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <set>
#include <functional>
namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;


// Drop-in replacement for the `std::unordered_map<tableint, tableint>` that used
// to back `entry_map`. The keys are always internal element ids in
// [0, max_elements_), so we store membership/value in a pre-sized array indexed
// by the key instead of a hash table. This removes the rehash that happened on
// concurrent `entry_map[cur_c] = ...` inserts during parallel cluster
// construction, which corrupted the heap ("double free or corruption").
//
// Membership is tracked with a generation stamp so that clear() is O(1):
// bumping the generation logically empties the container without touching the
// backing arrays. Concurrent writes target distinct keys (each insertion owns a
// unique cur_c), and membership reads only load the per-key atomic generation,
// so the build-time access pattern is data-race free.
class ClusterEntryMap {
 public:
    struct const_iterator {
        bool present;
        bool operator==(const const_iterator& o) const { return present == o.present; }
        bool operator!=(const const_iterator& o) const { return present != o.present; }
    };

    class reference {
     public:
        reference(ClusterEntryMap* m, tableint k) : m_(m), k_(k) {}
        operator tableint() const { return m_->get(k_); }
        reference& operator=(tableint v) {
            m_->set(k_, v);
            return *this;
        }

     private:
        ClusterEntryMap* m_;
        tableint k_;
    };

    void resize(size_t n) {
        gen_ = std::unique_ptr<std::atomic<uint32_t>[]>(new std::atomic<uint32_t>[n]);
        val_ = std::unique_ptr<tableint[]>(new tableint[n]);
        for (size_t i = 0; i < n; i++) gen_[i].store(0, std::memory_order_relaxed);
        cap_ = n;
        cur_gen_ = 1;
        count_.store(0, std::memory_order_relaxed);
    }

    // O(1) logical clear via generation bump.
    void clear() {
        if (++cur_gen_ == 0) {  // generation wrap-around: hard reset
            for (size_t i = 0; i < cap_; i++) gen_[i].store(0, std::memory_order_relaxed);
            cur_gen_ = 1;
        }
        count_.store(0, std::memory_order_relaxed);
    }

    bool contains(tableint k) const {
        return k < cap_ && gen_[k].load(std::memory_order_acquire) == cur_gen_;
    }

    const_iterator find(tableint k) const { return const_iterator{contains(k)}; }
    const_iterator end() const { return const_iterator{false}; }

    reference operator[](tableint k) { return reference(this, k); }

    size_t size() const { return count_.load(std::memory_order_relaxed); }

 private:
    void set(tableint k, tableint v) {
        if (gen_[k].load(std::memory_order_relaxed) != cur_gen_)
            count_.fetch_add(1, std::memory_order_relaxed);
        val_[k] = v;
        gen_[k].store(cur_gen_, std::memory_order_release);
    }

    tableint get(tableint k) const {
        return contains(k) ? val_[k] : static_cast<tableint>(0);
    }

    std::unique_ptr<std::atomic<uint32_t>[]> gen_;
    std::unique_ptr<tableint[]> val_;
    size_t cap_{0};
    uint32_t cur_gen_{1};
    std::atomic<size_t> count_{0};
};


inline std::mutex cout_mutex;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};
    int maxLevel_ty=3;

    const int fineEdgeTopk = 1;
    const int fineEdgeMaxlen = 0;
    const int fineEdgeSize = fineEdgeMaxlen * 2 * fineEdgeTopk;
    const int multi_entry_thread_num = 1;
    const int inner_search_thread_num = 8;
    const int local_rounds = 10;

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    float (*fstdistfunc_)(const vectorset*, const vectorset*, int level) ;
    float (*fstdistfunc4search_)(const vectorset*, const vectorset*, int level) ;
    float (*fstdistfuncCF)(const vectorset*, const vectorset*, int level) ;
    float (*fstdistfuncEMD)(const vectorset*, const vectorset*, int level) ;
    std::function<float(const vectorset*, const vectorset*, const float*)> fstdistfuncClusterEMD;
    float (*fstdistfuncMap_)(const vectorset* , const vectorset* , const vectorset* , const uint8_t* , const uint8_t* , uint8_t* , int level);
    float (*fstdistfuncMapCalc_)(const vectorset* , const vectorset* , const vectorset* , const uint8_t* , const uint8_t* , std::vector<std::vector<float>>&, int level);
    float (*fstdistfuncInit_)(const vectorset* , const vectorset* , uint8_t* , int level);
    float (*fstdistfuncInitEMD)(const vectorset* , const vectorset* , uint8_t* , int level);
    float (*fstdistfuncCluster)(const vectorset*, const vectorset*, int level) ;
    std::pair<float, float> (*fstdistfuncInit2_)(const vectorset* , const vectorset* , uint8_t* , int level);
    float (*fstdistfuncInitPre_)(const vectorset* , const vectorset* , uint8_t* , std::vector<std::vector<float>>&, int level);
    std::pair<float, float>  (*fstdistfuncInitPre2_) (const vectorset* , const vectorset* , uint8_t* , std::vector<std::vector<float>>&, int level);
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;
    ClusterEntryMap entry_map;
    std::vector<bool>search_set;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 48,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        //data_size_ = s->get_data_size();
        data_size_ = sizeof(vectorset);
        fstdistfunc_ = L2SqrVecSet;
        fstdistfuncInit_ = L2SqrVecSetInit;
        fstdistfuncInit2_ = L2SqrVecSetInitReturn2;
        fstdistfuncMap_ = L2SqrVecSetMap;
        fstdistfuncMapCalc_ = L2SqrVecSetMapCalc;
        fstdistfuncInitPre_ = L2SqrVecSetInitPreCalc;
        fstdistfuncInitPre2_ = L2SqrVecSetInitPreCalcReturn2;
        fstdistfunc4search_ = L2SqrVecSet4Search;
        fstdistfuncCF = L2SqrVecCF;
        fstdistfuncEMD = L2SqrVecEMD;
        fstdistfuncInitEMD = L2SqrVecSetInitEMD;
        fstdistfuncCluster = L2SqrCluster4Search;
        fstdistfuncClusterEMD = L2SqrVecClusterEMD;
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 80;



        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * (sizeof(tableint) + fineEdgeSize * sizeof(uint8_t)) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        entry_map.resize(max_elements_);

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        // size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_links_per_element_ = maxM0_ * (sizeof(tableint) + fineEdgeSize * sizeof(uint8_t)) + sizeof(linklistsizeint);
        // size_links_level0_ = maxM0_ * (sizeof(tableint) + 240 * sizeof(uint8_t)) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    struct CompareByFirstFirst {
        constexpr bool operator()(std::pair<std::pair<dist_t, dist_t>, tableint> const& a,
            std::pair<std::pair<dist_t, dist_t>, tableint> const& b) const noexcept {
            return a.first.first < b.first.first;
        }
    };

    struct CompareTupleByFirst {
        constexpr bool operator()(std::tuple<dist_t, tableint, uint8_t*> const& a,
            std::tuple<dist_t, tableint, uint8_t*> const& b) const noexcept {
            return std::get<0>(a) < std::get<0>(b);
        }
    };

    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) 0;
        // return maxLevel_ty;

    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

#include "detail/layer_search.inl"
#include "detail/cluster_search.inl"
#include "detail/entry_search.inl"
#include "detail/parallel_search.inl"
#include "detail/neighbors.inl"
#include "detail/edges.inl"
#include "detail/serialization.inl"
#include "detail/labels.inl"
#include "detail/cluster_insert.inl"
#include "detail/insert_update.inl"
#include "detail/query.inl"
#include "detail/repair.inl"
};
}  // namespace hnswlib
