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
namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;


std::mutex cout_mutex;

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
    float (*fstdistfuncClusterEMD)(const vectorset*, const vectorset*, const float*) ;
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
    std::unordered_map<tableint, tableint> entry_map;
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

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchKnnParaForConstruction(tableint ep_id, const void *data_point) const {
        tableint currObj = enterpoint_node_;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerFullPreST<true>(currObj, data_point, ef_construction_);
        while (top_candidates.size() > ef_construction_) {
            top_candidates.pop();
        }
        return top_candidates;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) const {
        return searchKnnParaForConstruction(ep_id, data_point);
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerClusterAppr(tableint ep_id, const void *data_point, const float *cluster_distance, int layer) {
        tableint currObj = ep_id;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerClusterConsPara<true>(currObj, data_point, cluster_distance, ef_construction_);
        while (top_candidates.size() > ef_construction_) {
            top_candidates.pop();
        }
        return top_candidates;
    }



//     // old version
//     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
//     searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
//         VisitedList *vl = visited_list_pool_->getFreeVisitedList();
//         vl_type *visited_array = vl->mass;
//         vl_type visited_array_tag = vl->curV;

//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

//         dist_t lowerBound;
//         if (!isMarkedDeleted(ep_id)) {
//             // std::cout << "searchBaseLayer: " << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)getDataByInternalId(ep_id))->vecnum << std::endl;
//             dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)getDataByInternalId(ep_id), layer);
//             top_candidates.emplace(dist, ep_id);
//             lowerBound = dist;
//             candidateSet.emplace(-dist, ep_id);
//         } else {
//             lowerBound = std::numeric_limits<dist_t>::max();
//             candidateSet.emplace(-lowerBound, ep_id);
//         }
//         visited_array[ep_id] = visited_array_tag;

//         while (!candidateSet.empty()) {
//             std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
//             if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
//                 break;
//             }
//             candidateSet.pop();

//             tableint curNodeNum = curr_el_pair.second;

//             std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

//             int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
//             if (layer == 0) {
//                 data = (int*)get_linklist0(curNodeNum);
//             } else {
//                 data = (int*)get_linklist(curNodeNum, layer);
// //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
//             }
//             size_t size = getListCount((linklistsizeint*)data);
//             tableint *datal = (tableint *) (data + 1);
// #ifdef USE_SSE
//             _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
//             _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
//             _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
//             _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
// #endif
//             for (size_t j = 0; j < size; j++) {
//                 tableint candidate_id = *(datal + j);
// //                    if (candidate_id == 0) continue;
// #ifdef USE_SSE
//                 _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
//                 _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
// #endif
//                 if (visited_array[candidate_id] == visited_array_tag) continue;
//                 visited_array[candidate_id] = visited_array_tag;
//                 char *currObj1 = (getDataByInternalId(candidate_id));
//                 // if (((vectorset*)currObj1)->vecnum == 0) {
//                 //     std::cout << curNodeNum << " neighbor size: " << size << std::endl;
//                 //     std::cout << j << " " << candidate_id << std::endl;
//                 //     std::cout << "searchBaseLayer: " << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)currObj1)->vecnum << std::endl;
//                 //     std::cout << "searchBaseLayer: " << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)getDataByInternalId(candidate_id))->vecnum << std::endl;
//                 //     // std::cout << "searchBaseLayer: " << candidate_id << " " << getExternalLabel(candidate_id) << std::endl;
//                 // }
//                 dist_t dist1 = fstdistfunc_((vectorset*)data_point, (vectorset*)currObj1, layer);
//                 if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
//                     candidateSet.emplace(-dist1, candidate_id);
// #ifdef USE_SSE
//                     _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
// #endif

//                     if (!isMarkedDeleted(candidate_id))
//                         top_candidates.emplace(dist1, candidate_id);

//                     if (top_candidates.size() > ef_construction_)
//                         top_candidates.pop();

//                     if (!top_candidates.empty())
//                         lowerBound = top_candidates.top().first;
//                 }
//             }
//         }
//         visited_list_pool_->releaseVisitedList(vl);

//         return top_candidates;
//     }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<std::pair<dist_t, dist_t>, tableint>, std::vector<std::pair<std::pair<dist_t, dist_t>, tableint>>, CompareByFirstFirst>
    searchBaseLayerSTCF(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        uint8_t* mapEP = (uint8_t*)malloc(fineEdgeSize);
        std::priority_queue<std::pair<std::pair<dist_t, dist_t>, tableint>, std::vector<std::pair<std::pair<dist_t, dist_t>, tableint>>, CompareByFirstFirst> top_candidates;
        std::priority_queue<std::tuple<dist_t, tableint, uint8_t*>, std::vector<std::tuple<dist_t, tableint, uint8_t*>>, CompareTupleByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            std::pair<dist_t, dist_t> dist2 = fstdistfuncInit2_((vectorset*)data_point, (vectorset*)ep_data, mapEP, 0);
            // size_t an = std::min(((vectorset*)data_point)->vecnum, (size_t)120);
            // size_t bn = std::min(((vectorset*)ep_data)->vecnum, (size_t)120);
            // for (uint8_t i = 0; i < an; i++) {
            //     std::cout << (uint16_t) i << " " << (uint16_t) mapEP[i] << " " << bn << std::endl;
            //     assert(mapEP[i] < (uint8_t)bn);
            // }
            // for (uint8_t i = 0; i < bn; i++) {
            //     std::cout << (uint16_t) i << " " << (uint16_t) mapEP[i + 120] << " " << an << std::endl;
            //     assert(mapEP[i + 120] < (uint8_t)an);
            // }
            // std::cout << "===== outer ====" << an << " " << bn << std::endl;
            // std::cout << "====" << std::endl;
            // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)ep_data);
            dist_t dist = dist2.first;
            lowerBound = dist;
            top_candidates.emplace(dist2, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist2.first, ep_id, mapEP);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id, mapEP);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::tuple<dist_t, tableint, uint8_t*> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -std::get<0>(current_node_pair);
            uint8_t* mapAB = std::get<2>(current_node_pair);
            // for (uint8_t i = 0; i < 120; ++i) {
            //     std::cout << (u_int16_t)i << " " << (u_int16_t)mapAB[i] << std::endl;
            // }
            // for (uint8_t i = 0; i < 120; ++i) {
            //     std::cout << (u_int16_t)i << " " << (u_int16_t)mapAB[i + 120 * 1] << std::endl;
            // }
            // std::cout << "==MapAB==" << std::endl;
            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = std::get<1>(current_node_pair);
            // std::cout << candidate_dist << ' ' <<  current_node_id << std::endl;

            char *nodeObj = (getDataByInternalId(current_node_id));

            int *data = (int *) get_linklist0(current_node_id);
            uint8_t *distancelistl = (uint8_t *) ((tableint *)data + 1 + maxM0_);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            // if (collect_metrics) {
            metric_hops++;
            metric_distance_computations+=size;
            // }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                uint8_t* mapBC = distancelistl + fineEdgeSize * (j - 1);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    // {
                    //     std::lock_guard<std::mutex> lock(cout_mutex); 
                    //     std::cout << "search ==" << current_node_id << " " << candidate_id << " " << j - 1 << std::endl;
                    //     size_t bn = std::min(((vectorset*)nodeObj)->vecnum, (size_t)120);
                    //     size_t cn = std::min(((vectorset*)currObj1)->vecnum, (size_t)120);    
                    //     std::cout << "== MapBC == " << bn << " " << cn << std::endl;
                    //     bool flag = true;
                    //     for (uint8_t i = 0; i < bn; ++i) {
                    //         std::cout << (u_int16_t)i << " " << (u_int16_t)(mapBC[i]) << std::endl;
                    //         flag = flag & ((mapBC[i]) < cn);
                    //     }
                    //     for (uint8_t i = 0; i < cn; ++i) {
                    //         std::cout << (u_int16_t)i << " " << (u_int16_t)(mapBC[i + 120 * 1]) << std::endl;
                    //         flag = flag & ((mapBC[i + 120]) < bn);
                    //     }
                    //     assert (flag);
                    //     std::cout << "== MapBC ==" << std::endl;
                    // }
                    
                    // dist_t dist = fstdistfuncMap_((vectorset*)data_point, (vectorset*)nodeObj, (vectorset*)currObj1, mapAB, mapBC, mapAC, 0);
                    std::vector<std::vector<float>> dist_matrix(((vectorset*)data_point)->vecnum, std::vector<float>(((vectorset*)currObj1)->vecnum));
                    dist_t dist = fstdistfuncMapCalc_((vectorset*)data_point, (vectorset*)nodeObj, (vectorset*)currObj1, mapAB, mapBC, dist_matrix, 0);
                    dist = dist * 0.85;
                    bool flag_consider_candidate;
                    bool estimate_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        estimate_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        estimate_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }
                    if (!estimate_consider_candidate) {
                        continue;
                    }
                    uint8_t* mapAC = (uint8_t*)malloc(fineEdgeSize);
                    std::pair<dist_t, dist_t> dist2 = fstdistfuncInitPre2_((vectorset*)data_point, (vectorset*)currObj1, mapAC, dist_matrix, 0);
                    // std::cout << dist << std::endl;
                    dist = dist2.first;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }
                    // std::cout << flag_consider_candidate << std::endl;

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id, mapAC);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + std::get<1>(candidate_set.top()) * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist2, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first.first;
                    } else {
                        free(mapAC);
                    }
                }
            }
            // std::cout << " free" << std::endl;
            free(mapAB);
            // std::cout << " free" << std::endl;
        }

        visited_list_pool_->releaseVisitedList(vl);
        // std::cout << top_candidates.size() << std::endl;
        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        uint8_t* mapEP = (uint8_t*)malloc(fineEdgeSize);
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::tuple<dist_t, tableint, uint8_t*>, std::vector<std::tuple<dist_t, tableint, uint8_t*>>, CompareTupleByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfuncInit_((vectorset*)data_point, (vectorset*)ep_data, mapEP, 0);
            // size_t an = std::min(((vectorset*)data_point)->vecnum, (size_t)120);
            // size_t bn = std::min(((vectorset*)ep_data)->vecnum, (size_t)120);
            // for (uint8_t i = 0; i < an; i++) {
            //     std::cout << (uint16_t) i << " " << (uint16_t) mapEP[i] << " " << bn << std::endl;
            //     assert(mapEP[i] < (uint8_t)bn);
            // }
            // for (uint8_t i = 0; i < bn; i++) {
            //     std::cout << (uint16_t) i << " " << (uint16_t) mapEP[i + 120] << " " << an << std::endl;
            //     assert(mapEP[i + 120] < (uint8_t)an);
            // }
            // std::cout << "===== outer ====" << an << " " << bn << std::endl;
            // std::cout << "====" << std::endl;
            // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)ep_data);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id, mapEP);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id, mapEP);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::tuple<dist_t, tableint, uint8_t*> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -std::get<0>(current_node_pair);
            uint8_t* mapAB = std::get<2>(current_node_pair);
            // for (uint8_t i = 0; i < 120; ++i) {
            //     std::cout << (u_int16_t)i << " " << (u_int16_t)mapAB[i] << std::endl;
            // }
            // for (uint8_t i = 0; i < 120; ++i) {
            //     std::cout << (u_int16_t)i << " " << (u_int16_t)mapAB[i + 120 * 1] << std::endl;
            // }
            // std::cout << "==MapAB==" << std::endl;
            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = std::get<1>(current_node_pair);
            // std::cout << candidate_dist << ' ' <<  current_node_id << std::endl;

            char *nodeObj = (getDataByInternalId(current_node_id));

            int *data = (int *) get_linklist0(current_node_id);
            uint8_t *distancelistl = (uint8_t *) ((tableint *)data + 1 + maxM0_);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                uint8_t* mapBC = distancelistl + fineEdgeSize * (j - 1);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    // {
                    //     std::lock_guard<std::mutex> lock(cout_mutex); 
                    //     std::cout << "search ==" << current_node_id << " " << candidate_id << " " << j - 1 << std::endl;
                    //     size_t bn = std::min(((vectorset*)nodeObj)->vecnum, (size_t)120);
                    //     size_t cn = std::min(((vectorset*)currObj1)->vecnum, (size_t)120);    
                    //     std::cout << "== MapBC == " << bn << " " << cn << std::endl;
                    //     bool flag = true;
                    //     for (uint8_t i = 0; i < bn; ++i) {
                    //         std::cout << (u_int16_t)i << " " << (u_int16_t)(mapBC[i]) << std::endl;
                    //         flag = flag & ((mapBC[i]) < cn);
                    //     }
                    //     for (uint8_t i = 0; i < cn; ++i) {
                    //         std::cout << (u_int16_t)i << " " << (u_int16_t)(mapBC[i + 120 * 1]) << std::endl;
                    //         flag = flag & ((mapBC[i + 120]) < bn);
                    //     }
                    //     assert (flag);
                    //     std::cout << "== MapBC ==" << std::endl;
                    // }
                    
                    // dist_t dist = fstdistfuncMap_((vectorset*)data_point, (vectorset*)nodeObj, (vectorset*)currObj1, mapAB, mapBC, mapAC, 0);
                    std::vector<std::vector<float>> dist_matrix(((vectorset*)data_point)->vecnum, std::vector<float>(((vectorset*)currObj1)->vecnum));
                    dist_t dist = fstdistfuncMapCalc_((vectorset*)data_point, (vectorset*)nodeObj, (vectorset*)currObj1, mapAB, mapBC, dist_matrix, 0);
                    dist = dist * 0.75;
                    bool flag_consider_candidate;
                    bool estimate_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        estimate_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        estimate_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }
                    if (!estimate_consider_candidate) {
                        continue;
                    }
                    uint8_t* mapAC = (uint8_t*)malloc(fineEdgeSize);
                    dist = fstdistfuncInitPre_((vectorset*)data_point, (vectorset*)currObj1, mapAC, dist_matrix, 0);
                    // std::cout << dist << std::endl;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }
                    // std::cout << flag_consider_candidate << std::endl;

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id, mapAC);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + std::get<1>(candidate_set.top()) * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    } else {
                        free(mapAC);
                    }
                }
            }
            // std::cout << " free" << std::endl;
            free(mapAB);
            // std::cout << " free" << std::endl;
        }

        visited_list_pool_->releaseVisitedList(vl);
        // std::cout << top_candidates.size() << std::endl;
        return top_candidates;
    }



    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerFullPreST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            // dist_t dist = fstdistfuncEMD((vectorset*)data_point, (vectorset*)ep_data, 0);
            dist_t dist = fstdistfuncEMD((vectorset*)data_point, (vectorset*)ep_data, 0);
            // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)ep_data);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    // dist_t dist = fstdistfuncEMD((vectorset*)data_point, (vectorset*)ep_data, 0);
                    dist_t dist = fstdistfuncEMD((vectorset*)data_point, (vectorset*)currObj1, 0);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        // std::cout << top_candidates.size() << std::endl;
        return top_candidates;
    }

    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerClusterCons(
        tableint ep_id, 
        const void *data_point,
        const float *cluster_distance, 
        size_t ef) {
        int layer = 0;
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            // std::cout << "searchBaseLayer: " << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)getDataByInternalId(ep_id))->vecnum << std::endl;
            // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)getDataByInternalId(ep_id), layer);
            dist_t dist = fstdistfuncClusterEMD((vectorset*)data_point, (vectorset*)getDataByInternalId(ep_id), cluster_distance);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif
            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                if (entry_map.find(candidate_id) == entry_map.end()) continue;
                
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));
                // if (((vectorset*)currObj1)->vecnum == 0) {
                //     std::cout << curNodeNum << " neighbor size: " << size << std::endl;
                //     std::cout << j << " " << candidate_id << std::endl;
                //     std::cout << "searchBaseLayer: " << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)currObj1)->vecnum << std::endl;
                //     std::cout << "searchBaseLayer: " << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)getDataByInternalId(candidate_id))->vecnum << std::endl;
                //     // std::cout << "searchBaseLayer: " << candidate_id << " " << getExternalLabel(candidate_id) << std::endl;
                // }
                // dist_t dist1 = fstdistfunc_((vectorset*)data_point, (vectorset*)currObj1, layer);
                dist_t dist1 = fstdistfuncClusterEMD((vectorset*)data_point, (vectorset*)currObj1, cluster_distance);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }

    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerClusterConsPara(
        tableint ep_id,
        const void *data_point,
        const float *cluster_distance,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        // std::cout << bare_bone_search << std::endl;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
        
        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            // dist_t dist = fstdistfuncEMD((vectorset*)data_point, (vectorset*)ep_data, 0);
            // std::cout << ep_id << " " << ((vectorset*)data_point)->vecnum << " " << ((vectorset*)ep_data)->vecnum << std::endl;
            dist_t dist = fstdistfuncClusterEMD((vectorset*)data_point, (vectorset*)ep_data, cluster_distance);
            // std::cout << ep_id << " " << dist << std::endl;
            // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)ep_data);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;
            // std::cout << current_node_pair.second << " " << candidate_dist << " " << lowerBound << std::endl;
            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif
            // std::cout << size << " ";
            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                // if (!(visited_array[candidate_id] == visited_array_tag)) {
                if (!(visited_array[candidate_id] == visited_array_tag) && (entry_map.find((tableint)candidate_id) != entry_map.end())) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    // dist_t dist = fstdistfuncEMD((vectorset*)data_point, (vectorset*)ep_data, 0);
                    dist_t dist = fstdistfuncClusterEMD((vectorset*)data_point, (vectorset*)currObj1, cluster_distance);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        // std::cout << top_candidates.size() << std::endl;
        return top_candidates;
    }

template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerFullChamferST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            // dist_t dist = L2SqrVecEMD((vectorset*)data_point, (vectorset*)ep_data, 0);
            dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)ep_data, 0);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)currObj1);
                    dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)currObj1, 0);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        // std::cout << top_candidates.size() << std::endl;
        return top_candidates;
    }


    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerClusterEntries(
        const std::vector<labeltype>& entry_points,
        // tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound = -1;
        for (labeltype ep: entry_points) {
            tableint ep_id = label_lookup_.find(ep)->second;
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfuncCluster((vectorset*)data_point, (vectorset*)ep_data, 0);
            lowerBound = std::max(lowerBound, dist);
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            // std::cout << ep << "(" << dist << ") ";
            candidate_set.emplace(-dist, ep_id);
            visited_array[ep_id] = visited_array_tag;
        }
        // std::cout << std::endl;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;
            // std::cout << top_candidates.size() << " " << lowerBound << " " << getExternalLabel(current_node_pair.second) << "(" << candidate_dist << "): ";
            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)currObj1);
                    dist_t dist = fstdistfuncCluster((vectorset*)data_point, (vectorset*)currObj1, 0);
                    // std::cout << getExternalLabel(candidate_id) << "(" << dist << ") ";
                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
                        // std::cout << "T ";
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            // std::cout << std::endl;
        }

        visited_list_pool_->releaseVisitedList(vl);
        // std::cout << top_candidates.size() << std::endl;
        return top_candidates;
    }



    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerClusterEntriesMulti(
        const std::vector<labeltype>& entry_points,
        // tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
            
        // std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> top_candidates_list(entry_points.size());
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        
        std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> candidate_set_list(entry_points.size());
        std::vector<bool> stop_flag_list(entry_points.size());
        // std::vector<dist_t> lower_bound_list(entry_points.size());
        dist_t lowerBound;
        for (int i = 0; i < entry_points.size(); i++) {
            labeltype ep = entry_points[i];
            tableint ep_id = label_lookup_.find(ep)->second;
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfuncCluster((vectorset*)data_point, (vectorset*)ep_data, 0);
            lowerBound = std::max(lowerBound, dist);
            // lower_bound_list[i] = std::max(lower_bound_list[i], dist);
            top_candidates.emplace(dist, ep_id);
            candidate_set_list[i].emplace(-dist, ep_id);
            visited_array[ep_id] = visited_array_tag;
            stop_flag_list[i] = false;
        }
        bool all_empty = false;
        while (!all_empty) {
            all_empty = true;
            for (int i = 0; i < candidate_set_list.size(); i++) {
                if (!stop_flag_list[i] && !candidate_set_list[i].empty()) {
                    std::pair<dist_t, tableint> current_node_pair = candidate_set_list[i].top();
                    dist_t candidate_dist = -current_node_pair.first;
                    // std::cout << top_candidates.size() << " " << lowerBound << " " << getExternalLabel(current_node_pair.second) << "(" << candidate_dist << "): ";
                    bool flag_stop_search;
                    if (bare_bone_search) {
                        flag_stop_search = candidate_dist > lowerBound;
                    } else {
                        if (stop_condition) {
                            flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                        } else {
                            flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                        }
                    }
                    if (flag_stop_search) {
                        stop_flag_list[i] = true;
                        continue;
                    }
                    all_empty = false;
                    candidate_set_list[i].pop();

                    tableint current_node_id = current_node_pair.second;
                    int *data = (int *) get_linklist0(current_node_id);
                    size_t size = getListCount((linklistsizeint*)data);
        //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                    if (collect_metrics) {
                        metric_hops++;
                        metric_distance_computations+=size;
                    }

        #ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                    _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                    _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
        #endif

                    for (size_t j = 1; j <= size; j++) {
                        int candidate_id = *(data + j);
        //                    if (candidate_id == 0) continue;
        #ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                        _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                        _MM_HINT_T0);  ////////////
        #endif
                        if (!search_set[candidate_id]) {
                            continue;
                        }
                        if (!(visited_array[candidate_id] == visited_array_tag)) {
                            visited_array[candidate_id] = visited_array_tag;

                            char *currObj1 = (getDataByInternalId(candidate_id));
                            // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)currObj1);
                            dist_t dist = fstdistfuncCluster((vectorset*)data_point, (vectorset*)currObj1, 0);
                            // std::cout << getExternalLabel(candidate_id) << "(" << dist << ") ";
                            bool flag_consider_candidate;
                            if (!bare_bone_search && stop_condition) {
                                flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                            } else {
                                flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                            }

                            if (flag_consider_candidate) {
                                candidate_set_list[i].emplace(-dist, candidate_id);
                                // std::cout << "T ";
        #ifdef USE_SSE
                                _mm_prefetch(data_level0_memory_ + candidate_set_list[i].top().second * size_data_per_element_ +
                                                offsetLevel0_,  ///////////
                                                _MM_HINT_T0);  ////////////////////////
        #endif

                                if (bare_bone_search || 
                                    (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                                        top_candidates.emplace(dist, candidate_id);
                                    if (!bare_bone_search && stop_condition) {
                                        stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                                    }
                                }

                                bool flag_remove_extra = false;
                                if (!bare_bone_search && stop_condition) {
                                    flag_remove_extra = stop_condition->should_remove_extra();
                                } else {
                                    flag_remove_extra = top_candidates.size() > ef;
                                }
                                while (flag_remove_extra) {
                                    tableint id = top_candidates.top().second;
                                    top_candidates.pop();
                                    if (!bare_bone_search && stop_condition) {
                                        stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                        flag_remove_extra = stop_condition->should_remove_extra();
                                    } else {
                                        flag_remove_extra = top_candidates.size() > ef;
                                    }
                                }

                                if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                            }
                        }
                    }
                    // std::cout << std::endl;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);
        // std::cout << top_candidates.size() << std::endl;
        return top_candidates;
    }    

    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerClusterEntriesMultiMerge(
        const std::vector<labeltype>& entry_points,
        // tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
            
        std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> top_candidates_list(entry_points.size());
        // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        
        std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> candidate_set_list(entry_points.size());
        std::vector<bool> stop_flag_list(entry_points.size());
        std::vector<dist_t> lower_bound_list(entry_points.size());
        dist_t lowerBound;
        for (int i = 0; i < entry_points.size(); i++) {
            labeltype ep = entry_points[i];
            tableint ep_id = label_lookup_.find(ep)->second;
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfuncCluster((vectorset*)data_point, (vectorset*)ep_data, 0);
            lowerBound = std::max(lowerBound, dist);
            lower_bound_list[i] = std::max(lower_bound_list[i], dist);
            top_candidates_list[i].emplace(dist, ep_id);
            candidate_set_list[i].emplace(-dist, ep_id);
            visited_array[ep_id] = visited_array_tag;
            stop_flag_list[i] = false;
        }
        bool all_empty = false;
        while (!all_empty) {
            all_empty = true;
            for (int i = 0; i < candidate_set_list.size(); i++) {
                if (!stop_flag_list[i] && !candidate_set_list[i].empty()) {
                    std::pair<dist_t, tableint> current_node_pair = candidate_set_list[i].top();
                    dist_t candidate_dist = -current_node_pair.first;
                    // std::cout << top_candidates.size() << " " << lowerBound << " " << getExternalLabel(current_node_pair.second) << "(" << candidate_dist << "): ";
                    bool flag_stop_search;
                    if (bare_bone_search) {
                        flag_stop_search = candidate_dist > lower_bound_list[i];
                    } else {
                        if (stop_condition) {
                            flag_stop_search = stop_condition->should_stop_search(candidate_dist, lower_bound_list[i]);
                        } else {
                            flag_stop_search = candidate_dist > lower_bound_list[i] && top_candidates_list[i].size() == ef;
                        }
                    }
                    if (flag_stop_search) {
                        stop_flag_list[i] = true;
                        continue;
                    }
                    all_empty = false;
                    candidate_set_list[i].pop();

                    tableint current_node_id = current_node_pair.second;
                    int *data = (int *) get_linklist0(current_node_id);
                    size_t size = getListCount((linklistsizeint*)data);
        //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                    if (collect_metrics) {
                        metric_hops++;
                        metric_distance_computations+=size;
                    }

        #ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                    _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                    _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
        #endif

                    for (size_t j = 1; j <= size; j++) {
                        int candidate_id = *(data + j);
        //                    if (candidate_id == 0) continue;
        #ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                        _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                        _MM_HINT_T0);  ////////////
        #endif
                        if (entry_map.find((tableint)candidate_id) == entry_map.end()) {
                            continue;
                        }
                        if (!(visited_array[candidate_id] == visited_array_tag)) {
                            visited_array[candidate_id] = visited_array_tag;

                            char *currObj1 = (getDataByInternalId(candidate_id));
                            // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)currObj1);
                            dist_t dist = fstdistfuncCluster((vectorset*)data_point, (vectorset*)currObj1, 0);
                            // std::cout << getExternalLabel(candidate_id) << "(" << dist << ") ";
                            bool flag_consider_candidate;
                            if (!bare_bone_search && stop_condition) {
                                flag_consider_candidate = stop_condition->should_consider_candidate(dist, lower_bound_list[i]);
                            } else {
                                flag_consider_candidate = top_candidates_list[i].size() < ef || lower_bound_list[i] > dist;
                            }

                            if (flag_consider_candidate) {
                                candidate_set_list[i].emplace(-dist, candidate_id);
                                // std::cout << "T ";
        #ifdef USE_SSE
                                _mm_prefetch(data_level0_memory_ + candidate_set_list[i].top().second * size_data_per_element_ +
                                                offsetLevel0_,  ///////////
                                                _MM_HINT_T0);  ////////////////////////
        #endif

                                if (bare_bone_search || 
                                    (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                                        top_candidates_list[i].emplace(dist, candidate_id);
                                    if (!bare_bone_search && stop_condition) {
                                        stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                                    }
                                }

                                bool flag_remove_extra = false;
                                if (!bare_bone_search && stop_condition) {
                                    flag_remove_extra = stop_condition->should_remove_extra();
                                } else {
                                    flag_remove_extra = top_candidates_list[i].size() > ef;
                                }
                                while (flag_remove_extra) {
                                    tableint id = top_candidates_list[i].top().second;
                                    top_candidates_list[i].pop();
                                    if (!bare_bone_search && stop_condition) {
                                        stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                        flag_remove_extra = stop_condition->should_remove_extra();
                                    } else {
                                        flag_remove_extra = top_candidates_list[i].size() > ef;
                                    }
                                }

                                if (!top_candidates_list[i].empty())
                                lower_bound_list[i] = top_candidates_list[i].top().first;
                            }
                        }
                    }
                    // std::cout << std::endl;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);
        // std::cout << top_candidates.size() << std::endl;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        for (int i = 0; i < candidate_set_list.size(); i++) {
            while(!top_candidates_list[i].empty()) {
                top_candidates.push(top_candidates_list[i].top());
                top_candidates_list[i].pop();
            }
        }
        return top_candidates;
    }

    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerCluster(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            // dist_t dist = L2SqrVecEMD((vectorset*)data_point, (vectorset*)ep_data, 0);
            dist_t dist = fstdistfuncCluster((vectorset*)data_point, (vectorset*)ep_data, 0);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;
            // std::cout << lowerBound << " " << -current_node_pair.first << " " << getExternalLabel(current_node_pair.second) << std::endl;
            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)currObj1);
                    dist_t dist = fstdistfuncCluster((vectorset*)data_point, (vectorset*)currObj1, 0);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        // std::cout << top_candidates.size() << std::endl;
        return top_candidates;
    }

    inline void update_top_candidate_para(std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_cand, std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> &top_local, size_t ef){
        std::set<int> seen;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> min_heap;
        while(!top_cand.empty()){
            if(seen.find(top_cand.top().second) == seen.end()){
                min_heap.emplace(-top_cand.top().first, top_cand.top().second);
                seen.insert(top_cand.top().second);
            }
            top_cand.pop();
        }

        for(int i=0 ;i<top_local.size(); i++){
            while(!top_local[i].empty()){
                if(seen.find(top_local[i].top().second) == seen.end()){
                    min_heap.emplace(-top_local[i].top().first, top_local[i].top().second);
                    seen.insert(top_local[i].top().second);
                }
                top_local[i].pop();
            }
        }

        while(!min_heap.empty() && top_cand.size() < ef){
            top_cand.emplace(-min_heap.top().first, min_heap.top().second);
            min_heap.pop();
        }
    }

    inline void update_candidate_set_para(std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &cand_total, std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> &cand_local, bool delete_tail, dist_t lowerbound, size_t ef){
        std::set<int> seen;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> max_heap;
        // std::cout << cand_total.size() << std::endl;
        while(!cand_total.empty()){
            if(seen.find(cand_total.top().second) == seen.end()){
                max_heap.emplace(-cand_total.top().first, cand_total.top().second);
                seen.insert(cand_total.top().second);
            }
            cand_total.pop();
        }

        for(int i=0 ;i<cand_local.size(); i++){
            // std::cout << cand_local[i].size() << std::endl;
            while(!cand_local[i].empty()){
                if(seen.find(cand_local[i].top().second) == seen.end()){
                    max_heap.emplace(-cand_local[i].top().first, cand_local[i].top().second);
                    seen.insert(cand_local[i].top().second);
                }
                cand_local[i].pop();
            }
        }

        // while(!max_heap.empty() && max_heap.size() < ef_){
        //     if(delete_tail && max_heap.top().first < -lowerbound) break;
        //     cand_total.emplace(max_heap.top().first, max_heap.top().second);
        //     max_heap.pop();
        // }

        // std::cout << max_heap.top().first << " " << max_heap.size() << " " << lowerbound << std::endl;
        while(!max_heap.empty() && max_heap.size() > ef){
            max_heap.pop();
        }
        
        while(!max_heap.empty() && cand_total.size() < ef){
            // std::cout << max_heap.top().first << " " << lowerbound << std::endl;
            if(delete_tail && max_heap.top().first >= lowerbound) {
                max_heap.pop();
            }
            else {
                cand_total.emplace(-max_heap.top().first, max_heap.top().second);
                max_heap.pop();
            }
        }
    }

    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerSTPara(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) {
        // std::cout<< "Parallel Search Knn" << "bare bone search" << bare_bone_search <<std::endl;
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc4search_((vectorset*)data_point, (vectorset*)ep_data, 0);
            // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)ep_data);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> candidate_local(inner_search_thread_num);
        std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> local_result(inner_search_thread_num);

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            candidate_local[0].emplace(-candidate_dist, current_node_pair.second);

            while(!candidate_set.empty()){
                for(int i = 0; i < inner_search_thread_num; i++){
                    if(candidate_set.empty()) break;
                    candidate_local[i].emplace(candidate_set.top().first, candidate_set.top().second);
                    candidate_set.pop();
                }
            }

            #pragma omp parallel num_threads(inner_search_thread_num)
            // for (int tid = 0; tid < thread_num; tid++)
            {
                int curr_thread_id = omp_get_thread_num();
                // int curr_thread_id = tid;
                // std::cout<<"curr_thread_id: " << curr_thread_id<<std::endl;
                for(int t = 0; t < local_rounds; t++){
                    if(candidate_local[curr_thread_id].empty())
                        break;
                    // std::cout<<"curr_thread_id: " << curr_thread_id<< " " << t << " "<< candidate_local[curr_thread_id].size() << std::endl;
                    std::pair<dist_t, tableint> current_node_pair_local = candidate_local[curr_thread_id].top();
                    // std::cout<<"curr_thread_id: " << curr_thread_id<< " top " << t <<std::endl;
                    tableint current_node_id_local = current_node_pair_local.second;
                    int *data = (int *) get_linklist0(current_node_id_local);
                    size_t size = getListCount((linklistsizeint*)data);
                    candidate_local[curr_thread_id].pop();
                    // std::cout<<"curr_thread_id: " << curr_thread_id<< " pop " << t <<std::endl;
                    for (size_t j = 1; j <= size; j++) {
                        int candidate_id = *(data + j);
        //                    if (candidate_id == 0) continue;
                        if (!(visited_array[candidate_id] == visited_array_tag)) {
                            visited_array[candidate_id] = visited_array_tag;

                            char *currObj1 = (getDataByInternalId(candidate_id));
                            // dist_t dist = fstdistfunc_((vectorset*)data_point, (vectorset*)currObj1);
                            dist_t dist = fstdistfunc4search_((vectorset*)data_point, (vectorset*)currObj1, 0);
                            // bool flag_consider_candidate;
                            // if (!bare_bone_search && stop_condition) {
                            //     flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                            // } else {
                            //     flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                            // }

                            if(top_candidates.size() < ef || lowerBound > dist){
                                candidate_local[curr_thread_id].emplace(-dist, candidate_id);
                                local_result[curr_thread_id].emplace(dist, candidate_id);
                                if(local_result[curr_thread_id].size() == ef)
                                    local_result[curr_thread_id].pop();
                            }
                        }
                    }
                }
            }
            
            update_top_candidate_para(top_candidates, local_result, ef);

            lowerBound = top_candidates.top().first;

            bool delete_tail = false;
            if(top_candidates.size() >= ef)
                delete_tail = true;
            // std::cout<<candidate_set.size()<<std::endl;
            update_candidate_set_para(candidate_set, candidate_local, delete_tail, lowerBound, ef);
            // std::cout<<candidate_set.size()<<std::endl;
            
        }

        visited_list_pool_->releaseVisitedList(vl);
        // std::cout<< top_candidates.size() << std::endl;
        return top_candidates;
    }

    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M, int level) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfuncEMD((vectorset*)getDataByInternalId(second_pair.second),
                                        (vectorset*)getDataByInternalId(curent_pair.second), 0);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    // new for logical h
    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return get_linklist0(internal_id);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return get_linklist0(internal_id);
    }
    // end for logical h

    // original for recover
    // linklistsizeint *get_linklist0(tableint internal_id) const {
    //     return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    // }


    // linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
    //     return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    // }


    // linklistsizeint *get_linklist(tableint internal_id, int level) const {
    //     return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    // }


    // linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
    //     return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    // }
    // end recover


    void getNeighborsByHeuristic2Cluster(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const float* cluster_distance,
        const size_t M, int level) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfuncClusterEMD((vectorset*)getDataByInternalId(second_pair.second),
                                        (vectorset*)getDataByInternalId(curent_pair.second), cluster_distance);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }



    void getNeighborsByHeuristic2ClusterKeep(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const float* cluster_distance,
        const size_t M, int level) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfuncClusterEMD((vectorset*)getDataByInternalId(second_pair.second),
                                        (vectorset*)getDataByInternalId(curent_pair.second), cluster_distance);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }    

    tableint mutuallyConnectUpdateElementCluster(
        const void *data_point,
        const float *cluster_distance,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        volatile int temp = 6;
        size_t Mcurmax = level ? maxM_ : maxM0_;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> cur_candidates;
        std::unordered_set<tableint>visi_set;
        // std::cout << std::to_string(temp) + " " + std::to_string(cur_c) + " " + "stage1 " << std::endl;
        // std::cout << temp << " " << cur_c << std::endl;
        while (top_candidates.size() > 0) {
            if (top_candidates.top().second != cur_c && visi_set.find(top_candidates.top().second) == visi_set.end()) {
                visi_set.insert(top_candidates.top().second);
                cur_candidates.push(top_candidates.top());
            }
            top_candidates.pop();
        }
        // std::cout << " " << temp << " "  << cur_candidates.size() << std::endl;
        {
            // std::cout << " " << temp << " "  << cur_c << std::flush;
            //std::unique_lock <std::mutex> lock(link_list_locks_[cur_c]);
            linklistsizeint *ll_cur = get_linklist0(cur_c);
            size_t sz_link_list_cur = getListCount(ll_cur);
            tableint *data = (tableint *) (ll_cur + 1);
            // std::cout << " " << temp << " "  << sz_link_list_cur << std::flush;
            for (size_t j = 0; j < sz_link_list_cur; j++) {
                // std::cout << " " << temp << " " << data[j] << std::flush;
                if (visi_set.find(data[j]) == visi_set.end()) {
                    cur_candidates.emplace(
                        fstdistfuncClusterEMD((vectorset*)getDataByInternalId(data[j]), (vectorset*)getDataByInternalId(cur_c), cluster_distance), data[j]);
                }
            }
            // std::cout << " " << temp << " " << cur_candidates.size() << std::flush;
        }
        // std::cout << temp << " " << cur_candidates.size() << std::endl;
        getNeighborsByHeuristic2Cluster(cur_candidates, cluster_distance, M_, level);
        // std::cout << " " << cur_c << " ";
        // std::cout << " " << cur_candidates.size() << std::flush;

        if (cur_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (cur_candidates.size() > 0) {
            // std::cout << cur_candidates.top().second << " " << cur_candidates.top().first << " ";
            selectedNeighbors.push_back(cur_candidates.top().second);
            cur_candidates.pop();
        }
        // std::cout << temp << " " << selectedNeighbors.size() << std::endl;
        // std::cout << " " << selectedNeighbors.size() << std::flush;
        // std::cout << std::endl;

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            // std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            // if (isUpdate) {
            //     lock.lock();
            // }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            // uint8_t* data_list = (uint8_t*) ((tableint *) data + maxM0_);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                // std::cout << temp << " " << idx <<  " "  << selectedNeighbors[idx] << " ";
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");
                data[idx] = selectedNeighbors[idx];
            }
        }
        // std::cout << temp << " ok " << selectedNeighbors.size() << std::endl;

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            // std::cout << temp << " ?? " << cur_c << " ?? " << idx << " " << selectedNeighbors[idx] << std::endl;
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);
            // std::cout << temp << " ?? " << cur_c << " ?? " << idx << " " << selectedNeighbors[idx] << std::endl;
            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);
            // uint8_t* data_list = (uint8_t*) ((tableint *) data + maxM0_);
            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }
            // std::cout << temp << " " << selectedNeighbors[idx] << std::endl;
            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    // dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(selectedNeighbors[idx]), (vectorset*)getDataByInternalId(cur_c), data_list + fineEdgeSize * sz_link_list_other, 0);
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    // dist_t tmp = fstdistfuncInit_((vectorset*)getDataByInternalId(cur_c), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), data_list + fineEdgeSize * idx, 0);
                
                    dist_t d_max = fstdistfuncClusterEMD((vectorset*)getDataByInternalId(cur_c), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), cluster_distance);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                            fstdistfuncClusterEMD((vectorset*)getDataByInternalId(data[j]), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), cluster_distance), data[j]);
                    }

                    getNeighborsByHeuristic2Cluster(candidates, cluster_distance, Mcurmax, level);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        // dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(selectedNeighbors[idx]), (vectorset*)getDataByInternalId(data[indx]), data_list + fineEdgeSize * indx, 0);
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
            // std::cout << temp << " " << selectedNeighbors[idx] << std::endl;
        }
        // std::cout << temp << " finish" << std::endl;
        return next_closest_entry_point;
    }

    tableint mutuallyConnectNewElementCluster(
        const void *data_point,
        const float *cluster_distance,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2Cluster(top_candidates, cluster_distance, M_, level);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            if (top_candidates.top().second == cur_c) {
                top_candidates.pop();
                continue;
            }
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }
        // std::cout << " " << selectedNeighbors.size();
        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            // uint8_t* data_list = (uint8_t*) ((tableint *) data + maxM0_);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");
                data[idx] = selectedNeighbors[idx];
            }
        }
        // std::cout << " " << selectedNeighbors.size();
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);
            // std::cout << " " << selectedNeighbors[idx];
            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);
            // uint8_t* data_list = (uint8_t*) ((tableint *) data + maxM0_);
            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    // dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(selectedNeighbors[idx]), (vectorset*)getDataByInternalId(cur_c), data_list + fineEdgeSize * sz_link_list_other, 0);
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    // dist_t tmp = fstdistfuncInit_((vectorset*)getDataByInternalId(cur_c), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), data_list + fineEdgeSize * idx, 0);
                
                    dist_t d_max = fstdistfuncClusterEMD((vectorset*)getDataByInternalId(cur_c), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), cluster_distance);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                            fstdistfuncClusterEMD((vectorset*)getDataByInternalId(data[j]), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), cluster_distance), data[j]);
                    }

                    getNeighborsByHeuristic2Cluster(candidates, cluster_distance, Mcurmax, level);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        // dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(selectedNeighbors[idx]), (vectorset*)getDataByInternalId(data[indx]), data_list + fineEdgeSize * indx, 0);
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }
        // std::cout << " " << selectedNeighbors.size();
        return next_closest_entry_point;
    }

    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_, level);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            // uint8_t* data_list = (uint8_t*) ((tableint *) data + maxM0_);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");
                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);
            // uint8_t* data_list = (uint8_t*) ((tableint *) data + maxM0_);
            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    // dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(selectedNeighbors[idx]), (vectorset*)getDataByInternalId(cur_c), data_list + fineEdgeSize * sz_link_list_other, 0);
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    // dist_t tmp = fstdistfuncInit_((vectorset*)getDataByInternalId(cur_c), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), data_list + fineEdgeSize * idx, 0);
                
                    dist_t d_max = fstdistfuncEMD((vectorset*)getDataByInternalId(cur_c), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), 0);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                            fstdistfuncEMD((vectorset*)getDataByInternalId(data[j]), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), 0), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax, level);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        // dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(selectedNeighbors[idx]), (vectorset*)getDataByInternalId(data[indx]), data_list + fineEdgeSize * indx, 0);
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }

    // tableint mutuallyConnectNewElement(
    //     const void *data_point,
    //     tableint cur_c,
    //     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    //     int level,
    //     bool isUpdate) {
    //     size_t Mcurmax = level ? maxM_ : maxM0_;
    //     getNeighborsByHeuristic2(top_candidates, M_, level);
    //     if (top_candidates.size() > M_)
    //         throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

    //     std::vector<tableint> selectedNeighbors;
    //     selectedNeighbors.reserve(M_);
    //     while (top_candidates.size() > 0) {
    //         selectedNeighbors.push_back(top_candidates.top().second);
    //         top_candidates.pop();
    //     }

    //     tableint next_closest_entry_point = selectedNeighbors.back();

    //     {
    //         // lock only during the update
    //         // because during the addition the lock for cur_c is already acquired
    //         std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
    //         if (isUpdate) {
    //             lock.lock();
    //         }
    //         linklistsizeint *ll_cur;
    //         if (level == 0)
    //             ll_cur = get_linklist0(cur_c);
    //         else
    //             ll_cur = get_linklist(cur_c, level);

    //         if (*ll_cur && !isUpdate) {
    //             throw std::runtime_error("The newly inserted element should have blank link list");
    //         }
    //         setListCount(ll_cur, selectedNeighbors.size());
    //         tableint *data = (tableint *) (ll_cur + 1);
    //         uint8_t* data_list = (uint8_t*) ((tableint *) data + maxM0_);
    //         for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
    //             if (data[idx] && !isUpdate)
    //                 throw std::runtime_error("Possible memory corruption");
    //             if (level > element_levels_[selectedNeighbors[idx]])
    //                 throw std::runtime_error("Trying to make a link on a non-existent level");
    //             dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(cur_c), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), data_list + fineEdgeSize * idx, 0);
    //             // {
    //             //     std::lock_guard<std::mutex> lock(cout_mutex); 
    //             //     size_t bn = std::min(((vectorset*)getDataByInternalId(cur_c))->vecnum, (size_t)120);
    //             //     size_t cn = std::min(((vectorset*)getDataByInternalId(selectedNeighbors[idx]))->vecnum, (size_t)120);   
    //             //     std::cout << "connect ==" << cur_c << " " << selectedNeighbors[idx] << " " << idx << std::endl;
    //             //     std::cout << "== MapBC == " << bn << " " << cn << std::endl;
    //             //     bool flag = true;
    //             //     for (uint8_t i = 0; i < bn; ++i) {
    //             //         std::cout << (u_int16_t)i << " " << (u_int16_t)((data_list + fineEdgeSize * idx)[i]) << std::endl;
    //             //         flag = flag & (((data_list + fineEdgeSize * idx)[i]) < cn);
    //             //     }
    //             //     for (uint8_t i = 0; i < cn; ++i) {
    //             //         std::cout << (u_int16_t)i << " " << (u_int16_t)((data_list + fineEdgeSize * idx)[i + 120 * 1]) << std::endl;
    //             //         flag = flag & (((data_list + fineEdgeSize * idx)[i + 120]) < bn);
    //             //     }
    //             //     assert (flag);
    //             //     std::cout << "== MapBC ==" << std::endl;
    //             // }
    //             data[idx] = selectedNeighbors[idx];
    //         }
    //     }

    //     for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
    //         std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

    //         linklistsizeint *ll_other;
    //         if (level == 0)
    //             ll_other = get_linklist0(selectedNeighbors[idx]);
    //         else
    //             ll_other = get_linklist(selectedNeighbors[idx], level);

    //         size_t sz_link_list_other = getListCount(ll_other);

    //         if (sz_link_list_other > Mcurmax)
    //             throw std::runtime_error("Bad value of sz_link_list_other");
    //         if (selectedNeighbors[idx] == cur_c)
    //             throw std::runtime_error("Trying to connect an element to itself");
    //         if (level > element_levels_[selectedNeighbors[idx]])
    //             throw std::runtime_error("Trying to make a link on a non-existent level");

    //         tableint *data = (tableint *) (ll_other + 1);
    //         uint8_t* data_list = (uint8_t*) ((tableint *) data + maxM0_);
    //         bool is_cur_c_present = false;
    //         if (isUpdate) {
    //             for (size_t j = 0; j < sz_link_list_other; j++) {
    //                 if (data[j] == cur_c) {
    //                     is_cur_c_present = true;
    //                     break;
    //                 }
    //             }
    //         }

    //         // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
    //         if (!is_cur_c_present) {
    //             if (sz_link_list_other < Mcurmax) {
    //                 data[sz_link_list_other] = cur_c;
    //                 dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(selectedNeighbors[idx]), (vectorset*)getDataByInternalId(cur_c), data_list + fineEdgeSize * sz_link_list_other, 0);
    //                 // {
    //                 //     std::lock_guard<std::mutex> lock(cout_mutex); 
    //                 //     size_t bn = std::min(((vectorset*)getDataByInternalId(selectedNeighbors[idx]))->vecnum, (size_t)120);
    //                 //     size_t cn = std::min(((vectorset*)getDataByInternalId(cur_c))->vecnum, (size_t)120);    
    //                 //     std::cout << "connect ==" << selectedNeighbors[idx] << " " << cur_c << " " << sz_link_list_other << std::endl;
    //                 //     std::cout << "== MapBC == " << bn << " " << cn << std::endl;
    //                 //     bool flag = true;
    //                 //     for (uint8_t i = 0; i < bn; ++i) {
    //                 //         std::cout << (u_int16_t)i << " " << (u_int16_t)((data_list + fineEdgeSize * sz_link_list_other)[i]) << std::endl;
    //                 //         flag = flag & (((data_list + fineEdgeSize * sz_link_list_other)[i]) < cn);
    //                 //     }
    //                 //     for (uint8_t i = 0; i < cn; ++i) {
    //                 //         std::cout << (u_int16_t)i << " " << (u_int16_t)((data_list + fineEdgeSize * sz_link_list_other)[i + 120 * 1]) << std::endl;
    //                 //         flag = flag & (((data_list + fineEdgeSize * sz_link_list_other)[i + 120]) < bn);
    //                 //     }
    //                 //     assert (flag);
    //                 //     std::cout << "== MapBC ==" << std::endl;
    //                 // }
    //                 setListCount(ll_other, sz_link_list_other + 1);
    //             } else {
    //                 // finding the "weakest" element to replace it with the new one
    //                 // dist_t tmp = fstdistfuncInit_((vectorset*)getDataByInternalId(cur_c), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), data_list + fineEdgeSize * idx, 0);
                
    //                 dist_t d_max = fstdistfuncEMD((vectorset*)getDataByInternalId(cur_c), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), level);
    //                 // Heuristic:
    //                 std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
    //                 candidates.emplace(d_max, cur_c);

    //                 for (size_t j = 0; j < sz_link_list_other; j++) {
    //                     candidates.emplace(
    //                             fstdistfuncEMD((vectorset*)getDataByInternalId(data[j]), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), level), data[j]);
    //                 }

    //                 getNeighborsByHeuristic2(candidates, Mcurmax, level);

    //                 int indx = 0;
    //                 while (candidates.size() > 0) {
    //                     data[indx] = candidates.top().second;
    //                     dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(selectedNeighbors[idx]), (vectorset*)getDataByInternalId(data[indx]), data_list + fineEdgeSize * indx, 0);
    //                     // {
    //                     //     std::lock_guard<std::mutex> lock(cout_mutex); 
    //                     //     size_t bn = std::min(((vectorset*)getDataByInternalId(selectedNeighbors[idx]))->vecnum, (size_t)120);
    //                     //     size_t cn = std::min(((vectorset*)getDataByInternalId(data[indx]))->vecnum, (size_t)120);    
    //                     //     std::cout << "connect ==" << selectedNeighbors[idx] << " " << data[indx] << " " << indx << std::endl;
    //                     //     std::cout << "== MapBC == " << bn << " " << cn << std::endl;
    //                     //     bool flag = true;
    //                     //     for (uint8_t i = 0; i < bn; ++i) {
    //                     //         std::cout << (u_int16_t)i << " " << (u_int16_t)((data_list + fineEdgeSize * indx)[i]) << std::endl;
    //                     //         flag = flag & ((((data_list + fineEdgeSize * indx)[i]) < cn));
    //                     //     }
    //                     //     for (uint8_t i = 0; i < cn; ++i) {
    //                     //         std::cout << (u_int16_t)i << " " << (u_int16_t)((data_list + fineEdgeSize * indx)[i + 120 * 1]) << std::endl;
    //                     //         flag = flag & (((data_list + fineEdgeSize * indx)[i + 120]) < bn);
    //                     //     }
    //                     //     assert (flag);
    //                     //     std::cout << "== MapBC ==" << std::endl;
    //                     // }
    //                     candidates.pop();
    //                     indx++;
    //                 }

    //                 setListCount(ll_other, indx);
    //                 // Nearest K:
    //                 /*int indx = -1;
    //                 for (int j = 0; j < sz_link_list_other; j++) {
    //                     dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
    //                     if (d > d_max) {
    //                         indx = j;
    //                         d_max = d;
    //                     }
    //                 }
    //                 if (indx >= 0) {
    //                     data[indx] = cur_c;
    //                 } */
    //             }
    //         }
    //     }

    //     return next_closest_entry_point;
    // }

    void mutuallyConnectTwoElement(
        labeltype label1,
        labeltype label2,
        bool isUpdate = true) {
        int level = 0;
        size_t Mcurmax = level ? maxM_ : maxM0_;
        // std::cout << label1 << ' ' << label2 << std::endl;
        tableint p1, p2;
        {
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label1);
            if (search != label_lookup_.end()) {
                p1 = search->second;
            }
            search = label_lookup_.find(label2);
            if (search != label_lookup_.end()) {
                p2 = search->second;
            }
            lock_table.unlock();

        }
        tableint cur_c = p2;
        {
            // std::cout << p1 << ' ' << p2 << std::endl;
            std::unique_lock <std::mutex> lock(link_list_locks_[p1]);
            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(p1);
            else
                ll_other = get_linklist(p1, level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");

            tableint *data = (tableint *) (ll_other + 1);
            // uint8_t* data_list = (uint8_t*) ((tableint *) data + maxM0_);
            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    // dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(p1), (vectorset*)getDataByInternalId(cur_c), data_list + fineEdgeSize * sz_link_list_other, 0);
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    dist_t d_max = 0;
                    int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfuncEMD((vectorset*)getDataByInternalId(p1), (vectorset*)getDataByInternalId(data[j]), 0);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                        // dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(p1), (vectorset*)getDataByInternalId(cur_c), data_list + fineEdgeSize * indx, 0);
                    } 
                }
            }
        }

        // linklistsizeint *ll_other;
        // ll_other = get_linklist0(p1);
        // size_t sz_link_list_other = getListCount(ll_other);
        // tableint *data = (tableint *) (ll_other + 1);
        // std::cout << getExternalLabel(p1) << " " << sz_link_list_other << ": ";
        // for (size_t j = 0; j < sz_link_list_other; j++) {
        //     std::cout << getExternalLabel(data[j]) << " "; 
        // }
        // std::cout << std::endl;
        return;
    }


    void mutuallyConnectTwoInterElement(
        tableint p1,
        tableint p2,
        bool isUpdate = true) {
        int level = 0;
        size_t Mcurmax = level ? maxM_ : maxM0_;
        tableint cur_c = p2;
        {
            // std::cout << p1 << ' ' << p2 << std::endl;
            std::unique_lock <std::mutex> lock(link_list_locks_[p1]);
            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(p1);
            else
                ll_other = get_linklist(p1, level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");

            tableint *data = (tableint *) (ll_other + 1);
            // uint8_t* data_list = (uint8_t*) ((tableint *) data + maxM0_);
            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    // dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(p1), (vectorset*)getDataByInternalId(cur_c), data_list + fineEdgeSize * sz_link_list_other, 0);
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    dist_t d_max = 0;
                    int indx = -1;
                    data[cur_c % sz_link_list_other] = cur_c;

                    // for (int j = 0; j < sz_link_list_other; j++) {
                    //     dist_t d = fstdistfuncEMD((vectorset*)getDataByInternalId(p1), (vectorset*)getDataByInternalId(data[j]), 0);
                    //     if (d > d_max) {
                    //         indx = j;
                    //         d_max = d;
                    //     }
                    // }
                    // if (indx >= 0) {
                    //     data[indx] = cur_c;
                    //     // dist_t tmp = fstdistfuncInitEMD((vectorset*)getDataByInternalId(p1), (vectorset*)getDataByInternalId(cur_c), data_list + fineEdgeSize * indx, 0);
                    // } 
                }
            }
        }

        // linklistsizeint *ll_other;
        // ll_other = get_linklist0(p1);
        // size_t sz_link_list_other = getListCount(ll_other);
        // tableint *data = (tableint *) (ll_other + 1);
        // std::cout << getExternalLabel(p1) << " " << sz_link_list_other << ": ";
        // for (size_t j = 0; j < sz_link_list_other; j++) {
        //     std::cout << getExternalLabel(data[j]) << " "; 
        // }
        // std::cout << std::endl;
        return;
    }

    bool canAddEdge(
        labeltype label1) {
        int level = 0;
        size_t Mcurmax = level ? maxM_ : maxM0_;
        // std::cout << label1 << ' ' << label2 << std::endl;
        tableint p1;
        {
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label1);
            if (search != label_lookup_.end()) {
                p1 = search->second;
            }
            lock_table.unlock();

        }
        bool canadd = false;
        {
            std::unique_lock <std::mutex> lock(link_list_locks_[p1]);
            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(p1);
            else
                ll_other = get_linklist(p1, level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other < Mcurmax)
                canadd = true;
            // else
            //     std::cout << label1 << " " << sz_link_list_other << " " << Mcurmax << std::endl;
        }
        
        return canadd;
    }


    bool canAddEdgeinter(
        tableint p1) {
        int level = 0;
        size_t Mcurmax = level ? maxM_ : maxM0_;
        bool canadd = false;
        {
            std::unique_lock <std::mutex> lock(link_list_locks_[p1]);
            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(p1);
            else
                ll_other = get_linklist(p1, level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other < Mcurmax)
                canadd = true;
            // else
            //     std::cout << label1 << " " << sz_link_list_other << " " << Mcurmax << std::endl;
        }
        
        return canadd;
    }
    // tableint mutuallyConnectNewElement(
    //     const void *data_point,
    //     tableint cur_c,
    //     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    //     int level,
    //     bool isUpdate) {
    //     size_t Mcurmax = level ? maxM_ : maxM0_;
    //     getNeighborsByHeuristic2(top_candidates, M_, level);
    //     if (top_candidates.size() > M_)
    //         throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

    //     std::vector<tableint> selectedNeighbors;
    //     selectedNeighbors.reserve(M_);
    //     while (top_candidates.size() > 0) {
    //         selectedNeighbors.push_back(top_candidates.top().second);
    //         top_candidates.pop();
    //     }

    //     tableint next_closest_entry_point = selectedNeighbors.back();

    //     {
    //         // lock only during the update
    //         // because during the addition the lock for cur_c is already acquired
    //         std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
    //         if (isUpdate) {
    //             lock.lock();
    //         }
    //         linklistsizeint *ll_cur;
    //         if (level == 0)
    //             ll_cur = get_linklist0(cur_c);
    //         else
    //             ll_cur = get_linklist(cur_c, level);

    //         if (*ll_cur && !isUpdate) {
    //             throw std::runtime_error("The newly inserted element should have blank link list");
    //         }
    //         setListCount(ll_cur, selectedNeighbors.size());
    //         tableint *data = (tableint *) (ll_cur + 1);
    //         for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
    //             if (data[idx] && !isUpdate)
    //                 throw std::runtime_error("Possible memory corruption");
    //             if (level > element_levels_[selectedNeighbors[idx]])
    //                 throw std::runtime_error("Trying to make a link on a non-existent level");

    //             data[idx] = selectedNeighbors[idx];
    //         }
    //     }

    //     for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
    //         std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

    //         linklistsizeint *ll_other;
    //         if (level == 0)
    //             ll_other = get_linklist0(selectedNeighbors[idx]);
    //         else
    //             ll_other = get_linklist(selectedNeighbors[idx], level);

    //         size_t sz_link_list_other = getListCount(ll_other);

    //         if (sz_link_list_other > Mcurmax)
    //             throw std::runtime_error("Bad value of sz_link_list_other");
    //         if (selectedNeighbors[idx] == cur_c)
    //             throw std::runtime_error("Trying to connect an element to itself");
    //         if (level > element_levels_[selectedNeighbors[idx]])
    //             throw std::runtime_error("Trying to make a link on a non-existent level");

    //         tableint *data = (tableint *) (ll_other + 1);

    //         bool is_cur_c_present = false;
    //         if (isUpdate) {
    //             for (size_t j = 0; j < sz_link_list_other; j++) {
    //                 if (data[j] == cur_c) {
    //                     is_cur_c_present = true;
    //                     break;
    //                 }
    //             }
    //         }

    //         // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
    //         if (!is_cur_c_present) {
    //             if (sz_link_list_other < Mcurmax) {
    //                 data[sz_link_list_other] = cur_c;
    //                 setListCount(ll_other, sz_link_list_other + 1);
    //             } else {
    //                 // finding the "weakest" element to replace it with the new one
    //                 dist_t d_max = fstdistfunc_((vectorset*)getDataByInternalId(cur_c), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), level);
    //                 // Heuristic:
    //                 std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
    //                 candidates.emplace(d_max, cur_c);

    //                 for (size_t j = 0; j < sz_link_list_other; j++) {
    //                     candidates.emplace(
    //                             fstdistfunc_((vectorset*)getDataByInternalId(data[j]), (vectorset*)getDataByInternalId(selectedNeighbors[idx]), level), data[j]);
    //                 }

    //                 getNeighborsByHeuristic2(candidates, Mcurmax, level);

    //                 int indx = 0;
    //                 while (candidates.size() > 0) {
    //                     data[indx] = candidates.top().second;
    //                     candidates.pop();
    //                     indx++;
    //                 }

    //                 setListCount(ll_other, indx);
    //                 // Nearest K:
    //                 /*int indx = -1;
    //                 for (int j = 0; j < sz_link_list_other; j++) {
    //                     dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
    //                     if (d > d_max) {
    //                         indx = j;
    //                         d_max = d;
    //                     }
    //                 }
    //                 if (indx >= 0) {
    //                     data[indx] = cur_c;
    //                 } */
    //             }
    //         }
    //     }

    //     return next_closest_entry_point;
    // }



    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        // data_size_ = s->get_data_size();
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
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        // 修改这里
        size_links_per_element_ = maxM0_ * (sizeof(tableint) + fineEdgeSize * sizeof(uint8_t))  + sizeof(linklistsizeint);
        size_links_level0_ = maxM0_ * (sizeof(tableint) + fineEdgeSize * sizeof(uint8_t))  + sizeof(linklistsizeint);
        //修改这里
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 80;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }

    void testDataLabel(labeltype label) {
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                lock_table.unlock();
                // std::cout << getExternalLabel(existingInternalId) << " " << existingInternalId << " " << label << std::endl;
                // memcpy(getDataByInternalId(existingInternalId), data_point, data_size_);
                // setExternalLabel(existingInternalId, label);
                return ;
            }
        }
    }

    void loadDataAddress(const void *data_point , labeltype label) {
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                lock_table.unlock();
                // std::cout << getExternalLabel(existingInternalId) << " " << existingInternalId << " " << label << std::endl;
                memcpy(getDataByInternalId(existingInternalId), data_point, data_size_);
                // std::cout << getExternalLabel(existingInternalId) << " " << existingInternalId << " " << label << std::endl;
                // setExternalLabel(existingInternalId, label);
                return ;
            }
        }
    }

    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }

    void addClusterPoint(const void *data_point, const float *cluster_distance, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addClusterPoint(data_point, cluster_distance, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addClusterPoint(data_point, cluster_distance, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }

    // new pipeline for para
    bool addPointMem(const void *data_point, labeltype label, labeltype entry) {
        tableint cur_c = 0;
        bool new_point = true;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                cur_c = existingInternalId;
                lock_table.unlock();
                new_point = false;
                // std::cout << label << " " << cur_c << " " << entry << " ? " << std::flush; 
            }
            else {
                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                }
                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
                // std::cout << label << " " << cur_c << " " << entry << " new " << std::flush;
            }
        }
        tableint enterpoint_copy = 0;
        if (label == entry) {
            enterpoint_copy = cur_c;
        } else {
            enterpoint_copy = label_lookup_[entry];
        }
        entry_map[cur_c] = enterpoint_copy;
        if (new_point) {
            int curlevel = getRandomLevel(mult_);
            element_levels_[cur_c] = curlevel;
            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        }
        return new_point;
    }

    bool updateOldPointClusterEntry(const void *data_point, const float *cluster_distance, labeltype label, labeltype entry) {
        tableint cur_c = label_lookup_[label];
        if (label == entry) {
            return cur_c;
        }
        tableint currObj = label_lookup_[entry];
        int level = 0;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayerClusterAppr(
                currObj, data_point, cluster_distance, level);
        currObj = mutuallyConnectUpdateElementCluster(data_point, cluster_distance, cur_c, top_candidates, level, true);
        return currObj;
    }


    tableint updateNewPointClusterEntry(const void *data_point, const float *cluster_distance, labeltype label, labeltype entry) {
        tableint cur_c = label_lookup_[label];
        if (label == entry) {
            return cur_c;
        }
        tableint currObj = label_lookup_[entry];
        int level = 0;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayerClusterAppr(
                currObj, data_point, cluster_distance, level);
        currObj = mutuallyConnectNewElementCluster(data_point, cluster_distance, cur_c, top_candidates, level, true);
        return currObj;
    }

    void addClusterPointEntry(const void *data_point, const float *cluster_distance, labeltype label, labeltype enrty, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addClusterPointEntry(data_point, cluster_distance, label, enrty, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addClusterPointEntry(data_point, cluster_distance, label, enrty, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }

    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);
        // std::cout << "???" << std::endl;
        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_((vectorset*)getDataByInternalId(neigh), (vectorset*)getDataByInternalId(cand),layer);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_, layer);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_((vectorset*)dataPoint, (vectorset*)getDataByInternalId(currObj), dataPointLevel);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_((vectorset*)dataPoint, (vectorset*)getDataByInternalId(cand), level);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_((vectorset*)dataPoint, (vectorset*)getDataByInternalId(entryPointInternalId), level), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }

    void getListCountForOutId() {
        int mean_size = 0;
        int max_size = 0;
        int min_size = 100;
        for (tableint i = 0; i < cur_element_count; i++) {
            int size = getListCount((unsigned int *) get_linklist0(i));
            mean_size += size;
            max_size = std::max(max_size, size);
            min_size = std::min(min_size, size);
        }
        std::cout << mean_size / cur_element_count << ' ' << max_size << ' ' << min_size << std::endl;
    }

    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;
        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        // std::cout << "Before memcpy: " << *getExternalLabeLp(cur_c) << " " << getExternalLabel(cur_c) <<  " " << label << std::endl;
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        // std::cout << "After memcpy: " << *getExternalLabeLp(cur_c) << " " << getExternalLabel(cur_c) <<  " " << label << std::endl;
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        // *getExternalLabeLp(cur_c) = label;
        // std::cout << data_size_ << " " << sizeof(vectorset) << std::endl;
        // std::cout << *getExternalLabeLp(cur_c) << " " << (labeltype)getExternalLabel(cur_c) << " " << label << " " << sizeof(label) << " " << sizeof(labeltype) << std::endl;
        // // std::cout << ((vectorset*)data_point)->vecnum << " " << (float)(((vectorset*)data_point)->data)[0] << " " << (float)(((vectorset*)data_point)->data)[1] <<  std::endl;
        // std::cout << ((vectorset*)getDataByInternalId(cur_c))->vecnum << " " << (float)(((vectorset*)getDataByInternalId(cur_c))->data)[0]  << " " << (float)(((vectorset*)getDataByInternalId(cur_c))->data)[1] << std::endl;
        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            // for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
            //     if (level > maxlevelcopy || level < 0)  // possible?
            //         throw std::runtime_error("Level error");
            //     // std::cout << "search " << std::endl;
            //     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
            //             currObj, data_point, level);
            //     if (epDeleted) {
            //         // std::cout << "add Point: emplace " << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)getDataByInternalId(enterpoint_copy))->vecnum << std::endl;
            //         top_candidates.emplace(fstdistfunc_((vectorset*)data_point, (vectorset*)getDataByInternalId(enterpoint_copy), 0), enterpoint_copy);
            //         if (top_candidates.size() > ef_construction_)
            //             top_candidates.pop();
            //     }
            //     currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            // }
            int level = 0;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                    currObj, data_point, level);
            if (epDeleted) {
                // std::cout << "add Point: emplace " << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)getDataByInternalId(enterpoint_copy))->vecnum << std::endl;
                top_candidates.emplace(fstdistfunc_((vectorset*)data_point, (vectorset*)getDataByInternalId(enterpoint_copy), 0), enterpoint_copy);
                if (top_candidates.size() > ef_construction_)
                    top_candidates.pop();
            }
            currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    tableint addClusterPoint(const void *data_point, const float *cluster_distance, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;
        // entry_map[cur_c] = enterpoint_copy;
        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        // std::cout << "Before memcpy: " << *getExternalLabeLp(cur_c) << " " << getExternalLabel(cur_c) <<  " " << label << std::endl;
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        // std::cout << "After memcpy: " << *getExternalLabeLp(cur_c) << " " << getExternalLabel(cur_c) <<  " " << label << std::endl;
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        // *getExternalLabeLp(cur_c) = label;
        // std::cout << data_size_ << " " << sizeof(vectorset) << std::endl;
        // std::cout << *getExternalLabeLp(cur_c) << " " << (labeltype)getExternalLabel(cur_c) << " " << label << " " << sizeof(label) << " " << sizeof(labeltype) << std::endl;
        // // std::cout << ((vectorset*)data_point)->vecnum << " " << (float)(((vectorset*)data_point)->data)[0] << " " << (float)(((vectorset*)data_point)->data)[1] <<  std::endl;
        // std::cout << ((vectorset*)getDataByInternalId(cur_c))->vecnum << " " << (float)(((vectorset*)getDataByInternalId(cur_c))->data)[0]  << " " << (float)(((vectorset*)getDataByInternalId(cur_c))->data)[1] << std::endl;
        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            int level = 0;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayerClusterAppr(
                    currObj, data_point, cluster_distance, level);
            if (epDeleted) {
                // std::cout << "add Point: emplace " << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)getDataByInternalId(enterpoint_copy))->vecnum << std::endl;
                top_candidates.emplace(fstdistfuncClusterEMD((vectorset*)data_point, (vectorset*)getDataByInternalId(enterpoint_copy), cluster_distance), enterpoint_copy);
                if (top_candidates.size() > ef_construction_)
                    top_candidates.pop();
            }
            currObj = mutuallyConnectNewElementCluster(data_point, cluster_distance, cur_c, top_candidates, level, false);
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    tableint addClusterPointEntry(const void *data_point, const float *cluster_distance, labeltype label, labeltype entry, int level) {
        tableint cur_c = 0;
        bool new_point = true;
        volatile int temp = 6;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                cur_c = existingInternalId;
                lock_table.unlock();
                new_point = false;
                // std::cout << label << " " << cur_c << " " << entry << " ? " << std::flush; 
                // std::cout << std::to_string(temp) + " " + std::to_string(new_point) + " " + std::to_string(cur_c) << std::endl;
            }
            else {
                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                }
                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
                // std::cout << label << " " << cur_c << " " << entry << " new " << std::flush;
                // std::cout << std::to_string(temp) + " " + std::to_string(new_point) + " " + std::to_string(cur_c) << std::endl;
            }
        }
        // if (!new_point) {
        // std::cout << " " << new_point << " " << cur_c << " " << std::flush;
        // }
        tableint currObj = 0;
        tableint enterpoint_copy = 0;
        if (label == entry) {
            currObj = cur_c;
        } else {
            currObj = label_lookup_[entry];
        }
        enterpoint_copy = currObj;
        entry_map[cur_c] = enterpoint_copy;
        // std::cout << std::to_string(temp) + " " + std::to_string(new_point) + " " + std::to_string(cur_c) << std::endl;
        if (new_point) { 
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;
            element_levels_[cur_c] = curlevel;
            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);
            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }
            // std::cout << " memcpopy success " << std::flush;
            if (label == entry) {
                maxlevel_ = curlevel;
            } else {
                int level = 0;
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayerClusterAppr(
                        currObj, data_point, cluster_distance, level);
                // std::cout << " search success " << std::flush;
                currObj = mutuallyConnectNewElementCluster(data_point, cluster_distance, cur_c, top_candidates, level, false);
            }
        } else {
            if (label != entry) {
                int level = 0;
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayerClusterAppr(
                        currObj, data_point, cluster_distance, level);
                currObj = mutuallyConnectUpdateElementCluster(data_point, cluster_distance, cur_c, top_candidates, level, true);
            }
        }
        return cur_c;
    }


    tableint addPointLogicHierarchical(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;
        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        // std::cout << "Before memcpy: " << *getExternalLabeLp(cur_c) << " " << getExternalLabel(cur_c) <<  " " << label << std::endl;
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        // std::cout << "After memcpy: " << *getExternalLabeLp(cur_c) << " " << getExternalLabel(cur_c) <<  " " << label << std::endl;
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        // *getExternalLabeLp(cur_c) = label;
        // std::cout << data_size_ << " " << sizeof(vectorset) << std::endl;
        // std::cout << *getExternalLabeLp(cur_c) << " " << (labeltype)getExternalLabel(cur_c) << " " << label << " " << sizeof(label) << " " << sizeof(labeltype) << std::endl;
        // // std::cout << ((vectorset*)data_point)->vecnum << " " << (float)(((vectorset*)data_point)->data)[0] << " " << (float)(((vectorset*)data_point)->data)[1] <<  std::endl;
        // std::cout << ((vectorset*)getDataByInternalId(cur_c))->vecnum << " " << (float)(((vectorset*)getDataByInternalId(cur_c))->data)[0]  << " " << (float)(((vectorset*)getDataByInternalId(cur_c))->data)[1] << std::endl;
        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                // std::cout << "add Point:" << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)getDataByInternalId(currObj))->vecnum << std::endl;
                dist_t curdist = fstdistfunc_((vectorset*)data_point, (vectorset*)getDataByInternalId(currObj), curlevel);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            // std::cout << "add Point:" << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)getDataByInternalId(cand))->vecnum << std::endl;
                            dist_t d = fstdistfunc_((vectorset*)data_point, (vectorset*)getDataByInternalId(cand), level);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");
                // std::cout << "search " << std::endl;
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    // std::cout << "add Point: emplace " << ((vectorset*)data_point)->vecnum << " " <<  ((vectorset*)getDataByInternalId(enterpoint_copy))->vecnum << std::endl;
                    top_candidates.emplace(fstdistfunc_((vectorset*)data_point, (vectorset*)getDataByInternalId(enterpoint_copy), 0), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;
        // std::cout<< cur_element_count << std::endl;
        tableint currObj = enterpoint_node_;
        // if (getExternalLabel(enterpoint_node_) == 10) {
        //     std::cout<< "=============================" << std::endl;
        // }
        // dist_t curdist = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(enterpoint_node_), 0);
        // dist_t curdist = fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(enterpoint_node_), 0);
        // // if (getExternalLabel(enterpoint_node_) == 10) {
        // //     std::cout<< "=============================" << std::endl;
        // // }
        // for (int level = maxlevel_; level > 0; level--) {
        //     bool changed = true;
        //     while (changed) {
        //         changed = false;
        //         unsigned int *data;

        //         data = (unsigned int *) get_linklist(currObj, level);
        //         int size = getListCount(data);
        //         metric_hops++;
        //         metric_distance_computations+=size;

        //         tableint *datal = (tableint *) (data + 1);
        //         for (int i = 0; i < size; i++) {
        //             tableint cand = datal[i];
        //             if (cand < 0 || cand > max_elements_)
        //                 throw std::runtime_error("cand error");
        //             // if (getExternalLabel(cand) == 10) {
        //             //     std::cout<< "=============================" << std::endl;
        //             // }
        //             // dist_t d = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand), level);
        //             dist_t d = fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand), level);
        //             // if (getExternalLabel(cand) == 10) {
        //             //     std::cout<< "=============================" << std::endl;
        //             // }

        //             if (d < curdist) {
        //                 curdist = d;
        //                 currObj = cand;
        //                 changed = true;
        //             }
        //         }
        //     }
        // }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerFullChamferST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerFullChamferST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    inline std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    getMinKUnique(
        std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>>& heaps, 
        int k
    ) {
        // 1. 将所有元素弹出到一个vector中
        std::vector<std::pair<dist_t, labeltype>> allElements;
        for (auto &h : heaps) {
            while (!h.empty()) {
                // if (h.size() == 1) {
                //     std::cout << "hello: " << getExternalLabel(h.top().second) << " " << h.top().first << std::endl;
                // }
                allElements.push_back(h.top());
                h.pop();
            }
        }

        // 2. 按float值从小到大排序
        std::sort(allElements.begin(), allElements.end(), [](const std::pair<dist_t, labeltype> &a, const std::pair<dist_t, labeltype> &b) {
            return a.first < b.first;
        });

        // 3. 去重并获取前k个最小元素
        std::unordered_set<labeltype> seen;
        std::vector<std::pair<dist_t, labeltype>> selected;
        for (auto &elem : allElements) {
            if (seen.find(elem.second) == seen.end()) {
                seen.insert(elem.second);
                selected.push_back(elem);
                if ((int)selected.size() == k) {
                    break;
                }
            }
        }

        // 如果最终unique的元素不足k个，selected里就是所有unique元素

        // 4. 将结果压入一个最大堆并返回
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> resultHeap;
        for (auto &elem : selected) {
            resultHeap.push(elem);
        }

        return resultHeap;
    }

    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnPara2(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr)  {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;
        // std::cout<< cur_element_count << std::endl;
        tableint currObj = enterpoint_node_;
        // if (getExternalLabel(enterpoint_node_) == 10) {
        //     std::cout<< "=============================" << std::endl;
        // }
        // dist_t curdist = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(enterpoint_node_), 0);
        dist_t curdist = fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(enterpoint_node_), 0);
        // if (getExternalLabel(enterpoint_node_) == 10) {
        //     std::cout<< "=============================" << std::endl;
        // }
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    // if (getExternalLabel(cand) == 10) {
                    //     std::cout<< "=============================" << std::endl;
                    // }
                    // dist_t d = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand), level);
                    dist_t d = fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand), level);
                    // if (getExternalLabel(cand) == 10) {
                    //     std::cout<< "=============================" << std::endl;
                    // }

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> top_candidate_local(multi_entry_thread_num);
        //bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        // if (bare_bone_search) {
        //     top_candidates = searchBaseLayerST<true>(
        //             currObj, query_data, std::max(ef_, k), isIdAllowed);
        // } else {
        //     top_candidates = searchBaseLayerST<false>(
        //             currObj, query_data, std::max(ef_, k), isIdAllowed);
        // }
        std::vector<tableint> obj_list(multi_entry_thread_num);
        
        obj_list[0] = currObj;

        for(int i = 1; i < multi_entry_thread_num; i++){
            obj_list[i] = rand() % cur_element_count;
        }
        // for(int i = 0; i < thread_num; i++)
        // #pragma omp parallel num_threads(thread_num) schedule(dynamic)
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < multi_entry_thread_num; i++)
        {
            // int i = omp_get_thread_num();
            top_candidate_local[i] = searchBaseLayerST<true>(obj_list[i], query_data, std::max(ef_, k), isIdAllowed);
            while (top_candidate_local[i].size() > k) 
                top_candidate_local[i].pop();
        } 

        top_candidates = getMinKUnique(top_candidate_local, k);
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnFineEdge(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr)  {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;
        // std::cout<< cur_element_count << std::endl;
        tableint currObj = enterpoint_node_;
        // std::cout << "? ? ? ?" << std::endl;
        // std::cout << currObj << std::endl;
        uint8_t* mapAB = (uint8_t*)malloc(fineEdgeSize);
        dist_t curdist = fstdistfuncInit_((vectorset*)query_data, (vectorset*)getDataByInternalId(currObj), mapAB, 0);
        bool changed = true;
        while (changed) {
            changed = false;
            // std::cout << currObj << std::endl;
            char *nodeObj = (getDataByInternalId(currObj));
            int *data = (int *) get_linklist0(currObj);
            uint8_t *distancelistl = (uint8_t *) ((tableint *)data + 1 + maxM0_);
            size_t size = getListCount((linklistsizeint*)data);
            metric_hops++;
            metric_distance_computations+=size;
#ifdef USE_SSE
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif
            uint8_t* new_mapAC = (uint8_t*)malloc(fineEdgeSize);
            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                uint8_t* mapBC = distancelistl + fineEdgeSize * (j - 1);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                char *currObj1 = (getDataByInternalId(candidate_id));
                uint8_t* cand_mapAC = (uint8_t*)malloc(fineEdgeSize);
                dist_t d = fstdistfuncMap_((vectorset*)query_data, (vectorset*)nodeObj, (vectorset*)currObj1, mapAB, mapBC, cand_mapAC, 0);
                dist_t estimated_d = d * 0.9;
                // std::cout << d << " " << curdist << std::endl;
                // std::cout << j << ' ' << candidate_id << ' ' << curdist << ' ' << estimated_d<< std::endl;
                if (curdist - estimated_d > 0.0001) {
                    // std::cout << "change: " <<  j << ' ' << candidate_id << ' ' << curdist << ' ' << estimated_d << std::endl;
                    curdist = estimated_d;
                    currObj = candidate_id;
                    //free(new_mapAC);
                    new_mapAC = cand_mapAC;
                    changed = true;
                }
            }
            if (changed) {
                //free(mapAB);
                mapAB = new_mapAC;
            }
        }

        // std::cout << currObj << std::endl;
        std::priority_queue<std::pair<std::pair<dist_t, dist_t>, tableint>, std::vector<std::pair<std::pair<dist_t, dist_t>, tableint>>, CompareByFirstFirst> top_candidates;
        // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> refine_candidates;
        // std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> top_candidate_local(multi_entry_thread_num);
        top_candidates = searchBaseLayerSTCF<true>(currObj, query_data, std::max(ef_, k), isIdAllowed);
        // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> refine_candidates;
        // while (!top_candidates.empty()) {
        //     std::pair<dist_t, tableint> rez = top_candidates.top();
        //     refine_candidates.push(std::pair<dist_t, labeltype>(fstdistfuncCF((vectorset*)query_data, (vectorset*)getDataByInternalId(rez.second), 0), rez.second));
        //     top_candidates.pop();
        // }
        while (!top_candidates.empty()) {
            std::pair<std::pair<dist_t, dist_t>, tableint> rez = top_candidates.top();
            refine_candidates.push(std::pair<dist_t, labeltype>(rez.first.second, rez.second));
            top_candidates.pop();
        }
        while (refine_candidates.size() > k) {
            refine_candidates.pop();
        }
        while (refine_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = refine_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            refine_candidates.pop();
        }
        return result;
    }



    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnCluster(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr)  {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;
        tableint currObj = enterpoint_node_;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerCluster<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerCluster<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnClusterEntries(const void *query_data, size_t k, const std::vector<labeltype>& entry_points, BaseFilterFunctor* isIdAllowed = nullptr)  {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerClusterEntriesMulti<true>(
                entry_points, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerClusterEntriesMulti<false>(
                entry_points, query_data, std::max(ef_, k), isIdAllowed);
        }
        // if (bare_bone_search) {
        //     top_candidates = searchBaseLayerClusterEntries<true>(
        //         entry_points, query_data, std::max(ef_, k), isIdAllowed);
        // } else {
        //     top_candidates = searchBaseLayerClusterEntries<false>(
        //         entry_points, query_data, std::max(ef_, k), isIdAllowed);
        // }
        // while (top_candidates.size() > k) {
        //     top_candidates.pop();
        // }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnParaFromEntries(const void *query_data, size_t k, std::vector<labeltype>& entry_points, BaseFilterFunctor* isIdAllowed = nullptr)  {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;
        // std::cout<< cur_element_count << std::endl;
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(enterpoint_node_), 0);
        // if (getExternalLabel(enterpoint_node_) == 10) {
        //     std::cout<< "=============================" << std::endl;
        // }
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    // if (getExternalLabel(cand) == 10) {
                    //     std::cout<< "=============================" << std::endl;
                    // }
                    // dist_t d = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand), level);
                    dist_t d = fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand), level);
                    // if (getExternalLabel(cand) == 10) {
                    //     std::cout<< "=============================" << std::endl;
                    // }

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        // std::cout << "? ? ? ?" << std::endl;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> top_candidate_local(multi_entry_thread_num);
        //bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        // if (bare_bone_search) {
        //     top_candidates = searchBaseLayerST<true>(
        //             currObj, query_data, std::max(ef_, k), isIdAllowed);
        // } else {
        //     top_candidates = searchBaseLayerST<false>(
        //             currObj, query_data, std::max(ef_, k), isIdAllowed);
        // }

        // std::cout << "? ? ? ?" << std::endl;
        std::vector<tableint> obj_list(multi_entry_thread_num);
        // std::cout << "? ? ? ?" << std::endl;
        obj_list[0] = currObj;
        // std::cout << entry_points.size() << std::endl;
        
        for(int i = 1; i < multi_entry_thread_num; i++){
            // obj_list[i] = rand() % cur_element_count;
            if (i <= entry_points.size()) {
                std::unique_lock <std::mutex> lock_table(label_lookup_lock);
                // std::cout << i << std::endl;
                // std::cout << " " << entry_points[i - 1] << std::endl;
                auto search = label_lookup_.find(entry_points[i - 1]);
                obj_list[i] = search->second;
            } else {
                obj_list[i] = rand() % cur_element_count;
            }
        }
        // std::cout << std::endl;
        // #pragma omp parallel num_threads(thread_num)
        // #pragma omp parallel for schedule(dynamic)
        #pragma omp parallel num_threads(multi_entry_thread_num)
        // for(int i = 0; i < thread_num; i++)
        {
            int i = omp_get_thread_num();
            // std::cout << i << std::endl;
            top_candidate_local[i] = searchBaseLayerST<true>(obj_list[i], query_data, std::max(ef_, k), isIdAllowed);
            // top_candidate_local[i] = searchBaseLayerST<true>(obj_list[i], query_data, std::max(ef_, k), isIdAllowed);
            while (top_candidate_local[i].size() > k) 
                top_candidate_local[i].pop();
        } 
        // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> refine_candidates;
        // while (!top_candidate_local[0].empty()) {
        //     std::pair<dist_t, tableint> rez = top_candidate_local[0].top();
        //     top_candidates.push(std::pair<dist_t, labeltype>(fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(rez.second), 0), rez.second));
        //     top_candidate_local[0].pop();
        // }
        top_candidates = getMinKUnique(top_candidate_local, k);
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnPara(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr)  {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;
        // std::cout<< cur_element_count << std::endl;
        tableint currObj = enterpoint_node_;
        // if (getExternalLabel(enterpoint_node_) == 10) {
        //     std::cout<< "=============================" << std::endl;
        // }
        // dist_t curdist = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(enterpoint_node_));
        // dist_t curdist = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(enterpoint_node_), 0);
        dist_t curdist = fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(enterpoint_node_), 0);
        // if (getExternalLabel(enterpoint_node_) == 10) {
        //     std::cout<< "=============================" << std::endl;
        // }
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    // if (getExternalLabel(cand) == 10) {
                    //     std::cout<< "=============================" << std::endl;
                    // }
                    // dist_t d = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand), level);
                    dist_t d = fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand), level);
                    // if (getExternalLabel(cand) == 10) {
                    //     std::cout<< "=============================" << std::endl;
                    // }

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerSTPara<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerSTPara<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }



    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnFromId(const void *query_data, size_t k, labeltype label, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;
        tableint currObj = enterpoint_node_;
        // std::cout<< currObj << " " << element_levels_[currObj] << std::endl;
        {
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            // std::cout<< label << std::endl;
            currObj = search->second;
            // std::cout<< currObj << " " << element_levels_[currObj] << std::endl;
        }
        // if (getExternalLabel(enterpoint_node_) == 10) {
        //     std::cout<< "=============================" << std::endl;
        // }
        // dist_t curdist = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(currObj));
        // dist_t curdist = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(enterpoint_node_), 0);
        dist_t curdist = fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(currObj), 0);
        // std::cout<< curdist << std::endl;
        // if (getExternalLabel(enterpoint_node_) == 10) {
        //     std::cout<< "=============================" << std::endl;
        // }
        for (int level = element_levels_[currObj]; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;
                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    // if (getExternalLabel(cand) == 10) {
                    //     std::cout<< "=============================" << std::endl;
                    // }
                    // dist_t d = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand));
                    // dist_t d = fstdistfunc_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand), level);
                    dist_t d = fstdistfunc4search_((vectorset*)query_data, (vectorset*)getDataByInternalId(cand), level);
                    // if (getExternalLabel(cand) == 10) {
                    //     std::cout<< "=============================" << std::endl;
                    // }

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        // dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_));
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), 0);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    // dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand));
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), level);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        int diff_size = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        int out_bound_zero_size = 0;
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                // std::cout << i << " " << size << " ";
                if (size == 0) {
                    out_bound_zero_size += 1;
                }
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    // std::cout << data[j] << " ";
                    connections_checked++;
                }
                // std::cout << std::endl;
                if (s.size() != size) {
                    // std::cout << i << " ";
                    // for (int j = 0; j < size; j++) {
                    //     std::cout << data[j] << " ";
                    // }
                    // std::cout << std::endl;
                    // std::cout << s.size() << ' ' << size << std::endl;
                    diff_size += 1;
                }
                //assert(s.size() == size);
            }
        }
        int all_inbound_0_count  = 0;
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                // assert(inbound_connections_num[i] > 0);
                // if (getExternalLabel(i) == 7258856) {
                //     std::cout << " ?? " <<  inbound_connections_num[i] << " ?? " << std::endl;
                // }
                if (inbound_connections_num[i] == 0) { 
                    // std::cout << getExternalLabel(i) << " " ;
                    all_inbound_0_count += 1;
                }
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            std::cout << cur_element_count << " " << all_inbound_0_count << " " << diff_size << " " << out_bound_zero_size << std::endl;
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }


    void addIntegrity() {
        int connections_checked = 0;
        int diff_size = 0;
        std::vector <int> inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                // std::cout << std::endl;
                //assert(s.size() == size);
            }
        }
        int add_count = 0;
        for (int i=0; i < cur_element_count; i++) {
            if (inbound_connections_num[i] == 0) { 
                // std::cout << getExternalLabel(i) << " " ;
                linklistsizeint *ll_cur = get_linklist_at_level(i, 0);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::vector<tableint> s;
                bool addsuccess = false;
                for (int j = 0; j < size; j++) {
                    tableint nei_c = data[j];
                    linklistsizeint *ll_nei = get_linklist_at_level(nei_c, 0);
                    int size_nei = getListCount(ll_nei);   
                    if (size_nei < maxM0_) {
                        tableint *data_nei = (tableint *) (ll_nei + 1);
                        data_nei[size_nei] = i;
                        setListCount(ll_nei, size_nei + 1);
                        addsuccess = true;
                        add_count += 1;
                    }
                }
                if (addsuccess) {
                    continue;
                }
            }
        }
        for (int i=cur_element_count - 1; i >= 0; i--) {
            linklistsizeint *ll_cur = get_linklist_at_level(i, 0);
            int size = getListCount(ll_cur);
            tableint *data = (tableint *) (ll_cur + 1);
            std::vector<tableint> s;
            for (int j = 0; j < size; j++) {
                tableint nei_c = data[j];
                linklistsizeint *ll_nei = get_linklist_at_level(nei_c, 0);
                int size_nei = getListCount(ll_nei);   
                if (size_nei < maxM0_) {
                    tableint *data_nei = (tableint *) (ll_nei + 1);
                    bool flag = true;
                    for (int k = 0; k < size_nei; k++) {
                        if (data_nei[k] == i) {
                            flag = false;
                            break;
                        }
                    }
                    if (flag) {
                        data_nei[size_nei] = i;
                        setListCount(ll_nei, size_nei + 1);
                        add_count += 1;
                    }
                }
            }
        }
        std::cout << add_count << std::endl;
    }

    std::vector<std::pair<tableint, int>> searchNodes(labeltype entry, int cid) {
        tableint epid = label_lookup_[entry];
        std::vector<std::pair<tableint, int>> queue;
        int l = 0;
        int r = 1;
        int count = 0;
        queue.push_back(std::make_pair(epid, 0));
        entry_map[epid] = cid + 1;
        while (l < r) {
            tableint topnode = queue[l].first;
            int curhop = queue[l].second;
            l++;
            linklistsizeint *ll_cur = get_linklist_at_level(topnode, 0);
            int size = getListCount(ll_cur);
            tableint *data = (tableint *) (ll_cur + 1);
            // std::cout << getExternalLabel(topnode) << " " << size << std::endl;
            for (int j = 0; j < size; j++) {
                tableint nei_c = data[j];
                // std::cout << getExternalLabel(nei_c) << " ";
                if (entry_map.find(nei_c) != entry_map.end() && entry_map[nei_c] != cid + 1) {
                    entry_map[nei_c] = cid + 1;
                    queue.push_back(std::make_pair(nei_c, curhop + 1));
                    r++;
                }
            }
        }
        return queue;
    }


    std::vector<std::pair<tableint, int>> searchNodesForFix(labeltype entry, int cid) {
        tableint epid = label_lookup_[entry];
        std::vector<std::pair<tableint, int>> queue;
        int l = 0;
        int r = 1;
        int count = 0;
        queue.push_back(std::make_pair(epid, 0));
        entry_map[epid] = cid + 1;
        while (l < r) {
            tableint topnode = queue[l].first;
            int curhop = queue[l].second;
            linklistsizeint *ll_cur = get_linklist_at_level(topnode, 0);
            int size = getListCount(ll_cur);
            queue[l].second = size;
            tableint *data = (tableint *) (ll_cur + 1);
            // std::cout << getExternalLabel(topnode) << " " << size << std::endl;
            for (int j = 0; j < size; j++) {
                tableint nei_c = data[j];
                // std::cout << getExternalLabel(nei_c) << " ";
                if (entry_map.find(nei_c) != entry_map.end() && entry_map[nei_c] != cid + 1) {
                    entry_map[nei_c] = cid + 1;
                    queue.push_back(std::make_pair(nei_c, curhop + 1));
                    r++;
                }
            }
            l++;
        }
        return queue;
    }


    std::vector<std::pair<tableint, int>> searchNodesWithHop(labeltype entry, int entry_layer, int cid) {
        tableint epid = label_lookup_[entry];
        std::vector<std::pair<tableint, int>> queue;
        int l = 0;
        int r = 1;
        int count = 0;
        queue.push_back(std::make_pair(epid, entry_layer));
        entry_map[epid] = cid + 1;
        while (l < r) {
            tableint topnode = queue[l].first;
            int curhop = queue[l].second;
            linklistsizeint *ll_cur = get_linklist_at_level(topnode, 0);
            int size = getListCount(ll_cur);
            tableint *data = (tableint *) (ll_cur + 1);
            // std::cout << getExternalLabel(topnode) << " " << size << std::endl;
            for (int j = 0; j < size; j++) {
                tableint nei_c = data[j];
                // std::cout << getExternalLabel(nei_c) << " ";
                if (entry_map.find(nei_c) != entry_map.end() && entry_map[nei_c] != cid + 1) {
                    entry_map[nei_c] = cid + 1;
                    queue.push_back(std::make_pair(nei_c, curhop + 1));
                    r++;
                }
            }
            l++;
        }
        return queue;
    }
};
}  // namespace hnswlib
