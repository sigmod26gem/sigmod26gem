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
