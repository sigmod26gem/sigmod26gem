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
