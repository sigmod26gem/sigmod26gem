    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerClusterEntriesMulti(
        const std::vector<labeltype>& entry_points,
        // tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr,
        const std::vector<bool>* query_mask = nullptr) const {
        EntrySearchScratch<dist_t> scratch;
        searchBaseLayerClusterEntriesInto<bare_bone_search, collect_metrics>(
            entry_points, data_point, ef, scratch, isIdAllowed, stop_condition, query_mask);
        return std::move(scratch.top);
    }

    template <bool bare_bone_search = true, bool collect_metrics = false>
    void searchBaseLayerClusterEntriesInto(
        const std::vector<labeltype>& entry_points, const void* data_point, size_t ef,
        EntrySearchScratch<dist_t>& scratch, BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr,
        const std::vector<bool>* query_mask = nullptr) const {
        const auto score = [this, data_point](tableint id) {
            return fstdistfuncCluster(static_cast<const vectorset*>(data_point),
                                     reinterpret_cast<const vectorset*>(getDataByInternalId(id)), 0);
        };
        searchBaseLayerClusterEntriesScored<bare_bone_search, collect_metrics>(
            entry_points, ef, scratch, score, isIdAllowed, stop_condition, query_mask);
    }

    template <bool bare_bone_search = true, bool collect_metrics = false, class Score>
    void searchBaseLayerClusterEntriesScored(
        const std::vector<labeltype>& entry_points, size_t ef,
        EntrySearchScratch<dist_t>& scratch, const Score& score,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr,
        const std::vector<bool>* query_mask = nullptr) const {
        scratch.reset(max_elements_, entry_points.size());
        const auto* visited_words = scratch.visited.data();
        auto& top_candidates = scratch.top;
        auto& candidate_set_list = scratch.frontiers;
        auto& stop_flag_list = scratch.stopped;
        // std::vector<dist_t> lower_bound_list(entry_points.size());
        dist_t lowerBound = std::numeric_limits<dist_t>::lowest();
        for (int i = 0; i < entry_points.size(); i++) {
            labeltype ep = entry_points[i];
            tableint ep_id = label_lookup_.at(ep);
            if (scratch.visited.contains(ep_id)) {
                stop_flag_list[i] = true;
                continue;
            }
            dist_t dist = score(ep_id);
            lowerBound = std::max(lowerBound, dist);
            // lower_bound_list[i] = std::max(lower_bound_list[i], dist);
            top_candidates.emplace(dist, ep_id);
            candidate_set_list[i].emplace(-dist, ep_id);
            scratch.visited.test_and_set(ep_id);
            stop_flag_list[i] = false;
        }
        bool all_empty = false;
        while (!all_empty) {
            all_empty = true;
            for (int i = 0; i < entry_points.size(); i++) {
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
                    if (size) {
                        _mm_prefetch((const char*) (visited_words + (*(data + 1) / 64)), _MM_HINT_T0);
                        _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                    }
                    _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
        #endif

                    for (size_t j = 1; j <= size; j++) {
                        int candidate_id = *(data + j);
        //                    if (candidate_id == 0) continue;
        #ifdef USE_SSE
                        if (j < size) {
                            _mm_prefetch((const char*) (visited_words + (*(data + j + 1) / 64)), _MM_HINT_T0);
                            _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                        }
        #endif
                        if (!(query_mask ? (*query_mask)[candidate_id] : search_set[candidate_id])) {
                            continue;
                        }
                        if (!scratch.visited.test_and_set(candidate_id)) {

                            dist_t dist = score(candidate_id);
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
                                        stop_condition->add_point_to_result(getExternalLabel(candidate_id), getDataByInternalId(candidate_id), dist);
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
    }
