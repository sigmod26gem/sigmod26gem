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
    searchKnnClusterEntries(const void *query_data, size_t k, const std::vector<labeltype>& entry_points,
                            BaseFilterFunctor* isIdAllowed = nullptr,
                            const std::vector<bool>* query_mask = nullptr,
                            size_t query_ef = 0) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerClusterEntriesMulti<true>(
                entry_points, query_data, std::max(query_ef ? query_ef : ef_, k), isIdAllowed, nullptr, query_mask);
        } else {
            top_candidates = searchBaseLayerClusterEntriesMulti<false>(
                entry_points, query_data, std::max(query_ef ? query_ef : ef_, k), isIdAllowed, nullptr, query_mask);
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
