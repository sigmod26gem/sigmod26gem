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
