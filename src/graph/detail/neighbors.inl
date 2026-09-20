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
