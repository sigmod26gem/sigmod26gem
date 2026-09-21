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
