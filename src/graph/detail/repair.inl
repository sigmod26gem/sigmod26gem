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

    // std::vector<std::pair<tableint, int>> searchNodes(labeltype entry, int cid) {
    //     tableint epid = label_lookup_[entry];
    //     std::vector<std::pair<tableint, int>> queue;
    //     int l = 0;
    //     int r = 1;
    //     int count = 0;
    //     queue.push_back(std::make_pair(epid, 0));
    //     entry_map[epid] = cid + 1;
    //     while (l < r) {
    //         tableint topnode = queue[l].first;
    //         int curhop = queue[l].second;
    //         l++;
    //         linklistsizeint *ll_cur = get_linklist_at_level(topnode, 0);
    //         int size = getListCount(ll_cur);
    //         tableint *data = (tableint *) (ll_cur + 1);
    //         // std::cout << getExternalLabel(topnode) << " " << size << std::endl;
    //         for (int j = 0; j < size; j++) {
    //             tableint nei_c = data[j];
    //             // std::cout << getExternalLabel(nei_c) << " ";
    //             if (entry_map.find(nei_c) != entry_map.end() && entry_map[nei_c] != cid + 1) {
    //                 entry_map[nei_c] = cid + 1;
    //                 queue.push_back(std::make_pair(nei_c, curhop + 1));
    //                 r++;
    //             }
    //         }
    //     }
    //     return queue;
    // }


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


    // std::vector<std::pair<tableint, int>> searchNodesWithHop(labeltype entry, int entry_layer, int cid) {
    //     tableint epid = label_lookup_[entry];
    //     std::vector<std::pair<tableint, int>> queue;
    //     int l = 0;
    //     int r = 1;
    //     int count = 0;
    //     queue.push_back(std::make_pair(epid, entry_layer));
    //     entry_map[epid] = cid + 1;
    //     while (l < r) {
    //         tableint topnode = queue[l].first;
    //         int curhop = queue[l].second;
    //         linklistsizeint *ll_cur = get_linklist_at_level(topnode, 0);
    //         int size = getListCount(ll_cur);
    //         tableint *data = (tableint *) (ll_cur + 1);
    //         // std::cout << getExternalLabel(topnode) << " " << size << std::endl;
    //         for (int j = 0; j < size; j++) {
    //             tableint nei_c = data[j];
    //             // std::cout << getExternalLabel(nei_c) << " ";
    //             if (entry_map.find(nei_c) != entry_map.end() && entry_map[nei_c] != cid + 1) {
    //                 entry_map[nei_c] = cid + 1;
    //                 queue.push_back(std::make_pair(nei_c, curhop + 1));
    //                 r++;
    //             }
    //         }
    //         l++;
    //     }
    //     return queue;
    // }
