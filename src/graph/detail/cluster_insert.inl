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
    // bool addPointMem(const void *data_point, labeltype label, labeltype entry) {
    //     tableint cur_c = 0;
    //     bool new_point = true;
    //     {
    //         // Checking if the element with the same label already exists
    //         // if so, updating it *instead* of creating a new element.
    //         std::unique_lock <std::mutex> lock_table(label_lookup_lock);
    //         auto search = label_lookup_.find(label);
    //         if (search != label_lookup_.end()) {
    //             tableint existingInternalId = search->second;
    //             cur_c = existingInternalId;
    //             lock_table.unlock();
    //             new_point = false;
    //             // std::cout << label << " " << cur_c << " " << entry << " ? " << std::flush;
    //         }
    //         else {
    //             if (cur_element_count >= max_elements_) {
    //                 throw std::runtime_error("The number of elements exceeds the specified limit");
    //             }
    //             cur_c = cur_element_count;
    //             cur_element_count++;
    //             label_lookup_[label] = cur_c;
    //             // std::cout << label << " " << cur_c << " " << entry << " new " << std::flush;
    //         }
    //     }
    //     tableint enterpoint_copy = 0;
    //     if (label == entry) {
    //         enterpoint_copy = cur_c;
    //     } else {
    //         enterpoint_copy = label_lookup_[entry];
    //     }
    //     entry_map[cur_c] = enterpoint_copy;
    //     if (new_point) {
    //         int curlevel = getRandomLevel(mult_);
    //         element_levels_[cur_c] = curlevel;
    //         memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
    //         memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    //         memcpy(getDataByInternalId(cur_c), data_point, data_size_);
    //     }
    //     return new_point;
    // }

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
