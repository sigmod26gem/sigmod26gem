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
        if (!output) throw std::runtime_error("Cannot create graph file: " + location);
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
        if (!output) throw std::runtime_error("Cannot write graph file: " + location);
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) try {
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

        if (!cur_element_count || cur_element_count > max_elements_ || cur_element_count > INT32_MAX)
            throw std::runtime_error("Invalid graph element count");
        size_t max_elements = std::max(max_elements_i, cur_element_count.load());
        if (max_elements > INT32_MAX) throw std::runtime_error("Graph capacity exceeds int32");
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

        // GEM's cluster-only build leaves the global HNSW entry unset.
        const bool cluster_only = maxlevel_ == 0 && enterpoint_node_ == std::numeric_limits<tableint>::max();
        if (M_ < 2 || M_ > 10000 || maxM_ != M_ || maxM0_ != 2 * M_ ||
            !std::isfinite(mult_) || mult_ <= 0 || ef_construction_ < M_ ||
            (!cluster_only && (maxlevel_ < 0 || enterpoint_node_ >= cur_element_count)))
            throw std::runtime_error("Invalid graph header");
        size_links_per_element_ = size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        if (offsetLevel0_ != 0 || offsetData_ != size_links_level0_ ||
            label_offset_ != offsetData_ + sizeof(vectorset) ||
            size_data_per_element_ != label_offset_ + sizeof(labeltype) ||
            max_elements > std::numeric_limits<size_t>::max() / size_data_per_element_)
            throw std::runtime_error("Invalid graph record layout");

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
        const auto base_bytes = cur_element_count * size_data_per_element_;
        if (total_filesize < pos || base_bytes > static_cast<size_t>(total_filesize - pos) ||
            cur_element_count > (static_cast<size_t>(total_filesize - pos) - base_bytes) / sizeof(unsigned int))
            throw std::runtime_error("Truncated graph records");

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize % size_links_per_element_ != 0 ||
                linkListSize / size_links_per_element_ > static_cast<size_t>(std::max(0, maxlevel_)) ||
                input.tellg() > total_filesize || linkListSize > static_cast<size_t>(total_filesize - input.tellg()))
                throw std::runtime_error("Invalid graph upper layer size");
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
        if (!input.read(data_level0_memory_, base_bytes)) throw std::runtime_error("Truncated graph data");

        auto validate_links = [&](const char* record) {
            unsigned short degree;
            std::memcpy(&degree, record, sizeof(degree));
            if (degree > maxM0_) throw std::runtime_error("Invalid graph degree");
            for (size_t j = 0; j < degree; ++j) {
                tableint neighbor;
                std::memcpy(&neighbor, record + sizeof(linklistsizeint) + j * sizeof(tableint), sizeof(neighbor));
                if (neighbor >= cur_element_count) throw std::runtime_error("Graph neighbor outside element range");
            }
        };
        for (size_t i = 0; i < cur_element_count; ++i)
            validate_links(data_level0_memory_ + i * size_data_per_element_);

        // 修改这里
        size_links_per_element_ = maxM0_ * (sizeof(tableint) + fineEdgeSize * sizeof(uint8_t))  + sizeof(linklistsizeint);
        size_links_level0_ = maxM0_ * (sizeof(tableint) + fineEdgeSize * sizeof(uint8_t))  + sizeof(linklistsizeint);
        //修改这里
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) calloc(max_elements, sizeof(void *));
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        entry_map.resize(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 80;
        for (size_t i = 0; i < cur_element_count; i++) {
            if (!label_lookup_.emplace(getExternalLabel(i), i).second)
                throw std::runtime_error("Duplicate graph label");
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
                if (!input.read(linkLists_[i], linkListSize)) throw std::runtime_error("Truncated graph upper layer");
                for (size_t offset = 0; offset < linkListSize; offset += size_links_per_element_)
                    validate_links(linkLists_[i] + offset);
            }
        }

        if (!cluster_only && element_levels_[enterpoint_node_] != maxlevel_)
            throw std::runtime_error("Graph entry level mismatch");
        for (size_t i = 0; i < cur_element_count; ++i)
            for (int level = 1; level <= element_levels_[i]; ++level) {
                auto* links = reinterpret_cast<linklistsizeint*>(linkLists_[i] + (level - 1) * size_links_per_element_);
                for (size_t j = 0; j < getListCount(links); ++j)
                    if (element_levels_[reinterpret_cast<tableint*>(links + 1)[j]] < level)
                        throw std::runtime_error("Graph neighbor has no requested upper layer");
            }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    } catch (...) {
        clear();
        throw;
    }
