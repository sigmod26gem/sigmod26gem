#pragma once

#include <mutex>
#include <string.h>
#include <deque>
#include <memory>

namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
 public:
    vl_type curV;
    vl_type *mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList *> pool;
    std::mutex poolguard;
    int numelements;

 public:
    struct ReturnToPool {
        VisitedListPool* pool;
        void operator()(VisitedList* list) const noexcept { pool->releaseVisitedList(list); }
    };
    using Lease = std::unique_ptr<VisitedList, ReturnToPool>;

    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        try {
            for (int i = 0; i < initmaxpools; i++) {
                auto list = std::make_unique<VisitedList>(numelements);
                pool.push_front(list.get());
                list.release();
            }
        } catch (...) {
            for (auto* list : pool) delete list;
            throw;
        }
    }

    Lease acquire() { return Lease(getFreeVisitedList(), ReturnToPool{this}); }

    VisitedList *getFreeVisitedList() {
        VisitedList *rez = nullptr;
        {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            }
        }
        if (!rez) rez = new VisitedList(numelements);
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList *vl) noexcept {
        try {
            std::unique_lock <std::mutex> lock(poolguard);
            pool.push_front(vl);
        } catch (...) {
            delete vl;
        }
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib
