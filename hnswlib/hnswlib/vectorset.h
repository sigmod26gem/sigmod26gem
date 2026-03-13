#pragma once
#include <cstdlib>
struct vectorset
{
    float* data = nullptr;
    size_t dim, vecnum;
    int* codes = nullptr;
    vectorset(float* _data, int _dim, int _vecnum):data(_data), dim(_dim), vecnum(_vecnum){}
    vectorset(float* _data, int* _codes, int _dim, int _vecnum): data(_data), codes(_codes), dim(_dim), vecnum(_vecnum){}
    ~vectorset(){
    }
};
