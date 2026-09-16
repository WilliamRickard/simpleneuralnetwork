#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

extern "C" __m512d _ZGVeN8v_exp(__m512d);
extern "C" __m512d v30_external_vec_noop(__m512d);

alignas(64) static volatile double v30_direct_sink[8];

__attribute__((target("avx512f,avx512dq")))
static double runExpV30(std::size_t iterations){
    alignas(64) const double values[8][8]={
        {-8.0,-4.0,-2.0,-1.0,-.5,.25,.75,1.25},
        {-7.5,-3.5,-1.75,-.75,-.25,.5,1.0,1.5},
        {-7.0,-3.0,-1.5,-.5,0.0,.625,1.25,1.75},
        {-6.5,-2.5,-1.25,-.25,.125,.75,1.5,2.0},
        {-6.0,-2.0,-1.0,0.0,.25,1.0,1.75,2.25},
        {-5.5,-1.5,-.75,.25,.5,1.25,2.0,2.5},
        {-5.0,-1.0,-.5,.5,.75,1.5,2.25,2.75},
        {-4.5,-.5,-.25,.75,1.0,1.75,2.5,3.0}
    };
    __m512d x[8];
    for(int i=0;i<8;i++)x[i]=_mm512_load_pd(values[i]);
    __m512d acc=_mm512_setzero_pd();
    const auto start=std::chrono::steady_clock::now();
    for(std::size_t r=0;r<iterations;r++){
        acc=_mm512_xor_pd(acc,_ZGVeN8v_exp(x[0]));
        acc=_mm512_xor_pd(acc,_ZGVeN8v_exp(x[1]));
        acc=_mm512_xor_pd(acc,_ZGVeN8v_exp(x[2]));
        acc=_mm512_xor_pd(acc,_ZGVeN8v_exp(x[3]));
        acc=_mm512_xor_pd(acc,_ZGVeN8v_exp(x[4]));
        acc=_mm512_xor_pd(acc,_ZGVeN8v_exp(x[5]));
        acc=_mm512_xor_pd(acc,_ZGVeN8v_exp(x[6]));
        acc=_mm512_xor_pd(acc,_ZGVeN8v_exp(x[7]));
    }
    const auto finish=std::chrono::steady_clock::now();
    alignas(64) double sink[8];
    _mm512_store_pd(sink,acc);
    for(int i=0;i<8;i++)v30_direct_sink[i]=sink[i];
    return std::chrono::duration<double>(finish-start).count();
}

__attribute__((target("avx512f,avx512dq")))
static double runNoopV30(std::size_t iterations){
    alignas(64) const double values[8][8]={
        {-8.0,-4.0,-2.0,-1.0,-.5,.25,.75,1.25},
        {-7.5,-3.5,-1.75,-.75,-.25,.5,1.0,1.5},
        {-7.0,-3.0,-1.5,-.5,0.0,.625,1.25,1.75},
        {-6.5,-2.5,-1.25,-.25,.125,.75,1.5,2.0},
        {-6.0,-2.0,-1.0,0.0,.25,1.0,1.75,2.25},
        {-5.5,-1.5,-.75,.25,.5,1.25,2.0,2.5},
        {-5.0,-1.0,-.5,.5,.75,1.5,2.25,2.75},
        {-4.5,-.5,-.25,.75,1.0,1.75,2.5,3.0}
    };
    __m512d x[8];
    for(int i=0;i<8;i++)x[i]=_mm512_load_pd(values[i]);
    __m512d acc=_mm512_setzero_pd();
    const auto start=std::chrono::steady_clock::now();
    for(std::size_t r=0;r<iterations;r++){
        acc=_mm512_xor_pd(acc,v30_external_vec_noop(x[0]));
        acc=_mm512_xor_pd(acc,v30_external_vec_noop(x[1]));
        acc=_mm512_xor_pd(acc,v30_external_vec_noop(x[2]));
        acc=_mm512_xor_pd(acc,v30_external_vec_noop(x[3]));
        acc=_mm512_xor_pd(acc,v30_external_vec_noop(x[4]));
        acc=_mm512_xor_pd(acc,v30_external_vec_noop(x[5]));
        acc=_mm512_xor_pd(acc,v30_external_vec_noop(x[6]));
        acc=_mm512_xor_pd(acc,v30_external_vec_noop(x[7]));
    }
    const auto finish=std::chrono::steady_clock::now();
    alignas(64) double sink[8];
    _mm512_store_pd(sink,acc);
    for(int i=0;i<8;i++)v30_direct_sink[i]=sink[i];
    return std::chrono::duration<double>(finish-start).count();
}

static double median(std::vector<double> values){
    std::sort(values.begin(),values.end());
    return values[values.size()/2];
}

int main(){
    constexpr std::size_t iterations=250000;
    runExpV30(1000);
    runNoopV30(1000);
    std::vector<double> expTimes,noopTimes;
    for(int repeat=0;repeat<11;repeat++){
        if((repeat&1)==0){
            expTimes.push_back(runExpV30(iterations));
            noopTimes.push_back(runNoopV30(iterations));
        }else{
            noopTimes.push_back(runNoopV30(iterations));
            expTimes.push_back(runExpV30(iterations));
        }
    }
    const double expMedian=median(expTimes),noopMedian=median(noopTimes);
    const double calls=static_cast<double>(iterations)*8.0;
    std::cout<<std::setprecision(12)
             <<"DIRECT_CALL exp_ns_per_call="<<(1e9*expMedian/calls)
             <<" external_noop_ns_per_call="<<(1e9*noopMedian/calls)
             <<" common_fraction_of_exp="<<(noopMedian/expMedian)
             <<" exp_over_noop="<<(expMedian/noopMedian)<<'\n';
    return 0;
}
