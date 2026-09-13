#include<iostream>
#include<algorithm>
#include<array>
#include<chrono>
#include<cmath>
#include<cstddef>
#include<ctime>
#include<exception>
#include<fstream>
#include<iomanip>
#if defined(__x86_64__) && defined(__GNUC__)
#include<immintrin.h>
#endif
#include<random>
#include<stdexcept>
#include<string>
#include<vector>
#ifdef _OPENMP
#include<omp.h>
#endif

namespace v11_v10 {
#include "../v10/main.cpp"
}
using namespace v11_v10;

/*
 * Simple Neural Network - v11 relaxed floating-point
 * ==================================================
 *
 * V11 keeps the v10 model, full-batch momentum training equations and double
 * precision weights. It relaxes bit-for-bit equivalence only in sigmoid
 * evaluation: complete AVX-512 tiles use a bounded piecewise polynomial
 * approximation. Small/unsupported/untested paths delegate to v10.
 *
 * The common |x| <= 1 path is a degree-4 polynomial. Degree-8 fallbacks cover
 * 1 < |x| <= 12, and values outside that range saturate to 0/1. A dense-grid
 * validation over [-20,20] gives maximum absolute sigmoid error < 6.57e-6.
 */

#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d sigmoidCentralV11(__m512d x) {
    const __m512d c4=_mm512_set1_pd(0.0041853293217438935);
    const __m512d c3=_mm512_set1_pd(-0.024120015689114357);
    const __m512d c2=_mm512_set1_pd(0.0011384450152525012);
    const __m512d c1=_mm512_set1_pd(0.24984606999988493);
    const __m512d c0=_mm512_set1_pd(0.5000049283417471);
    __m512d y=_mm512_fmadd_pd(c4,x,c3);
    y=_mm512_fmadd_pd(y,x,c2);
    y=_mm512_fmadd_pd(y,x,c1);
    return _mm512_fmadd_pd(y,x,c0);
}

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d sigmoidHorner8V11(__m512d x, const double *coefficients) {
    __m512d y=_mm512_set1_pd(coefficients[8]);
    for(int i=7;i>=0;i--) y=_mm512_fmadd_pd(y,x,_mm512_set1_pd(coefficients[i]));
    return y;
}

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d sigmoidApproxAVX512V11(__m512d x) {
    static const double middle[9]={
        0.5000011358677788,0.24999835406259627,-0.00016539918722939355,
        -0.019899021588728794,-0.002045765106270335,0.0043622465752078555,
        -0.001373869927816452,0.0001903321352260143,-1.0315836568979578e-05
    };
    static const double outer[9]={
        0.35947614514228465,0.5002804028401375,-0.17657318540721206,
        0.03651949030026908,-0.004810100518711358,0.0004109220099530741,
        -2.213677938171726e-05,6.851190548062953e-07,-9.301037994539489e-09
    };
    const __m512d signMask=_mm512_set1_pd(-0.0),one=_mm512_set1_pd(1.0);
    const __m512d four=_mm512_set1_pd(4.0),twelve=_mm512_set1_pd(12.0);
    const __m512d absolute=_mm512_andnot_pd(signMask,x);
    const __mmask8 centralMask=_mm512_cmp_pd_mask(absolute,one,_CMP_LE_OQ);
    __m512d positive;
    if(centralMask==0xFF) {
        positive=sigmoidCentralV11(absolute);
    }
    else {
        const __mmask8 middleMask=_mm512_cmp_pd_mask(absolute,four,_CMP_LE_OQ);
        if(middleMask==0xFF) {
            positive=sigmoidHorner8V11(absolute,middle);
        }
        else {
            const __m512d middleValue=sigmoidHorner8V11(absolute,middle);
            const __m512d outerValue=sigmoidHorner8V11(absolute,outer);
            positive=_mm512_mask_blend_pd(middleMask,outerValue,middleValue);
            const __mmask8 finiteApproxMask=_mm512_cmp_pd_mask(absolute,twelve,_CMP_LE_OQ);
            positive=_mm512_mask_mov_pd(one,finiteApproxMask,positive);
        }
    }
    const __mmask8 negativeMask=_mm512_cmp_pd_mask(x,_mm512_setzero_pd(),_CMP_LT_OQ);
    return _mm512_mask_sub_pd(positive,negativeMask,one,positive);
}

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void sigmoidVectorV11(double *values, size_t count) {
    for(size_t i=0;i<count;i+=8) _mm512_storeu_pd(values+i,sigmoidApproxAVX512V11(_mm512_loadu_pd(values+i)));
}

#define V11_DECLARE_W1_ACCUMULATORS \
    __m512d g00=_mm512_setzero_pd(),g01=_mm512_setzero_pd(),g10=_mm512_setzero_pd(),g11=_mm512_setzero_pd(),g20w=_mm512_setzero_pd(),g21w=_mm512_setzero_pd(),g30=_mm512_setzero_pd(),g31=_mm512_setzero_pd(),g40=_mm512_setzero_pd(),g41=_mm512_setzero_pd(),g50=_mm512_setzero_pd(),g51=_mm512_setzero_pd(),g60=_mm512_setzero_pd(),g61=_mm512_setzero_pd(),g70=_mm512_setzero_pd(),g71=_mm512_setzero_pd(),g80=_mm512_setzero_pd(),g81=_mm512_setzero_pd(),g90=_mm512_setzero_pd(),g91=_mm512_setzero_pd(),g100=_mm512_setzero_pd(),g101=_mm512_setzero_pd()
#define V11_ACCUMULATE_W1(K,G0,G1) do { const __m512d xv=_mm512_set1_pd(x[K]); G0=_mm512_fmadd_pd(xv,delta0,G0); G1=_mm512_fmadd_pd(xv,delta1,G1); } while(false)
#define V11_STORE_W1(K,G0,G1) do { double*g=accumulator.dJdWone.data()+(K)*HIDDEN_NODES; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),G0)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),G1)); } while(false)

template<bool CalculatePercentage>
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void processTrainingFullTile4V11(const Dataset &data, size_t startRow, const Network &network, ThreadAccumulator &accumulator, bool detailed) {
    alignas(64) array<double,BLOCK_SIZE*HIDDEN_NODES> hidden;
    alignas(64) array<double,BLOCK_SIZE> output,deltaThree,percentage;
    const __m512d one=_mm512_set1_pd(1.0);
    for(size_t row=0;row<BLOCK_SIZE;row+=4) {
        const double*x0=data.x.rowData(startRow+row),*x1=data.x.rowData(startRow+row+1),*x2=data.x.rowData(startRow+row+2),*x3=data.x.rowData(startRow+row+3);
        __m512d a0=_mm512_setzero_pd(),a1=_mm512_setzero_pd(),b0=_mm512_setzero_pd(),b1=_mm512_setzero_pd();
        __m512d c0=_mm512_setzero_pd(),c1=_mm512_setzero_pd(),d0=_mm512_setzero_pd(),d1=_mm512_setzero_pd();
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const double*w=network.wOne.data()+k*HIDDEN_NODES;
            const __m512d w0=_mm512_loadu_pd(w),w1=_mm512_loadu_pd(w+8);
            const __m512d v0=_mm512_set1_pd(x0[k]),v1=_mm512_set1_pd(x1[k]),v2=_mm512_set1_pd(x2[k]),v3=_mm512_set1_pd(x3[k]);
            a0=_mm512_fmadd_pd(v0,w0,a0);a1=_mm512_fmadd_pd(v0,w1,a1);b0=_mm512_fmadd_pd(v1,w0,b0);b1=_mm512_fmadd_pd(v1,w1,b1);
            c0=_mm512_fmadd_pd(v2,w0,c0);c1=_mm512_fmadd_pd(v2,w1,c1);d0=_mm512_fmadd_pd(v3,w0,d0);d1=_mm512_fmadd_pd(v3,w1,d1);
        }
        double*h0=hidden.data()+row*HIDDEN_NODES,*h1=hidden.data()+(row+1)*HIDDEN_NODES,*h2=hidden.data()+(row+2)*HIDDEN_NODES,*h3=hidden.data()+(row+3)*HIDDEN_NODES;
        _mm512_store_pd(h0,a0);_mm512_store_pd(h0+8,a1);_mm512_store_pd(h1,b0);_mm512_store_pd(h1+8,b1);
        _mm512_store_pd(h2,c0);_mm512_store_pd(h2+8,c1);_mm512_store_pd(h3,d0);_mm512_store_pd(h3+8,d1);
    }
    sigmoidVectorV11(hidden.data(),BLOCK_SIZE*HIDDEN_NODES);
    const __m512d w20=_mm512_loadu_pd(network.wTwo.data()),w21=_mm512_loadu_pd(network.wTwo.data()+8);
    for(size_t i=0;i<BLOCK_SIZE;i++) {
        const double*h=hidden.data()+i*HIDDEN_NODES;
        const __m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));
        output[i]=_mm512_reduce_add_pd(sum);
    }
    sigmoidVectorV11(output.data(),BLOCK_SIZE);
    computeOutputDeltasV9<CalculatePercentage>(data,startRow,output,deltaThree,percentage,accumulator,detailed);
    __m512d g20=_mm512_loadu_pd(accumulator.dJdWtwo.data()),g21=_mm512_loadu_pd(accumulator.dJdWtwo.data()+8);V11_DECLARE_W1_ACCUMULATORS;
    for(size_t q=0;q<BLOCK_SIZE;q++) {
        const double*h=hidden.data()+q*HIDDEN_NODES;const __m512d d=_mm512_set1_pd(deltaThree[q]),a0=_mm512_load_pd(h),a1=_mm512_load_pd(h+8);
        g20=_mm512_fmadd_pd(a0,d,g20);g21=_mm512_fmadd_pd(a1,d,g21);
        const __m512d delta0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0))),delta1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));
        const double*x=data.x.rowData(startRow+q);
        V11_ACCUMULATE_W1(0,g00,g01);V11_ACCUMULATE_W1(1,g10,g11);V11_ACCUMULATE_W1(2,g20w,g21w);V11_ACCUMULATE_W1(3,g30,g31);V11_ACCUMULATE_W1(4,g40,g41);V11_ACCUMULATE_W1(5,g50,g51);
        V11_ACCUMULATE_W1(6,g60,g61);V11_ACCUMULATE_W1(7,g70,g71);V11_ACCUMULATE_W1(8,g80,g81);V11_ACCUMULATE_W1(9,g90,g91);V11_ACCUMULATE_W1(10,g100,g101);
    }
    _mm512_storeu_pd(accumulator.dJdWtwo.data(),g20);_mm512_storeu_pd(accumulator.dJdWtwo.data()+8,g21);
    V11_STORE_W1(0,g00,g01);V11_STORE_W1(1,g10,g11);V11_STORE_W1(2,g20w,g21w);V11_STORE_W1(3,g30,g31);V11_STORE_W1(4,g40,g41);V11_STORE_W1(5,g50,g51);
    V11_STORE_W1(6,g60,g61);V11_STORE_W1(7,g70,g71);V11_STORE_W1(8,g80,g81);V11_STORE_W1(9,g90,g91);V11_STORE_W1(10,g100,g101);
}

template<bool CalculatePercentage>
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void processTrainingFullTile8V11(const Dataset &data, size_t startRow, const Network &network, ThreadAccumulator &accumulator, bool detailed) {
    alignas(64) array<double,BLOCK_SIZE*HIDDEN_NODES> hidden;
    alignas(64) array<double,BLOCK_SIZE> output,deltaThree,percentage;
    const __m512d one=_mm512_set1_pd(1.0);
    for(size_t row=0;row<BLOCK_SIZE;row+=8) {
        const double*x0=data.x.rowData(startRow+row),*x1=data.x.rowData(startRow+row+1),*x2=data.x.rowData(startRow+row+2),*x3=data.x.rowData(startRow+row+3);
        const double*x4=data.x.rowData(startRow+row+4),*x5=data.x.rowData(startRow+row+5),*x6=data.x.rowData(startRow+row+6),*x7=data.x.rowData(startRow+row+7);
        __m512d a0=_mm512_setzero_pd(),a1=_mm512_setzero_pd(),b0=_mm512_setzero_pd(),b1=_mm512_setzero_pd(),c0=_mm512_setzero_pd(),c1=_mm512_setzero_pd(),d0v=_mm512_setzero_pd(),d1v=_mm512_setzero_pd();
        __m512d e0=_mm512_setzero_pd(),e1=_mm512_setzero_pd(),f0=_mm512_setzero_pd(),f1=_mm512_setzero_pd(),g0=_mm512_setzero_pd(),g1=_mm512_setzero_pd(),h0v=_mm512_setzero_pd(),h1v=_mm512_setzero_pd();
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const double*w=network.wOne.data()+k*HIDDEN_NODES;const __m512d w0=_mm512_loadu_pd(w),w1=_mm512_loadu_pd(w+8);__m512d value=_mm512_set1_pd(x0[k]);
            a0=_mm512_fmadd_pd(value,w0,a0);a1=_mm512_fmadd_pd(value,w1,a1);value=_mm512_set1_pd(x1[k]);b0=_mm512_fmadd_pd(value,w0,b0);b1=_mm512_fmadd_pd(value,w1,b1);
            value=_mm512_set1_pd(x2[k]);c0=_mm512_fmadd_pd(value,w0,c0);c1=_mm512_fmadd_pd(value,w1,c1);value=_mm512_set1_pd(x3[k]);d0v=_mm512_fmadd_pd(value,w0,d0v);d1v=_mm512_fmadd_pd(value,w1,d1v);
            value=_mm512_set1_pd(x4[k]);e0=_mm512_fmadd_pd(value,w0,e0);e1=_mm512_fmadd_pd(value,w1,e1);value=_mm512_set1_pd(x5[k]);f0=_mm512_fmadd_pd(value,w0,f0);f1=_mm512_fmadd_pd(value,w1,f1);
            value=_mm512_set1_pd(x6[k]);g0=_mm512_fmadd_pd(value,w0,g0);g1=_mm512_fmadd_pd(value,w1,g1);value=_mm512_set1_pd(x7[k]);h0v=_mm512_fmadd_pd(value,w0,h0v);h1v=_mm512_fmadd_pd(value,w1,h1v);
        }
        double*ha=hidden.data()+(row+0)*HIDDEN_NODES,*hb=hidden.data()+(row+1)*HIDDEN_NODES,*hc=hidden.data()+(row+2)*HIDDEN_NODES,*hd=hidden.data()+(row+3)*HIDDEN_NODES;
        double*he=hidden.data()+(row+4)*HIDDEN_NODES,*hf=hidden.data()+(row+5)*HIDDEN_NODES,*hg=hidden.data()+(row+6)*HIDDEN_NODES,*hh=hidden.data()+(row+7)*HIDDEN_NODES;
        _mm512_store_pd(ha,a0);_mm512_store_pd(ha+8,a1);_mm512_store_pd(hb,b0);_mm512_store_pd(hb+8,b1);_mm512_store_pd(hc,c0);_mm512_store_pd(hc+8,c1);_mm512_store_pd(hd,d0v);_mm512_store_pd(hd+8,d1v);
        _mm512_store_pd(he,e0);_mm512_store_pd(he+8,e1);_mm512_store_pd(hf,f0);_mm512_store_pd(hf+8,f1);_mm512_store_pd(hg,g0);_mm512_store_pd(hg+8,g1);_mm512_store_pd(hh,h0v);_mm512_store_pd(hh+8,h1v);
    }
    sigmoidVectorV11(hidden.data(),BLOCK_SIZE*HIDDEN_NODES);
    const __m512d w20=_mm512_loadu_pd(network.wTwo.data()),w21=_mm512_loadu_pd(network.wTwo.data()+8);
    for(size_t row=0;row<BLOCK_SIZE;row++) {
        const double*h=hidden.data()+row*HIDDEN_NODES;
        const __m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));output[row]=_mm512_reduce_add_pd(sum);
    }
    sigmoidVectorV11(output.data(),BLOCK_SIZE);
    computeOutputDeltasV9<CalculatePercentage>(data,startRow,output,deltaThree,percentage,accumulator,detailed);
    __m512d g20=_mm512_loadu_pd(accumulator.dJdWtwo.data()),g21=_mm512_loadu_pd(accumulator.dJdWtwo.data()+8);V11_DECLARE_W1_ACCUMULATORS;
    for(size_t q=0;q<BLOCK_SIZE;q++) {
        const double*h=hidden.data()+q*HIDDEN_NODES;const __m512d d=_mm512_set1_pd(deltaThree[q]),a0=_mm512_load_pd(h),a1=_mm512_load_pd(h+8);
        g20=_mm512_fmadd_pd(a0,d,g20);g21=_mm512_fmadd_pd(a1,d,g21);
        const __m512d delta0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0))),delta1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));
        const double*x=data.x.rowData(startRow+q);
        V11_ACCUMULATE_W1(0,g00,g01);V11_ACCUMULATE_W1(1,g10,g11);V11_ACCUMULATE_W1(2,g20w,g21w);V11_ACCUMULATE_W1(3,g30,g31);V11_ACCUMULATE_W1(4,g40,g41);V11_ACCUMULATE_W1(5,g50,g51);
        V11_ACCUMULATE_W1(6,g60,g61);V11_ACCUMULATE_W1(7,g70,g71);V11_ACCUMULATE_W1(8,g80,g81);V11_ACCUMULATE_W1(9,g90,g91);V11_ACCUMULATE_W1(10,g100,g101);
    }
    _mm512_storeu_pd(accumulator.dJdWtwo.data(),g20);_mm512_storeu_pd(accumulator.dJdWtwo.data()+8,g21);
    V11_STORE_W1(0,g00,g01);V11_STORE_W1(1,g10,g11);V11_STORE_W1(2,g20w,g21w);V11_STORE_W1(3,g30,g31);V11_STORE_W1(4,g40,g41);V11_STORE_W1(5,g50,g51);
    V11_STORE_W1(6,g60,g61);V11_STORE_W1(7,g70,g71);V11_STORE_W1(8,g80,g81);V11_STORE_W1(9,g90,g91);V11_STORE_W1(10,g100,g101);
}

template<bool EightRows>
__attribute__((target("avx512f,avx512dq,fma"))) static void processTrainingBlocksV11(const Dataset &data, size_t startRow, size_t batchSize, size_t firstBlock, size_t lastBlock, const Network &network, ThreadAccumulator &accumulator, bool detailed, double percentageLimit) {
    for(size_t block=firstBlock;block<lastBlock;block++) {
        const size_t offset=block*BLOCK_SIZE,rowsInBlock=min(BLOCK_SIZE,batchSize-offset);
        const bool calculatePercentage=detailed || accumulator.percentageErrorSum<=percentageLimit;
        if(rowsInBlock==BLOCK_SIZE) {
            if(EightRows) {
                if(calculatePercentage) processTrainingFullTile8V11<true>(data,startRow+offset,network,accumulator,detailed); else processTrainingFullTile8V11<false>(data,startRow+offset,network,accumulator,false);
            }
            else {
                if(calculatePercentage) processTrainingFullTile4V11<true>(data,startRow+offset,network,accumulator,detailed); else processTrainingFullTile4V11<false>(data,startRow+offset,network,accumulator,false);
            }
        }
        else {
            if(calculatePercentage) processTrainingTileAVX512V8<true>(data,startRow+offset,rowsInBlock,network,accumulator,detailed); else processTrainingTileAVX512V8<false>(data,startRow+offset,rowsInBlock,network,accumulator,false);
        }
    }
}
#undef V11_DECLARE_W1_ACCUMULATORS
#undef V11_ACCUMULATE_W1
#undef V11_STORE_W1
#endif

static bool finishV11Pass(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config, vector<ThreadAccumulator> &accumulators, Gradients &gradients, TrainResult &result, bool detailed) {
    result.metrics=combineAccumulators(accumulators,gradients,batchSize,detailed);
    bool targetReached=result.metrics.percentageError<=config.percentageErrorTarget;
    const bool updateLimitReached=result.updates>=config.maxDescents;
    if(targetReached || updateLimitReached) {
        result.metrics=calculateMetrics(data,startRow,batchSize,network);
        targetReached=result.metrics.percentageError<=config.percentageErrorTarget;
    }
    const bool logNow=config.logEvery!=0 && result.updates%config.logEvery==0;
    if(logNow || targetReached || updateLimitReached) printProgress(batchIndex,result.updates,result.metrics);
    if(targetReached || updateLimitReached) { result.reachedTarget=targetReached; return true; }
    applyMomentumUpdate(network,gradients,config.learningRate,config.momentum);result.updates++;return false;
}

static TrainResult trainSingleRangeV11(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config) {
    Gradients gradients;TrainResult result;vector<ThreadAccumulator> accumulators(1);ThreadAccumulator &accumulator=accumulators[0];
    const double percentageLimit=config.percentageErrorTarget*static_cast<double>(batchSize);
    while(true) {
        const bool detailed=(config.logEvery!=0 && result.updates%config.logEvery==0) || result.updates>=config.maxDescents;accumulator.clear();
        const size_t fullBlocks=batchSize/BLOCK_SIZE;processTrainingBlocksV11<false>(data,startRow,batchSize,0,fullBlocks+(batchSize%BLOCK_SIZE?1:0),network,accumulator,detailed,percentageLimit);
        if(finishV11Pass(data,startRow,batchSize,batchIndex,network,config,accumulators,gradients,result,detailed)) break;
    }
    writePredictions(data,startRow,batchSize,network,"ybar.txt");return result;
}

#ifdef _OPENMP
static TrainResult trainParallelRangeV11(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config) {
    Gradients gradients;TrainResult result;vector<ThreadAccumulator> accumulators(config.threads);bool stop=false;
    const size_t blockCount=(batchSize+BLOCK_SIZE-1)/BLOCK_SIZE;const double percentageLimit=config.percentageErrorTarget*static_cast<double>(batchSize);
#pragma omp parallel num_threads(static_cast<int>(config.threads)) shared(stop,result,gradients,network,accumulators)
    {
        const size_t thread=static_cast<size_t>(omp_get_thread_num()),baseCount=blockCount/config.threads,remainder=blockCount%config.threads;
        const size_t firstBlock=thread*baseCount+min(thread,remainder),ownedCount=baseCount+(thread<remainder?1:0),lastBlock=firstBlock+ownedCount;
        while(true) {
            const bool detailed=(config.logEvery!=0 && result.updates%config.logEvery==0) || result.updates>=config.maxDescents;accumulators[thread].clear();
            processTrainingBlocksV11<true>(data,startRow,batchSize,firstBlock,lastBlock,network,accumulators[thread],detailed,percentageLimit);
#pragma omp barrier
#pragma omp single
            stop=finishV11Pass(data,startRow,batchSize,batchIndex,network,config,accumulators,gradients,result,detailed);
            if(stop) break;
        }
    }
    writePredictions(data,startRow,batchSize,network,"ybar.txt");return result;
}
#endif

static TrainResult trainRangeV11(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config, const V9FeatureBounds &bounds) {
#if !defined(__x86_64__) || !defined(__GNUC__)
    return trainRangeV10(data,startRow,batchSize,batchIndex,network,config,bounds);
#else
    const bool supported=supportsV9Kernel();
    const bool useSingle=config.threads==1 && batchSize>=V8_SINGLE_THRESHOLD && supported;
#ifdef _OPENMP
    const bool useParallel=config.threads==4 && batchSize>=V9_PARALLEL_THRESHOLD && supported;
    if(useParallel) return trainParallelRangeV11(data,startRow,batchSize,batchIndex,network,config);
#endif
    if(useSingle) return trainSingleRangeV11(data,startRow,batchSize,batchIndex,network,config);
    return trainRangeV10(data,startRow,batchSize,batchIndex,network,config,bounds);
#endif
}

static void batchOnlineRunV11(size_t iterations,size_t exampleSize,const OptimiserConfig &optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937 &generator) {
    const size_t totalRows=iterations*exampleSize;V9FeatureBounds bounds;Dataset data=loadDatasetV9(totalRows,bounds);Network network;
    if(randomiseWeights) setmatrixrandom(network,rangeWone,rangeWtwo,generator); else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++) { const size_t startRow=batch*exampleSize;const TrainResult result=trainRangeV11(data,startRow,exampleSize,batch,network,optimiser,bounds);writeWeights(network);if(!result.reachedTarget) cout << "Batch " << batch << " reached the maximum number of descents before the target.\n"; }
}

static void offlineRunV11(size_t rowCount,const OptimiserConfig &optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937 &generator) {
    V9FeatureBounds bounds;Dataset data=loadDatasetV9(rowCount,bounds);Network network;if(randomiseWeights) setmatrixrandom(network,rangeWone,rangeWtwo,generator); else loadWeights(network);
    const TrainResult result=trainRangeV11(data,0,rowCount,0,network,optimiser,bounds);writeWeights(network);if(!result.reachedTarget) cout << "Offline training reached the maximum number of descents before the target.\n";
}

int main() {
    try {
        bool batchOnline=false,offline=false,test=true,randomiseWeights=false;
        double rangeWone=4.0,rangeWtwo=4.0,percentageErrorTarget=3.9,learningRate=0.0001,momentum=0.75;
        size_t iterations=1,exampleSize=13853,numberOfDescents=1000000,times=1,logEvery=1000,trainingThreads=1,offlineRows=10000,testRows=13853;
#ifndef _OPENMP
        if(trainingThreads>1) throw runtime_error("trainingThreads > 1 requires compiling with -fopenmp");
#endif
        mt19937 generator(static_cast<unsigned int>(time(nullptr)));const auto startTime=chrono::steady_clock::now();
        if(batchOnline) for(size_t i=0;i<times;i++) { OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.learningRate=learningRate;optimiser.momentum=momentum;batchOnlineRunV11(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);numberOfDescents+=numberOfDescents; }
        if(offline) { OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.learningRate=1.0;optimiser.momentum=0.0;offlineRunV11(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator); }
        if(test) testRun(testRows);
        const chrono::duration<double> elapsed=chrono::steady_clock::now()-startTime;cout << "Elapsed time = " << elapsed.count() << " seconds\n";return 0;
    }
    catch(const exception &error) { cerr << "Error: " << error.what() << '\n';return 1; }
}
