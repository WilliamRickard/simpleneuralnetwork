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

namespace v10_v9 {
#include "../v9/main.cpp"
}
using namespace v10_v9;

/*
 * Simple Neural Network - v10 exact frontier
 * ===========================================
 *
 * V10 keeps v9 for every path except large four-thread AVX-512/libmvec
 * training. The retained v10 kernel processes eight observations at a time in
 * the 11 x 16 forward multiply instead of four. Each observation still executes
 * the same 11 FMAs in the same order, so the arithmetic for every activation is
 * unchanged. The backpropagation, gradient reduction, sigmoid implementation,
 * percentage cutoff and weight update order are inherited exactly from v9.
 */

constexpr size_t V10_THREADS = 4;

#if defined(__x86_64__) && defined(__GNUC__)
#define V10_DECLARE_W1_ACCUMULATORS \
    __m512d g00=_mm512_setzero_pd(),g01=_mm512_setzero_pd(),g10=_mm512_setzero_pd(),g11=_mm512_setzero_pd(),g20w=_mm512_setzero_pd(),g21w=_mm512_setzero_pd(),g30=_mm512_setzero_pd(),g31=_mm512_setzero_pd(),g40=_mm512_setzero_pd(),g41=_mm512_setzero_pd(),g50=_mm512_setzero_pd(),g51=_mm512_setzero_pd(),g60=_mm512_setzero_pd(),g61=_mm512_setzero_pd(),g70=_mm512_setzero_pd(),g71=_mm512_setzero_pd(),g80=_mm512_setzero_pd(),g81=_mm512_setzero_pd(),g90=_mm512_setzero_pd(),g91=_mm512_setzero_pd(),g100=_mm512_setzero_pd(),g101=_mm512_setzero_pd()
#define V10_ACCUMULATE_W1(K,G0,G1) do { const __m512d xv=_mm512_set1_pd(x[K]); G0=_mm512_fmadd_pd(xv,delta0,G0); G1=_mm512_fmadd_pd(xv,delta1,G1); } while(false)
#define V10_STORE_W1(K,G0,G1) do { double*g=accumulator.dJdWone.data()+(K)*HIDDEN_NODES; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),G0)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),G1)); } while(false)

template<bool CalculatePercentage, bool UncheckedSigmoid>
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void processTrainingFullTileAVX512V10(const Dataset &data, size_t startRow, const Network &network, ThreadAccumulator &accumulator, bool detailed) {
    alignas(64) array<double,BLOCK_SIZE*HIDDEN_NODES> hidden;
    alignas(64) array<double,BLOCK_SIZE> output,deltaThree,percentage;
    const __m512d one=_mm512_set1_pd(1.0);

    for(size_t row=0;row<BLOCK_SIZE;row+=8) {
        const double*x0=data.x.rowData(startRow+row),*x1=data.x.rowData(startRow+row+1),*x2=data.x.rowData(startRow+row+2),*x3=data.x.rowData(startRow+row+3);
        const double*x4=data.x.rowData(startRow+row+4),*x5=data.x.rowData(startRow+row+5),*x6=data.x.rowData(startRow+row+6),*x7=data.x.rowData(startRow+row+7);
        __m512d a0=_mm512_setzero_pd(),a1=_mm512_setzero_pd(),b0=_mm512_setzero_pd(),b1=_mm512_setzero_pd();
        __m512d c0=_mm512_setzero_pd(),c1=_mm512_setzero_pd(),d0v=_mm512_setzero_pd(),d1v=_mm512_setzero_pd();
        __m512d e0=_mm512_setzero_pd(),e1=_mm512_setzero_pd(),f0=_mm512_setzero_pd(),f1=_mm512_setzero_pd();
        __m512d g0=_mm512_setzero_pd(),g1=_mm512_setzero_pd(),h0v=_mm512_setzero_pd(),h1v=_mm512_setzero_pd();
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const double*w=network.wOne.data()+k*HIDDEN_NODES;
            const __m512d w0=_mm512_loadu_pd(w),w1=_mm512_loadu_pd(w+8);
            __m512d value=_mm512_set1_pd(x0[k]);a0=_mm512_fmadd_pd(value,w0,a0);a1=_mm512_fmadd_pd(value,w1,a1);
            value=_mm512_set1_pd(x1[k]);b0=_mm512_fmadd_pd(value,w0,b0);b1=_mm512_fmadd_pd(value,w1,b1);
            value=_mm512_set1_pd(x2[k]);c0=_mm512_fmadd_pd(value,w0,c0);c1=_mm512_fmadd_pd(value,w1,c1);
            value=_mm512_set1_pd(x3[k]);d0v=_mm512_fmadd_pd(value,w0,d0v);d1v=_mm512_fmadd_pd(value,w1,d1v);
            value=_mm512_set1_pd(x4[k]);e0=_mm512_fmadd_pd(value,w0,e0);e1=_mm512_fmadd_pd(value,w1,e1);
            value=_mm512_set1_pd(x5[k]);f0=_mm512_fmadd_pd(value,w0,f0);f1=_mm512_fmadd_pd(value,w1,f1);
            value=_mm512_set1_pd(x6[k]);g0=_mm512_fmadd_pd(value,w0,g0);g1=_mm512_fmadd_pd(value,w1,g1);
            value=_mm512_set1_pd(x7[k]);h0v=_mm512_fmadd_pd(value,w0,h0v);h1v=_mm512_fmadd_pd(value,w1,h1v);
        }
        double*ha=hidden.data()+(row+0)*HIDDEN_NODES,*hb=hidden.data()+(row+1)*HIDDEN_NODES,*hc=hidden.data()+(row+2)*HIDDEN_NODES,*hd=hidden.data()+(row+3)*HIDDEN_NODES;
        double*he=hidden.data()+(row+4)*HIDDEN_NODES,*hf=hidden.data()+(row+5)*HIDDEN_NODES,*hg=hidden.data()+(row+6)*HIDDEN_NODES,*hh=hidden.data()+(row+7)*HIDDEN_NODES;
        _mm512_store_pd(ha,a0);_mm512_store_pd(ha+8,a1);_mm512_store_pd(hb,b0);_mm512_store_pd(hb+8,b1);
        _mm512_store_pd(hc,c0);_mm512_store_pd(hc+8,c1);_mm512_store_pd(hd,d0v);_mm512_store_pd(hd+8,d1v);
        _mm512_store_pd(he,e0);_mm512_store_pd(he+8,e1);_mm512_store_pd(hf,f0);_mm512_store_pd(hf+8,f1);
        _mm512_store_pd(hg,g0);_mm512_store_pd(hg+8,g1);_mm512_store_pd(hh,h0v);_mm512_store_pd(hh+8,h1v);
    }
    sigmoidVectorFullV9<UncheckedSigmoid>(hidden.data(),BLOCK_SIZE*HIDDEN_NODES);

    const __m512d w20=_mm512_loadu_pd(network.wTwo.data()),w21=_mm512_loadu_pd(network.wTwo.data()+8);
    for(size_t row=0;row<BLOCK_SIZE;row++) {
        const double*h=hidden.data()+row*HIDDEN_NODES;
        const __m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));
        output[row]=_mm512_reduce_add_pd(sum);
    }
    sigmoidVectorFullV9<UncheckedSigmoid>(output.data(),BLOCK_SIZE);
    computeOutputDeltasV9<CalculatePercentage>(data,startRow,output,deltaThree,percentage,accumulator,detailed);

    __m512d g20=_mm512_loadu_pd(accumulator.dJdWtwo.data()),g21=_mm512_loadu_pd(accumulator.dJdWtwo.data()+8);V10_DECLARE_W1_ACCUMULATORS;
    for(size_t q=0;q<BLOCK_SIZE;q++) {
        const double*h=hidden.data()+q*HIDDEN_NODES;const __m512d d=_mm512_set1_pd(deltaThree[q]),a0=_mm512_load_pd(h),a1=_mm512_load_pd(h+8);
        g20=_mm512_fmadd_pd(a0,d,g20);g21=_mm512_fmadd_pd(a1,d,g21);
        const __m512d delta0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0))),delta1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));
        const double*x=data.x.rowData(startRow+q);
        V10_ACCUMULATE_W1(0,g00,g01);V10_ACCUMULATE_W1(1,g10,g11);V10_ACCUMULATE_W1(2,g20w,g21w);V10_ACCUMULATE_W1(3,g30,g31);V10_ACCUMULATE_W1(4,g40,g41);V10_ACCUMULATE_W1(5,g50,g51);
        V10_ACCUMULATE_W1(6,g60,g61);V10_ACCUMULATE_W1(7,g70,g71);V10_ACCUMULATE_W1(8,g80,g81);V10_ACCUMULATE_W1(9,g90,g91);V10_ACCUMULATE_W1(10,g100,g101);
    }
    _mm512_storeu_pd(accumulator.dJdWtwo.data(),g20);_mm512_storeu_pd(accumulator.dJdWtwo.data()+8,g21);
    V10_STORE_W1(0,g00,g01);V10_STORE_W1(1,g10,g11);V10_STORE_W1(2,g20w,g21w);V10_STORE_W1(3,g30,g31);V10_STORE_W1(4,g40,g41);V10_STORE_W1(5,g50,g51);
    V10_STORE_W1(6,g60,g61);V10_STORE_W1(7,g70,g71);V10_STORE_W1(8,g80,g81);V10_STORE_W1(9,g90,g91);V10_STORE_W1(10,g100,g101);
}

template<bool UncheckedSigmoid>
__attribute__((target("avx512f,avx512dq,fma"))) static void processTrainingBlocksAVX512V10(const Dataset &data, size_t startRow, size_t batchSize, size_t firstBlock, size_t lastBlock, const Network &network, ThreadAccumulator &accumulator, bool detailed, double percentageLimit) {
    for(size_t block=firstBlock;block<lastBlock;block++) {
        const size_t offset=block*BLOCK_SIZE,rowsInBlock=min(BLOCK_SIZE,batchSize-offset);
        const bool calculatePercentage=detailed || accumulator.percentageErrorSum<=percentageLimit;
        if(rowsInBlock==BLOCK_SIZE) {
            if(calculatePercentage) processTrainingFullTileAVX512V10<true,UncheckedSigmoid>(data,startRow+offset,network,accumulator,detailed);
            else processTrainingFullTileAVX512V10<false,UncheckedSigmoid>(data,startRow+offset,network,accumulator,false);
        }
        else {
            if(calculatePercentage) processTrainingTileAVX512V8<true>(data,startRow+offset,rowsInBlock,network,accumulator,detailed);
            else processTrainingTileAVX512V8<false>(data,startRow+offset,rowsInBlock,network,accumulator,false);
        }
    }
}
#undef V10_DECLARE_W1_ACCUMULATORS
#undef V10_ACCUMULATE_W1
#undef V10_STORE_W1
#endif

#ifdef _OPENMP
static TrainResult trainParallelRangeV10(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config, const V9FeatureBounds &bounds) {
    Gradients gradients;
    TrainResult result;
    vector<ThreadAccumulator> accumulators(config.threads);
    bool stop=false;
    const size_t blockCount=(batchSize+BLOCK_SIZE-1)/BLOCK_SIZE;
    const double percentageLimit=config.percentageErrorTarget*static_cast<double>(batchSize);
#pragma omp parallel num_threads(static_cast<int>(config.threads)) shared(stop,result,gradients,network,accumulators)
    {
        const size_t thread=static_cast<size_t>(omp_get_thread_num());
        const size_t baseCount=blockCount/config.threads,remainder=blockCount%config.threads;
        const size_t firstBlock=thread*baseCount+min(thread,remainder),ownedCount=baseCount+(thread<remainder?1:0),lastBlock=firstBlock+ownedCount;
        while(true) {
            const bool detailed=(config.logEvery!=0 && result.updates%config.logEvery==0) || result.updates>=config.maxDescents;
            accumulators[thread].clear();
            if(v9UncheckedSigmoidSafe(bounds,network)) processTrainingBlocksAVX512V10<true>(data,startRow,batchSize,firstBlock,lastBlock,network,accumulators[thread],detailed,percentageLimit);
            else processTrainingBlocksAVX512V10<false>(data,startRow,batchSize,firstBlock,lastBlock,network,accumulators[thread],detailed,percentageLimit);
#pragma omp barrier
#pragma omp single
            stop=finishV9Pass(data,startRow,batchSize,batchIndex,network,config,accumulators,gradients,result,detailed);
            if(stop) break;
        }
    }
    writePredictions(data,startRow,batchSize,network,"ybar.txt");
    return result;
}
#endif

static TrainResult trainRangeV10(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config, const V9FeatureBounds &bounds) {
#if !defined(__x86_64__) || !defined(__GNUC__)
    (void)bounds;
    return trainRangeV9(data,startRow,batchSize,batchIndex,network,config,bounds);
#else
    const bool useV10=config.threads==V10_THREADS && batchSize>=V9_PARALLEL_THRESHOLD && supportsV9Kernel();
#ifdef _OPENMP
    if(useV10) return trainParallelRangeV10(data,startRow,batchSize,batchIndex,network,config,bounds);
#else
    (void)useV10;
#endif
    return trainRangeV9(data,startRow,batchSize,batchIndex,network,config,bounds);
#endif
}

static void batchOnlineRunV10(size_t iterations, size_t exampleSize, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator) {
    const size_t totalRows=iterations*exampleSize;V9FeatureBounds bounds;Dataset data=loadDatasetV9(totalRows,bounds);Network network;
    if(randomiseWeights) setmatrixrandom(network,rangeWone,rangeWtwo,generator); else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++) { const size_t startRow=batch*exampleSize;const TrainResult result=trainRangeV10(data,startRow,exampleSize,batch,network,optimiser,bounds);writeWeights(network);if(!result.reachedTarget) cout << "Batch " << batch << " reached the maximum number of descents before the target.\n"; }
}

static void offlineRunV10(size_t rowCount, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator) {
    V9FeatureBounds bounds;Dataset data=loadDatasetV9(rowCount,bounds);Network network;if(randomiseWeights) setmatrixrandom(network,rangeWone,rangeWtwo,generator); else loadWeights(network);
    const TrainResult result=trainRangeV10(data,0,rowCount,0,network,optimiser,bounds);writeWeights(network);if(!result.reachedTarget) cout << "Offline training reached the maximum number of descents before the target.\n";
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
        if(batchOnline) for(size_t i=0;i<times;i++) { OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.learningRate=learningRate;optimiser.momentum=momentum;batchOnlineRunV10(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);numberOfDescents+=numberOfDescents; }
        if(offline) { OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.learningRate=1.0;optimiser.momentum=0.0;offlineRunV10(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator); }
        if(test) testRun(testRows);
        const chrono::duration<double> elapsed=chrono::steady_clock::now()-startTime;cout << "Elapsed time = " << elapsed.count() << " seconds\n";return 0;
    }
    catch(const exception &error) { cerr << "Error: " << error.what() << '\n';return 1; }
}
