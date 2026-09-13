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
 * Simple Neural Network - v11 relaxed floating point
 * ==================================================
 *
 * V11 keeps the v10/v9 network, full-batch gradient descent, momentum,
 * reduction structure and stopping target. It relaxes bit-for-bit equivalence
 * only inside the sigmoid. Vectors wholly inside [-1,1] use a degree-5 odd
 * minimax-style polynomial. Any vector with a lane outside that interval falls
 * back to the existing checked libmvec sigmoid.
 *
 * The approximation is selected only on measured winning production paths:
 * one thread at >= 50k rows and exactly four threads at >= 1m rows. All other
 * configurations delegate to v10 unchanged.
 */

constexpr size_t V11_THREADS = 4;

#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx512f,avx512dq,fma"),always_inline))
static inline void sigmoidVectorAdaptiveV11(double *values, size_t count) {
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__)
    const __m512d signless=_mm512_castsi512_pd(_mm512_set1_epi64(0x7fffffffffffffffLL));
    const __m512d limit=_mm512_set1_pd(1.0),half=_mm512_set1_pd(0.5);
    const __m512d c1=_mm512_set1_pd(0.24998101634651657);
    const __m512d c3=_mm512_set1_pd(-0.020677835421401624);
    const __m512d c5=_mm512_set1_pd(0.0017580292406143272);
    for(size_t i=0;i<count;i+=8) {
        const __m512d x=_mm512_loadu_pd(values+i),absolute=_mm512_and_pd(x,signless);
        if(_mm512_cmp_pd_mask(absolute,limit,_CMP_GT_OQ)) {
            sigmoidVectorFullV9<false>(values+i,8);
            continue;
        }
        const __m512d x2=_mm512_mul_pd(x,x);
        __m512d polynomial=_mm512_fmadd_pd(c5,x2,c3);
        polynomial=_mm512_fmadd_pd(polynomial,x2,c1);
        _mm512_storeu_pd(values+i,_mm512_fmadd_pd(x,polynomial,half));
    }
#else
    sigmoidVectorCached(values,count);
#endif
}

#define V11_DECLARE_W1_ACCUMULATORS \
    __m512d g00=_mm512_setzero_pd(),g01=_mm512_setzero_pd(),g10=_mm512_setzero_pd(),g11=_mm512_setzero_pd(),g20w=_mm512_setzero_pd(),g21w=_mm512_setzero_pd(),g30=_mm512_setzero_pd(),g31=_mm512_setzero_pd(),g40=_mm512_setzero_pd(),g41=_mm512_setzero_pd(),g50=_mm512_setzero_pd(),g51=_mm512_setzero_pd(),g60=_mm512_setzero_pd(),g61=_mm512_setzero_pd(),g70=_mm512_setzero_pd(),g71=_mm512_setzero_pd(),g80=_mm512_setzero_pd(),g81=_mm512_setzero_pd(),g90=_mm512_setzero_pd(),g91=_mm512_setzero_pd(),g100=_mm512_setzero_pd(),g101=_mm512_setzero_pd()
#define V11_ACCUMULATE_W1(K,G0,G1) do { const __m512d xv=_mm512_set1_pd(x[K]); G0=_mm512_fmadd_pd(xv,delta0,G0); G1=_mm512_fmadd_pd(xv,delta1,G1); } while(false)
#define V11_STORE_W1(K,G0,G1) do { double*g=accumulator.dJdWone.data()+(K)*HIDDEN_NODES; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),G0)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),G1)); } while(false)

template<size_t RowsAtOnce>
__attribute__((target("avx512f,avx512dq,fma"),always_inline))
static inline void forwardFullTileV11(const Dataset &data, size_t startRow, const Network &network, array<double,BLOCK_SIZE*HIDDEN_NODES> &hidden) {
    for(size_t row=0;row<BLOCK_SIZE;row+=RowsAtOnce) {
        __m512d low[8],high[8];
        const double* x[8];
        for(size_t r=0;r<RowsAtOnce;r++) { low[r]=_mm512_setzero_pd();high[r]=_mm512_setzero_pd();x[r]=data.x.rowData(startRow+row+r); }
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const double*w=network.wOne.data()+k*HIDDEN_NODES;
            const __m512d w0=_mm512_loadu_pd(w),w1=_mm512_loadu_pd(w+8);
            for(size_t r=0;r<RowsAtOnce;r++) {
                const __m512d value=_mm512_set1_pd(x[r][k]);
                low[r]=_mm512_fmadd_pd(value,w0,low[r]);high[r]=_mm512_fmadd_pd(value,w1,high[r]);
            }
        }
        for(size_t r=0;r<RowsAtOnce;r++) {
            double*h=hidden.data()+(row+r)*HIDDEN_NODES;
            _mm512_storeu_pd(h,low[r]);_mm512_storeu_pd(h+8,high[r]);
        }
    }
}

template<bool CalculatePercentage, size_t RowsAtOnce>
__attribute__((target("avx512f,avx512dq,fma"),always_inline))
static inline void processTrainingFullTileAVX512V11(const Dataset &data, size_t startRow, const Network &network, ThreadAccumulator &accumulator, bool detailed) {
    alignas(64) array<double,BLOCK_SIZE*HIDDEN_NODES> hidden;
    alignas(64) array<double,BLOCK_SIZE> output,deltaThree,percentage;
    const __m512d one=_mm512_set1_pd(1.0);
    forwardFullTileV11<RowsAtOnce>(data,startRow,network,hidden);
    sigmoidVectorAdaptiveV11(hidden.data(),BLOCK_SIZE*HIDDEN_NODES);
    const __m512d w20=_mm512_loadu_pd(network.wTwo.data()),w21=_mm512_loadu_pd(network.wTwo.data()+8);
    for(size_t row=0;row<BLOCK_SIZE;row++) {
        const double*h=hidden.data()+row*HIDDEN_NODES;
        const __m512d sum=_mm512_fmadd_pd(_mm512_loadu_pd(h+8),w21,_mm512_mul_pd(_mm512_loadu_pd(h),w20));
        output[row]=_mm512_reduce_add_pd(sum);
    }
    sigmoidVectorAdaptiveV11(output.data(),BLOCK_SIZE);
    computeOutputDeltasV9<CalculatePercentage>(data,startRow,output,deltaThree,percentage,accumulator,detailed);
    __m512d g20=_mm512_loadu_pd(accumulator.dJdWtwo.data()),g21=_mm512_loadu_pd(accumulator.dJdWtwo.data()+8);V11_DECLARE_W1_ACCUMULATORS;
    for(size_t q=0;q<BLOCK_SIZE;q++) {
        const double*h=hidden.data()+q*HIDDEN_NODES;const __m512d d=_mm512_set1_pd(deltaThree[q]),a0=_mm512_loadu_pd(h),a1=_mm512_loadu_pd(h+8);
        g20=_mm512_fmadd_pd(a0,d,g20);g21=_mm512_fmadd_pd(a1,d,g21);
        const __m512d delta0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0)));
        const __m512d delta1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));
        const double*x=data.x.rowData(startRow+q);
        V11_ACCUMULATE_W1(0,g00,g01);V11_ACCUMULATE_W1(1,g10,g11);V11_ACCUMULATE_W1(2,g20w,g21w);V11_ACCUMULATE_W1(3,g30,g31);V11_ACCUMULATE_W1(4,g40,g41);V11_ACCUMULATE_W1(5,g50,g51);
        V11_ACCUMULATE_W1(6,g60,g61);V11_ACCUMULATE_W1(7,g70,g71);V11_ACCUMULATE_W1(8,g80,g81);V11_ACCUMULATE_W1(9,g90,g91);V11_ACCUMULATE_W1(10,g100,g101);
    }
    _mm512_storeu_pd(accumulator.dJdWtwo.data(),g20);_mm512_storeu_pd(accumulator.dJdWtwo.data()+8,g21);
    V11_STORE_W1(0,g00,g01);V11_STORE_W1(1,g10,g11);V11_STORE_W1(2,g20w,g21w);V11_STORE_W1(3,g30,g31);V11_STORE_W1(4,g40,g41);V11_STORE_W1(5,g50,g51);
    V11_STORE_W1(6,g60,g61);V11_STORE_W1(7,g70,g71);V11_STORE_W1(8,g80,g81);V11_STORE_W1(9,g90,g91);V11_STORE_W1(10,g100,g101);
}

template<size_t RowsAtOnce>
__attribute__((target("avx512f,avx512dq,fma")))
static void processTrainingRangeAVX512V11(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, ThreadAccumulator &accumulator, bool detailed, double percentageLimit) {
    size_t offset=0;
    for(;offset+BLOCK_SIZE<=rowCount;offset+=BLOCK_SIZE) {
        if(detailed || accumulator.percentageErrorSum<=percentageLimit) processTrainingFullTileAVX512V11<true,RowsAtOnce>(data,startRow+offset,network,accumulator,detailed);
        else processTrainingFullTileAVX512V11<false,RowsAtOnce>(data,startRow+offset,network,accumulator,false);
    }
    if(offset<rowCount) {
        const size_t tail=rowCount-offset;
        if(detailed || accumulator.percentageErrorSum<=percentageLimit) processTrainingTileAVX512V8<true>(data,startRow+offset,tail,network,accumulator,detailed);
        else processTrainingTileAVX512V8<false>(data,startRow+offset,tail,network,accumulator,false);
    }
}

template<size_t RowsAtOnce>
__attribute__((target("avx512f,avx512dq,fma")))
static void processTrainingBlocksAVX512V11(const Dataset &data, size_t startRow, size_t batchSize, size_t firstBlock, size_t lastBlock, const Network &network, ThreadAccumulator &accumulator, bool detailed, double percentageLimit) {
    for(size_t block=firstBlock;block<lastBlock;block++) {
        const size_t offset=block*BLOCK_SIZE,rowsInBlock=min(BLOCK_SIZE,batchSize-offset);
        const bool calculatePercentage=detailed || accumulator.percentageErrorSum<=percentageLimit;
        if(rowsInBlock==BLOCK_SIZE) {
            if(calculatePercentage) processTrainingFullTileAVX512V11<true,RowsAtOnce>(data,startRow+offset,network,accumulator,detailed);
            else processTrainingFullTileAVX512V11<false,RowsAtOnce>(data,startRow+offset,network,accumulator,false);
        }
        else {
            if(calculatePercentage) processTrainingTileAVX512V8<true>(data,startRow+offset,rowsInBlock,network,accumulator,detailed);
            else processTrainingTileAVX512V8<false>(data,startRow+offset,rowsInBlock,network,accumulator,false);
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
    const bool logNow=config.logEvery!=0 && result.updates%config.logEvery==0;
    if(targetReached) { result.metrics=calculateMetrics(data,startRow,batchSize,network);targetReached=result.metrics.percentageError<=config.percentageErrorTarget; }
    if(logNow || targetReached || updateLimitReached) printProgress(batchIndex,result.updates,result.metrics);
    if(targetReached || updateLimitReached) { result.reachedTarget=targetReached;return true; }
    applyMomentumUpdate(network,gradients,config.learningRate,config.momentum);result.updates++;return false;
}

static TrainResult trainSingleRangeV11(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config) {
    Gradients gradients;TrainResult result;vector<ThreadAccumulator> accumulators(1);ThreadAccumulator &accumulator=accumulators[0];
    const double percentageLimit=config.percentageErrorTarget*static_cast<double>(batchSize);
    while(true) {
        const bool detailed=(config.logEvery!=0 && result.updates%config.logEvery==0) || result.updates>=config.maxDescents;
        accumulator.clear();processTrainingRangeAVX512V11<4>(data,startRow,batchSize,network,accumulator,detailed,percentageLimit);
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
        const size_t thread=static_cast<size_t>(omp_get_thread_num());const size_t baseCount=blockCount/config.threads,remainder=blockCount%config.threads;
        const size_t firstBlock=thread*baseCount+min(thread,remainder),ownedCount=baseCount+(thread<remainder?1:0),lastBlock=firstBlock+ownedCount;
        while(true) {
            const bool detailed=(config.logEvery!=0 && result.updates%config.logEvery==0) || result.updates>=config.maxDescents;
            accumulators[thread].clear();processTrainingBlocksAVX512V11<8>(data,startRow,batchSize,firstBlock,lastBlock,network,accumulators[thread],detailed,percentageLimit);
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
    if(config.threads==1 && batchSize>=V8_SINGLE_THRESHOLD && supported) return trainSingleRangeV11(data,startRow,batchSize,batchIndex,network,config);
#ifdef _OPENMP
    if(config.threads==V11_THREADS && batchSize>=V9_PARALLEL_THRESHOLD && supported) return trainParallelRangeV11(data,startRow,batchSize,batchIndex,network,config);
#endif
    return trainRangeV10(data,startRow,batchSize,batchIndex,network,config,bounds);
#endif
}

static void batchOnlineRunV11(size_t iterations, size_t exampleSize, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator) {
    const size_t totalRows=iterations*exampleSize;V9FeatureBounds bounds;Dataset data=loadDatasetV9(totalRows,bounds);Network network;
    if(randomiseWeights) setmatrixrandom(network,rangeWone,rangeWtwo,generator); else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++) { const size_t startRow=batch*exampleSize;const TrainResult result=trainRangeV11(data,startRow,exampleSize,batch,network,optimiser,bounds);writeWeights(network);if(!result.reachedTarget) cout << "Batch " << batch << " reached the maximum number of descents before the target.\n"; }
}

static void offlineRunV11(size_t rowCount, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator) {
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
