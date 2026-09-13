#include<iostream>
#include<algorithm>
#include<array>
#include<chrono>
#include<cmath>
#include<cstddef>
#include<cstdint>
#include<cstring>
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

namespace v13_v12 {
#include "../v12/main.cpp"
}
using namespace v13_v12;

/*
 * Simple Neural Network - v13 BF16 first-layer operands
 * =====================================================
 *
 * V13 keeps v12's FP64 master weights/momentum and FP32 activation,
 * backpropagation and gradient accumulation. Only the first-layer multiply
 * operands are quantised to BF16. AVX512_BF16 VDPBF16PS accumulates paired
 * BF16 products directly into FP32 hidden preactivations.
 *
 * The retained path is deliberately single-thread only because the measured
 * four-thread benefit was not stable enough across paired runs. Unsupported
 * and other configurations delegate to v12 unchanged.
 */

constexpr size_t V13_PAIR_COUNT=(NUMBER_OF_VARIABLES+1)/2;
constexpr size_t V13_SINGLE_ROWS_AT_ONCE=16;

static inline uint16_t floatToBfloatV13(float value) {
    uint32_t bits=0;
    memcpy(&bits,&value,sizeof(bits));
    const uint32_t leastSignificant=(bits>>16)&1u;
    bits+=0x7fffu+leastSignificant;
    return static_cast<uint16_t>(bits>>16);
}

struct V13TrainingData {
    V12FloatDataset floatData;
    vector<uint32_t> xPairs;
    V13TrainingData() {}
    explicit V13TrainingData(const Dataset &data):floatData(data),xPairs(data.y.size()*V13_PAIR_COUNT) {
        for(size_t row=0;row<data.y.size();row++) {
            const float*x=floatData.rowData(row);
            for(size_t pair=0;pair<V13_PAIR_COUNT;pair++) {
                const size_t first=pair*2;
                const uint16_t low=floatToBfloatV13(x[first]);
                const uint16_t high=first+1<NUMBER_OF_VARIABLES?floatToBfloatV13(x[first+1]):0;
                xPairs[row*V13_PAIR_COUNT+pair]=static_cast<uint32_t>(low)|(static_cast<uint32_t>(high)<<16);
            }
        }
    }
    const uint32_t* pairData(size_t row) const { return xPairs.data()+row*V13_PAIR_COUNT; }
};

struct alignas(64) V13PackedWeights {
    array<uint32_t,V13_PAIR_COUNT*HIDDEN_NODES> wOne{};
    array<float,WONE_SIZE> wOneFloat{};
    array<float,HIDDEN_NODES> wTwo{};
};

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
static bool supportsV13Kernel() {
    __builtin_cpu_init();
    return supportsV12Kernel() && __builtin_cpu_supports("avx512bf16");
}

static inline void syncWeightsV13(const Network &network,V13PackedWeights &packed) {
    for(size_t pair=0;pair<V13_PAIR_COUNT;pair++) {
        const size_t first=pair*2;
        for(size_t hidden=0;hidden<HIDDEN_NODES;hidden++) {
            const float lowFloat=static_cast<float>(network.wOne[first*HIDDEN_NODES+hidden]);
            packed.wOneFloat[first*HIDDEN_NODES+hidden]=lowFloat;
            const uint16_t low=floatToBfloatV13(lowFloat);
            uint16_t high=0;
            if(first+1<NUMBER_OF_VARIABLES) {
                const float highFloat=static_cast<float>(network.wOne[(first+1)*HIDDEN_NODES+hidden]);
                packed.wOneFloat[(first+1)*HIDDEN_NODES+hidden]=highFloat;
                high=floatToBfloatV13(highFloat);
            }
            packed.wOne[pair*HIDDEN_NODES+hidden]=static_cast<uint32_t>(low)|(static_cast<uint32_t>(high)<<16);
        }
    }
    for(size_t hidden=0;hidden<HIDDEN_NODES;hidden++) packed.wTwo[hidden]=static_cast<float>(network.wTwo[hidden]);
}

template<bool CalculatePercentage,size_t RowsAtOnce>
__attribute__((target("avx512f,avx512dq,avx512bf16,fma"))) static void processFullTileV13(
    const Dataset &data,const V13TrainingData &training,size_t startRow,
    const V13PackedWeights &weights,V12FloatAccumulator &accumulator,bool detailed) {
    alignas(64) float hidden[BLOCK_SIZE*HIDDEN_NODES],output[BLOCK_SIZE],deltaThree[BLOCK_SIZE];
    const __m512 one=_mm512_set1_ps(1.0f),wTwoVector=_mm512_loadu_ps(weights.wTwo.data());

    for(size_t row=0;row<BLOCK_SIZE;row+=RowsAtOnce) {
        __m512 preactivation[RowsAtOnce];
        for(size_t r=0;r<RowsAtOnce;r++) preactivation[r]=_mm512_setzero_ps();
        for(size_t pair=0;pair<V13_PAIR_COUNT;pair++) {
            const __m512bh packedWeights=(__m512bh)_mm512_loadu_si512((const void*)(weights.wOne.data()+pair*HIDDEN_NODES));
            for(size_t r=0;r<RowsAtOnce;r++) {
                const uint32_t packedInput=training.pairData(startRow+row+r)[pair];
                const __m512bh inputPair=(__m512bh)_mm512_set1_epi32(static_cast<int>(packedInput));
                preactivation[r]=_mm512_dpbf16_ps(preactivation[r],inputPair,packedWeights);
            }
        }
        for(size_t r=0;r<RowsAtOnce;r++)
            _mm512_store_ps(hidden+(row+r)*HIDDEN_NODES,sigmoidVectorV12(preactivation[r]));
    }

    for(size_t row=0;row<BLOCK_SIZE;row++) {
        const __m512 activation=_mm512_load_ps(hidden+row*HIDDEN_NODES);
        output[row]=sigmoidScalarV12(_mm512_reduce_add_ps(_mm512_mul_ps(activation,wTwoVector)));
        const float error=output[row]-training.floatData.y[startRow+row];
        deltaThree[row]=error*output[row]*(1.0f-output[row]);
        if(CalculatePercentage) {
            const double actual=data.y[startRow+row],prediction=static_cast<double>(output[row]),doubleError=prediction-actual;
            const double percentage=actual==0.0?0.0:abs(doubleError/actual)*100.0;
            accumulator.percentageErrorSum+=percentage;
            if(detailed) {
                accumulator.squaredError+=doubleError*doubleError;
                accumulator.maxPercentageError=max(accumulator.maxPercentageError,percentage);
            }
        }
    }

    __m512 gradientTwo=_mm512_setzero_ps(),gradientOne[NUMBER_OF_VARIABLES];
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) gradientOne[k]=_mm512_setzero_ps();
    for(size_t row=0;row<BLOCK_SIZE;row++) {
        const float*x=training.floatData.rowData(startRow+row);
        const __m512 activation=_mm512_load_ps(hidden+row*HIDDEN_NODES),delta=_mm512_set1_ps(deltaThree[row]);
        gradientTwo=_mm512_fmadd_ps(activation,delta,gradientTwo);
        const __m512 hiddenDelta=_mm512_mul_ps(_mm512_mul_ps(delta,wTwoVector),_mm512_mul_ps(activation,_mm512_sub_ps(one,activation)));
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)
            gradientOne[k]=_mm512_fmadd_ps(_mm512_set1_ps(x[k]),hiddenDelta,gradientOne[k]);
    }
    _mm512_storeu_ps(accumulator.dJdWtwo.data(),_mm512_add_ps(_mm512_loadu_ps(accumulator.dJdWtwo.data()),gradientTwo));
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)
        _mm512_storeu_ps(accumulator.dJdWone.data()+k*HIDDEN_NODES,_mm512_add_ps(_mm512_loadu_ps(accumulator.dJdWone.data()+k*HIDDEN_NODES),gradientOne[k]));
}

static void processBlocksV13(const Dataset &data,const V13TrainingData &training,size_t startRow,size_t batchSize,
    const V13PackedWeights &weights,V12FloatAccumulator &accumulator,bool detailed,double percentageLimit) {
    const size_t blockCount=(batchSize+BLOCK_SIZE-1)/BLOCK_SIZE;
    for(size_t block=0;block<blockCount;block++) {
        const size_t offset=block*BLOCK_SIZE,rowsInBlock=min(BLOCK_SIZE,batchSize-offset);
        const bool calculatePercentage=detailed||accumulator.percentageErrorSum<=percentageLimit;
        if(rowsInBlock==BLOCK_SIZE) {
            if(calculatePercentage) processFullTileV13<true,V13_SINGLE_ROWS_AT_ONCE>(data,training,startRow+offset,weights,accumulator,detailed);
            else processFullTileV13<false,V13_SINGLE_ROWS_AT_ONCE>(data,training,startRow+offset,weights,accumulator,false);
        }
        else {
            if(calculatePercentage) processTailV12<true>(data,training.floatData,startRow+offset,rowsInBlock,
                weights.wOneFloat,weights.wTwo,accumulator,detailed);
            else processTailV12<false>(data,training.floatData,startRow+offset,rowsInBlock,
                weights.wOneFloat,weights.wTwo,accumulator,false);
        }
    }
}

static TrainResult trainSingleRangeV13(const Dataset&data,const V13TrainingData&training,size_t startRow,size_t batchSize,
    size_t batchIndex,Network&network,const OptimiserConfig&config) {
    Gradients gradients;TrainResult result;vector<V12FloatAccumulator>accumulators(1);V13PackedWeights packed;
    const double percentageLimit=config.percentageErrorTarget*static_cast<double>(batchSize);
    while(true) {
        const bool detailed=(config.logEvery!=0&&result.updates%config.logEvery==0)||result.updates>=config.maxDescents;
        syncWeightsV13(network,packed);accumulators[0].clear();
        processBlocksV13(data,training,startRow,batchSize,packed,accumulators[0],detailed,percentageLimit);
        if(finishV12Pass(data,startRow,batchSize,batchIndex,network,config,accumulators,gradients,result,detailed)) break;
    }
    writePredictions(data,startRow,batchSize,network,"ybar.txt");return result;
}
#endif

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
static bool useV13Path(size_t rows,const OptimiserConfig&config) {
    return supportsV13Kernel()&&config.threads==1&&rows>=V8_SINGLE_THRESHOLD;
}
#endif

static void batchOnlineRunV13(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
    bool randomiseWeights,mt19937&generator) {
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(!useV13Path(exampleSize,optimiser)) { batchOnlineRunV12(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);return; }
    const size_t totalRows=iterations*exampleSize;V9FeatureBounds bounds;Dataset data=loadDatasetV9(totalRows,bounds);V13TrainingData training(data);Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++) {
        const size_t startRow=batch*exampleSize;const TrainResult result=trainSingleRangeV13(data,training,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of descents before the target.\n";
    }
#else
    batchOnlineRunV12(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
#endif
}

static void offlineRunV13(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator) {
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(!useV13Path(rowCount,optimiser)) { offlineRunV12(rowCount,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);return; }
    V9FeatureBounds bounds;Dataset data=loadDatasetV9(rowCount,bounds);V13TrainingData training(data);Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainSingleRangeV13(data,training,0,rowCount,0,network,optimiser);writeWeights(network);
    if(!result.reachedTarget)cout<<"Offline training reached the maximum number of descents before the target.\n";
#else
    offlineRunV12(rowCount,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
#endif
}

int main() {
    try {
        bool batchOnline=false,offline=false,test=true,randomiseWeights=false;
        double rangeWone=4.0,rangeWtwo=4.0,percentageErrorTarget=3.9,learningRate=0.0001,momentum=0.75;
        size_t iterations=1,exampleSize=13853,numberOfDescents=1000000,times=1,logEvery=1000,trainingThreads=1,offlineRows=10000,testRows=13853;
#ifndef _OPENMP
        if(trainingThreads>1)throw runtime_error("trainingThreads > 1 requires compiling with -fopenmp");
#endif
        mt19937 generator(static_cast<unsigned int>(time(nullptr)));const auto startTime=chrono::steady_clock::now();
        if(batchOnline)for(size_t i=0;i<times;i++){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.learningRate=learningRate;optimiser.momentum=momentum;batchOnlineRunV13(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);numberOfDescents+=numberOfDescents;}
        if(offline){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.learningRate=1.0;optimiser.momentum=0.0;offlineRunV13(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);}
        if(test)testRun(testRows);
        const chrono::duration<double>elapsed=chrono::steady_clock::now()-startTime;
        cout<<"Elapsed time = "<<elapsed.count()<<" seconds\n";
        return 0;
    }
    catch(const exception&error){cerr<<"Error: "<<error.what()<<'\n';return 1;}
}
