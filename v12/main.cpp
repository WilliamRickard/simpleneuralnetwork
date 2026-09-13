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

namespace v12_v11 {
#include "../v11/main.cpp"
}
using namespace v12_v11;

/*
 * Simple Neural Network - v12 mixed precision
 * ===========================================
 * FP64 master weights, FP64 momentum and FP64 updates are retained. The hot
 * training path stores a float copy of the dataset and evaluates forward pass,
 * sigmoid, backpropagation and per-thread gradient accumulation in FP32.
 * Gradients are promoted back to FP64 before the historical momentum update.
 */

struct V12FloatDataset {
    size_t rows=0;
    vector<float> x,y;
    V12FloatDataset() {}
    explicit V12FloatDataset(const Dataset &data):rows(data.y.size()),x(rows*NUMBER_OF_VARIABLES),y(rows) {
        for(size_t row=0;row<rows;row++) {
            const double*source=data.x.rowData(row);
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) x[row*NUMBER_OF_VARIABLES+k]=static_cast<float>(source[k]);
            y[row]=static_cast<float>(data.y[row]);
        }
    }
    const float* rowData(size_t row) const { return x.data()+row*NUMBER_OF_VARIABLES; }
};

struct alignas(64) V12FloatAccumulator {
    array<float,WONE_SIZE> dJdWone{};
    array<float,HIDDEN_NODES> dJdWtwo{};
    double squaredError=0.0,percentageErrorSum=0.0,maxPercentageError=0.0;
    void clear(){dJdWone.fill(0.0f);dJdWtwo.fill(0.0f);squaredError=percentageErrorSum=maxPercentageError=0.0;}
};

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
extern "C" __m512 _ZGVeN16v_expf(__m512);

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512 sigmoidVectorV12(__m512 value) {
    const __m512 signless=_mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512 limit=_mm512_set1_ps(1.0f),half=_mm512_set1_ps(0.5f),one=_mm512_set1_ps(1.0f),zero=_mm512_setzero_ps();
    const __m512 c1=_mm512_set1_ps(0.24998101634651657f),c3=_mm512_set1_ps(-0.020677835421401624f),c5=_mm512_set1_ps(0.0017580292406143272f);
    const __m512 absolute=_mm512_and_ps(value,signless);
    if(_mm512_cmp_ps_mask(absolute,limit,_CMP_GT_OQ)) {
        const __m512 exponent=_ZGVeN16v_expf(_mm512_sub_ps(zero,value));
        return _mm512_div_ps(one,_mm512_add_ps(one,exponent));
    }
    const __m512 squared=_mm512_mul_ps(value,value);
    __m512 polynomial=_mm512_fmadd_ps(c5,squared,c3);
    polynomial=_mm512_fmadd_ps(polynomial,squared,c1);
    return _mm512_fmadd_ps(value,polynomial,half);
}

static inline float sigmoidScalarV12(float value) {
    if(fabsf(value)<=1.0f) {
        const float c1=0.24998101634651657f,c3=-0.020677835421401624f,c5=0.0017580292406143272f;
        const float squared=value*value;
        return fmaf(value,fmaf(squared,fmaf(c5,squared,c3),c1),0.5f);
    }
    return 1.0f/(1.0f+expf(-value));
}

static bool supportsV12Kernel(){return supportsV9Kernel() && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");}

static inline void syncWeightsV12(const Network &network, array<float,WONE_SIZE> &wOne, array<float,HIDDEN_NODES> &wTwo) {
    for(size_t i=0;i<WONE_SIZE;i++) wOne[i]=static_cast<float>(network.wOne[i]);
    for(size_t j=0;j<HIDDEN_NODES;j++) wTwo[j]=static_cast<float>(network.wTwo[j]);
}

template<bool CalculatePercentage>
__attribute__((target("avx512f,avx512dq,fma"))) static void processFullTileV12(
    const Dataset &data,const V12FloatDataset &floatData,size_t startRow,
    const array<float,WONE_SIZE> &wOne,const array<float,HIDDEN_NODES> &wTwo,
    V12FloatAccumulator &accumulator,bool detailed) {
    alignas(64) float hidden[BLOCK_SIZE*HIDDEN_NODES],output[BLOCK_SIZE],deltaThree[BLOCK_SIZE];
    const __m512 one=_mm512_set1_ps(1.0f),wTwoVector=_mm512_loadu_ps(wTwo.data());
    for(size_t row=0;row<BLOCK_SIZE;row+=4) {
        __m512 z0=_mm512_setzero_ps(),z1=_mm512_setzero_ps(),z2=_mm512_setzero_ps(),z3=_mm512_setzero_ps();
        const float*x0=floatData.rowData(startRow+row),*x1=floatData.rowData(startRow+row+1),*x2=floatData.rowData(startRow+row+2),*x3=floatData.rowData(startRow+row+3);
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const __m512 weights=_mm512_loadu_ps(wOne.data()+k*HIDDEN_NODES);
            z0=_mm512_fmadd_ps(_mm512_set1_ps(x0[k]),weights,z0);z1=_mm512_fmadd_ps(_mm512_set1_ps(x1[k]),weights,z1);
            z2=_mm512_fmadd_ps(_mm512_set1_ps(x2[k]),weights,z2);z3=_mm512_fmadd_ps(_mm512_set1_ps(x3[k]),weights,z3);
        }
        _mm512_store_ps(hidden+(row+0)*HIDDEN_NODES,sigmoidVectorV12(z0));_mm512_store_ps(hidden+(row+1)*HIDDEN_NODES,sigmoidVectorV12(z1));
        _mm512_store_ps(hidden+(row+2)*HIDDEN_NODES,sigmoidVectorV12(z2));_mm512_store_ps(hidden+(row+3)*HIDDEN_NODES,sigmoidVectorV12(z3));
    }
    for(size_t row=0;row<BLOCK_SIZE;row++) {
        const __m512 activation=_mm512_load_ps(hidden+row*HIDDEN_NODES);
        output[row]=sigmoidScalarV12(_mm512_reduce_add_ps(_mm512_mul_ps(activation,wTwoVector)));
        const float error=output[row]-floatData.y[startRow+row];
        deltaThree[row]=error*output[row]*(1.0f-output[row]);
        if(CalculatePercentage) {
            const double actual=data.y[startRow+row],prediction=static_cast<double>(output[row]),doubleError=prediction-actual;
            const double percentage=actual==0.0?0.0:abs(doubleError/actual)*100.0;
            accumulator.percentageErrorSum+=percentage;
            if(detailed){accumulator.squaredError+=doubleError*doubleError;accumulator.maxPercentageError=max(accumulator.maxPercentageError,percentage);}
        }
    }
    __m512 gradientTwo=_mm512_setzero_ps(),gradientOne[NUMBER_OF_VARIABLES];
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) gradientOne[k]=_mm512_setzero_ps();
    for(size_t row=0;row<BLOCK_SIZE;row++) {
        const float*x=floatData.rowData(startRow+row);const __m512 activation=_mm512_load_ps(hidden+row*HIDDEN_NODES),delta=_mm512_set1_ps(deltaThree[row]);
        gradientTwo=_mm512_fmadd_ps(activation,delta,gradientTwo);
        const __m512 hiddenDelta=_mm512_mul_ps(_mm512_mul_ps(delta,wTwoVector),_mm512_mul_ps(activation,_mm512_sub_ps(one,activation)));
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) gradientOne[k]=_mm512_fmadd_ps(_mm512_set1_ps(x[k]),hiddenDelta,gradientOne[k]);
    }
    _mm512_storeu_ps(accumulator.dJdWtwo.data(),_mm512_add_ps(_mm512_loadu_ps(accumulator.dJdWtwo.data()),gradientTwo));
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) _mm512_storeu_ps(accumulator.dJdWone.data()+k*HIDDEN_NODES,_mm512_add_ps(_mm512_loadu_ps(accumulator.dJdWone.data()+k*HIDDEN_NODES),gradientOne[k]));
}

template<bool CalculatePercentage>
static void processTailV12(const Dataset &data,const V12FloatDataset &floatData,size_t startRow,size_t rowCount,
    const array<float,WONE_SIZE> &wOne,const array<float,HIDDEN_NODES> &wTwo,V12FloatAccumulator &accumulator,bool detailed) {
    for(size_t offset=0;offset<rowCount;offset++) {
        const size_t row=startRow+offset;const float*x=floatData.rowData(row);float hidden[HIDDEN_NODES];
        for(size_t j=0;j<HIDDEN_NODES;j++){float value=0.0f;for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)value=fmaf(x[k],wOne[k*HIDDEN_NODES+j],value);hidden[j]=sigmoidScalarV12(value);}
        float raw=0.0f;for(size_t j=0;j<HIDDEN_NODES;j++)raw=fmaf(hidden[j],wTwo[j],raw);const float prediction=sigmoidScalarV12(raw),error=prediction-floatData.y[row],deltaThree=error*prediction*(1.0f-prediction);
        if(CalculatePercentage){const double actual=data.y[row],doubleError=static_cast<double>(prediction)-actual,percentage=actual==0.0?0.0:abs(doubleError/actual)*100.0;accumulator.percentageErrorSum+=percentage;if(detailed){accumulator.squaredError+=doubleError*doubleError;accumulator.maxPercentageError=max(accumulator.maxPercentageError,percentage);}}
        for(size_t j=0;j<HIDDEN_NODES;j++){accumulator.dJdWtwo[j]+=hidden[j]*deltaThree;const float hiddenDelta=deltaThree*wTwo[j]*hidden[j]*(1.0f-hidden[j]);for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)accumulator.dJdWone[k*HIDDEN_NODES+j]+=x[k]*hiddenDelta;}
    }
}

static void processBlocksV12(const Dataset &data,const V12FloatDataset &floatData,size_t startRow,size_t batchSize,size_t firstBlock,size_t lastBlock,
    const array<float,WONE_SIZE>&wOne,const array<float,HIDDEN_NODES>&wTwo,V12FloatAccumulator &accumulator,bool detailed,double percentageLimit) {
    for(size_t block=firstBlock;block<lastBlock;block++) {
        const size_t offset=block*BLOCK_SIZE,rowsInBlock=min(BLOCK_SIZE,batchSize-offset);const bool calculatePercentage=detailed||accumulator.percentageErrorSum<=percentageLimit;
        if(rowsInBlock==BLOCK_SIZE){if(calculatePercentage)processFullTileV12<true>(data,floatData,startRow+offset,wOne,wTwo,accumulator,detailed);else processFullTileV12<false>(data,floatData,startRow+offset,wOne,wTwo,accumulator,false);}
        else {if(calculatePercentage)processTailV12<true>(data,floatData,startRow+offset,rowsInBlock,wOne,wTwo,accumulator,detailed);else processTailV12<false>(data,floatData,startRow+offset,rowsInBlock,wOne,wTwo,accumulator,false);}
    }
}

static Metrics combineV12(const vector<V12FloatAccumulator>&accumulators,Gradients&gradients,size_t rowCount,bool detailed) {
    gradients.clear();double squaredError=0.0,percentageErrorSum=0.0,maxPercentageError=0.0;
    for(const V12FloatAccumulator&a:accumulators){for(size_t i=0;i<WONE_SIZE;i++)gradients.dJdWone[i]+=static_cast<double>(a.dJdWone[i]);for(size_t j=0;j<HIDDEN_NODES;j++)gradients.dJdWtwo[j]+=static_cast<double>(a.dJdWtwo[j]);percentageErrorSum+=a.percentageErrorSum;if(detailed){squaredError+=a.squaredError;maxPercentageError=max(maxPercentageError,a.maxPercentageError);}}
    Metrics metrics;metrics.cost=.5*squaredError;metrics.percentageError=percentageErrorSum/static_cast<double>(rowCount);metrics.maxPercentageError=maxPercentageError;return metrics;
}

static bool finishV12Pass(const Dataset &data,size_t startRow,size_t batchSize,size_t batchIndex,Network &network,const OptimiserConfig &config,
    vector<V12FloatAccumulator>&accumulators,Gradients&gradients,TrainResult&result,bool detailed) {
    result.metrics=combineV12(accumulators,gradients,batchSize,detailed);bool targetReached=result.metrics.percentageError<=config.percentageErrorTarget;
    const bool updateLimitReached=result.updates>=config.maxDescents,logNow=config.logEvery!=0&&result.updates%config.logEvery==0;
    if(targetReached){result.metrics=calculateMetrics(data,startRow,batchSize,network);targetReached=result.metrics.percentageError<=config.percentageErrorTarget;}
    if(logNow||targetReached||updateLimitReached)printProgress(batchIndex,result.updates,result.metrics);
    if(targetReached||updateLimitReached){result.reachedTarget=targetReached;return true;}
    applyMomentumUpdate(network,gradients,config.learningRate,config.momentum);result.updates++;return false;
}

static TrainResult trainSingleRangeV12(const Dataset&data,const V12FloatDataset&floatData,size_t startRow,size_t batchSize,size_t batchIndex,Network&network,const OptimiserConfig&config) {
    Gradients gradients;TrainResult result;vector<V12FloatAccumulator>accumulators(1);array<float,WONE_SIZE>wOne;array<float,HIDDEN_NODES>wTwo;const double percentageLimit=config.percentageErrorTarget*static_cast<double>(batchSize);const size_t blockCount=(batchSize+BLOCK_SIZE-1)/BLOCK_SIZE;
    while(true){const bool detailed=(config.logEvery!=0&&result.updates%config.logEvery==0)||result.updates>=config.maxDescents;syncWeightsV12(network,wOne,wTwo);accumulators[0].clear();processBlocksV12(data,floatData,startRow,batchSize,0,blockCount,wOne,wTwo,accumulators[0],detailed,percentageLimit);if(finishV12Pass(data,startRow,batchSize,batchIndex,network,config,accumulators,gradients,result,detailed))break;}
    writePredictions(data,startRow,batchSize,network,"ybar.txt");return result;
}

#ifdef _OPENMP
static TrainResult trainParallelRangeV12(const Dataset&data,const V12FloatDataset&floatData,size_t startRow,size_t batchSize,size_t batchIndex,Network&network,const OptimiserConfig&config) {
    Gradients gradients;TrainResult result;vector<V12FloatAccumulator>accumulators(config.threads);array<float,WONE_SIZE>wOne;array<float,HIDDEN_NODES>wTwo;bool stop=false;const size_t blockCount=(batchSize+BLOCK_SIZE-1)/BLOCK_SIZE;const double percentageLimit=config.percentageErrorTarget*static_cast<double>(batchSize);
#pragma omp parallel num_threads(static_cast<int>(config.threads)) shared(stop,result,gradients,network,accumulators,wOne,wTwo)
    {const size_t thread=static_cast<size_t>(omp_get_thread_num()),baseCount=blockCount/config.threads,remainder=blockCount%config.threads,firstBlock=thread*baseCount+min(thread,remainder),ownedCount=baseCount+(thread<remainder?1:0),lastBlock=firstBlock+ownedCount;
        while(true){const bool detailed=(config.logEvery!=0&&result.updates%config.logEvery==0)||result.updates>=config.maxDescents;
#pragma omp single
            syncWeightsV12(network,wOne,wTwo);
            accumulators[thread].clear();processBlocksV12(data,floatData,startRow,batchSize,firstBlock,lastBlock,wOne,wTwo,accumulators[thread],detailed,percentageLimit);
#pragma omp barrier
#pragma omp single
            stop=finishV12Pass(data,startRow,batchSize,batchIndex,network,config,accumulators,gradients,result,detailed);
            if(stop)break;
        }
    }
    writePredictions(data,startRow,batchSize,network,"ybar.txt");return result;
}
#endif

static TrainResult trainRangeV12(const Dataset&data,const V12FloatDataset*floatData,size_t startRow,size_t batchSize,size_t batchIndex,Network&network,const OptimiserConfig&config,const V9FeatureBounds&bounds) {
    if(floatData==nullptr)return trainRangeV11(data,startRow,batchSize,batchIndex,network,config,bounds);
    if(config.threads==1)return trainSingleRangeV12(data,*floatData,startRow,batchSize,batchIndex,network,config);
#ifdef _OPENMP
    if(config.threads==4)return trainParallelRangeV12(data,*floatData,startRow,batchSize,batchIndex,network,config);
#endif
    return trainRangeV11(data,startRow,batchSize,batchIndex,network,config,bounds);
}
#endif

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
static bool useV12Path(size_t rows,const OptimiserConfig&config){return supportsV12Kernel()&&((config.threads==1&&rows>=V8_SINGLE_THRESHOLD)||(config.threads==4&&rows>=V9_PARALLEL_THRESHOLD));}
#endif

static void batchOnlineRunV12(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator) {
    const size_t totalRows=iterations*exampleSize;V9FeatureBounds bounds;Dataset data=loadDatasetV9(totalRows,bounds);Network network;if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    const bool accelerated=useV12Path(exampleSize,optimiser);V12FloatDataset floatData;if(accelerated)floatData=V12FloatDataset(data);
#endif
    for(size_t batch=0;batch<iterations;batch++){const size_t startRow=batch*exampleSize;
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
        const V12FloatDataset*floatPtr=accelerated?&floatData:nullptr;const TrainResult result=trainRangeV12(data,floatPtr,startRow,exampleSize,batch,network,optimiser,bounds);
#else
        const TrainResult result=trainRangeV11(data,startRow,exampleSize,batch,network,optimiser,bounds);
#endif
        writeWeights(network);if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of descents before the target.\n";}
}

static void offlineRunV12(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator) {
    V9FeatureBounds bounds;Dataset data=loadDatasetV9(rowCount,bounds);Network network;if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    V12FloatDataset floatData;const V12FloatDataset*floatPtr=nullptr;if(useV12Path(rowCount,optimiser)){floatData=V12FloatDataset(data);floatPtr=&floatData;}const TrainResult result=trainRangeV12(data,floatPtr,0,rowCount,0,network,optimiser,bounds);
#else
    const TrainResult result=trainRangeV11(data,0,rowCount,0,network,optimiser,bounds);
#endif
    writeWeights(network);if(!result.reachedTarget)cout<<"Offline training reached the maximum number of descents before the target.\n";
}

int main(){
    try{bool batchOnline=false,offline=false,test=true,randomiseWeights=false;double rangeWone=4.0,rangeWtwo=4.0,percentageErrorTarget=3.9,learningRate=0.0001,momentum=0.75;size_t iterations=1,exampleSize=13853,numberOfDescents=1000000,times=1,logEvery=1000,trainingThreads=1,offlineRows=10000,testRows=13853;
#ifndef _OPENMP
        if(trainingThreads>1)throw runtime_error("trainingThreads > 1 requires compiling with -fopenmp");
#endif
        mt19937 generator(static_cast<unsigned int>(time(nullptr)));const auto startTime=chrono::steady_clock::now();if(batchOnline)for(size_t i=0;i<times;i++){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.learningRate=learningRate;optimiser.momentum=momentum;batchOnlineRunV12(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);numberOfDescents+=numberOfDescents;}if(offline){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.learningRate=1.0;optimiser.momentum=0.0;offlineRunV12(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);}if(test)testRun(testRows);const chrono::duration<double>elapsed=chrono::steady_clock::now()-startTime;cout<<"Elapsed time = "<<elapsed.count()<<" seconds\n";return 0;}
    catch(const exception&error){cerr<<"Error: "<<error.what()<<'\n';return 1;}
}
