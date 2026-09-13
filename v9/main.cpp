#define main v8_reference_main
#define trainRange trainRangeV8
#define batchOnlineRun batchOnlineRunV8
#define offlineRun offlineRunV8
#include "../v8/main.cpp"
#undef offlineRun
#undef batchOnlineRun
#undef trainRange
#undef main

/*
 * Simple Neural Network - v9
 * ==========================
 *
 * V9 layers a specialised exact AVX-512 training path over the frozen v8
 * implementation. All unsupported CPUs, small batches and unsafe sigmoid
 * ranges delegate to the v8 arithmetic. Complete 16-row tiles eliminate tail
 * control, and the libmvec exp path omits the per-vector -700 guard only after
 * a conservative pass-level bound proves that guard cannot trigger.
 */

constexpr size_t V9_PARALLEL_THRESHOLD = 1000000;
constexpr long double V9_SIGMOID_SAFE_BOUND = 699.0L;

struct V9FeatureBounds {
    array<double,NUMBER_OF_VARIABLES> maxAbsX{};
};

static Dataset loadDatasetV9(size_t rows, V9FeatureBounds &bounds) {
    ifstream input("InputVariables.txt");
    if(!input) throw runtime_error("Failed to open file: InputVariables.txt");
    Dataset data;
    data.x = Matrix(rows,NUMBER_OF_VARIABLES);
    for(size_t row=0;row<rows;row++) {
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            double value=0.0;
            if(!(input >> value)) throw runtime_error("Insufficient or invalid data in file: InputVariables.txt");
            data.x.data[row*NUMBER_OF_VARIABLES+k]=value;
            bounds.maxAbsX[k]=max(bounds.maxAbsX[k],abs(value));
        }
    }
    data.y=readVector("OutputVariables.txt",rows);
    return data;
}

static bool supportsV9Kernel() {
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    return supportsV8Kernel();
#else
    return false;
#endif
}

static bool v9UncheckedSigmoidSafe(const V9FeatureBounds &bounds, const Network &network) {
    for(size_t j=0;j<HIDDEN_NODES;j++) {
        long double magnitudeBound=0.0L;
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            magnitudeBound+=static_cast<long double>(bounds.maxAbsX[k]) * abs(static_cast<long double>(network.wOne[k*HIDDEN_NODES+j]));
        }
        if(!(magnitudeBound < V9_SIGMOID_SAFE_BOUND)) return false;
    }
    long double outputLowerBound=0.0L;
    for(size_t j=0;j<HIDDEN_NODES;j++) if(network.wTwo[j] < 0.0) outputLowerBound+=static_cast<long double>(network.wTwo[j]);
    return outputLowerBound > -V9_SIGMOID_SAFE_BOUND;
}

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx512f"),always_inline)) static inline void sigmoidVectorAVX512UncheckedV9(double *values, size_t count) {
    const __m512d one=_mm512_set1_pd(1.0),zero=_mm512_setzero_pd();
    for(size_t i=0;i<count;i+=8) {
        const __m512d x=_mm512_loadu_pd(values+i);
        const __m512d exponent=_ZGVeN8v_exp(_mm512_sub_pd(zero,x));
        _mm512_storeu_pd(values+i,_mm512_div_pd(one,_mm512_add_pd(one,exponent)));
    }
}
#endif

template<bool Unchecked>
__attribute__((target("avx512f"),always_inline)) static inline void sigmoidVectorFullV9(double *values, size_t count) {
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(Unchecked) sigmoidVectorAVX512UncheckedV9(values,count);
    else sigmoidVectorAVX512(values,count);
#else
    (void)Unchecked;
    sigmoidVectorCached(values,count);
#endif
}

#if defined(__x86_64__) && defined(__GNUC__)
#define V9_DECLARE_W1_ACCUMULATORS \
    __m512d g00=_mm512_setzero_pd(),g01=_mm512_setzero_pd(),g10=_mm512_setzero_pd(),g11=_mm512_setzero_pd(),g20w=_mm512_setzero_pd(),g21w=_mm512_setzero_pd(),g30=_mm512_setzero_pd(),g31=_mm512_setzero_pd(),g40=_mm512_setzero_pd(),g41=_mm512_setzero_pd(),g50=_mm512_setzero_pd(),g51=_mm512_setzero_pd(),g60=_mm512_setzero_pd(),g61=_mm512_setzero_pd(),g70=_mm512_setzero_pd(),g71=_mm512_setzero_pd(),g80=_mm512_setzero_pd(),g81=_mm512_setzero_pd(),g90=_mm512_setzero_pd(),g91=_mm512_setzero_pd(),g100=_mm512_setzero_pd(),g101=_mm512_setzero_pd()
#define V9_ACCUMULATE_W1(K,G0,G1) do { const __m512d xv=_mm512_set1_pd(x[K]); G0=_mm512_fmadd_pd(xv,delta0,G0); G1=_mm512_fmadd_pd(xv,delta1,G1); } while(false)
#define V9_STORE_W1(K,G0,G1) do { double*g=accumulator.dJdWone.data()+(K)*HIDDEN_NODES; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),G0)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),G1)); } while(false)

template<bool CalculatePercentage>
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void computeOutputDeltasV9(const Dataset &data, size_t startRow, array<double,BLOCK_SIZE> &output, array<double,BLOCK_SIZE> &deltaThree, array<double,BLOCK_SIZE> &percentage, ThreadAccumulator &accumulator, bool detailed) {
    const __m512d one=_mm512_set1_pd(1.0),hundred=_mm512_set1_pd(100.0),signMask=_mm512_set1_pd(-0.0);
    if(CalculatePercentage) {
        for(size_t i=0;i<BLOCK_SIZE;i+=8) {
            const __m512d prediction=_mm512_loadu_pd(output.data()+i),actual=_mm512_loadu_pd(data.y.data()+startRow+i),error=_mm512_sub_pd(prediction,actual);
            const __m512d absoluteError=_mm512_andnot_pd(signMask,error);
            __m512d p=_mm512_mul_pd(_mm512_div_pd(absoluteError,actual),hundred);
            const __mmask8 zeroActual=_mm512_cmp_pd_mask(actual,_mm512_setzero_pd(),_CMP_EQ_OQ);
            p=_mm512_mask_mov_pd(p,zeroActual,_mm512_setzero_pd());
            _mm512_storeu_pd(percentage.data()+i,p);
            _mm512_storeu_pd(deltaThree.data()+i,_mm512_mul_pd(_mm512_mul_pd(error,prediction),_mm512_sub_pd(one,prediction)));
        }
        for(size_t q=0;q<BLOCK_SIZE;q++) {
            accumulator.percentageErrorSum+=percentage[q];
            if(detailed) { const double error=output[q]-data.y[startRow+q]; accumulator.squaredError+=error*error; accumulator.maxPercentageError=max(accumulator.maxPercentageError,percentage[q]); }
        }
    }
    else {
        for(size_t i=0;i<BLOCK_SIZE;i+=8) {
            const __m512d prediction=_mm512_loadu_pd(output.data()+i),actual=_mm512_loadu_pd(data.y.data()+startRow+i),error=_mm512_sub_pd(prediction,actual);
            _mm512_storeu_pd(deltaThree.data()+i,_mm512_mul_pd(_mm512_mul_pd(error,prediction),_mm512_sub_pd(one,prediction)));
        }
    }
}

template<bool CalculatePercentage, bool UncheckedSigmoid>
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void processTrainingFullTileAVX512V9(const Dataset &data, size_t startRow, const Network &network, ThreadAccumulator &accumulator, bool detailed) {
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
        _mm512_storeu_pd(h0,a0);_mm512_storeu_pd(h0+8,a1);_mm512_storeu_pd(h1,b0);_mm512_storeu_pd(h1+8,b1);
        _mm512_storeu_pd(h2,c0);_mm512_storeu_pd(h2+8,c1);_mm512_storeu_pd(h3,d0);_mm512_storeu_pd(h3+8,d1);
    }
    sigmoidVectorFullV9<UncheckedSigmoid>(hidden.data(),BLOCK_SIZE*HIDDEN_NODES);

    const __m512d w20=_mm512_loadu_pd(network.wTwo.data()),w21=_mm512_loadu_pd(network.wTwo.data()+8);
    for(size_t i=0;i<BLOCK_SIZE;i++) {
        const double*h=hidden.data()+i*HIDDEN_NODES;
        const __m512d sum=_mm512_fmadd_pd(_mm512_loadu_pd(h+8),w21,_mm512_mul_pd(_mm512_loadu_pd(h),w20));
        output[i]=_mm512_reduce_add_pd(sum);
    }
    sigmoidVectorFullV9<UncheckedSigmoid>(output.data(),BLOCK_SIZE);

    computeOutputDeltasV9<CalculatePercentage>(data,startRow,output,deltaThree,percentage,accumulator,detailed);

    __m512d g20=_mm512_loadu_pd(accumulator.dJdWtwo.data()),g21=_mm512_loadu_pd(accumulator.dJdWtwo.data()+8);V9_DECLARE_W1_ACCUMULATORS;
    for(size_t q=0;q<BLOCK_SIZE;q++) {
        const double*h=hidden.data()+q*HIDDEN_NODES;const __m512d d=_mm512_set1_pd(deltaThree[q]),a0=_mm512_loadu_pd(h),a1=_mm512_loadu_pd(h+8);
        g20=_mm512_fmadd_pd(a0,d,g20);g21=_mm512_fmadd_pd(a1,d,g21);
        const __m512d delta0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0))),delta1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));
        const double*x=data.x.rowData(startRow+q);
        V9_ACCUMULATE_W1(0,g00,g01);V9_ACCUMULATE_W1(1,g10,g11);V9_ACCUMULATE_W1(2,g20w,g21w);V9_ACCUMULATE_W1(3,g30,g31);V9_ACCUMULATE_W1(4,g40,g41);V9_ACCUMULATE_W1(5,g50,g51);
        V9_ACCUMULATE_W1(6,g60,g61);V9_ACCUMULATE_W1(7,g70,g71);V9_ACCUMULATE_W1(8,g80,g81);V9_ACCUMULATE_W1(9,g90,g91);V9_ACCUMULATE_W1(10,g100,g101);
    }
    _mm512_storeu_pd(accumulator.dJdWtwo.data(),g20);_mm512_storeu_pd(accumulator.dJdWtwo.data()+8,g21);
    V9_STORE_W1(0,g00,g01);V9_STORE_W1(1,g10,g11);V9_STORE_W1(2,g20w,g21w);V9_STORE_W1(3,g30,g31);V9_STORE_W1(4,g40,g41);V9_STORE_W1(5,g50,g51);
    V9_STORE_W1(6,g60,g61);V9_STORE_W1(7,g70,g71);V9_STORE_W1(8,g80,g81);V9_STORE_W1(9,g90,g91);V9_STORE_W1(10,g100,g101);
}

template<bool UncheckedSigmoid>
__attribute__((target("avx512f,avx512dq,fma"))) static void processTrainingRangeAVX512V9(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, ThreadAccumulator &accumulator, bool detailed, double percentageLimit) {
    size_t offset=0;
    for(;offset+BLOCK_SIZE<=rowCount;offset+=BLOCK_SIZE) {
        if(detailed || accumulator.percentageErrorSum<=percentageLimit) processTrainingFullTileAVX512V9<true,UncheckedSigmoid>(data,startRow+offset,network,accumulator,detailed);
        else processTrainingFullTileAVX512V9<false,UncheckedSigmoid>(data,startRow+offset,network,accumulator,false);
    }
    if(offset<rowCount) {
        const size_t tail=rowCount-offset;
        if(detailed || accumulator.percentageErrorSum<=percentageLimit) processTrainingTileAVX512V8<true>(data,startRow+offset,tail,network,accumulator,detailed);
        else processTrainingTileAVX512V8<false>(data,startRow+offset,tail,network,accumulator,false);
    }
}

template<bool UncheckedSigmoid>
__attribute__((target("avx512f,avx512dq,fma"))) static void processTrainingBlocksAVX512V9(const Dataset &data, size_t startRow, size_t batchSize, size_t firstBlock, size_t lastBlock, const Network &network, ThreadAccumulator &accumulator, bool detailed) {
    for(size_t block=firstBlock;block<lastBlock;block++) {
        const size_t offset=block*BLOCK_SIZE,rowsInBlock=min(BLOCK_SIZE,batchSize-offset);
        if(rowsInBlock==BLOCK_SIZE) processTrainingFullTileAVX512V9<true,UncheckedSigmoid>(data,startRow+offset,network,accumulator,detailed);
        else processTrainingTileAVX512V8<true>(data,startRow+offset,rowsInBlock,network,accumulator,detailed);
    }
}
#undef V9_DECLARE_W1_ACCUMULATORS
#undef V9_ACCUMULATE_W1
#undef V9_STORE_W1
#endif

static bool finishV9Pass(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config, vector<ThreadAccumulator> &accumulators, Gradients &gradients, TrainResult &result, bool detailed) {
    result.metrics=combineAccumulators(accumulators,gradients,batchSize,detailed);
    const bool targetReached=result.metrics.percentageError<=config.percentageErrorTarget;
    const bool updateLimitReached=result.updates>=config.maxDescents;
    const bool logNow=config.logEvery!=0 && result.updates%config.logEvery==0;
    if(targetReached && !detailed) result.metrics=calculateMetrics(data,startRow,batchSize,network);
    if(logNow || targetReached || updateLimitReached) printProgress(batchIndex,result.updates,result.metrics);
    if(targetReached || updateLimitReached) {
        result.reachedTarget=targetReached;
        return true;
    }
    applyMomentumUpdate(network,gradients,config.learningRate,config.momentum);
    result.updates++;
    return false;
}

static TrainResult trainSingleRangeV9(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config, const V9FeatureBounds &bounds) {
    Gradients gradients;
    TrainResult result;
    vector<ThreadAccumulator> accumulators(1);
    ThreadAccumulator &accumulator=accumulators[0];
    const double percentageLimit=config.percentageErrorTarget*static_cast<double>(batchSize);
    while(true) {
        const bool detailed=(config.logEvery!=0 && result.updates%config.logEvery==0) || result.updates>=config.maxDescents;
        accumulator.clear();
        if(v9UncheckedSigmoidSafe(bounds,network)) processTrainingRangeAVX512V9<true>(data,startRow,batchSize,network,accumulator,detailed,percentageLimit);
        else processTrainingRangeAVX512V9<false>(data,startRow,batchSize,network,accumulator,detailed,percentageLimit);
        if(finishV9Pass(data,startRow,batchSize,batchIndex,network,config,accumulators,gradients,result,detailed)) break;
    }
    writePredictions(data,startRow,batchSize,network,"ybar.txt");
    return result;
}

#ifdef _OPENMP
static TrainResult trainParallelRangeV9(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config, const V9FeatureBounds &bounds) {
    Gradients gradients;
    TrainResult result;
    vector<ThreadAccumulator> accumulators(config.threads);
    bool stop=false;
    const size_t blockCount=(batchSize+BLOCK_SIZE-1)/BLOCK_SIZE;
#pragma omp parallel num_threads(static_cast<int>(config.threads)) shared(stop,result,gradients,network,accumulators)
    {
        const size_t thread=static_cast<size_t>(omp_get_thread_num());
        const size_t baseCount=blockCount/config.threads,remainder=blockCount%config.threads;
        const size_t firstBlock=thread*baseCount+min(thread,remainder),ownedCount=baseCount+(thread<remainder?1:0),lastBlock=firstBlock+ownedCount;
        while(true) {
            const bool detailed=(config.logEvery!=0 && result.updates%config.logEvery==0) || result.updates>=config.maxDescents;
            accumulators[thread].clear();
            if(v9UncheckedSigmoidSafe(bounds,network)) processTrainingBlocksAVX512V9<true>(data,startRow,batchSize,firstBlock,lastBlock,network,accumulators[thread],detailed);
            else processTrainingBlocksAVX512V9<false>(data,startRow,batchSize,firstBlock,lastBlock,network,accumulators[thread],detailed);
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

static TrainResult trainRangeV9(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config, const V9FeatureBounds &bounds) {
#if !defined(__x86_64__) || !defined(__GNUC__)
    (void)bounds;
    return trainRangeV8(data,startRow,batchSize,batchIndex,network,config);
#else
    const bool useSingleV9=config.threads==1 && batchSize>=V8_SINGLE_THRESHOLD && supportsV9Kernel();
    const bool useParallelV9=config.threads>1 && batchSize>=V9_PARALLEL_THRESHOLD && supportsV9Kernel();
    if(!useSingleV9 && !useParallelV9) return trainRangeV8(data,startRow,batchSize,batchIndex,network,config);
#ifdef _OPENMP
    if(useParallelV9) return trainParallelRangeV9(data,startRow,batchSize,batchIndex,network,config,bounds);
#endif
    return trainSingleRangeV9(data,startRow,batchSize,batchIndex,network,config,bounds);
#endif
}

static void batchOnlineRunV9(size_t iterations, size_t exampleSize, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator) {
    const size_t totalRows=iterations*exampleSize;V9FeatureBounds bounds;Dataset data=loadDatasetV9(totalRows,bounds);Network network;
    if(randomiseWeights) setmatrixrandom(network,rangeWone,rangeWtwo,generator); else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++) { const size_t startRow=batch*exampleSize;const TrainResult result=trainRangeV9(data,startRow,exampleSize,batch,network,optimiser,bounds);writeWeights(network);if(!result.reachedTarget) cout << "Batch " << batch << " reached the maximum number of descents before the target.\n"; }
}

static void offlineRunV9(size_t rowCount, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator) {
    V9FeatureBounds bounds;Dataset data=loadDatasetV9(rowCount,bounds);Network network;if(randomiseWeights) setmatrixrandom(network,rangeWone,rangeWtwo,generator); else loadWeights(network);
    const TrainResult result=trainRangeV9(data,0,rowCount,0,network,optimiser,bounds);writeWeights(network);if(!result.reachedTarget) cout << "Offline training reached the maximum number of descents before the target.\n";
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
        if(batchOnline) for(size_t i=0;i<times;i++) { OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.learningRate=learningRate;optimiser.momentum=momentum;batchOnlineRunV9(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);numberOfDescents+=numberOfDescents; }
        if(offline) { OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.learningRate=1.0;optimiser.momentum=0.0;offlineRunV9(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator); }
        if(test) testRun(testRows);
        const chrono::duration<double> elapsed=chrono::steady_clock::now()-startTime;cout << "Elapsed time = " << elapsed.count() << " seconds\n";return 0;
    }
    catch(const exception &error) { cerr << "Error: " << error.what() << '\n';return 1; }
}
