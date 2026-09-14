#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define main v26_embedded_main
#include "../v26/main.cpp"
#undef main
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V27: accelerate sub-50k Gauss-Newton refreshes without changing their floating-point result. */
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
constexpr size_t V27_GN_BLOCK_ROWS=8192;
constexpr size_t V27_GN_PARALLEL_ROW_LIMIT=50000;

/**
 * Compute one row's 192 Gauss-Newton Jacobian components exactly as v20.
 *
 * Each AVX-512 lane represents one hidden node. The input loop remains
 * k=0..10, multiplication and addition remain separate, and scalar exp is
 * retained. The stored doubles therefore match v20 before its long-double
 * squaring and accumulation.
 */
__attribute__((target("avx512f,avx512dq"),optimize("fp-contract=off")))
static void gaussNewtonRowV27(const double*x,const Network&network,double*out){
    alignas(64) double raw[HIDDEN_NODES],hidden[HIDDEN_NODES];
    __m512d low=_mm512_setzero_pd(),high=_mm512_setzero_pd();
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
        const __m512d value=_mm512_set1_pd(x[k]);
        low=_mm512_add_pd(low,_mm512_mul_pd(value,_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES)));
        high=_mm512_add_pd(high,_mm512_mul_pd(value,_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES+8)));
    }
    _mm512_store_pd(raw,low);
    _mm512_store_pd(raw+8,high);
    for(size_t j=0;j<HIDDEN_NODES;j++)hidden[j]=sigmoidV15(raw[j]);

    double rawOutput=0.0;
    for(size_t j=0;j<HIDDEN_NODES;j++)rawOutput+=hidden[j]*network.wTwo[j];
    const double prediction=sigmoidV15(rawOutput);
    const double outputDerivative=prediction*(1.0-prediction);

    const __m512d one=_mm512_set1_pd(1.0);
    const __m512d derivative=_mm512_set1_pd(outputDerivative);
    const __m512d hiddenLow=_mm512_load_pd(hidden);
    const __m512d hiddenHigh=_mm512_load_pd(hidden+8);
    const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
    const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);

    _mm512_storeu_pd(out+WONE_SIZE,_mm512_mul_pd(derivative,hiddenLow));
    _mm512_storeu_pd(out+WONE_SIZE+8,_mm512_mul_pd(derivative,hiddenHigh));

    const __m512d baseLow=_mm512_mul_pd(
        _mm512_mul_pd(_mm512_mul_pd(derivative,wTwoLow),hiddenLow),
        _mm512_sub_pd(one,hiddenLow));
    const __m512d baseHigh=_mm512_mul_pd(
        _mm512_mul_pd(_mm512_mul_pd(derivative,wTwoHigh),hiddenHigh),
        _mm512_sub_pd(one,hiddenHigh));
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
        const __m512d value=_mm512_set1_pd(x[k]);
        _mm512_storeu_pd(out+k*HIDDEN_NODES,_mm512_mul_pd(baseLow,value));
        _mm512_storeu_pd(out+k*HIDDEN_NODES+8,_mm512_mul_pd(baseHigh,value));
    }
}

/** Prepare one bounded block in parallel, then reduce rows in v20's exact order. */
static void accumulateGaussNewtonBlockV27(const Dataset&data,size_t startRow,size_t count,
                                          const Network&network,size_t threads,
                                          vector<double>&scratch,V20DiagonalPartial&total){
#ifdef _OPENMP
#pragma omp parallel for num_threads(static_cast<int>(threads)) schedule(static)
#endif
    for(long long rr=0;rr<static_cast<long long>(count);rr++){
        const size_t row=startRow+static_cast<size_t>(rr);
        gaussNewtonRowV27(data.x.rowData(row),network,
                          scratch.data()+static_cast<size_t>(rr)*V14_PARAMETER_COUNT);
    }
    for(size_t rr=0;rr<count;rr++){
        const double*row=scratch.data()+rr*V14_PARAMETER_COUNT;
        for(size_t j=0;j<HIDDEN_NODES;j++){
            const double outputJacobian=row[WONE_SIZE+j];
            total.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
                const double jacobian=row[k*HIDDEN_NODES+j];
                total.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
            }
        }
    }
}

/** Convert an exactly accumulated v20 diagonal into the inherited scale. */
static V20DiagonalScale finishGaussNewtonScaleV27(const V20DiagonalPartial&total,size_t batchSize){
    V20DiagonalScale diagonal{},scale{};
    vector<double>ordered(V14_PARAMETER_COUNT);
    const long double inv=1.0L/static_cast<long double>(batchSize);
    for(size_t i=0;i<V14_PARAMETER_COUNT;i++){
        diagonal[i]=static_cast<double>(total.values[i]*inv);
        ordered[i]=diagonal[i];
    }
    nth_element(ordered.begin(),ordered.begin()+V14_PARAMETER_COUNT/2,ordered.end());
    const double reference=ordered[V14_PARAMETER_COUNT/2];
    if(!(reference>0.0)||!isfinite(reference)){
        scale.fill(1.0);
        return scale;
    }
    const double floorValue=reference*V20_GN_FLOOR_RATIO;
    const double minimumScale=1.0/V20_GN_CLIP;
    for(size_t i=0;i<V14_PARAMETER_COUNT;i++){
        const double safeDiagonal=max(diagonal[i],floorValue);
        const double rawScale=pow(reference/safeDiagonal,V20_GN_EXPONENT);
        scale[i]=min(V20_GN_CLIP,max(minimumScale,rawScale));
    }
    return scale;
}

/**
 * Build v20's Gauss-Newton scale with parallel row preparation and exact reduction.
 * Scratch is bounded to 8,192 rows (about 12.6 MB for 192 doubles per row).
 */
static V20DiagonalScale gaussNewtonScaleV27(const Dataset&data,size_t startRow,size_t batchSize,
                                            Network&network,const vector<double>&parameters,
                                            size_t requestedThreads){
    unpackV14(parameters,network);
    const size_t threads=evaluationThreadsV25(requestedThreads,batchSize);
    const size_t scratchRows=min(V27_GN_BLOCK_ROWS,batchSize);
    vector<double>scratch(scratchRows*V14_PARAMETER_COUNT);
    V20DiagonalPartial total;
    for(size_t offset=0;offset<batchSize;offset+=V27_GN_BLOCK_ROWS){
        const size_t count=min(V27_GN_BLOCK_ROWS,batchSize-offset);
        accumulateGaussNewtonBlockV27(data,startRow+offset,count,network,threads,scratch,total);
    }
    return finishGaussNewtonScaleV27(total,batchSize);
}

/** Refresh the inherited target-aware scale using the exact v27 implementation. */
static void refreshScaleV27(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                            const vector<double>&parameters,const OptimiserConfig&config,
                            double percentageError,unsigned&completedStage,V20DiagonalScale&scale){
    const unsigned requiredStage=refreshStageV21(percentageError,config.percentageErrorTarget);
    if(requiredStage<=completedStage)return;
    scale=gaussNewtonScaleV27(data,startRow,batchSize,network,parameters,config.threads);
    completedStage=requiredStage;
}

/** Prepare deep-stage state in the same order as v26. */
static void prepareDeepStateV27(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                                const vector<double>&parameters,const OptimiserConfig&config,
                                double percentageError,deque<V14HistoryPair>&history,bool&deepStage,
                                bool&scaleReady,bool&denseReady,unsigned&refreshStage,
                                V20DiagonalScale&scale,vector<double>&denseInverse){
    deepStage=deepStageV19(deepStage,percentageError);
    if(deepStage&&!scaleReady){
        scale=gaussNewtonScaleV27(data,startRow,batchSize,network,parameters,config.threads);
        scaleReady=true;
        refreshStage=refreshStageV21(percentageError,config.percentageErrorTarget);
    }
    if(scaleReady&&!denseReady){
        refreshScaleV27(data,startRow,batchSize,network,parameters,config,
                        percentageError,refreshStage,scale);
        if(percentageError<=V22_DENSE_SWITCH_PERCENTAGE){
            initialiseDenseInverseV23(history,scale,denseInverse);
            denseReady=true;
        }
    }
}

/** Build v26's direction while retaining its exact dense fallbacks. */
static void buildDirectionV27(const vector<double>&gradient,deque<V14HistoryPair>&history,
                              const V20DiagonalScale&scale,bool scaleReady,bool denseReady,
                              vector<double>&denseInverse,vector<double>&direction,double&directional){
    if(!denseReady){
        descentDirectionV20(gradient,history,scaleReady?&scale:nullptr,direction,directional);
        return;
    }
    denseDirectionV26(gradient,denseInverse,direction,directional);
    if(!(directional<0.0)||!isfinite(directional)){
        initialiseDenseInverseV23(history,scale,denseInverse);
        denseDirectionV26(gradient,denseInverse,direction,directional);
    }
    if(!(directional<0.0)||!isfinite(directional)){
        direction=gradient;
        for(double&value:direction)value=-value;
        directional=-dotV14(gradient,gradient);
    }
}

/** Apply v26's accepted-step curvature update. */
static void updateCurvatureV27(const vector<double>&parameters,const vector<double>&candidate,
                               const vector<double>&gradient,const vector<double>&nextGradient,
                               size_t historyLimit,bool denseReady,deque<V14HistoryPair>&history,
                               vector<double>&denseInverse){
    if(denseReady)
        updateDenseInverseV26(parameters,candidate,gradient,nextGradient,denseInverse);
    else
        updateHistoryV19(parameters,candidate,gradient,nextGradient,historyLimit,history);
}
#endif

/** Preserve v26 completely except for exact sub-50k Gauss-Newton refresh preparation. */
static TrainResult trainRangeV27(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(config.percentageErrorTarget>V21_REFRESH_TARGET||batchSize>=V27_GN_PARALLEL_ROW_LIMIT||!supportsV16Kernel())
        return trainRangeV26(data,startRow,batchSize,batchIndex,network,config);

    TrainResult result;
    deque<V14HistoryPair>history;
    vector<double>parameters=packV14(network);
    V14Evaluation current=evaluateV25(data,startRow,batchSize,network,parameters,config.threads);
    bool deepStage=false,scaleReady=false,denseReady=false;
    unsigned refreshStage=0;
    V20DiagonalScale scale{};
    vector<double>denseInverse;
    if(config.logEvery!=0)printProgress(batchIndex,0,current.metrics);

    while(current.metrics.percentageError>config.percentageErrorTarget&&result.updates<config.maxDescents){
        prepareDeepStateV27(data,startRow,batchSize,network,parameters,config,
                            current.metrics.percentageError,history,deepStage,scaleReady,denseReady,
                            refreshStage,scale,denseInverse);
        const size_t historyLimit=historyLimitV19(deepStage);
        vector<double>direction;
        double directional=0.0;
        buildDirectionV27(current.gradient,history,scale,scaleReady,denseReady,
                          denseInverse,direction,directional);

        vector<double>candidate(parameters.size());
        V14Evaluation next;
        const bool accepted=denseReady
            ?armijoDenseStepV25(data,startRow,batchSize,network,config,parameters,direction,directional,
                                current.objective,candidate,next)
            :armijoStepV25(data,startRow,batchSize,network,config,parameters,direction,directional,
                           current.objective,candidate,next);
        if(!accepted){
            unpackV14(parameters,network);
            break;
        }

        updateCurvatureV27(parameters,candidate,current.gradient,next.gradient,historyLimit,
                           denseReady,history,denseInverse);
        parameters.swap(candidate);
        current=std::move(next);
        result.updates++;
        if(config.logEvery!=0&&result.updates%config.logEvery==0)
            printProgress(batchIndex,result.updates,current.metrics);
    }

    unpackV14(parameters,network);
    network.deltaWone.fill(0.0);
    network.deltaWtwo.fill(0.0);
    result.metrics=confirmMetricsV25(data,startRow,batchSize,network,config.threads);
    result.reachedTarget=result.metrics.percentageError<=config.percentageErrorTarget;
    if(config.logEvery!=0||result.reachedTarget)printProgress(batchIndex,result.updates,result.metrics);
    writePredictions(data,startRow,batchSize,network,"ybar.txt");
    return result;
#else
    return trainRangeV26(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV27(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV27(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of BFGS iterations before the target.\n";
    }
}

static void offlineRunV27(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV27(data,0,rowCount,0,network,optimiser);
    writeWeights(network);
    if(!result.reachedTarget)cout<<"Offline BFGS reached the maximum number of iterations before the target.\n";
}

int main(){
    try{
        bool batchOnline=false,offline=false,test=true,randomiseWeights=false;
        double rangeWone=4.0,rangeWtwo=4.0,percentageErrorTarget=3.9;
        size_t iterations=1,exampleSize=13853,numberOfDescents=1000000,times=1,logEvery=1,trainingThreads=4,offlineRows=10000,testRows=13853;
#ifndef _OPENMP
        trainingThreads=1;
#endif
        mt19937 generator(static_cast<unsigned int>(time(nullptr)));
        const auto startTime=chrono::steady_clock::now();
        if(batchOnline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            batchOnlineRunV27(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV27(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(test)for(size_t i=0;i<times;i++)testRun(testRows);
        const auto endTime=chrono::steady_clock::now();
        cout<<"Elapsed seconds: "<<chrono::duration<double>(endTime-startTime).count()<<'\n';
        return 0;
    }catch(const exception&error){
        cerr<<"Error: "<<error.what()<<'\n';
        return 1;
    }
}
