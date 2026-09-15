#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define SIMPLE_NN_V29_NO_MAIN
#include "../v29/main.cpp"
#undef SIMPLE_NN_V29_NO_MAIN
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V30 candidate: reduce exact GN intermediate traffic and reuse final scalar predictions. */
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
constexpr size_t V30_GN_BASES_PER_ROW=2*HIDDEN_NODES;
constexpr size_t V30_GN_BLOCK_ROWS=8192;

/**
 * Prepare the 16 output Jacobians and 16 hidden Jacobian bases exactly as v27.
 * Input-specific Jacobians are reconstructed during the inherited ordered
 * long-double reduction, cutting scratch from 192 to 32 doubles per row.
 */
__attribute__((target("avx512f,avx512dq"),optimize("fp-contract=off")))
static void gaussNewtonBasesRowV30(const double*x,const Network&network,double*out){
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

    _mm512_storeu_pd(out,_mm512_mul_pd(derivative,hiddenLow));
    _mm512_storeu_pd(out+8,_mm512_mul_pd(derivative,hiddenHigh));
    _mm512_storeu_pd(out+HIDDEN_NODES,_mm512_mul_pd(
        _mm512_mul_pd(_mm512_mul_pd(derivative,wTwoLow),hiddenLow),
        _mm512_sub_pd(one,hiddenLow)));
    _mm512_storeu_pd(out+HIDDEN_NODES+8,_mm512_mul_pd(
        _mm512_mul_pd(_mm512_mul_pd(derivative,wTwoHigh),hiddenHigh),
        _mm512_sub_pd(one,hiddenHigh)));
}

/** Accumulate one row from exact stored bases in v20's original operation order. */
static inline void accumulateGaussNewtonBasesRowV30(const double*x,const double*bases,
                                                    V20DiagonalPartial&total){
    for(size_t j=0;j<HIDDEN_NODES;j++){
        const double outputJacobian=bases[j];
        total.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
        const double hiddenJacobianBase=bases[HIDDEN_NODES+j];
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            const double jacobian=hiddenJacobianBase*x[k];
            total.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
        }
    }
}

/** Exact sub-50k GN scale with six-times smaller row scratch than v27. */
static V20DiagonalScale gaussNewtonScaleReducedScratchV30(
    const Dataset&data,size_t startRow,size_t batchSize,Network&network,
    const vector<double>&parameters,size_t requestedThreads){
    unpackV14(parameters,network);
    const size_t threads=evaluationThreadsV25(requestedThreads,batchSize);
    const size_t scratchRows=min(V30_GN_BLOCK_ROWS,batchSize);
    vector<double>scratch(scratchRows*V30_GN_BASES_PER_ROW);
    V20DiagonalPartial total;
    for(size_t offset=0;offset<batchSize;offset+=V30_GN_BLOCK_ROWS){
        const size_t count=min(V30_GN_BLOCK_ROWS,batchSize-offset);
#ifdef _OPENMP
#pragma omp parallel for num_threads(static_cast<int>(threads)) schedule(static)
#endif
        for(long long rr=0;rr<static_cast<long long>(count);rr++){
            const size_t local=static_cast<size_t>(rr);
            gaussNewtonBasesRowV30(data.x.rowData(startRow+offset+local),network,
                                  scratch.data()+local*V30_GN_BASES_PER_ROW);
        }
        for(size_t rr=0;rr<count;rr++)
            accumulateGaussNewtonBasesRowV30(
                data.x.rowData(startRow+offset+rr),
                scratch.data()+rr*V30_GN_BASES_PER_ROW,total);
    }
    return finishGaussNewtonScaleV27(total,batchSize);
}

/** Exact SIMD row calculation inside one inherited v20 thread slice. */
static void accumulateGaussNewtonSliceSimdV30(const Dataset&data,size_t firstRow,size_t lastRow,
                                              const Network&network,V20DiagonalPartial&out){
    alignas(64) double bases[V30_GN_BASES_PER_ROW];
    for(size_t row=firstRow;row<lastRow;row++){
        const double*x=data.x.rowData(row);
        gaussNewtonBasesRowV30(x,network,bases);
        accumulateGaussNewtonBasesRowV30(x,bases,out);
    }
}

/** Exact 50k+ GN scale preserving v20's thread partition and partial-reduction order. */
static V20DiagonalScale gaussNewtonScaleDirectSimdV30(
    const Dataset&data,size_t startRow,size_t batchSize,Network&network,
    const vector<double>&parameters,size_t requestedThreads){
    unpackV14(parameters,network);
    const size_t threads=evaluationThreadsV15(requestedThreads,batchSize);
    vector<V20DiagonalPartial>partials(threads);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(threads))
    {
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
        const size_t base=batchSize/threads,remainder=batchSize%threads;
        const size_t offset=tid*base+min(tid,remainder),count=base+(tid<remainder?1:0);
        accumulateGaussNewtonSliceSimdV30(data,startRow+offset,startRow+offset+count,network,partials[tid]);
    }
#else
    accumulateGaussNewtonSliceSimdV30(data,startRow,startRow+batchSize,network,partials[0]);
#endif
    V20DiagonalScale diagonal{},scale{};
    vector<double>ordered(V14_PARAMETER_COUNT);
    const long double inv=1.0L/static_cast<long double>(batchSize);
    for(size_t i=0;i<V14_PARAMETER_COUNT;i++){
        long double sum=0.0L;
        for(size_t t=0;t<threads;t++)sum+=partials[t].values[i];
        diagonal[i]=static_cast<double>(sum*inv);
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

static V20DiagonalScale gaussNewtonScaleV30(
    const Dataset&data,size_t startRow,size_t batchSize,Network&network,
    const vector<double>&parameters,size_t requestedThreads){
    return batchSize<V27_GN_PARALLEL_ROW_LIMIT
        ?gaussNewtonScaleReducedScratchV30(data,startRow,batchSize,network,parameters,requestedThreads)
        :gaussNewtonScaleDirectSimdV30(data,startRow,batchSize,network,parameters,requestedThreads);
}

/** Refresh the target-aware diagonal using the exact v30 GN implementation. */
static void refreshScaleV30(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                            const vector<double>&parameters,const OptimiserConfig&config,
                            double percentageError,unsigned&completedStage,V20DiagonalScale&scale){
    const unsigned requiredStage=refreshStageV21(percentageError,config.percentageErrorTarget);
    if(requiredStage<=completedStage)return;
    scale=gaussNewtonScaleV30(data,startRow,batchSize,network,parameters,config.threads);
    completedStage=requiredStage;
}

/** Preserve v29's deep-state policy while replacing only the exact GN implementation. */
static void prepareDeepStateV30(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                                const vector<double>&parameters,const OptimiserConfig&config,
                                double percentageError,deque<V14HistoryPair>&history,bool&deepStage,
                                bool&scaleReady,bool&denseReady,unsigned&refreshStage,
                                V20DiagonalScale&scale,vector<double>&denseInverse){
    deepStage=deepStageV19(deepStage,percentageError);
    if(deepStage&&!scaleReady){
        scale=gaussNewtonScaleV30(data,startRow,batchSize,network,parameters,config.threads);
        scaleReady=true;
        refreshStage=refreshStageV21(percentageError,config.percentageErrorTarget);
    }
    if(scaleReady&&!denseReady){
        refreshScaleV30(data,startRow,batchSize,network,parameters,config,
                        percentageError,refreshStage,scale);
        if(percentageError<=V22_DENSE_SWITCH_PERCENTAGE){
            initialiseDenseInverseV23(history,scale,denseInverse);
            denseReady=true;
        }
    }
}

/**
 * Confirm final metrics and retain the exact scalar predictions already needed
 * for deterministic aggregation, avoiding a second final forward pass.
 */
static Metrics confirmMetricsAndPredictionsV30(
    const Dataset&data,size_t startRow,size_t batchSize,const Network&network,
    size_t requestedThreads,vector<double>&predictions){
    predictions.resize(batchSize);
    const size_t threads=evaluationThreadsV25(requestedThreads,batchSize);
#ifndef _OPENMP
    (void)threads;
#endif
#ifdef _OPENMP
#pragma omp parallel for num_threads(static_cast<int>(threads)) schedule(static)
#endif
    for(size_t row=0;row<batchSize;row++)
        predictions[row]=predictRow(data.x.rowData(startRow+row),network);

    Metrics result;
    long double squaredError=0.0L,percentageErrorSum=0.0L;
    double maxPercentageError=0.0;
    for(size_t row=0;row<batchSize;row++){
        const double actual=data.y[startRow+row],error=predictions[row]-actual;
        const double percentage=actual==0.0?0.0:abs(error/actual)*100.0;
        squaredError+=error*error;
        percentageErrorSum+=percentage;
        maxPercentageError=max(maxPercentageError,percentage);
    }
    result.cost=.5*static_cast<double>(squaredError);
    result.percentageError=static_cast<double>(percentageErrorSum)/static_cast<double>(batchSize);
    result.maxPercentageError=maxPercentageError;
    return result;
}

/** Write precomputed scalar predictions with the inherited text format. */
static void writePredictionValuesV30(const vector<double>&predictions,const string&path){
    ofstream output(path.c_str());
    if(!output)throw runtime_error("Failed to open output file: "+path);
    output<<setprecision(10);
    for(double prediction:predictions)output<<prediction<<'\n';
}
#endif

/** Preserve v29 exactly except for exact GN preparation and final prediction reuse. */
static TrainResult trainRangeV30(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(config.percentageErrorTarget>V21_REFRESH_TARGET||!supportsV16Kernel())
        return trainRangeV29(data,startRow,batchSize,batchIndex,network,config);

    TrainResult result;
    deque<V14HistoryPair>history;
    vector<double>parameters=packV14(network);
    V14Evaluation current=evaluateV29(data,startRow,batchSize,network,parameters,config.threads);
    bool deepStage=false,scaleReady=false,denseReady=false;
    unsigned refreshStage=0;
    V20DiagonalScale scale{};
    vector<double>denseInverse;
    if(config.logEvery!=0)printProgress(batchIndex,0,current.metrics);

    while(current.metrics.percentageError>config.percentageErrorTarget&&result.updates<config.maxDescents){
        prepareDeepStateV30(data,startRow,batchSize,network,parameters,config,
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
            ?armijoDenseStepV29(data,startRow,batchSize,network,config,parameters,direction,directional,
                                current.objective,candidate,next)
            :armijoStepV29(data,startRow,batchSize,network,config,parameters,direction,directional,
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
    vector<double>predictions;
    result.metrics=confirmMetricsAndPredictionsV30(
        data,startRow,batchSize,network,config.threads,predictions);
    result.reachedTarget=result.metrics.percentageError<=config.percentageErrorTarget;
    if(config.logEvery!=0||result.reachedTarget)
        printProgress(batchIndex,result.updates,result.metrics);
    writePredictionValuesV30(predictions,"ybar.txt");
    return result;
#else
    return trainRangeV29(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV30(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV30(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)
            cout<<"Batch "<<batch<<" reached the maximum number of BFGS iterations before the target.\n";
    }
}

static void offlineRunV30(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV30(data,0,rowCount,0,network,optimiser);
    writeWeights(network);
    if(!result.reachedTarget)
        cout<<"Offline BFGS reached the maximum number of iterations before the target.\n";
}

#ifndef SIMPLE_NN_V30_NO_MAIN
int main(){
    try{
        bool batchOnline=false,offline=false,test=true,randomiseWeights=false;
        double rangeWone=4.0,rangeWtwo=4.0,percentageErrorTarget=3.9;
        size_t iterations=1,exampleSize=13853,numberOfDescents=1000000,times=1,logEvery=1;
        size_t trainingThreads=4,offlineRows=10000,testRows=13853;
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
            batchOnlineRunV30(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,
                              randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV30(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(test)for(size_t i=0;i<times;i++)testRun(testRows);
        const chrono::duration<double> elapsed=chrono::steady_clock::now()-startTime;
        cout<<"Elapsed time = "<<elapsed.count()<<" seconds\n";
        return 0;
    }catch(const exception&error){
        cerr<<"Error: "<<error.what()<<'\n';
        return 1;
    }
}
#endif
