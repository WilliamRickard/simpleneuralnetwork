#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define SIMPLE_NN_V24_NO_MAIN
#include "../v24/main.cpp"
#undef SIMPLE_NN_V24_NO_MAIN
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V25: remove the stale 50k-row forced-serial evaluator threshold. */
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
constexpr size_t V25_MIN_ROWS_PER_EVALUATION_THREAD=1024;

/**
 * Choose evaluator workers from the caller request and available row work.
 *
 * V15 forced every evaluation below 50,000 rows to one thread. The AVX-512
 * evaluator is now large enough that this cutoff leaves substantial CPU
 * parallelism unused. V25 allows one worker per 1,024 rows, capped by the
 * existing V15 maximum and the caller's requested thread count.
 */
static size_t evaluationThreadsV25(size_t requested,size_t rows){
#ifdef _OPENMP
    const size_t wanted=requested==0?V15_MAX_EVALUATION_THREADS:requested;
    const size_t capped=max<size_t>(1,min(V15_MAX_EVALUATION_THREADS,wanted));
    const size_t useful=max<size_t>(1,rows/V25_MIN_ROWS_PER_EVALUATION_THREAD);
    return min(capped,useful);
#else
    (void)requested;
    (void)rows;
    return 1;
#endif
}

/** Evaluate objective, PE and gradient with the V17 AVX-512 kernel using V25 worker selection. */
static V14Evaluation evaluateV25(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                                 const vector<double>&parameters,size_t requestedThreads){
    if(!supportsV16Kernel())return evaluateV17(data,startRow,batchSize,network,parameters,requestedThreads);
    unpackV14(parameters,network);
    const size_t threads=evaluationThreadsV25(requestedThreads,batchSize);
    vector<V15ThreadEvaluation>partials(threads);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(threads))
    {
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
        const size_t base=batchSize/threads,remainder=batchSize%threads;
        const size_t offset=tid*base+min(tid,remainder),count=base+(tid<remainder?1:0);
        evaluateSliceV17(data,startRow+offset,startRow+offset+count,network,partials[tid]);
    }
#else
    evaluateSliceV17(data,startRow,startRow+batchSize,network,partials[0]);
#endif
    V14Evaluation result;
    result.gradient.assign(V14_PARAMETER_COUNT,0.0);
    long double squaredError=0.0L,percentageErrorSum=0.0L;
    double maxPercentageError=0.0;
    for(size_t t=0;t<threads;t++){
        squaredError+=partials[t].squaredError;
        percentageErrorSum+=partials[t].percentageErrorSum;
        maxPercentageError=max(maxPercentageError,partials[t].maxPercentageError);
        for(size_t i=0;i<V14_PARAMETER_COUNT;i++)result.gradient[i]+=partials[t].gradient[i];
    }
    const double inv=1.0/static_cast<double>(batchSize);
    result.metrics.cost=.5*static_cast<double>(squaredError);
    result.metrics.percentageError=static_cast<double>(percentageErrorSum)*inv;
    result.metrics.maxPercentageError=maxPercentageError;
    result.objective=result.metrics.cost*inv;
    for(double&value:result.gradient)value*=inv;
    return result;
}

/** Confirm final metrics with V25 worker selection while retaining scalar deterministic aggregation. */
static Metrics confirmMetricsV25(const Dataset&data,size_t startRow,size_t batchSize,const Network&network,
                                 size_t requestedThreads){
    vector<double>predictions(batchSize);
    const size_t threads=evaluationThreadsV25(requestedThreads,batchSize);
#ifndef _OPENMP
    (void)threads;
#endif
#ifdef _OPENMP
#pragma omp parallel for num_threads(static_cast<int>(threads)) schedule(static)
#endif
    for(size_t row=0;row<batchSize;row++)predictions[row]=predictRow(data.x.rowData(startRow+row),network);
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

/** V19 Armijo search with only the evaluator worker policy changed. */
static bool armijoStepV25(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                          const OptimiserConfig&config,const vector<double>&parameters,
                          const vector<double>&direction,double directional,double currentObjective,
                          vector<double>&candidate,V14Evaluation&next){
    double step=1.0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        for(size_t k=0;k<parameters.size();k++)candidate[k]=parameters[k]+step*direction[k];
        next=evaluateV25(data,startRow,batchSize,network,candidate,config.threads);
        if(next.objective<=currentObjective+V14_ARMIJO*step*directional)return true;
        step*=0.5;
    }
    return false;
}

/** V24 safeguarded dense-tail Armijo search using the V25 evaluator. */
static bool armijoDenseStepV25(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                               const OptimiserConfig&config,const vector<double>&parameters,
                               const vector<double>&direction,double directional,double currentObjective,
                               vector<double>&candidate,V14Evaluation&next){
    double step=1.0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        for(size_t k=0;k<parameters.size();k++)candidate[k]=parameters[k]+step*direction[k];
        next=evaluateV25(data,startRow,batchSize,network,candidate,config.threads);
        if(next.objective<=currentObjective+V14_ARMIJO*step*directional)return true;

        const double denominator=2.0*(next.objective-currentObjective-step*directional);
        if(denominator>0.0&&isfinite(denominator)){
            const double quadratic=-directional*step*step/denominator;
            if(quadratic>0.0&&isfinite(quadratic)){
                const double lower=V24_QUADRATIC_MIN_FRACTION*step;
                const double upper=V24_QUADRATIC_MAX_FRACTION*step;
                step=max(lower,min(upper,quadratic));
                continue;
            }
        }
        step*=0.5;
    }
    return false;
}
#endif

/** Preserve V24 optimiser policy, changing only evaluator worker selection. */
static TrainResult trainRangeV25(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(config.percentageErrorTarget>V21_REFRESH_TARGET)
        return trainRangeV24(data,startRow,batchSize,batchIndex,network,config);

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
        deepStage=deepStageV19(deepStage,current.metrics.percentageError);
        if(deepStage&&!scaleReady){
            scale=gaussNewtonScaleV20(data,startRow,batchSize,network,parameters,config.threads);
            scaleReady=true;
            refreshStage=refreshStageV21(current.metrics.percentageError,config.percentageErrorTarget);
        }
        if(scaleReady&&!denseReady){
            refreshScaleV21(data,startRow,batchSize,network,parameters,config,
                            current.metrics.percentageError,refreshStage,scale);
            if(current.metrics.percentageError<=V22_DENSE_SWITCH_PERCENTAGE){
                initialiseDenseInverseV23(history,scale,denseInverse);
                denseReady=true;
            }
        }

        const size_t historyLimit=historyLimitV19(deepStage);
        vector<double>direction;
        double directional=0.0;
        if(denseReady){
            denseDirectionV22(current.gradient,denseInverse,direction,directional);
            if(!(directional<0.0)||!isfinite(directional)){
                initialiseDenseInverseV23(history,scale,denseInverse);
                denseDirectionV22(current.gradient,denseInverse,direction,directional);
            }
            if(!(directional<0.0)||!isfinite(directional)){
                direction=current.gradient;
                for(double&value:direction)value=-value;
                directional=-dotV14(current.gradient,current.gradient);
            }
        }else{
            descentDirectionV20(current.gradient,history,scaleReady?&scale:nullptr,direction,directional);
        }

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

        if(denseReady){
            updateDenseInverseV22(parameters,candidate,current.gradient,next.gradient,denseInverse);
        }else{
            updateHistoryV19(parameters,candidate,current.gradient,next.gradient,historyLimit,history);
        }
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
    return trainRangeV24(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV25(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV25(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of BFGS iterations before the target.\n";
    }
}

static void offlineRunV25(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV25(data,0,rowCount,0,network,optimiser);
    writeWeights(network);
    if(!result.reachedTarget)cout<<"Offline BFGS reached the maximum number of iterations before the target.\n";
}

#ifndef SIMPLE_NN_V25_NO_MAIN
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
            batchOnlineRunV25(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV25(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
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
#endif
