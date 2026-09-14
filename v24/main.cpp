#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define main v23_embedded_main
#include "../v23/main.cpp"
#undef main
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V24: retain v23's dense handover and use safeguarded quadratic backtracking only in the dense tail. */
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
constexpr double V24_QUADRATIC_MIN_FRACTION=0.08;
constexpr double V24_QUADRATIC_MAX_FRACTION=0.50;

/**
 * Run Armijo backtracking in the dense-BFGS tail.
 *
 * The first trial remains alpha = 1 exactly as in v23. After a rejection, use
 * the one-point quadratic model when it is finite and positive, safeguarded to
 * [0.08 alpha, 0.50 alpha]. Invalid models fall back to v23's half-step rule.
 */
static bool armijoDenseStepV24(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                               const OptimiserConfig&config,const vector<double>&parameters,
                               const vector<double>&direction,double directional,double currentObjective,
                               vector<double>&candidate,V14Evaluation&next){
    double step=1.0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        for(size_t k=0;k<parameters.size();k++)candidate[k]=parameters[k]+step*direction[k];
        next=evaluateV17(data,startRow,batchSize,network,candidate,config.threads);
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

/** Preserve v23 before the dense handover, then use v24 dense-tail line search. */
static TrainResult trainRangeV24(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(config.percentageErrorTarget>V21_REFRESH_TARGET)
        return trainRangeV23(data,startRow,batchSize,batchIndex,network,config);

    TrainResult result;
    deque<V14HistoryPair>history;
    vector<double>parameters=packV14(network);
    V14Evaluation current=evaluateV17(data,startRow,batchSize,network,parameters,config.threads);
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
            ?armijoDenseStepV24(data,startRow,batchSize,network,config,parameters,direction,directional,
                                current.objective,candidate,next)
            :armijoStepV19(data,startRow,batchSize,network,config,parameters,direction,directional,
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
    result.metrics=confirmMetricsV17(data,startRow,batchSize,network,config.threads);
    result.reachedTarget=result.metrics.percentageError<=config.percentageErrorTarget;
    if(config.logEvery!=0||result.reachedTarget)printProgress(batchIndex,result.updates,result.metrics);
    writePredictions(data,startRow,batchSize,network,"ybar.txt");
    return result;
#else
    return trainRangeV23(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV24(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV24(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of BFGS iterations before the target.\n";
    }
}

static void offlineRunV24(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV24(data,0,rowCount,0,network,optimiser);
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
            batchOnlineRunV24(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV24(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
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
