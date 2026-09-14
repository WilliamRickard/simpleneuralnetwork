#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define main v20_embedded_main
#include "../v20/main.cpp"
#undef main
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V21: refresh v20's deep Gauss-Newton scale only for ultra-deep targets. */
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
constexpr double V21_REFRESH_TARGET=0.001;
constexpr double V21_FIRST_REFRESH_PERCENTAGE=0.0225;
constexpr double V21_SECOND_REFRESH_PERCENTAGE=0.01;
constexpr double V21_ULTRA_DEEP_TARGET=0.0005;
constexpr double V21_THIRD_REFRESH_PERCENTAGE=0.002;

/** Return how many post-v20 diagonal refreshes should already have occurred. */
static unsigned refreshStageV21(double percentageError,double target){
    if(target>V21_REFRESH_TARGET)return 0;
    unsigned stage=0;
    if(percentageError<=V21_FIRST_REFRESH_PERCENTAGE)stage=1;
    if(percentageError<=V21_SECOND_REFRESH_PERCENTAGE)stage=2;
    if(target<=V21_ULTRA_DEEP_TARGET&&percentageError<=V21_THIRD_REFRESH_PERCENTAGE)stage=3;
    return stage;
}

/** Refresh the fixed diagonal only when a new target-aware deep stage is crossed. */
static void refreshScaleV21(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                            const vector<double>&parameters,const OptimiserConfig&config,
                            double percentageError,unsigned&completedStage,V20DiagonalScale&scale){
    const unsigned requiredStage=refreshStageV21(percentageError,config.percentageErrorTarget);
    if(requiredStage<=completedStage)return;
    scale=gaussNewtonScaleV20(data,startRow,batchSize,network,parameters,config.threads);
    completedStage=requiredStage;
}
#endif

/** Train as v20, with sparse target-aware Gauss-Newton refreshes below 0.0225% PE. */
static TrainResult trainRangeV21(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(config.percentageErrorTarget>V21_REFRESH_TARGET)
        return trainRangeV20(data,startRow,batchSize,batchIndex,network,config);

    TrainResult result;
    deque<V14HistoryPair>history;
    vector<double>parameters=packV14(network);
    V14Evaluation current=evaluateV17(data,startRow,batchSize,network,parameters,config.threads);
    bool deepStage=false,scaleReady=false;
    unsigned refreshStage=0;
    V20DiagonalScale scale{};
    if(config.logEvery!=0)printProgress(batchIndex,0,current.metrics);

    while(current.metrics.percentageError>config.percentageErrorTarget&&result.updates<config.maxDescents){
        deepStage=deepStageV19(deepStage,current.metrics.percentageError);
        if(deepStage&&!scaleReady){
            scale=gaussNewtonScaleV20(data,startRow,batchSize,network,parameters,config.threads);
            scaleReady=true;
        }
        if(scaleReady){
            refreshScaleV21(data,startRow,batchSize,network,parameters,config,
                            current.metrics.percentageError,refreshStage,scale);
        }

        const size_t historyLimit=historyLimitV19(deepStage);
        vector<double>direction;
        double directional=0.0;
        descentDirectionV20(current.gradient,history,scaleReady?&scale:nullptr,direction,directional);

        vector<double>candidate(parameters.size());
        V14Evaluation next;
        if(!armijoStepV19(data,startRow,batchSize,network,config,parameters,direction,directional,
                          current.objective,candidate,next)){
            unpackV14(parameters,network);
            break;
        }

        updateHistoryV19(parameters,candidate,current.gradient,next.gradient,historyLimit,history);
        parameters.swap(candidate);
        current=std::move(next);
        result.updates++;
        if(config.logEvery!=0&&result.updates%config.logEvery==0)printProgress(batchIndex,result.updates,current.metrics);
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
    return trainRangeV20(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV21(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV21(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of L-BFGS iterations before the target.\n";
    }
}

static void offlineRunV21(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV21(data,0,rowCount,0,network,optimiser);
    writeWeights(network);
    if(!result.reachedTarget)cout<<"Offline L-BFGS reached the maximum number of iterations before the target.\n";
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
            batchOnlineRunV21(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV21(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
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
