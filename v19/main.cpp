#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#endif
#define main v18_embedded_main
#include "../v18/main.cpp"
#undef main
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V19: retain v18's fast path, then expand L-BFGS memory for deep convergence. */
constexpr size_t V19_DEEP_HISTORY=160;
constexpr double V19_DEEP_STAGE_PERCENTAGE=0.05;

/** Return the active L-BFGS history cap for the current optimisation stage. */
static size_t historyLimitV19(bool deepStage){
    return deepStage?V19_DEEP_HISTORY:V14_HISTORY;
}

/** Enter the deep stage once and keep it active for the rest of the run. */
static bool deepStageV19(bool alreadyDeep,double percentageError){
    return alreadyDeep||percentageError<=V19_DEEP_STAGE_PERCENTAGE;
}

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
/** Build a descent direction, falling back to steepest descent if L-BFGS loses descent. */
static void descentDirectionV19(const vector<double>&gradient,deque<V14HistoryPair>&history,
                                vector<double>&direction,double&directional){
    direction=directionV14(gradient,history);
    directional=dotV14(gradient,direction);
    if(directional<0.0)return;
    direction=gradient;
    for(double&value:direction)value=-value;
    directional=-dotV14(gradient,gradient);
    history.clear();
}

/** Run the unchanged v18 Armijo backtracking search. */
static bool armijoStepV19(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                          const OptimiserConfig&config,const vector<double>&parameters,
                          const vector<double>&direction,double directional,double currentObjective,
                          vector<double>&candidate,V14Evaluation&next){
    double step=1.0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        for(size_t k=0;k<parameters.size();k++)candidate[k]=parameters[k]+step*direction[k];
        next=evaluateV17(data,startRow,batchSize,network,candidate,config.threads);
        if(next.objective<=currentObjective+V14_ARMIJO*step*directional)return true;
        step*=0.5;
    }
    return false;
}

/** Retain a mathematically useful secant pair under the v18 scale-aware curvature rule. */
static void updateHistoryV19(const vector<double>&parameters,const vector<double>&candidate,
                             const vector<double>&gradient,const vector<double>&nextGradient,
                             size_t historyLimit,deque<V14HistoryPair>&history){
    vector<double>s(parameters.size()),y(parameters.size());
    for(size_t k=0;k<parameters.size();k++){
        s[k]=candidate[k]-parameters[k];
        y[k]=nextGradient[k]-gradient[k];
    }
    const double sy=dotV14(s,y);
    if(!acceptCurvatureV18(s,y,sy))return;
    if(history.size()>=historyLimit)history.pop_front();
    V14HistoryPair pair;
    pair.s.swap(s);
    pair.y.swap(y);
    pair.rho=1.0/sy;
    history.push_back(std::move(pair));
}
#endif

/** Train with v18 unchanged until 0.05% PE, then permit 160 L-BFGS curvature pairs. */
static TrainResult trainRangeV19(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    TrainResult result;
    deque<V14HistoryPair>history;
    vector<double>parameters=packV14(network);
    V14Evaluation current=evaluateV17(data,startRow,batchSize,network,parameters,config.threads);
    bool deepStage=false;
    if(config.logEvery!=0)printProgress(batchIndex,0,current.metrics);

    while(current.metrics.percentageError>config.percentageErrorTarget&&result.updates<config.maxDescents){
        deepStage=deepStageV19(deepStage,current.metrics.percentageError);
        const size_t historyLimit=historyLimitV19(deepStage);
        vector<double>direction;
        double directional=0.0;
        descentDirectionV19(current.gradient,history,direction,directional);

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
    return trainRangeV18(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV19(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV19(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of L-BFGS iterations before the target.\n";
    }
}

static void offlineRunV19(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV19(data,0,rowCount,0,network,optimiser);
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
            batchOnlineRunV19(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV19(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
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
