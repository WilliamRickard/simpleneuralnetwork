#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define main v21_embedded_main
#include "../v21/main.cpp"
#undef main
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V22: retain v21 through the deep tail, then use full-memory BFGS below 0.015% PE. */
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
constexpr double V22_DENSE_SWITCH_PERCENTAGE=0.015;

/** Scalar L-BFGS H0 multiplier represented by the newest accepted secant pair. */
static double denseInitialGammaV22(const deque<V14HistoryPair>&history){
    if(history.empty())return 1.0;
    const V14HistoryPair&pair=history.back();
    const double yy=dotV14(pair.y,pair.y);
    const double sy=dotV14(pair.s,pair.y);
    if(!(yy>0.0)||!(sy>0.0)||!isfinite(yy)||!isfinite(sy))return 1.0;
    return sy/yy;
}

/** Initialise the dense inverse-Hessian estimate from v21's current diagonal scale. */
static void initialiseDenseInverseV22(const deque<V14HistoryPair>&history,
                                      const V20DiagonalScale&scale,vector<double>&inverse){
    inverse.assign(V14_PARAMETER_COUNT*V14_PARAMETER_COUNT,0.0);
    const double gamma=denseInitialGammaV22(history);
    for(size_t i=0;i<V14_PARAMETER_COUNT;i++)inverse[i*V14_PARAMETER_COUNT+i]=gamma*scale[i];
}

/** Multiply the dense inverse-Hessian estimate by the gradient. */
static void denseDirectionV22(const vector<double>&gradient,const vector<double>&inverse,
                              vector<double>&direction,double&directional){
    direction.assign(V14_PARAMETER_COUNT,0.0);
    for(size_t i=0;i<V14_PARAMETER_COUNT;i++){
        double value=0.0;
        const size_t row=i*V14_PARAMETER_COUNT;
        for(size_t j=0;j<V14_PARAMETER_COUNT;j++)value+=inverse[row+j]*gradient[j];
        direction[i]=-value;
    }
    directional=dotV14(gradient,direction);
}

/** Apply the inverse-BFGS rank-two update when the v18 curvature safeguard accepts the pair. */
static bool updateDenseInverseV22(const vector<double>&parameters,const vector<double>&candidate,
                                  const vector<double>&gradient,const vector<double>&nextGradient,
                                  vector<double>&inverse){
    vector<double>s(V14_PARAMETER_COUNT),y(V14_PARAMETER_COUNT),hy(V14_PARAMETER_COUNT,0.0);
    for(size_t i=0;i<V14_PARAMETER_COUNT;i++){
        s[i]=candidate[i]-parameters[i];
        y[i]=nextGradient[i]-gradient[i];
    }
    const double sy=dotV14(s,y);
    if(!acceptCurvatureV18(s,y,sy))return false;

    for(size_t i=0;i<V14_PARAMETER_COUNT;i++){
        const size_t row=i*V14_PARAMETER_COUNT;
        double value=0.0;
        for(size_t j=0;j<V14_PARAMETER_COUNT;j++)value+=inverse[row+j]*y[j];
        hy[i]=value;
    }
    const double yhy=dotV14(y,hy);
    if(!isfinite(yhy))return false;
    const double ssCoefficient=(1.0+yhy/sy)/sy;
    const double crossCoefficient=1.0/sy;
    for(size_t i=0;i<V14_PARAMETER_COUNT;i++){
        const size_t row=i*V14_PARAMETER_COUNT;
        for(size_t j=0;j<V14_PARAMETER_COUNT;j++){
            inverse[row+j]+=ssCoefficient*s[i]*s[j]
                           -crossCoefficient*(hy[i]*s[j]+s[i]*hy[j]);
        }
    }
    return true;
}
#endif

/** Preserve v21 above 0.001%; below it, switch from diagonal L-BFGS to dense BFGS at 0.015% PE. */
static TrainResult trainRangeV22(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(config.percentageErrorTarget>V21_REFRESH_TARGET)
        return trainRangeV21(data,startRow,batchSize,batchIndex,network,config);

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
                initialiseDenseInverseV22(history,scale,denseInverse);
                denseReady=true;
            }
        }

        const size_t historyLimit=historyLimitV19(deepStage);
        vector<double>direction;
        double directional=0.0;
        if(denseReady){
            denseDirectionV22(current.gradient,denseInverse,direction,directional);
            if(!(directional<0.0)||!isfinite(directional)){
                initialiseDenseInverseV22(history,scale,denseInverse);
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
        if(!armijoStepV19(data,startRow,batchSize,network,config,parameters,direction,directional,
                          current.objective,candidate,next)){
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
    return trainRangeV21(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV22(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV22(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of BFGS iterations before the target.\n";
    }
}

static void offlineRunV22(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV22(data,0,rowCount,0,network,optimiser);
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
            batchOnlineRunV22(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV22(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
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
