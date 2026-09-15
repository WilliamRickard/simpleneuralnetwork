#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define SIMPLE_NN_V19_NO_MAIN
#include "../v19/main.cpp"
#undef SIMPLE_NN_V19_NO_MAIN
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V20: preserve v19, then diagonally precondition the deep L-BFGS stage. */
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
constexpr double V20_GN_EXPONENT=0.5;
constexpr double V20_GN_CLIP=28.0;
constexpr double V20_GN_FLOOR_RATIO=1e-18;
using V20DiagonalScale=array<double,V14_PARAMETER_COUNT>;

struct V20DiagonalPartial {
    array<long double,V14_PARAMETER_COUNT> values{};
};

/** Accumulate the Gauss-Newton diagonal for one contiguous row slice. */
static void accumulateGaussNewtonSliceV20(const Dataset&data,size_t firstRow,size_t lastRow,
                                          const Network&network,V20DiagonalPartial&out){
    array<double,HIDDEN_NODES> hidden{};
    for(size_t row=firstRow;row<lastRow;row++){
        const double*x=data.x.rowData(row);
        for(size_t j=0;j<HIDDEN_NODES;j++){
            double raw=0.0;
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)raw+=x[k]*network.wOne[k*HIDDEN_NODES+j];
            hidden[j]=sigmoidV15(raw);
        }
        double rawOutput=0.0;
        for(size_t j=0;j<HIDDEN_NODES;j++)rawOutput+=hidden[j]*network.wTwo[j];
        const double prediction=sigmoidV15(rawOutput);
        const double outputDerivative=prediction*(1.0-prediction);
        for(size_t j=0;j<HIDDEN_NODES;j++){
            const double outputJacobian=outputDerivative*hidden[j];
            out.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
            const double hiddenJacobianBase=outputDerivative*network.wTwo[j]*hidden[j]*(1.0-hidden[j]);
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
                const double jacobian=hiddenJacobianBase*x[k];
                out.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
            }
        }
    }
}

/** Build a positive, median-normalised square-root inverse Gauss-Newton diagonal. */
static V20DiagonalScale gaussNewtonScaleV20(const Dataset&data,size_t startRow,size_t batchSize,
                                            Network&network,const vector<double>&parameters,
                                            size_t requestedThreads){
    unpackV14(parameters,network);
    const size_t threads=evaluationThreadsV15(requestedThreads,batchSize);
    vector<V20DiagonalPartial>partials(threads);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(threads))
    {
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
        const size_t base=batchSize/threads,remainder=batchSize%threads;
        const size_t offset=tid*base+min(tid,remainder),count=base+(tid<remainder?1:0);
        accumulateGaussNewtonSliceV20(data,startRow+offset,startRow+offset+count,network,partials[tid]);
    }
#else
    accumulateGaussNewtonSliceV20(data,startRow,startRow+batchSize,network,partials[0]);
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

/** Apply v14's two-loop recursion with a diagonal initial inverse-Hessian scale. */
static vector<double> preconditionedDirectionV20(const vector<double>&gradient,
                                                 const deque<V14HistoryPair>&history,
                                                 const V20DiagonalScale&scale){
    vector<double>q=gradient,alpha(history.size());
    for(size_t ii=history.size();ii>0;--ii){
        const size_t i=ii-1;
        alpha[i]=history[i].rho*dotV14(history[i].s,q);
        for(size_t k=0;k<q.size();k++)q[k]-=alpha[i]*history[i].y[k];
    }
    double gamma=1.0;
    if(!history.empty()){
        const auto&last=history.back();
        const double yy=dotV14(last.y,last.y);
        if(yy>0.0)gamma=dotV14(last.s,last.y)/yy;
    }
    vector<double>direction=q;
    for(size_t k=0;k<direction.size();k++)direction[k]*=gamma*scale[k];
    for(size_t i=0;i<history.size();i++){
        const double beta=history[i].rho*dotV14(history[i].y,direction);
        for(size_t k=0;k<direction.size();k++)direction[k]+=history[i].s[k]*(alpha[i]-beta);
    }
    for(double&value:direction)value=-value;
    return direction;
}

/** Preserve v19's exact ordinary direction and use diagonal scaling only in the deep stage. */
static void descentDirectionV20(const vector<double>&gradient,deque<V14HistoryPair>&history,
                                const V20DiagonalScale*scale,vector<double>&direction,
                                double&directional){
    direction=scale==nullptr?directionV14(gradient,history):preconditionedDirectionV20(gradient,history,*scale);
    directional=dotV14(gradient,direction);
    if(directional<0.0)return;
    direction=gradient;
    for(double&value:direction)value=-value;
    directional=-dotV14(gradient,gradient);
    history.clear();
}
#endif

/** Train exactly as v19 through 0.05% PE, then use one fixed Gauss-Newton diagonal scale. */
static TrainResult trainRangeV20(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    TrainResult result;
    deque<V14HistoryPair>history;
    vector<double>parameters=packV14(network);
    V14Evaluation current=evaluateV17(data,startRow,batchSize,network,parameters,config.threads);
    bool deepStage=false,scaleReady=false;
    V20DiagonalScale scale{};
    if(config.logEvery!=0)printProgress(batchIndex,0,current.metrics);

    while(current.metrics.percentageError>config.percentageErrorTarget&&result.updates<config.maxDescents){
        deepStage=deepStageV19(deepStage,current.metrics.percentageError);
        if(deepStage&&!scaleReady){
            scale=gaussNewtonScaleV20(data,startRow,batchSize,network,parameters,config.threads);
            scaleReady=true;
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
    return trainRangeV19(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV20(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV20(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of L-BFGS iterations before the target.\n";
    }
}

static void offlineRunV20(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV20(data,0,rowCount,0,network,optimiser);
    writeWeights(network);
    if(!result.reachedTarget)cout<<"Offline L-BFGS reached the maximum number of iterations before the target.\n";
}

#ifndef SIMPLE_NN_V20_NO_MAIN
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
            batchOnlineRunV20(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV20(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
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
