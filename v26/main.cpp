#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define SIMPLE_NN_V25_NO_MAIN
#include "../v25/main.cpp"
#undef SIMPLE_NN_V25_NO_MAIN
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V26: preserve v25 exactly while accelerating dense BFGS algebra with an exact AVX-512 column-major layout. */
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)

static_assert(V14_PARAMETER_COUNT%8==0,"V26 AVX-512 dense blocks require parameter count divisible by eight");

/**
 * Multiply a column-major dense inverse-Hessian by a vector.
 *
 * Each AVX-512 lane represents one matrix row. The inner j loop therefore
 * accumulates products in exactly the same order as v22's scalar row-major
 * implementation. FP contraction is disabled locally so multiply/add rounding
 * also matches v22 exactly.
 */
__attribute__((target("avx512f,avx512dq"),optimize("fp-contract=off")))
static void denseMatVecColumnMajorV26(const vector<double>&inverse,const vector<double>&x,
                                      vector<double>&out,bool negate){
    const size_t n=V14_PARAMETER_COUNT;
    const __m512d zero=_mm512_setzero_pd();
    for(size_t i=0;i<n;i+=8){
        __m512d value=zero;
        for(size_t j=0;j<n;j++){
            const __m512d h=_mm512_loadu_pd(inverse.data()+j*n+i);
            const __m512d product=_mm512_mul_pd(h,_mm512_set1_pd(x[j]));
            value=_mm512_add_pd(value,product);
        }
        if(negate)value=_mm512_sub_pd(zero,value);
        _mm512_storeu_pd(out.data()+i,value);
    }
}

/** Build the dense BFGS descent direction without changing v22's FP result. */
static void denseDirectionV26(const vector<double>&gradient,const vector<double>&inverse,
                              vector<double>&direction,double&directional){
    direction.assign(V14_PARAMETER_COUNT,0.0);
    denseMatVecColumnMajorV26(inverse,gradient,direction,true);
    directional=dotV14(gradient,direction);
}

/**
 * Apply the elementwise inverse-BFGS update to a column-major matrix.
 *
 * The arithmetic expression mirrors v22 term-for-term, again with contraction
 * disabled, while eight independent rows are updated at once.
 */
__attribute__((target("avx512f,avx512dq"),optimize("fp-contract=off")))
static void updateDenseElementsColumnMajorV26(const vector<double>&s,const vector<double>&hy,
                                               double ssCoefficient,double crossCoefficient,
                                               vector<double>&inverse){
    const size_t n=V14_PARAMETER_COUNT;
    const __m512d ss=_mm512_set1_pd(ssCoefficient);
    const __m512d cross=_mm512_set1_pd(crossCoefficient);
    for(size_t j=0;j<n;j++){
        double*column=inverse.data()+j*n;
        const __m512d sj=_mm512_set1_pd(s[j]);
        const __m512d hyj=_mm512_set1_pd(hy[j]);
        for(size_t i=0;i<n;i+=8){
            const __m512d si=_mm512_loadu_pd(s.data()+i);
            const __m512d hyi=_mm512_loadu_pd(hy.data()+i);
            const __m512d old=_mm512_loadu_pd(column+i);
            const __m512d scaledSi=_mm512_mul_pd(ss,si);
            const __m512d first=_mm512_mul_pd(scaledSi,sj);
            const __m512d inner=_mm512_add_pd(_mm512_mul_pd(hyi,sj),_mm512_mul_pd(si,hyj));
            const __m512d second=_mm512_mul_pd(cross,inner);
            const __m512d delta=_mm512_sub_pd(first,second);
            _mm512_storeu_pd(column+i,_mm512_add_pd(old,delta));
        }
    }
}

/** Apply v22's safeguarded inverse-BFGS update using the exact column-major SIMD representation. */
static bool updateDenseInverseV26(const vector<double>&parameters,const vector<double>&candidate,
                                  const vector<double>&gradient,const vector<double>&nextGradient,
                                  vector<double>&inverse){
    vector<double>s(V14_PARAMETER_COUNT),y(V14_PARAMETER_COUNT),hy(V14_PARAMETER_COUNT,0.0);
    for(size_t i=0;i<V14_PARAMETER_COUNT;i++){
        s[i]=candidate[i]-parameters[i];
        y[i]=nextGradient[i]-gradient[i];
    }
    const double sy=dotV14(s,y);
    if(!acceptCurvatureV18(s,y,sy))return false;

    denseMatVecColumnMajorV26(inverse,y,hy,false);
    const double yhy=dotV14(y,hy);
    if(!isfinite(yhy))return false;
    const double ssCoefficient=(1.0+yhy/sy)/sy;
    const double crossCoefficient=1.0/sy;
    updateDenseElementsColumnMajorV26(s,hy,ssCoefficient,crossCoefficient,inverse);
    return true;
}
#endif

/** Preserve v25 completely except for the dense inverse-Hessian storage/algebra. */
static TrainResult trainRangeV26(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(config.percentageErrorTarget>V21_REFRESH_TARGET||!supportsV16Kernel())
        return trainRangeV25(data,startRow,batchSize,batchIndex,network,config);

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
                /* A diagonal matrix has the same flat representation in row- and column-major order. */
                initialiseDenseInverseV23(history,scale,denseInverse);
                denseReady=true;
            }
        }

        const size_t historyLimit=historyLimitV19(deepStage);
        vector<double>direction;
        double directional=0.0;
        if(denseReady){
            denseDirectionV26(current.gradient,denseInverse,direction,directional);
            if(!(directional<0.0)||!isfinite(directional)){
                initialiseDenseInverseV23(history,scale,denseInverse);
                denseDirectionV26(current.gradient,denseInverse,direction,directional);
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
            updateDenseInverseV26(parameters,candidate,current.gradient,next.gradient,denseInverse);
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
    return trainRangeV25(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV26(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV26(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of BFGS iterations before the target.\n";
    }
}

static void offlineRunV26(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV26(data,0,rowCount,0,network,optimiser);
    writeWeights(network);
    if(!result.reachedTarget)cout<<"Offline BFGS reached the maximum number of iterations before the target.\n";
}

#ifndef SIMPLE_NN_V26_NO_MAIN
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
            batchOnlineRunV26(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV26(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
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
