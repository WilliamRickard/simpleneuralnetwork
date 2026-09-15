#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define main v27_embedded_main
#include "../v27/main.cpp"
#undef main
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V28: reduce AVX-512 spill traffic across libmvec exp call boundaries. */
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)

static_assert(NUMBER_OF_VARIABLES==11&&HIDDEN_NODES==16,
              "V28 call-boundary kernel requires the fixed 11->16->1 architecture");

#define V28_DECLARE_W1(G) \
    __m512d g00=_mm512_loadu_pd((G)+0*HIDDEN_NODES), g01=_mm512_loadu_pd((G)+0*HIDDEN_NODES+8); \
    __m512d g10=_mm512_loadu_pd((G)+1*HIDDEN_NODES), g11=_mm512_loadu_pd((G)+1*HIDDEN_NODES+8); \
    __m512d g20=_mm512_loadu_pd((G)+2*HIDDEN_NODES), g21=_mm512_loadu_pd((G)+2*HIDDEN_NODES+8); \
    __m512d g30=_mm512_loadu_pd((G)+3*HIDDEN_NODES), g31=_mm512_loadu_pd((G)+3*HIDDEN_NODES+8); \
    __m512d g40=_mm512_loadu_pd((G)+4*HIDDEN_NODES), g41=_mm512_loadu_pd((G)+4*HIDDEN_NODES+8); \
    __m512d g50=_mm512_loadu_pd((G)+5*HIDDEN_NODES), g51=_mm512_loadu_pd((G)+5*HIDDEN_NODES+8); \
    __m512d g60=_mm512_loadu_pd((G)+6*HIDDEN_NODES), g61=_mm512_loadu_pd((G)+6*HIDDEN_NODES+8); \
    __m512d g70=_mm512_loadu_pd((G)+7*HIDDEN_NODES), g71=_mm512_loadu_pd((G)+7*HIDDEN_NODES+8); \
    __m512d g80=_mm512_loadu_pd((G)+8*HIDDEN_NODES), g81=_mm512_loadu_pd((G)+8*HIDDEN_NODES+8); \
    __m512d g90=_mm512_loadu_pd((G)+9*HIDDEN_NODES), g91=_mm512_loadu_pd((G)+9*HIDDEN_NODES+8); \
    __m512d g100=_mm512_loadu_pd((G)+10*HIDDEN_NODES), g101=_mm512_loadu_pd((G)+10*HIDDEN_NODES+8)

#define V28_ACCUM_W1(K,G0,G1) do { \
    const __m512d xv=_mm512_set1_pd(xq[(K)]); \
    (G0)=_mm512_fmadd_pd(xv,hiddenDeltaLow,(G0)); \
    (G1)=_mm512_fmadd_pd(xv,hiddenDeltaHigh,(G1)); \
} while(false)

#define V28_ACCUM_ALL_W1() do { \
    V28_ACCUM_W1(0,g00,g01); V28_ACCUM_W1(1,g10,g11); V28_ACCUM_W1(2,g20,g21); \
    V28_ACCUM_W1(3,g30,g31); V28_ACCUM_W1(4,g40,g41); V28_ACCUM_W1(5,g50,g51); \
    V28_ACCUM_W1(6,g60,g61); V28_ACCUM_W1(7,g70,g71); V28_ACCUM_W1(8,g80,g81); \
    V28_ACCUM_W1(9,g90,g91); V28_ACCUM_W1(10,g100,g101); \
} while(false)

#define V28_STORE_W1(G) do { \
    _mm512_storeu_pd((G)+0*HIDDEN_NODES,g00); _mm512_storeu_pd((G)+0*HIDDEN_NODES+8,g01); \
    _mm512_storeu_pd((G)+1*HIDDEN_NODES,g10); _mm512_storeu_pd((G)+1*HIDDEN_NODES+8,g11); \
    _mm512_storeu_pd((G)+2*HIDDEN_NODES,g20); _mm512_storeu_pd((G)+2*HIDDEN_NODES+8,g21); \
    _mm512_storeu_pd((G)+3*HIDDEN_NODES,g30); _mm512_storeu_pd((G)+3*HIDDEN_NODES+8,g31); \
    _mm512_storeu_pd((G)+4*HIDDEN_NODES,g40); _mm512_storeu_pd((G)+4*HIDDEN_NODES+8,g41); \
    _mm512_storeu_pd((G)+5*HIDDEN_NODES,g50); _mm512_storeu_pd((G)+5*HIDDEN_NODES+8,g51); \
    _mm512_storeu_pd((G)+6*HIDDEN_NODES,g60); _mm512_storeu_pd((G)+6*HIDDEN_NODES+8,g61); \
    _mm512_storeu_pd((G)+7*HIDDEN_NODES,g70); _mm512_storeu_pd((G)+7*HIDDEN_NODES+8,g71); \
    _mm512_storeu_pd((G)+8*HIDDEN_NODES,g80); _mm512_storeu_pd((G)+8*HIDDEN_NODES+8,g81); \
    _mm512_storeu_pd((G)+9*HIDDEN_NODES,g90); _mm512_storeu_pd((G)+9*HIDDEN_NODES+8,g91); \
    _mm512_storeu_pd((G)+10*HIDDEN_NODES,g100); _mm512_storeu_pd((G)+10*HIDDEN_NODES+8,g101); \
} while(false)

/**
 * Exact V17 evaluator arithmetic with explicit libmvec call boundaries.
 *
 * V17 keeps 22 first-layer gradient ZMM accumulators live across the 17 vector
 * exp calls in each eight-row tile. Under the SysV ABI those registers are
 * caller-saved, so the compiler spills and reloads a large part of that state
 * around every call. V28 stores only completed tile state in L1 across exp
 * calls, then loads the gradient accumulators after the tile's final sigmoid,
 * performs the complete eight-row backpropagation, and stores them once.
 *
 * Each individual gradient accumulator receives exactly the same FMA sequence
 * as V17, in the same row order. Forward arithmetic, metrics and libmvec calls
 * are unchanged.
 */
__attribute__((target("avx512f,avx512dq,fma"))) static void evaluateSliceV28(
    const Dataset&data,size_t firstRow,size_t lastRow,const Network&network,V15ThreadEvaluation&out){
    const __m512d one=_mm512_set1_pd(1.0),hundred=_mm512_set1_pd(100.0),zero=_mm512_setzero_pd();

    size_t row=firstRow;
    for(;row+V17_FORWARD_ROWS<=lastRow;row+=V17_FORWARD_ROWS){
        const double*x[V17_FORWARD_ROWS];
        __m512d rawLow[V17_FORWARD_ROWS],rawHigh[V17_FORWARD_ROWS];
        for(size_t q=0;q<V17_FORWARD_ROWS;q++){
            x[q]=data.x.rowData(row+q);
            rawLow[q]=zero;
            rawHigh[q]=zero;
        }
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            const __m512d weightsLow=_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES);
            const __m512d weightsHigh=_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES+8);
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                const __m512d value=_mm512_set1_pd(x[q][k]);
                rawLow[q]=_mm512_fmadd_pd(value,weightsLow,rawLow[q]);
                rawHigh[q]=_mm512_fmadd_pd(value,weightsHigh,rawHigh[q]);
            }
        }

        alignas(64) double hidden[V17_FORWARD_ROWS*HIDDEN_NODES];
        for(size_t q=0;q<V17_FORWARD_ROWS;q++){
            double*h=hidden+q*HIDDEN_NODES;
            _mm512_store_pd(h,rawLow[q]);
            _mm512_store_pd(h+8,rawHigh[q]);
        }

        alignas(64) double rawOutput[V17_FORWARD_ROWS],deltaThree[V17_FORWARD_ROWS];
        for(size_t q=0;q<V17_FORWARD_ROWS;q++){
            double*h=hidden+q*HIDDEN_NODES;
            const __m512d hiddenLow=sigmoidVectorV17(_mm512_load_pd(h));
            const __m512d hiddenHigh=sigmoidVectorV17(_mm512_load_pd(h+8));
            _mm512_store_pd(h,hiddenLow);
            _mm512_store_pd(h+8,hiddenHigh);
            const __m512d weighted=_mm512_add_pd(
                _mm512_mul_pd(hiddenLow,_mm512_loadu_pd(network.wTwo.data())),
                _mm512_mul_pd(hiddenHigh,_mm512_loadu_pd(network.wTwo.data()+8)));
            rawOutput[q]=_mm512_reduce_add_pd(weighted);
        }

        const __m512d prediction=sigmoidVectorV17(_mm512_load_pd(rawOutput));
        const __m512d actual=_mm512_loadu_pd(data.y.data()+row);
        const __m512d error=_mm512_sub_pd(prediction,actual);
        const __m512d delta=_mm512_mul_pd(
            _mm512_mul_pd(error,prediction),_mm512_sub_pd(one,prediction));
        _mm512_store_pd(deltaThree,delta);

        out.squaredError+=_mm512_reduce_add_pd(_mm512_mul_pd(error,error));
        __m512d percentage=_mm512_mul_pd(_mm512_div_pd(_mm512_abs_pd(error),actual),hundred);
        const __mmask8 zeroActual=_mm512_cmp_pd_mask(actual,zero,_CMP_EQ_OQ);
        percentage=_mm512_mask_mov_pd(percentage,zeroActual,zero);
        out.percentageErrorSum+=_mm512_reduce_add_pd(percentage);
        alignas(64) double percentageValues[V17_FORWARD_ROWS];
        _mm512_store_pd(percentageValues,percentage);
        for(size_t q=0;q<V17_FORWARD_ROWS;q++)
            out.maxPercentageError=max(out.maxPercentageError,percentageValues[q]);

        double*gradient=out.gradient.data();
        __m512d gradientTwoLow=_mm512_loadu_pd(gradient+WONE_SIZE);
        __m512d gradientTwoHigh=_mm512_loadu_pd(gradient+WONE_SIZE+8);
        V28_DECLARE_W1(gradient);
        const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
        const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);

        for(size_t q=0;q<V17_FORWARD_ROWS;q++){
            const double*h=hidden+q*HIDDEN_NODES;
            const __m512d hiddenLow=_mm512_load_pd(h);
            const __m512d hiddenHigh=_mm512_load_pd(h+8);
            const __m512d d=_mm512_set1_pd(deltaThree[q]);
            gradientTwoLow=_mm512_fmadd_pd(hiddenLow,d,gradientTwoLow);
            gradientTwoHigh=_mm512_fmadd_pd(hiddenHigh,d,gradientTwoHigh);
            const __m512d hiddenDeltaLow=_mm512_mul_pd(
                _mm512_mul_pd(d,wTwoLow),
                _mm512_mul_pd(hiddenLow,_mm512_sub_pd(one,hiddenLow)));
            const __m512d hiddenDeltaHigh=_mm512_mul_pd(
                _mm512_mul_pd(d,wTwoHigh),
                _mm512_mul_pd(hiddenHigh,_mm512_sub_pd(one,hiddenHigh)));
            const double*xq=x[q];
            V28_ACCUM_ALL_W1();
        }

        _mm512_storeu_pd(gradient+WONE_SIZE,gradientTwoLow);
        _mm512_storeu_pd(gradient+WONE_SIZE+8,gradientTwoHigh);
        V28_STORE_W1(gradient);
    }

    for(;row<lastRow;row++){
        const double*x=data.x.rowData(row);
        __m512d rawLow=zero,rawHigh=zero;
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            const __m512d value=_mm512_set1_pd(x[k]);
            rawLow=_mm512_fmadd_pd(
                value,_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES),rawLow);
            rawHigh=_mm512_fmadd_pd(
                value,_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES+8),rawHigh);
        }

        alignas(64) double hidden[HIDDEN_NODES];
        const __m512d hiddenLowBefore=sigmoidVectorV17(rawLow);
        const __m512d hiddenHighBefore=sigmoidVectorV17(rawHigh);
        _mm512_store_pd(hidden,hiddenLowBefore);
        _mm512_store_pd(hidden+8,hiddenHighBefore);
        const double rawOutput=_mm512_reduce_add_pd(_mm512_add_pd(
            _mm512_mul_pd(hiddenLowBefore,_mm512_loadu_pd(network.wTwo.data())),
            _mm512_mul_pd(hiddenHighBefore,_mm512_loadu_pd(network.wTwo.data()+8))));
        const double prediction=sigmoidV15(rawOutput);

        const double actual=data.y[row],error=prediction-actual;
        const double deltaThreeScalar=error*prediction*(1.0-prediction);
        out.squaredError+=error*error;
        const double percentage=actual==0.0?0.0:abs(error/actual)*100.0;
        out.percentageErrorSum+=percentage;
        out.maxPercentageError=max(out.maxPercentageError,percentage);

        double*gradient=out.gradient.data();
        __m512d gradientTwoLow=_mm512_loadu_pd(gradient+WONE_SIZE);
        __m512d gradientTwoHigh=_mm512_loadu_pd(gradient+WONE_SIZE+8);
        V28_DECLARE_W1(gradient);
        const __m512d hiddenLow=_mm512_load_pd(hidden);
        const __m512d hiddenHigh=_mm512_load_pd(hidden+8);
        const __m512d d=_mm512_set1_pd(deltaThreeScalar);
        const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
        const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);
        gradientTwoLow=_mm512_fmadd_pd(hiddenLow,d,gradientTwoLow);
        gradientTwoHigh=_mm512_fmadd_pd(hiddenHigh,d,gradientTwoHigh);
        const __m512d hiddenDeltaLow=_mm512_mul_pd(
            _mm512_mul_pd(d,wTwoLow),
            _mm512_mul_pd(hiddenLow,_mm512_sub_pd(one,hiddenLow)));
        const __m512d hiddenDeltaHigh=_mm512_mul_pd(
            _mm512_mul_pd(d,wTwoHigh),
            _mm512_mul_pd(hiddenHigh,_mm512_sub_pd(one,hiddenHigh)));
        const double*xq=x;
        V28_ACCUM_ALL_W1();

        _mm512_storeu_pd(gradient+WONE_SIZE,gradientTwoLow);
        _mm512_storeu_pd(gradient+WONE_SIZE+8,gradientTwoHigh);
        V28_STORE_W1(gradient);
    }
}

static V14Evaluation evaluateV28(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                                 const vector<double>&parameters,size_t requestedThreads){
    if(!supportsV16Kernel())
        return evaluateV25(data,startRow,batchSize,network,parameters,requestedThreads);
    unpackV14(parameters,network);
    const size_t threads=evaluationThreadsV25(requestedThreads,batchSize);
    vector<V15ThreadEvaluation>partials(threads);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(threads))
    {
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
        const size_t base=batchSize/threads,remainder=batchSize%threads;
        const size_t offset=tid*base+min(tid,remainder);
        const size_t count=base+(tid<remainder?1:0);
        evaluateSliceV28(data,startRow+offset,startRow+offset+count,network,partials[tid]);
    }
#else
    evaluateSliceV28(data,startRow,startRow+batchSize,network,partials[0]);
#endif
    V14Evaluation result;
    result.gradient.assign(V14_PARAMETER_COUNT,0.0);
    long double squaredError=0.0L,percentageErrorSum=0.0L;
    double maxPercentageError=0.0;
    for(size_t t=0;t<threads;t++){
        squaredError+=partials[t].squaredError;
        percentageErrorSum+=partials[t].percentageErrorSum;
        maxPercentageError=max(maxPercentageError,partials[t].maxPercentageError);
        for(size_t i=0;i<V14_PARAMETER_COUNT;i++)
            result.gradient[i]+=partials[t].gradient[i];
    }
    const double inv=1.0/static_cast<double>(batchSize);
    result.metrics.cost=.5*static_cast<double>(squaredError);
    result.metrics.percentageError=static_cast<double>(percentageErrorSum)*inv;
    result.metrics.maxPercentageError=maxPercentageError;
    result.objective=result.metrics.cost*inv;
    for(double&value:result.gradient)value*=inv;
    return result;
}

static bool armijoStepV28(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                          const OptimiserConfig&config,const vector<double>&parameters,
                          const vector<double>&direction,double directional,double currentObjective,
                          vector<double>&candidate,V14Evaluation&next){
    double step=1.0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        for(size_t k=0;k<parameters.size();k++)
            candidate[k]=parameters[k]+step*direction[k];
        next=evaluateV28(data,startRow,batchSize,network,candidate,config.threads);
        if(next.objective<=currentObjective+V14_ARMIJO*step*directional)return true;
        step*=0.5;
    }
    return false;
}

static bool armijoDenseStepV28(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                               const OptimiserConfig&config,const vector<double>&parameters,
                               const vector<double>&direction,double directional,double currentObjective,
                               vector<double>&candidate,V14Evaluation&next){
    double step=1.0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        for(size_t k=0;k<parameters.size();k++)
            candidate[k]=parameters[k]+step*direction[k];
        next=evaluateV28(data,startRow,batchSize,network,candidate,config.threads);
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

/** Match v27's GN policy exactly at both sides of its 50k boundary. */
static void prepareDeepStateV28(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                                const vector<double>&parameters,const OptimiserConfig&config,
                                double percentageError,deque<V14HistoryPair>&history,bool&deepStage,
                                bool&scaleReady,bool&denseReady,unsigned&refreshStage,
                                V20DiagonalScale&scale,vector<double>&denseInverse){
    if(batchSize<V27_GN_PARALLEL_ROW_LIMIT){
        prepareDeepStateV27(data,startRow,batchSize,network,parameters,config,percentageError,
                            history,deepStage,scaleReady,denseReady,refreshStage,scale,denseInverse);
        return;
    }

    deepStage=deepStageV19(deepStage,percentageError);
    if(deepStage&&!scaleReady){
        scale=gaussNewtonScaleV20(data,startRow,batchSize,network,parameters,config.threads);
        scaleReady=true;
        refreshStage=refreshStageV21(percentageError,config.percentageErrorTarget);
    }
    if(scaleReady&&!denseReady){
        refreshScaleV21(data,startRow,batchSize,network,parameters,config,
                        percentageError,refreshStage,scale);
        if(percentageError<=V22_DENSE_SWITCH_PERCENTAGE){
            initialiseDenseInverseV23(history,scale,denseInverse);
            denseReady=true;
        }
    }
}
#endif

static TrainResult trainRangeV28(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(config.percentageErrorTarget>V21_REFRESH_TARGET||!supportsV16Kernel())
        return trainRangeV27(data,startRow,batchSize,batchIndex,network,config);

    TrainResult result;
    deque<V14HistoryPair>history;
    vector<double>parameters=packV14(network);
    V14Evaluation current=evaluateV28(data,startRow,batchSize,network,parameters,config.threads);
    bool deepStage=false,scaleReady=false,denseReady=false;
    unsigned refreshStage=0;
    V20DiagonalScale scale{};
    vector<double>denseInverse;
    if(config.logEvery!=0)printProgress(batchIndex,0,current.metrics);

    while(current.metrics.percentageError>config.percentageErrorTarget&&result.updates<config.maxDescents){
        prepareDeepStateV28(data,startRow,batchSize,network,parameters,config,
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
            ?armijoDenseStepV28(data,startRow,batchSize,network,config,parameters,direction,directional,
                                current.objective,candidate,next)
            :armijoStepV28(data,startRow,batchSize,network,config,parameters,direction,directional,
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
    if(config.logEvery!=0||result.reachedTarget)
        printProgress(batchIndex,result.updates,result.metrics);
    writePredictions(data,startRow,batchSize,network,"ybar.txt");
    return result;
#else
    return trainRangeV27(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV28(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV28(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)
            cout<<"Batch "<<batch<<" reached the maximum number of BFGS iterations before the target.\n";
    }
}

static void offlineRunV28(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV28(data,0,rowCount,0,network,optimiser);
    writeWeights(network);
    if(!result.reachedTarget)
        cout<<"Offline BFGS reached the maximum number of iterations before the target.\n";
}

#ifndef SIMPLE_NN_V28_NO_MAIN
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
            batchOnlineRunV28(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,
                              randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV28(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }
        if(test)for(size_t i=0;i<times;i++)testRun(testRows);
        const chrono::duration<double> elapsed=chrono::steady_clock::now()-startTime;
        cout<<"Elapsed time = "<<elapsed.count()<<" seconds\n";
        return 0;
    }
    catch(const exception&error){
        cerr<<"Error: "<<error.what()<<'\n';
        return 1;
    }
}
#endif

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
#undef V28_DECLARE_W1
#undef V28_ACCUM_W1
#undef V28_ACCUM_ALL_W1
#undef V28_STORE_W1
#endif
