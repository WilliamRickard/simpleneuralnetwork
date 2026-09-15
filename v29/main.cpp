#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define SIMPLE_NN_V28_NO_MAIN
#include "../v28/main.cpp"
#undef SIMPLE_NN_V28_NO_MAIN
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* V29: amortise exact evaluator state across several eight-row tiles. */
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)

static_assert(NUMBER_OF_VARIABLES==11&&HIDDEN_NODES==16,
              "V29 phased evaluator requires the fixed 11->16->1 architecture");
constexpr size_t V29_GROUP_TILES=12;
constexpr size_t V29_GROUP_ROWS=V29_GROUP_TILES*V17_FORWARD_ROWS;

#define V29_DECLARE_W1(G) \
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

#define V29_ACCUM_W1(K,G0,G1) do { \
    const __m512d xv=_mm512_set1_pd(xq[(K)]); \
    (G0)=_mm512_fmadd_pd(xv,hiddenDeltaLow,(G0)); \
    (G1)=_mm512_fmadd_pd(xv,hiddenDeltaHigh,(G1)); \
} while(false)

#define V29_ACCUM_ALL_W1() do { \
    V29_ACCUM_W1(0,g00,g01); V29_ACCUM_W1(1,g10,g11); V29_ACCUM_W1(2,g20,g21); \
    V29_ACCUM_W1(3,g30,g31); V29_ACCUM_W1(4,g40,g41); V29_ACCUM_W1(5,g50,g51); \
    V29_ACCUM_W1(6,g60,g61); V29_ACCUM_W1(7,g70,g71); V29_ACCUM_W1(8,g80,g81); \
    V29_ACCUM_W1(9,g90,g91); V29_ACCUM_W1(10,g100,g101); \
} while(false)

#define V29_STORE_W1(G) do { \
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
 * Evaluate one slice with exactly the same floating-point operations as v28,
 * but schedule independent work in larger phases.
 *
 * For up to 12 eight-row tiles (96 rows):
 *  1. compute first-layer forward values and hidden sigmoids;
 *  2. compute output dot products with wTwo resident across the group;
 *  3. compute output sigmoids and metrics in the original tile order;
 *  4. load gradient state once, backpropagate every row in original order,
 *     then store gradient state once.
 *
 * Stores/loads removed between tiles are exact double round-trips, so deleting
 * them does not change any accumulator value. Metric reductions retain v28's
 * tile order, and every gradient accumulator receives the same FMA sequence.
 */
__attribute__((target("avx512f,avx512dq,fma"))) static void evaluateSliceV29(
    const Dataset&data,size_t firstRow,size_t lastRow,const Network&network,V15ThreadEvaluation&out){
    const __m512d one=_mm512_set1_pd(1.0);
    const __m512d hundred=_mm512_set1_pd(100.0);
    const __m512d zero=_mm512_setzero_pd();
    alignas(64) double hidden[V29_GROUP_ROWS*HIDDEN_NODES];
    alignas(64) double rawOutput[V29_GROUP_ROWS];
    alignas(64) double deltaThree[V29_GROUP_ROWS];

    size_t row=firstRow;
    while(row+V17_FORWARD_ROWS<=lastRow){
        const size_t fullTiles=(lastRow-row)/V17_FORWARD_ROWS;
        const size_t tiles=min(V29_GROUP_TILES,fullTiles);
        const size_t groupRows=tiles*V17_FORWARD_ROWS;

        /* Phase 1: first layer and hidden sigmoids, retaining v28 row arithmetic. */
        for(size_t tile=0;tile<tiles;tile++){
            const size_t tileRow=row+tile*V17_FORWARD_ROWS;
            const double*x[V17_FORWARD_ROWS];
            __m512d rawLow[V17_FORWARD_ROWS],rawHigh[V17_FORWARD_ROWS];
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                x[q]=data.x.rowData(tileRow+q);
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
            double*tileHidden=hidden+tile*V17_FORWARD_ROWS*HIDDEN_NODES;
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                double*h=tileHidden+q*HIDDEN_NODES;
                _mm512_store_pd(h,sigmoidVectorV17(rawLow[q]));
                _mm512_store_pd(h+8,sigmoidVectorV17(rawHigh[q]));
            }
        }

        /* Phase 2: output dot products with output weights live across the group. */
        {
            const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
            const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);
            for(size_t rr=0;rr<groupRows;rr++){
                const double*h=hidden+rr*HIDDEN_NODES;
                rawOutput[rr]=_mm512_reduce_add_pd(_mm512_add_pd(
                    _mm512_mul_pd(_mm512_load_pd(h),wTwoLow),
                    _mm512_mul_pd(_mm512_load_pd(h+8),wTwoHigh)));
            }
        }

        /* Phase 3: output sigmoid and metrics, preserving v28 tile order. */
        for(size_t tile=0;tile<tiles;tile++){
            const size_t tileRow=row+tile*V17_FORWARD_ROWS;
            const __m512d prediction=sigmoidVectorV17(
                _mm512_load_pd(rawOutput+tile*V17_FORWARD_ROWS));
            const __m512d actual=_mm512_loadu_pd(data.y.data()+tileRow);
            const __m512d error=_mm512_sub_pd(prediction,actual);
            const __m512d delta=_mm512_mul_pd(
                _mm512_mul_pd(error,prediction),_mm512_sub_pd(one,prediction));
            _mm512_store_pd(deltaThree+tile*V17_FORWARD_ROWS,delta);

            out.squaredError+=_mm512_reduce_add_pd(_mm512_mul_pd(error,error));
            __m512d percentage=_mm512_mul_pd(
                _mm512_div_pd(_mm512_abs_pd(error),actual),hundred);
            const __mmask8 zeroActual=_mm512_cmp_pd_mask(actual,zero,_CMP_EQ_OQ);
            percentage=_mm512_mask_mov_pd(percentage,zeroActual,zero);
            out.percentageErrorSum+=_mm512_reduce_add_pd(percentage);
            alignas(64) double percentageValues[V17_FORWARD_ROWS];
            _mm512_store_pd(percentageValues,percentage);
            for(size_t q=0;q<V17_FORWARD_ROWS;q++)
                out.maxPercentageError=max(out.maxPercentageError,percentageValues[q]);
        }

        /* Phase 4: one gradient state load/store for the complete group. */
        double*gradient=out.gradient.data();
        __m512d gradientTwoLow=_mm512_loadu_pd(gradient+WONE_SIZE);
        __m512d gradientTwoHigh=_mm512_loadu_pd(gradient+WONE_SIZE+8);
        V29_DECLARE_W1(gradient);
        const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
        const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);

        for(size_t rr=0;rr<groupRows;rr++){
            const double*h=hidden+rr*HIDDEN_NODES;
            const __m512d hiddenLow=_mm512_load_pd(h);
            const __m512d hiddenHigh=_mm512_load_pd(h+8);
            const __m512d d=_mm512_set1_pd(deltaThree[rr]);
            gradientTwoLow=_mm512_fmadd_pd(hiddenLow,d,gradientTwoLow);
            gradientTwoHigh=_mm512_fmadd_pd(hiddenHigh,d,gradientTwoHigh);
            const __m512d hiddenDeltaLow=_mm512_mul_pd(
                _mm512_mul_pd(d,wTwoLow),
                _mm512_mul_pd(hiddenLow,_mm512_sub_pd(one,hiddenLow)));
            const __m512d hiddenDeltaHigh=_mm512_mul_pd(
                _mm512_mul_pd(d,wTwoHigh),
                _mm512_mul_pd(hiddenHigh,_mm512_sub_pd(one,hiddenHigh)));
            const double*xq=data.x.rowData(row+rr);
            V29_ACCUM_ALL_W1();
        }

        _mm512_storeu_pd(gradient+WONE_SIZE,gradientTwoLow);
        _mm512_storeu_pd(gradient+WONE_SIZE+8,gradientTwoHigh);
        V29_STORE_W1(gradient);
        row+=groupRows;
    }

    /* Fewer than eight rows use the exact inherited scalar tail. */
    if(row<lastRow)evaluateSliceV28(data,row,lastRow,network,out);
}

static V14Evaluation evaluateV29(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                                 const vector<double>&parameters,size_t requestedThreads){
    if(!supportsV16Kernel())
        return evaluateV28(data,startRow,batchSize,network,parameters,requestedThreads);
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
        evaluateSliceV29(data,startRow+offset,startRow+offset+count,network,partials[tid]);
    }
#else
    evaluateSliceV29(data,startRow,startRow+batchSize,network,partials[0]);
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

static bool armijoStepV29(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                          const OptimiserConfig&config,const vector<double>&parameters,
                          const vector<double>&direction,double directional,double currentObjective,
                          vector<double>&candidate,V14Evaluation&next){
    double step=1.0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        for(size_t k=0;k<parameters.size();k++)
            candidate[k]=parameters[k]+step*direction[k];
        next=evaluateV29(data,startRow,batchSize,network,candidate,config.threads);
        if(next.objective<=currentObjective+V14_ARMIJO*step*directional)return true;
        step*=0.5;
    }
    return false;
}

static bool armijoDenseStepV29(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                               const OptimiserConfig&config,const vector<double>&parameters,
                               const vector<double>&direction,double directional,double currentObjective,
                               vector<double>&candidate,V14Evaluation&next){
    double step=1.0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        for(size_t k=0;k<parameters.size();k++)
            candidate[k]=parameters[k]+step*direction[k];
        next=evaluateV29(data,startRow,batchSize,network,candidate,config.threads);
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

static TrainResult trainRangeV29(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,
                                 Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(config.percentageErrorTarget>V21_REFRESH_TARGET||!supportsV16Kernel())
        return trainRangeV28(data,startRow,batchSize,batchIndex,network,config);

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
    result.metrics=confirmMetricsV25(data,startRow,batchSize,network,config.threads);
    result.reachedTarget=result.metrics.percentageError<=config.percentageErrorTarget;
    if(config.logEvery!=0||result.reachedTarget)
        printProgress(batchIndex,result.updates,result.metrics);
    writePredictions(data,startRow,batchSize,network,"ybar.txt");
    return result;
#else
    return trainRangeV28(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV29(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,
                              double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(totalRows,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){
        const size_t startRow=batch*exampleSize;
        const TrainResult result=trainRangeV29(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(!result.reachedTarget)
            cout<<"Batch "<<batch<<" reached the maximum number of BFGS iterations before the target.\n";
    }
}

static void offlineRunV29(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,
                          bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;
    Dataset data=loadDatasetV9(rowCount,bounds);
    Network network;
    if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    const TrainResult result=trainRangeV29(data,0,rowCount,0,network,optimiser);
    writeWeights(network);
    if(!result.reachedTarget)
        cout<<"Offline BFGS reached the maximum number of iterations before the target.\n";
}

#ifndef SIMPLE_NN_V29_NO_MAIN
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
            batchOnlineRunV29(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,
                              randomiseWeights,generator);
        }
        if(offline)for(size_t i=0;i<times;i++){
            OptimiserConfig optimiser;
            optimiser.maxDescents=numberOfDescents;
            optimiser.percentageErrorTarget=percentageErrorTarget;
            optimiser.logEvery=logEvery;
            optimiser.threads=trainingThreads;
            offlineRunV29(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
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
#undef V29_DECLARE_W1
#undef V29_ACCUM_W1
#undef V29_ACCUM_ALL_W1
#undef V29_STORE_W1
#endif
