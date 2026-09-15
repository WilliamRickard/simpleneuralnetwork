from pathlib import Path

path = Path('v30/main.cpp')
text = path.read_text()
marker = '/** Preserve v29 exactly except for exact GN preparation and final prediction reuse. */\nstatic TrainResult trainRangeV30'
if marker not in text:
    raise SystemExit('trainRangeV30 marker not found')

insert = r'''
/** Scratch-backed exact forward tile used by the experimental physical-worker evaluator. */
__attribute__((target("avx512f,avx512dq,fma")))
static void forwardTilePhysicalV30(
    const Dataset&data,size_t startRow,size_t tileRow,const Network&network,
    double*hidden,double*deltaThree,double*percentage,
    double&tileSquaredError,double&tilePercentageError){
    const __m512d one=_mm512_set1_pd(1.0);
    const __m512d hundred=_mm512_set1_pd(100.0);
    const __m512d zero=_mm512_setzero_pd();
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

    alignas(64) double rawOutput[V17_FORWARD_ROWS];
    const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
    const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);
    for(size_t q=0;q<V17_FORWARD_ROWS;q++){
        double*h=hidden+(tileRow-startRow+q)*HIDDEN_NODES;
        const __m512d hiddenLow=sigmoidVectorV17(rawLow[q]);
        const __m512d hiddenHigh=sigmoidVectorV17(rawHigh[q]);
        _mm512_storeu_pd(h,hiddenLow);
        _mm512_storeu_pd(h+8,hiddenHigh);
        rawOutput[q]=_mm512_reduce_add_pd(_mm512_add_pd(
            _mm512_mul_pd(hiddenLow,wTwoLow),_mm512_mul_pd(hiddenHigh,wTwoHigh)));
    }

    const __m512d prediction=sigmoidVectorV17(_mm512_load_pd(rawOutput));
    const __m512d actual=_mm512_loadu_pd(data.y.data()+tileRow);
    const __m512d error=_mm512_sub_pd(prediction,actual);
    const __m512d delta=_mm512_mul_pd(
        _mm512_mul_pd(error,prediction),_mm512_sub_pd(one,prediction));
    _mm512_storeu_pd(deltaThree+(tileRow-startRow),delta);
    tileSquaredError=_mm512_reduce_add_pd(_mm512_mul_pd(error,error));
    __m512d pct=_mm512_mul_pd(_mm512_div_pd(_mm512_abs_pd(error),actual),hundred);
    const __mmask8 zeroActual=_mm512_cmp_pd_mask(actual,zero,_CMP_EQ_OQ);
    pct=_mm512_mask_mov_pd(pct,zeroActual,zero);
    tilePercentageError=_mm512_reduce_add_pd(pct);
    _mm512_storeu_pd(percentage+(tileRow-startRow),pct);
}

/**
 * Reduce one logical evaluator slice from precomputed forward states.
 * Metric tile order and every gradient accumulator's row-order FMA sequence
 * are identical to evaluateSliceV29. The scalar tail remains evaluateSliceV28.
 */
__attribute__((target("avx512f,avx512dq,fma")))
static void reducePhysicalSliceV30(
    const Dataset&data,size_t startRow,size_t firstRow,size_t lastRow,const Network&network,
    const double*hidden,const double*deltaThree,const double*percentage,
    const double*tileSquaredError,const double*tilePercentageError,size_t firstTile,
    V15ThreadEvaluation&out){
    const __m512d one=_mm512_set1_pd(1.0);
    const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
    const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);
    size_t row=firstRow;
    size_t tileIndex=firstTile;
    while(row+V17_FORWARD_ROWS<=lastRow){
        const size_t fullTiles=(lastRow-row)/V17_FORWARD_ROWS;
        const size_t tiles=min(V29_GROUP_TILES,fullTiles);
        const size_t groupRows=tiles*V17_FORWARD_ROWS;

        for(size_t tile=0;tile<tiles;tile++){
            out.squaredError+=tileSquaredError[tileIndex+tile];
            out.percentageErrorSum+=tilePercentageError[tileIndex+tile];
            const size_t localRow=row-startRow+tile*V17_FORWARD_ROWS;
            for(size_t q=0;q<V17_FORWARD_ROWS;q++)
                out.maxPercentageError=max(out.maxPercentageError,percentage[localRow+q]);
        }

        double*gradient=out.gradient.data();
        __m512d gradientTwoLow=_mm512_loadu_pd(gradient+WONE_SIZE);
        __m512d gradientTwoHigh=_mm512_loadu_pd(gradient+WONE_SIZE+8);
        V29_DECLARE_W1(gradient);
        for(size_t rr=0;rr<groupRows;rr++){
            const size_t localRow=row-startRow+rr;
            const double*h=hidden+localRow*HIDDEN_NODES;
            const __m512d hiddenLow=_mm512_loadu_pd(h);
            const __m512d hiddenHigh=_mm512_loadu_pd(h+8);
            const __m512d d=_mm512_set1_pd(deltaThree[localRow]);
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
        tileIndex+=tiles;
    }
    if(row<lastRow)evaluateSliceV28(data,row,lastRow,network,out);
}

/**
 * Preserve v29's logical worker partition exactly while using up to twice as
 * many physical workers for independent eight-row forward tiles.
 */
static V14Evaluation evaluatePhysicalForwardV30(
    const Dataset&data,size_t startRow,size_t batchSize,Network&network,
    const vector<double>&parameters,size_t requestedThreads){
    if(!supportsV16Kernel())
        return evaluateV29(data,startRow,batchSize,network,parameters,requestedThreads);
    unpackV14(parameters,network);
    const size_t logicalThreads=evaluationThreadsV25(requestedThreads,batchSize);
#ifndef _OPENMP
    return evaluateV29(data,startRow,batchSize,network,parameters,requestedThreads);
#else
    const size_t physicalThreads=min(V15_MAX_EVALUATION_THREADS,
                                     max(logicalThreads,logicalThreads*2));
    if(physicalThreads==logicalThreads)
        return evaluateV29(data,startRow,batchSize,network,parameters,requestedThreads);

    vector<V15ThreadEvaluation>partials(logicalThreads);
    vector<size_t>sliceFirst(logicalThreads),sliceLast(logicalThreads),sliceFirstTile(logicalThreads);
    size_t totalTiles=0;
    const size_t base=batchSize/logicalThreads,remainder=batchSize%logicalThreads;
    for(size_t t=0;t<logicalThreads;t++){
        const size_t offset=t*base+min(t,remainder);
        const size_t count=base+(t<remainder?1:0);
        sliceFirst[t]=startRow+offset;
        sliceLast[t]=sliceFirst[t]+count;
        sliceFirstTile[t]=totalTiles;
        totalTiles+=count/V17_FORWARD_ROWS;
    }

    vector<size_t>tileRows(totalTiles);
    for(size_t t=0;t<logicalThreads;t++){
        size_t index=sliceFirstTile[t];
        for(size_t row=sliceFirst[t];row+V17_FORWARD_ROWS<=sliceLast[t];row+=V17_FORWARD_ROWS)
            tileRows[index++]=row;
    }
    vector<double>hidden(batchSize*HIDDEN_NODES);
    vector<double>deltaThree(batchSize);
    vector<double>percentage(batchSize);
    vector<double>tileSquaredError(totalTiles),tilePercentageError(totalTiles);

#pragma omp parallel num_threads(static_cast<int>(physicalThreads))
    {
#pragma omp for schedule(static)
        for(long long tile=0;tile<static_cast<long long>(totalTiles);tile++){
            const size_t i=static_cast<size_t>(tile);
            forwardTilePhysicalV30(data,startRow,tileRows[i],network,
                                   hidden.data(),deltaThree.data(),percentage.data(),
                                   tileSquaredError[i],tilePercentageError[i]);
        }
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
        if(tid<logicalThreads)
            reducePhysicalSliceV30(data,startRow,sliceFirst[tid],sliceLast[tid],network,
                                   hidden.data(),deltaThree.data(),percentage.data(),
                                   tileSquaredError.data(),tilePercentageError.data(),
                                   sliceFirstTile[tid],partials[tid]);
    }

    V14Evaluation result;
    result.gradient.assign(V14_PARAMETER_COUNT,0.0);
    long double squaredError=0.0L,percentageErrorSum=0.0L;
    double maxPercentageError=0.0;
    for(size_t t=0;t<logicalThreads;t++){
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
#endif
}

static bool armijoStepPhysicalV30(
    const Dataset&data,size_t startRow,size_t batchSize,Network&network,
    const OptimiserConfig&config,const vector<double>&parameters,
    const vector<double>&direction,double directional,double currentObjective,
    vector<double>&candidate,V14Evaluation&next){
    double step=1.0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        for(size_t k=0;k<parameters.size();k++)candidate[k]=parameters[k]+step*direction[k];
        next=evaluatePhysicalForwardV30(data,startRow,batchSize,network,candidate,config.threads);
        if(next.objective<=currentObjective+V14_ARMIJO*step*directional)return true;
        step*=0.5;
    }
    return false;
}

static bool armijoDenseStepPhysicalV30(
    const Dataset&data,size_t startRow,size_t batchSize,Network&network,
    const OptimiserConfig&config,const vector<double>&parameters,
    const vector<double>&direction,double directional,double currentObjective,
    vector<double>&candidate,V14Evaluation&next){
    double step=1.0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        for(size_t k=0;k<parameters.size();k++)candidate[k]=parameters[k]+step*direction[k];
        next=evaluatePhysicalForwardV30(data,startRow,batchSize,network,candidate,config.threads);
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

'''

text = text.replace(marker, insert + marker, 1)
text = text.replace(
    'V14Evaluation current=evaluateV29(data,startRow,batchSize,network,parameters,config.threads);',
    'V14Evaluation current=evaluatePhysicalForwardV30(data,startRow,batchSize,network,parameters,config.threads);',
    1)
text = text.replace(
    '?armijoDenseStepV29(data,startRow,batchSize,network,config,parameters,direction,directional,',
    '?armijoDenseStepPhysicalV30(data,startRow,batchSize,network,config,parameters,direction,directional,',
    1)
text = text.replace(
    ':armijoStepV29(data,startRow,batchSize,network,config,parameters,direction,directional,',
    ':armijoStepPhysicalV30(data,startRow,batchSize,network,config,parameters,direction,directional,',
    1)
path.write_text(text)
