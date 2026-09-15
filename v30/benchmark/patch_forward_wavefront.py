from pathlib import Path

path = Path('v30/main.cpp')
text = path.read_text()
start = text.index('static V14Evaluation evaluatePhysicalForwardV30(')
end = text.index('\nstatic bool armijoStepPhysicalV30(', start)

replacement = r'''static V14Evaluation evaluatePhysicalForwardV30(
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
    vector<size_t>sliceFirst(logicalThreads),sliceLast(logicalThreads),sliceFullEnd(logicalThreads);
    size_t maxGroups=0;
    const size_t base=batchSize/logicalThreads,remainder=batchSize%logicalThreads;
    for(size_t t=0;t<logicalThreads;t++){
        const size_t offset=t*base+min(t,remainder);
        const size_t count=base+(t<remainder?1:0);
        sliceFirst[t]=startRow+offset;
        sliceLast[t]=sliceFirst[t]+count;
        const size_t fullRows=(count/V17_FORWARD_ROWS)*V17_FORWARD_ROWS;
        sliceFullEnd[t]=sliceFirst[t]+fullRows;
        const size_t groups=(fullRows+V29_GROUP_ROWS-1)/V29_GROUP_ROWS;
        maxGroups=max(maxGroups,groups);
    }

    const size_t scratchRows=logicalThreads*V29_GROUP_ROWS;
    vector<double>hidden(scratchRows*HIDDEN_NODES);
    vector<double>deltaThree(scratchRows);
    vector<double>percentage(scratchRows);
    vector<double>tileSquaredError(logicalThreads*V29_GROUP_TILES);
    vector<double>tilePercentageError(logicalThreads*V29_GROUP_TILES);

#pragma omp parallel num_threads(static_cast<int>(physicalThreads))
    {
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
        for(size_t group=0;group<maxGroups;group++){
#pragma omp for schedule(static)
            for(long long task=0;task<static_cast<long long>(logicalThreads*V29_GROUP_TILES);task++){
                const size_t u=static_cast<size_t>(task);
                const size_t owner=u/V29_GROUP_TILES;
                const size_t tile=u%V29_GROUP_TILES;
                const size_t groupStart=sliceFirst[owner]+group*V29_GROUP_ROWS;
                const size_t tileRow=groupStart+tile*V17_FORWARD_ROWS;
                if(tileRow+V17_FORWARD_ROWS<=sliceFullEnd[owner]){
                    const size_t scratchBase=owner*V29_GROUP_ROWS;
                    forwardTilePhysicalV30(data,groupStart,tileRow,network,
                        hidden.data()+scratchBase*HIDDEN_NODES,
                        deltaThree.data()+scratchBase,
                        percentage.data()+scratchBase,
                        tileSquaredError[owner*V29_GROUP_TILES+tile],
                        tilePercentageError[owner*V29_GROUP_TILES+tile]);
                }
            }

            if(tid<logicalThreads){
                const size_t groupStart=sliceFirst[tid]+group*V29_GROUP_ROWS;
                if(groupStart<sliceFullEnd[tid]){
                    const size_t groupRows=min(V29_GROUP_ROWS,sliceFullEnd[tid]-groupStart);
                    const size_t scratchBase=tid*V29_GROUP_ROWS;
                    reducePhysicalSliceV30(data,groupStart,groupStart,groupStart+groupRows,network,
                        hidden.data()+scratchBase*HIDDEN_NODES,
                        deltaThree.data()+scratchBase,
                        percentage.data()+scratchBase,
                        tileSquaredError.data()+tid*V29_GROUP_TILES,
                        tilePercentageError.data()+tid*V29_GROUP_TILES,
                        0,partials[tid]);
                }
            }
#pragma omp barrier
        }
        if(tid<logicalThreads&&sliceFullEnd[tid]<sliceLast[tid])
            evaluateSliceV28(data,sliceFullEnd[tid],sliceLast[tid],network,partials[tid]);
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
'''

path.write_text(text[:start] + replacement + text[end:])
