#include<iostream>
#include<algorithm>
#include<array>
#include<chrono>
#include<cmath>
#include<cstddef>
#include<ctime>
#include<deque>
#include<exception>
#include<fstream>
#include<iomanip>
#include<random>
#include<stdexcept>
#include<string>
#include<vector>
#if defined(__x86_64__) && defined(__GNUC__)
#include<immintrin.h>
#endif
#ifdef _OPENMP
#include<omp.h>
#endif

namespace v17_v16 {
#include "../v16/main.cpp"
}
using namespace v17_v16;

/* V17: vectorised output layer plus parallel deterministic scalar confirmation. */
constexpr size_t V17_FORWARD_ROWS=8;

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
extern "C" __m512d _ZGVeN8v_exp(__m512d);

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d sigmoidVectorV17(__m512d value){
    const __m512d one=_mm512_set1_pd(1.0),zero=_mm512_setzero_pd();
    return _mm512_div_pd(one,_mm512_add_pd(one,_ZGVeN8v_exp(_mm512_sub_pd(zero,value))));
}

__attribute__((target("avx512f,avx512dq,fma"))) static void evaluateSliceV17(
    const Dataset&data,size_t firstRow,size_t lastRow,const Network&network,V15ThreadEvaluation&out){
    const __m512d one=_mm512_set1_pd(1.0),hundred=_mm512_set1_pd(100.0),zero=_mm512_setzero_pd();
    const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data()),wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);
    __m512d gradientTwoLow=_mm512_setzero_pd(),gradientTwoHigh=_mm512_setzero_pd();
    __m512d gradientOneLow[NUMBER_OF_VARIABLES],gradientOneHigh[NUMBER_OF_VARIABLES];
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){gradientOneLow[k]=_mm512_setzero_pd();gradientOneHigh[k]=_mm512_setzero_pd();}

    size_t row=firstRow;
    for(;row+V17_FORWARD_ROWS<=lastRow;row+=V17_FORWARD_ROWS){
        const double*x[V17_FORWARD_ROWS];__m512d rawLow[V17_FORWARD_ROWS],rawHigh[V17_FORWARD_ROWS];
        for(size_t q=0;q<V17_FORWARD_ROWS;q++){x[q]=data.x.rowData(row+q);rawLow[q]=zero;rawHigh[q]=zero;}
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            const __m512d weightsLow=_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES),weightsHigh=_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES+8);
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                const __m512d value=_mm512_set1_pd(x[q][k]);
                rawLow[q]=_mm512_fmadd_pd(value,weightsLow,rawLow[q]);rawHigh[q]=_mm512_fmadd_pd(value,weightsHigh,rawHigh[q]);
            }
        }
        __m512d hiddenLow[V17_FORWARD_ROWS],hiddenHigh[V17_FORWARD_ROWS];alignas(64) double rawOutput[V17_FORWARD_ROWS],deltaThree[V17_FORWARD_ROWS];
        for(size_t q=0;q<V17_FORWARD_ROWS;q++){
            hiddenLow[q]=sigmoidVectorV17(rawLow[q]);hiddenHigh[q]=sigmoidVectorV17(rawHigh[q]);
            rawOutput[q]=_mm512_reduce_add_pd(_mm512_add_pd(_mm512_mul_pd(hiddenLow[q],wTwoLow),_mm512_mul_pd(hiddenHigh[q],wTwoHigh)));
        }
        const __m512d prediction=sigmoidVectorV17(_mm512_load_pd(rawOutput));
        const __m512d actual=_mm512_loadu_pd(data.y.data()+row),error=_mm512_sub_pd(prediction,actual);
        const __m512d delta=_mm512_mul_pd(_mm512_mul_pd(error,prediction),_mm512_sub_pd(one,prediction));
        _mm512_store_pd(deltaThree,delta);
        out.squaredError+=_mm512_reduce_add_pd(_mm512_mul_pd(error,error));
        __m512d percentage=_mm512_mul_pd(_mm512_div_pd(_mm512_abs_pd(error),actual),hundred);
        const __mmask8 zeroActual=_mm512_cmp_pd_mask(actual,zero,_CMP_EQ_OQ);
        percentage=_mm512_mask_mov_pd(percentage,zeroActual,zero);
        out.percentageErrorSum+=_mm512_reduce_add_pd(percentage);
        alignas(64) double percentageValues[V17_FORWARD_ROWS];_mm512_store_pd(percentageValues,percentage);
        for(size_t q=0;q<V17_FORWARD_ROWS;q++)out.maxPercentageError=max(out.maxPercentageError,percentageValues[q]);

        for(size_t q=0;q<V17_FORWARD_ROWS;q++){
            const __m512d d=_mm512_set1_pd(deltaThree[q]);
            gradientTwoLow=_mm512_fmadd_pd(hiddenLow[q],d,gradientTwoLow);gradientTwoHigh=_mm512_fmadd_pd(hiddenHigh[q],d,gradientTwoHigh);
            const __m512d hiddenDeltaLow=_mm512_mul_pd(_mm512_mul_pd(d,wTwoLow),_mm512_mul_pd(hiddenLow[q],_mm512_sub_pd(one,hiddenLow[q])));
            const __m512d hiddenDeltaHigh=_mm512_mul_pd(_mm512_mul_pd(d,wTwoHigh),_mm512_mul_pd(hiddenHigh[q],_mm512_sub_pd(one,hiddenHigh[q])));
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
                const __m512d value=_mm512_set1_pd(x[q][k]);
                gradientOneLow[k]=_mm512_fmadd_pd(value,hiddenDeltaLow,gradientOneLow[k]);gradientOneHigh[k]=_mm512_fmadd_pd(value,hiddenDeltaHigh,gradientOneHigh[k]);
            }
        }
    }

    for(;row<lastRow;row++){
        const double*x=data.x.rowData(row);__m512d rawLow=zero,rawHigh=zero;
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            const __m512d value=_mm512_set1_pd(x[k]);
            rawLow=_mm512_fmadd_pd(value,_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES),rawLow);
            rawHigh=_mm512_fmadd_pd(value,_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES+8),rawHigh);
        }
        const __m512d hiddenLow=sigmoidVectorV17(rawLow),hiddenHigh=sigmoidVectorV17(rawHigh);
        const double prediction=sigmoidV15(_mm512_reduce_add_pd(_mm512_add_pd(_mm512_mul_pd(hiddenLow,wTwoLow),_mm512_mul_pd(hiddenHigh,wTwoHigh))));
        const double actual=data.y[row],error=prediction-actual,deltaThree=error*prediction*(1.0-prediction);
        out.squaredError+=error*error;const double percentage=actual==0.0?0.0:abs(error/actual)*100.0;out.percentageErrorSum+=percentage;out.maxPercentageError=max(out.maxPercentageError,percentage);
        const __m512d d=_mm512_set1_pd(deltaThree);gradientTwoLow=_mm512_fmadd_pd(hiddenLow,d,gradientTwoLow);gradientTwoHigh=_mm512_fmadd_pd(hiddenHigh,d,gradientTwoHigh);
        const __m512d hiddenDeltaLow=_mm512_mul_pd(_mm512_mul_pd(d,wTwoLow),_mm512_mul_pd(hiddenLow,_mm512_sub_pd(one,hiddenLow)));
        const __m512d hiddenDeltaHigh=_mm512_mul_pd(_mm512_mul_pd(d,wTwoHigh),_mm512_mul_pd(hiddenHigh,_mm512_sub_pd(one,hiddenHigh)));
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            const __m512d value=_mm512_set1_pd(x[k]);
            gradientOneLow[k]=_mm512_fmadd_pd(value,hiddenDeltaLow,gradientOneLow[k]);gradientOneHigh[k]=_mm512_fmadd_pd(value,hiddenDeltaHigh,gradientOneHigh[k]);
        }
    }
    _mm512_storeu_pd(out.gradient.data()+WONE_SIZE,gradientTwoLow);_mm512_storeu_pd(out.gradient.data()+WONE_SIZE+8,gradientTwoHigh);
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
        _mm512_storeu_pd(out.gradient.data()+k*HIDDEN_NODES,gradientOneLow[k]);_mm512_storeu_pd(out.gradient.data()+k*HIDDEN_NODES+8,gradientOneHigh[k]);
    }
}

static V14Evaluation evaluateV17(const Dataset&data,size_t startRow,size_t batchSize,Network&network,const vector<double>&parameters,size_t requestedThreads){
    if(!supportsV16Kernel())return evaluateV16(data,startRow,batchSize,network,parameters,requestedThreads);
    unpackV14(parameters,network);const size_t threads=evaluationThreadsV15(requestedThreads,batchSize);vector<V15ThreadEvaluation>partials(threads);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(threads))
    {
        const size_t tid=static_cast<size_t>(omp_get_thread_num());const size_t base=batchSize/threads,remainder=batchSize%threads;
        const size_t offset=tid*base+min(tid,remainder),count=base+(tid<remainder?1:0);
        evaluateSliceV17(data,startRow+offset,startRow+offset+count,network,partials[tid]);
    }
#else
    evaluateSliceV17(data,startRow,startRow+batchSize,network,partials[0]);
#endif
    V14Evaluation result;result.gradient.assign(V14_PARAMETER_COUNT,0.0);long double squaredError=0.0L,percentageErrorSum=0.0L;double maxPercentageError=0.0;
    for(size_t t=0;t<threads;t++){
        squaredError+=partials[t].squaredError;percentageErrorSum+=partials[t].percentageErrorSum;maxPercentageError=max(maxPercentageError,partials[t].maxPercentageError);
        for(size_t i=0;i<V14_PARAMETER_COUNT;i++)result.gradient[i]+=partials[t].gradient[i];
    }
    const double inv=1.0/static_cast<double>(batchSize);result.metrics.cost=.5*static_cast<double>(squaredError);result.metrics.percentageError=static_cast<double>(percentageErrorSum)*inv;result.metrics.maxPercentageError=maxPercentageError;result.objective=result.metrics.cost*inv;
    for(double&value:result.gradient)value*=inv;
    return result;
}
#endif

static Metrics confirmMetricsV17(const Dataset&data,size_t startRow,size_t batchSize,const Network&network,size_t requestedThreads){
    vector<double>predictions(batchSize);const size_t threads=evaluationThreadsV15(requestedThreads,batchSize);
#ifndef _OPENMP
    (void)threads;
#endif
#ifdef _OPENMP
#pragma omp parallel for num_threads(static_cast<int>(threads)) schedule(static)
#endif
    for(size_t row=0;row<batchSize;row++)predictions[row]=predictRow(data.x.rowData(startRow+row),network);
    Metrics result;long double squaredError=0.0L,percentageErrorSum=0.0L;double maxPercentageError=0.0;
    for(size_t row=0;row<batchSize;row++){
        const double actual=data.y[startRow+row],error=predictions[row]-actual;
        const double percentage=actual==0.0?0.0:abs(error/actual)*100.0;
        squaredError+=error*error;percentageErrorSum+=percentage;maxPercentageError=max(maxPercentageError,percentage);
    }
    result.cost=.5*static_cast<double>(squaredError);result.percentageError=static_cast<double>(percentageErrorSum)/static_cast<double>(batchSize);result.maxPercentageError=maxPercentageError;return result;
}

static TrainResult trainRangeV17(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,Network&network,const OptimiserConfig&config){
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    TrainResult result;deque<V14HistoryPair>history;vector<double>parameters=packV14(network);V14Evaluation current=evaluateV17(data,startRow,batchSize,network,parameters,config.threads);
    if(config.logEvery!=0)printProgress(batchIndex,0,current.metrics);
    while(current.metrics.percentageError>config.percentageErrorTarget&&result.updates<config.maxDescents){
        vector<double>direction=directionV14(current.gradient,history);double directional=dotV14(current.gradient,direction);
        if(!(directional<0.0)){direction=current.gradient;for(double&value:direction)value=-value;directional=-dotV14(current.gradient,current.gradient);history.clear();}
        double step=1.0;bool accepted=false;vector<double>candidate(parameters.size());V14Evaluation next;
        for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
            for(size_t k=0;k<parameters.size();k++)candidate[k]=parameters[k]+step*direction[k];
            next=evaluateV17(data,startRow,batchSize,network,candidate,config.threads);
            if(next.objective<=current.objective+V14_ARMIJO*step*directional){accepted=true;break;}step*=0.5;
        }
        if(!accepted){unpackV14(parameters,network);break;}
        vector<double>s(parameters.size()),y(parameters.size());for(size_t k=0;k<parameters.size();k++){s[k]=candidate[k]-parameters[k];y[k]=next.gradient[k]-current.gradient[k];}
        const double sy=dotV14(s,y);if(sy>1e-12){if(history.size()==V14_HISTORY)history.pop_front();V14HistoryPair pair;pair.s.swap(s);pair.y.swap(y);pair.rho=1.0/sy;history.push_back(std::move(pair));}
        parameters.swap(candidate);current=std::move(next);result.updates++;
        if(config.logEvery!=0&&result.updates%config.logEvery==0)printProgress(batchIndex,result.updates,current.metrics);
    }
    unpackV14(parameters,network);network.deltaWone.fill(0.0);network.deltaWtwo.fill(0.0);result.metrics=confirmMetricsV17(data,startRow,batchSize,network,config.threads);result.reachedTarget=result.metrics.percentageError<=config.percentageErrorTarget;
    if(config.logEvery!=0||result.reachedTarget)printProgress(batchIndex,result.updates,result.metrics);
    writePredictions(data,startRow,batchSize,network,"ybar.txt");
    return result;
#else
    return trainRangeV16(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV17(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;V9FeatureBounds bounds;Dataset data=loadDatasetV9(totalRows,bounds);Network network;if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){const size_t startRow=batch*exampleSize;const TrainResult result=trainRangeV17(data,startRow,exampleSize,batch,network,optimiser);writeWeights(network);if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of L-BFGS iterations before the target.\n";}
}
static void offlineRunV17(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;Dataset data=loadDatasetV9(rowCount,bounds);Network network;if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);const TrainResult result=trainRangeV17(data,0,rowCount,0,network,optimiser);writeWeights(network);if(!result.reachedTarget)cout<<"Offline L-BFGS reached the maximum number of iterations before the target.\n";
}
int main(){
    try{
        bool batchOnline=false,offline=false,test=true,randomiseWeights=false;double rangeWone=4.0,rangeWtwo=4.0,percentageErrorTarget=3.9;
        size_t iterations=1,exampleSize=13853,numberOfDescents=1000000,times=1,logEvery=1,trainingThreads=4,offlineRows=10000,testRows=13853;
#ifndef _OPENMP
        trainingThreads=1;
#endif
        mt19937 generator(static_cast<unsigned int>(time(nullptr)));const auto startTime=chrono::steady_clock::now();
        if(batchOnline)for(size_t i=0;i<times;i++){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;batchOnlineRunV17(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);}
        if(offline){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;offlineRunV17(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);}
        if(test)testRun(testRows);
        const chrono::duration<double>elapsed=chrono::steady_clock::now()-startTime;
        cout<<"Elapsed time = "<<elapsed.count()<<" seconds\n";
        return 0;
    }catch(const exception&error){cerr<<"Error: "<<error.what()<<'\n';return 1;}
}
