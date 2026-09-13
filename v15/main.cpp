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
#ifdef _OPENMP
#include<omp.h>
#endif

namespace v15_v14 {
#include "../v14/main.cpp"
}
using namespace v15_v14;

/* V15: parallel exact evaluator for the v14 L-BFGS optimiser. */
constexpr size_t V15_MAX_EVALUATION_THREADS=4;

struct alignas(64) V15ThreadEvaluation {
    array<double,V14_PARAMETER_COUNT> gradient{};
    long double squaredError=0.0L;
    long double percentageErrorSum=0.0L;
    double maxPercentageError=0.0;
};

static inline double sigmoidV15(double value){return 1.0/(1.0+exp(-value));}

static void evaluateSliceV15(const Dataset&data,size_t firstRow,size_t lastRow,const Network&network,V15ThreadEvaluation&out){
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
        const double prediction=sigmoidV15(rawOutput),actual=data.y[row],error=prediction-actual;
        const double deltaThree=error*prediction*(1.0-prediction);
        out.squaredError+=error*error;
        const double percentage=actual==0.0?0.0:abs(error/actual)*100.0;
        out.percentageErrorSum+=percentage;
        out.maxPercentageError=max(out.maxPercentageError,percentage);
        for(size_t j=0;j<HIDDEN_NODES;j++){
            out.gradient[WONE_SIZE+j]+=hidden[j]*deltaThree;
            const double hiddenDelta=deltaThree*network.wTwo[j]*hidden[j]*(1.0-hidden[j]);
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)out.gradient[k*HIDDEN_NODES+j]+=x[k]*hiddenDelta;
        }
    }
}

static size_t evaluationThreadsV15(size_t requested,size_t rows){
#ifdef _OPENMP
    if(rows<50000)return 1;
    const size_t wanted=requested==0?V15_MAX_EVALUATION_THREADS:requested;
    return max<size_t>(1,min(V15_MAX_EVALUATION_THREADS,wanted));
#else
    (void)requested;(void)rows;return 1;
#endif
}

static V14Evaluation evaluateV15(const Dataset&data,size_t startRow,size_t batchSize,Network&network,const vector<double>&parameters,size_t requestedThreads){
    unpackV14(parameters,network);
    const size_t threads=evaluationThreadsV15(requestedThreads,batchSize);
    vector<V15ThreadEvaluation> partials(threads);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(threads))
    {
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
        const size_t base=batchSize/threads,remainder=batchSize%threads;
        const size_t offset=tid*base+min(tid,remainder),count=base+(tid<remainder?1:0);
        evaluateSliceV15(data,startRow+offset,startRow+offset+count,network,partials[tid]);
    }
#else
    evaluateSliceV15(data,startRow,startRow+batchSize,network,partials[0]);
#endif
    V14Evaluation result;result.gradient.assign(V14_PARAMETER_COUNT,0.0);
    long double squaredError=0.0L,percentageErrorSum=0.0L;double maxPercentageError=0.0;
    for(size_t t=0;t<threads;t++){
        squaredError+=partials[t].squaredError;percentageErrorSum+=partials[t].percentageErrorSum;maxPercentageError=max(maxPercentageError,partials[t].maxPercentageError);
        for(size_t i=0;i<V14_PARAMETER_COUNT;i++)result.gradient[i]+=partials[t].gradient[i];
    }
    const double inv=1.0/static_cast<double>(batchSize);
    result.metrics.cost=.5*static_cast<double>(squaredError);
    result.metrics.percentageError=static_cast<double>(percentageErrorSum)*inv;
    result.metrics.maxPercentageError=maxPercentageError;
    result.objective=result.metrics.cost*inv;
    for(double&value:result.gradient)value*=inv;
    return result;
}

static TrainResult trainRangeV15(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,Network&network,const OptimiserConfig&config){
    TrainResult result;deque<V14HistoryPair>history;vector<double>parameters=packV14(network);
    V14Evaluation current=evaluateV15(data,startRow,batchSize,network,parameters,config.threads);
    if(config.logEvery!=0)printProgress(batchIndex,0,current.metrics);
    while(current.metrics.percentageError>config.percentageErrorTarget&&result.updates<config.maxDescents){
        vector<double>direction=directionV14(current.gradient,history);double directional=dotV14(current.gradient,direction);
        if(!(directional<0.0)){direction=current.gradient;for(double&value:direction)value=-value;directional=-dotV14(current.gradient,current.gradient);history.clear();}
        double step=1.0;bool accepted=false;vector<double>candidate(parameters.size());V14Evaluation next;
        for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
            for(size_t k=0;k<parameters.size();k++)candidate[k]=parameters[k]+step*direction[k];
            next=evaluateV15(data,startRow,batchSize,network,candidate,config.threads);
            if(next.objective<=current.objective+V14_ARMIJO*step*directional){accepted=true;break;}
            step*=0.5;
        }
        if(!accepted){unpackV14(parameters,network);break;}
        vector<double>s(parameters.size()),y(parameters.size());
        for(size_t k=0;k<parameters.size();k++){s[k]=candidate[k]-parameters[k];y[k]=next.gradient[k]-current.gradient[k];}
        const double sy=dotV14(s,y);
        if(sy>1e-12){if(history.size()==V14_HISTORY)history.pop_front();V14HistoryPair pair;pair.s.swap(s);pair.y.swap(y);pair.rho=1.0/sy;history.push_back(std::move(pair));}
        parameters.swap(candidate);current=std::move(next);result.updates++;
        if(config.logEvery!=0&&result.updates%config.logEvery==0)printProgress(batchIndex,result.updates,current.metrics);
    }
    unpackV14(parameters,network);network.deltaWone.fill(0.0);network.deltaWtwo.fill(0.0);
    result.metrics=calculateMetrics(data,startRow,batchSize,network);result.reachedTarget=result.metrics.percentageError<=config.percentageErrorTarget;
    if(config.logEvery!=0||result.reachedTarget)printProgress(batchIndex,result.updates,result.metrics);
    writePredictions(data,startRow,batchSize,network,"ybar.txt");return result;
}

static void batchOnlineRunV15(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;V9FeatureBounds bounds;Dataset data=loadDatasetV9(totalRows,bounds);Network network;if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){const size_t startRow=batch*exampleSize;const TrainResult result=trainRangeV15(data,startRow,exampleSize,batch,network,optimiser);writeWeights(network);if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of L-BFGS iterations before the target.\n";}
}
static void offlineRunV15(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;Dataset data=loadDatasetV9(rowCount,bounds);Network network;if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);const TrainResult result=trainRangeV15(data,0,rowCount,0,network,optimiser);writeWeights(network);if(!result.reachedTarget)cout<<"Offline L-BFGS reached the maximum number of iterations before the target.\n";
}
int main(){
    try{bool batchOnline=false,offline=false,test=true,randomiseWeights=false;double rangeWone=4.0,rangeWtwo=4.0,percentageErrorTarget=3.9;size_t iterations=1,exampleSize=13853,numberOfDescents=1000000,times=1,logEvery=1,trainingThreads=4,offlineRows=10000,testRows=13853;
#ifndef _OPENMP
        trainingThreads=1;
#endif
        mt19937 generator(static_cast<unsigned int>(time(nullptr)));const auto startTime=chrono::steady_clock::now();
        if(batchOnline)for(size_t i=0;i<times;i++){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;batchOnlineRunV15(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);}
        if(offline){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;optimiser.percentageErrorTarget=percentageErrorTarget;offlineRunV15(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);}
        if(test)testRun(testRows);
        const chrono::duration<double>elapsed=chrono::steady_clock::now()-startTime;
        cout<<"Elapsed time = "<<elapsed.count()<<" seconds\n";
        return 0;
    }catch(const exception&error){cerr<<"Error: "<<error.what()<<'\n';return 1;}
}
