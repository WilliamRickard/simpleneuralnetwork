#include<iostream>
#include<algorithm>
#include<array>
#include<chrono>
#include<cmath>
#include<cstddef>
#include<cstdint>
#include<cstring>
#include<ctime>
#include<deque>
#include<exception>
#include<fstream>
#include<iomanip>
#include<random>
#include<stdexcept>
#include<string>
#include<vector>

namespace v18_v17 {
#include "../v17/main.cpp"
}
using namespace v18_v17;

/* V18: scale-aware L-BFGS curvature acceptance to remove the low-error plateau. */
constexpr double V18_CURVATURE_RELATIVE_EPS=1e-8;

static bool acceptCurvatureV18(const vector<double>&s,const vector<double>&y,double sy){
    if(!(sy>0.0))return false;
    const double sNorm=sqrt(dotV14(s,s)),yNorm=sqrt(dotV14(y,y));
    if(!(sNorm>0.0)||!(yNorm>0.0))return false;
    return sy>V18_CURVATURE_RELATIVE_EPS*sNorm*yNorm;
}

static TrainResult trainRangeV18(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,Network&network,const OptimiserConfig&config){
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
        const double sy=dotV14(s,y);if(acceptCurvatureV18(s,y,sy)){if(history.size()==V14_HISTORY)history.pop_front();V14HistoryPair pair;pair.s.swap(s);pair.y.swap(y);pair.rho=1.0/sy;history.push_back(std::move(pair));}
        parameters.swap(candidate);current=std::move(next);result.updates++;
        if(config.logEvery!=0&&result.updates%config.logEvery==0)printProgress(batchIndex,result.updates,current.metrics);
    }
    unpackV14(parameters,network);network.deltaWone.fill(0.0);network.deltaWtwo.fill(0.0);result.metrics=confirmMetricsV17(data,startRow,batchSize,network,config.threads);result.reachedTarget=result.metrics.percentageError<=config.percentageErrorTarget;
    if(config.logEvery!=0||result.reachedTarget)printProgress(batchIndex,result.updates,result.metrics);writePredictions(data,startRow,batchSize,network,"ybar.txt");return result;
#else
    return trainRangeV17(data,startRow,batchSize,batchIndex,network,config);
#endif
}

static void batchOnlineRunV18(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;V9FeatureBounds bounds;Dataset data=loadDatasetV9(totalRows,bounds);Network network;if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){const size_t startRow=batch*exampleSize;const TrainResult result=trainRangeV18(data,startRow,exampleSize,batch,network,optimiser);writeWeights(network);if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of L-BFGS iterations before the target.\n";}
}
static void offlineRunV18(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;Dataset data=loadDatasetV9(rowCount,bounds);Network network;if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);const TrainResult result=trainRangeV18(data,0,rowCount,0,network,optimiser);writeWeights(network);if(!result.reachedTarget)cout<<"Offline L-BFGS reached the maximum number of iterations before the target.\n";
}
#ifndef SIMPLE_NN_V18_NO_MAIN
int main(){
    try{
        bool batchOnline=false,offline=false,test=true,randomiseWeights=false;double rangeWone=4.0,rangeWtwo=4.0,percentageErrorTarget=3.9;
        size_t iterations=1,exampleSize=13853,numberOfDescents=1000000,times=1,logEvery=1,trainingThreads=4,offlineRows=10000,testRows=13853;
#ifndef _OPENMP
        trainingThreads=1;
#endif
        mt19937 generator(static_cast<unsigned int>(time(nullptr)));const auto startTime=chrono::steady_clock::now();
        if(batchOnline)for(size_t i=0;i<times;i++){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;batchOnlineRunV18(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);}
        if(offline)for(size_t i=0;i<times;i++){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.percentageErrorTarget=percentageErrorTarget;optimiser.logEvery=logEvery;optimiser.threads=trainingThreads;offlineRunV18(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);}
        if(test)for(size_t i=0;i<times;i++)testRun(testRows);
        const auto endTime=chrono::steady_clock::now();cout<<"Elapsed seconds: "<<chrono::duration<double>(endTime-startTime).count()<<'\n';return 0;
    }catch(const exception&error){cerr<<"Error: "<<error.what()<<'\n';return 1;}
}
#endif
