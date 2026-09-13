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

namespace v14_v13 {
#include "../v13/main.cpp"
}
using namespace v14_v13;

/* V14: time-to-target optimisation with full-batch L-BFGS. */
constexpr size_t V14_PARAMETER_COUNT=WONE_SIZE+HIDDEN_NODES;
constexpr size_t V14_HISTORY=10;
constexpr size_t V14_MAX_LINE_SEARCH=20;
constexpr double V14_ARMIJO=1e-4;

struct V14Evaluation { double objective=0.0; Metrics metrics; vector<double> gradient; };
struct V14HistoryPair { vector<double> s,y; double rho=0.0; };

static vector<double> packV14(const Network &network){
    vector<double> p(V14_PARAMETER_COUNT);
    copy(network.wOne.begin(),network.wOne.end(),p.begin());
    copy(network.wTwo.begin(),network.wTwo.end(),p.begin()+WONE_SIZE);
    return p;
}
static void unpackV14(const vector<double>&p,Network&network){
    copy(p.begin(),p.begin()+WONE_SIZE,network.wOne.begin());
    copy(p.begin()+WONE_SIZE,p.end(),network.wTwo.begin());
}
static double dotV14(const vector<double>&a,const vector<double>&b){double s=0.0;for(size_t i=0;i<a.size();i++)s+=a[i]*b[i];return s;}
static V14Evaluation evaluateV14(const Dataset&data,size_t startRow,size_t batchSize,Network&network,const vector<double>&parameters){
    unpackV14(parameters,network);Gradients gradients;
    V14Evaluation e;e.metrics=calculateGradientsAndMetrics(data,startRow,batchSize,network,gradients,1);
    const double inv=1.0/static_cast<double>(batchSize);e.objective=e.metrics.cost*inv;e.gradient.resize(V14_PARAMETER_COUNT);
    for(size_t i=0;i<WONE_SIZE;i++)e.gradient[i]=gradients.dJdWone[i]*inv;
    for(size_t j=0;j<HIDDEN_NODES;j++)e.gradient[WONE_SIZE+j]=gradients.dJdWtwo[j]*inv;
    return e;
}
static vector<double> directionV14(const vector<double>&gradient,const deque<V14HistoryPair>&history){
    vector<double> q=gradient;vector<double> alpha(history.size());
    for(size_t ii=history.size();ii>0;--ii){const size_t i=ii-1;alpha[i]=history[i].rho*dotV14(history[i].s,q);for(size_t k=0;k<q.size();k++)q[k]-=alpha[i]*history[i].y[k];}
    double gamma=1.0;if(!history.empty()){const auto&h=history.back();const double yy=dotV14(h.y,h.y);if(yy>0.0)gamma=dotV14(h.s,h.y)/yy;}
    vector<double> r=q;for(double&v:r)v*=gamma;
    for(size_t i=0;i<history.size();i++){const double beta=history[i].rho*dotV14(history[i].y,r);for(size_t k=0;k<r.size();k++)r[k]+=history[i].s[k]*(alpha[i]-beta);}
    for(double&v:r){v=-v;}
    return r;
}
static TrainResult trainRangeV14(const Dataset&data,size_t startRow,size_t batchSize,size_t batchIndex,Network&network,const OptimiserConfig&config){
    TrainResult result;deque<V14HistoryPair>history;vector<double>p=packV14(network);V14Evaluation current=evaluateV14(data,startRow,batchSize,network,p);size_t evaluations=1;
    if(config.logEvery!=0)printProgress(batchIndex,0,current.metrics);
    while(current.metrics.percentageError>config.percentageErrorTarget && result.updates<config.maxDescents){
        vector<double>d=directionV14(current.gradient,history);double directional=dotV14(current.gradient,d);
        if(!(directional<0.0)){d=current.gradient;for(double&v:d)v=-v;directional=-dotV14(current.gradient,current.gradient);history.clear();}
        double step=1.0;bool accepted=false;vector<double>candidate(p.size());V14Evaluation next;
        for(size_t ls=0;ls<V14_MAX_LINE_SEARCH;ls++){
            for(size_t k=0;k<p.size();k++){candidate[k]=p[k]+step*d[k];}
            next=evaluateV14(data,startRow,batchSize,network,candidate);evaluations++;
            if(next.objective<=current.objective+V14_ARMIJO*step*directional){accepted=true;break;}step*=0.5;
        }
        if(!accepted){unpackV14(p,network);break;}
        vector<double>s(p.size()),y(p.size());for(size_t k=0;k<p.size();k++){s[k]=candidate[k]-p[k];y[k]=next.gradient[k]-current.gradient[k];}
        const double sy=dotV14(s,y);if(sy>1e-12){if(history.size()==V14_HISTORY)history.pop_front();V14HistoryPair pair;pair.s.swap(s);pair.y.swap(y);pair.rho=1.0/sy;history.push_back(std::move(pair));}
        p.swap(candidate);current=std::move(next);result.updates++;
        if(config.logEvery!=0 && result.updates%config.logEvery==0)printProgress(batchIndex,result.updates,current.metrics);
    }
    unpackV14(p,network);network.deltaWone.fill(0.0);network.deltaWtwo.fill(0.0);result.metrics=calculateMetrics(data,startRow,batchSize,network);result.reachedTarget=result.metrics.percentageError<=config.percentageErrorTarget;
    if(config.logEvery!=0 || result.reachedTarget)printProgress(batchIndex,result.updates,result.metrics);
    (void)evaluations;writePredictions(data,startRow,batchSize,network,"ybar.txt");return result;
}
static void batchOnlineRunV14(size_t iterations,size_t exampleSize,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    const size_t totalRows=iterations*exampleSize;V9FeatureBounds bounds;Dataset data=loadDatasetV9(totalRows,bounds);Network network;if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);
    for(size_t batch=0;batch<iterations;batch++){const size_t startRow=batch*exampleSize;const TrainResult result=trainRangeV14(data,startRow,exampleSize,batch,network,optimiser);writeWeights(network);if(!result.reachedTarget)cout<<"Batch "<<batch<<" reached the maximum number of L-BFGS iterations before the target.\n";}
}
static void offlineRunV14(size_t rowCount,const OptimiserConfig&optimiser,double rangeWone,double rangeWtwo,bool randomiseWeights,mt19937&generator){
    V9FeatureBounds bounds;Dataset data=loadDatasetV9(rowCount,bounds);Network network;if(randomiseWeights)setmatrixrandom(network,rangeWone,rangeWtwo,generator);else loadWeights(network);const TrainResult result=trainRangeV14(data,0,rowCount,0,network,optimiser);writeWeights(network);if(!result.reachedTarget)cout<<"Offline L-BFGS reached the maximum number of iterations before the target.\n";
}
int main(){
    try{bool batchOnline=false,offline=false,test=true,randomiseWeights=false;double rangeWone=4.0,rangeWtwo=4.0,percentageErrorTarget=3.9,learningRate=0.0001,momentum=0.75;size_t iterations=1,exampleSize=13853,numberOfDescents=1000000,times=1,logEvery=1,trainingThreads=1,offlineRows=10000,testRows=13853;
        (void)learningRate;(void)momentum;(void)trainingThreads;mt19937 generator(static_cast<unsigned int>(time(nullptr)));const auto startTime=chrono::steady_clock::now();
        if(batchOnline)for(size_t i=0;i<times;i++){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=1;optimiser.percentageErrorTarget=percentageErrorTarget;batchOnlineRunV14(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);}
        if(offline){OptimiserConfig optimiser;optimiser.maxDescents=numberOfDescents;optimiser.logEvery=logEvery;optimiser.threads=1;optimiser.percentageErrorTarget=percentageErrorTarget;offlineRunV14(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);}if(test)testRun(testRows);const chrono::duration<double>elapsed=chrono::steady_clock::now()-startTime;cout<<"Elapsed time = "<<elapsed.count()<<" seconds\n";return 0;}
    catch(const exception&error){cerr<<"Error: "<<error.what()<<'\n';return 1;}
}
