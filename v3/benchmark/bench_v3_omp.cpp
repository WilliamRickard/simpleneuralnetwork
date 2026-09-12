#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

constexpr size_t INPUTS=11, HIDDEN=16, W1_SIZE=INPUTS*HIDDEN;
struct Dataset { size_t rows; vector<double> x,y; explicit Dataset(size_t r):rows(r),x(r*INPUTS),y(r){} double* rowData(size_t r){return x.data()+r*INPUTS;} const double* rowData(size_t r) const{return x.data()+r*INPUTS;} };
struct Network { array<double,W1_SIZE>w1{}; array<double,HIDDEN>w2{}; array<double,W1_SIZE>velocityW1{}; array<double,HIDDEN>velocityW2{}; };
struct alignas(64) Accumulator { array<double,W1_SIZE>w1{}; array<double,HIDDEN>w2{}; double squaredError=0.0,percentageErrorSum=0.0,maxPercentageError=0.0; void clear(){w1.fill(0.0);w2.fill(0.0);squaredError=percentageErrorSum=maxPercentageError=0.0;} };
struct Metrics { double cost=0.0,percentageError=0.0,maxPercentageError=0.0; };
static inline double sigmoid(double x){if(x < -700.0){const double e=exp(x);return e/(1.0+e);}return 1.0/(1.0+exp(-x));}
static void fillData(Dataset&data){array<double,W1_SIZE>tw1{};array<double,HIDDEN>tw2{};for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)tw1[k*HIDDEN+j]=0.22*sin(0.17*(k+1)*(j+2));for(size_t j=0;j<HIDDEN;j++)tw2[j]=0.28*cos(0.31*(j+1));for(size_t i=0;i<data.rows;i++){double*x=data.rowData(i);for(size_t k=0;k<INPUTS;k++)x[k]=0.55*sin(0.013*(i+1)*(k+1))+0.35*cos(0.007*(i+3)*(k+2));double z3=0.0;for(size_t j=0;j<HIDDEN;j++){double z2=0.0;for(size_t k=0;k<INPUTS;k++)z2+=x[k]*tw1[k*HIDDEN+j];z3+=sigmoid(z2)*tw2[j];}data.y[i]=sigmoid(z3);}}
static void initWeights(Network&net){for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)net.w1[k*HIDDEN+j]=0.12*sin(0.43*(k+1)*(j+1));for(size_t j=0;j<HIDDEN;j++)net.w2[j]=0.15*cos(0.37*(j+1));net.velocityW1.fill(0.0);net.velocityW2.fill(0.0);}
static Metrics fusedParallel(const Dataset&data,const Network&net,array<double,W1_SIZE>&gradW1,array<double,HIDDEN>&gradW2,vector<Accumulator>&accs){
    const int threads=static_cast<int>(accs.size()); for(auto&a:accs)a.clear();
#pragma omp parallel num_threads(threads)
    {
        const int tid=omp_get_thread_num(); Accumulator&acc=accs[tid];
#pragma omp for schedule(static)
        for(long long ii=0;ii<static_cast<long long>(data.rows);ii++){
            const size_t i=static_cast<size_t>(ii); const double*x=data.rowData(i); array<double,HIDDEN>hidden{}; array<double,HIDDEN>delta2{};
            for(size_t k=0;k<INPUTS;k++){const double xv=x[k];const double*wr=net.w1.data()+k*HIDDEN;for(size_t j=0;j<HIDDEN;j++)hidden[j]+=xv*wr[j];}
            double z3=0.0;for(size_t j=0;j<HIDDEN;j++){hidden[j]=sigmoid(hidden[j]);z3+=hidden[j]*net.w2[j];}
            const double pred=sigmoid(z3),actual=data.y[i],error=pred-actual;acc.squaredError+=error*error;if(actual!=0.0){const double pct=abs(error/actual)*100.0;acc.percentageErrorSum+=pct;acc.maxPercentageError=max(acc.maxPercentageError,pct);}const double delta3=error*pred*(1.0-pred);
            for(size_t j=0;j<HIDDEN;j++){const double a=hidden[j];acc.w2[j]+=a*delta3;delta2[j]=delta3*net.w2[j]*a*(1.0-a);}for(size_t k=0;k<INPUTS;k++){const double xv=x[k];double*gr=acc.w1.data()+k*HIDDEN;for(size_t j=0;j<HIDDEN;j++)gr[j]+=xv*delta2[j];}
        }
    }
    gradW1.fill(0.0);gradW2.fill(0.0);double sq=0.0,pct=0.0,maxpct=0.0;
    for(int t=0;t<threads;t++){const auto&a=accs[t];for(size_t i=0;i<W1_SIZE;i++)gradW1[i]+=a.w1[i];for(size_t j=0;j<HIDDEN;j++)gradW2[j]+=a.w2[j];sq+=a.squaredError;pct+=a.percentageErrorSum;maxpct=max(maxpct,a.maxPercentageError);}
    Metrics m;m.cost=0.5*sq;m.percentageError=pct/static_cast<double>(data.rows);m.maxPercentageError=maxpct;return m;
}
static inline void applyUpdate(Network&net,const array<double,W1_SIZE>&g1,const array<double,HIDDEN>&g2,double lr,double momentum){for(size_t i=0;i<W1_SIZE;i++){net.velocityW1[i]=momentum*net.velocityW1[i]-lr*g1[i];net.w1[i]+=net.velocityW1[i];}for(size_t j=0;j<HIDDEN;j++){net.velocityW2[j]=momentum*net.velocityW2[j]-lr*g2[j];net.w2[j]+=net.velocityW2[j];}}
int main(int argc,char**argv){const size_t rows=argc>1?strtoull(argv[1],nullptr,10):13853;const size_t updates=argc>2?strtoull(argv[2],nullptr,10):10;const size_t reps=argc>3?strtoull(argv[3],nullptr,10):1;const int threads=argc>4?atoi(argv[4]):omp_get_max_threads();const double lr=0.0001/static_cast<double>(rows),momentum=0.75;Dataset data(rows);fillData(data);Network net;array<double,W1_SIZE>g1{};array<double,HIDDEN>g2{};vector<Accumulator>accs(static_cast<size_t>(threads));for(size_t rep=0;rep<reps;rep++){initWeights(net);Metrics m;auto start=chrono::steady_clock::now();for(size_t u=0;u<updates;u++){m=fusedParallel(data,net,g1,g2,accs);applyUpdate(net,g1,g2,lr,momentum);}auto end=chrono::steady_clock::now();double secs=chrono::duration<double>(end-start).count(),checksum=0.0;for(double v:net.w1)checksum+=v;for(double v:net.w2)checksum+=v;cout<<setprecision(17)<<"version=v3omp rows="<<rows<<" updates="<<updates<<" rep="<<rep<<" threads="<<threads<<" seconds="<<secs<<" cost="<<m.cost<<" pct="<<m.percentageError<<" max="<<m.maxPercentageError<<" checksum="<<checksum<<"\n";} }
