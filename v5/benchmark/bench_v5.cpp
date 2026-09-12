#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;
constexpr size_t INPUTS=11,HIDDEN=16,W1_SIZE=INPUTS*HIDDEN,BLOCK=64;
struct Dataset{size_t rows;vector<double>x,y;explicit Dataset(size_t r):rows(r),x(r*INPUTS),y(r){}double*rowData(size_t r){return x.data()+r*INPUTS;}const double*rowData(size_t r)const{return x.data()+r*INPUTS;}};
struct Network{array<double,W1_SIZE>w1{};array<double,HIDDEN>w2{};array<double,W1_SIZE>velocityW1{};array<double,HIDDEN>velocityW2{};};
struct alignas(64) Acc{array<double,W1_SIZE>g1{};array<double,HIDDEN>g2{};double sq=0,pct=0,maxpct=0;void clear(){g1.fill(0);g2.fill(0);sq=pct=maxpct=0;}};
struct Metrics{double cost=0,percentageError=0,maxPercentageError=0;};
static inline double sigmoid(double x){if(x < -700.0){const double e=exp(x);return e/(1.0+e);}return 1.0/(1.0+exp(-x));}
extern "C" __m256d _ZGVdN4v_exp(__m256d);
extern "C" __m512d _ZGVeN8v_exp(__m512d);
__attribute__((target("avx2"))) static void sigmoidVectorAVX2(double*a,size_t n){const __m256d one=_mm256_set1_pd(1.0),zero=_mm256_setzero_pd(),limit=_mm256_set1_pd(-700.0);size_t i=0;for(;i+4<=n;i+=4){__m256d x=_mm256_loadu_pd(a+i);__m256d cmp=_mm256_cmp_pd(x,limit,_CMP_LT_OQ);if(_mm256_movemask_pd(cmp)){for(size_t q=0;q<4;q++)a[i+q]=sigmoid(a[i+q]);continue;}__m256d e=_ZGVdN4v_exp(_mm256_sub_pd(zero,x));_mm256_storeu_pd(a+i,_mm256_div_pd(one,_mm256_add_pd(one,e)));}for(;i<n;i++)a[i]=sigmoid(a[i]);}
__attribute__((target("avx512f"))) static void sigmoidVectorAVX512(double*a,size_t n){const __m512d one=_mm512_set1_pd(1.0),zero=_mm512_setzero_pd(),limit=_mm512_set1_pd(-700.0);size_t i=0;for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(a+i);__mmask8 extreme=_mm512_cmp_pd_mask(x,limit,_CMP_LT_OQ);if(extreme){for(size_t q=0;q<8;q++)a[i+q]=sigmoid(a[i+q]);continue;}__m512d e=_ZGVeN8v_exp(_mm512_sub_pd(zero,x));_mm512_storeu_pd(a+i,_mm512_div_pd(one,_mm512_add_pd(one,e)));}for(;i<n;i++)a[i]=sigmoid(a[i]);}
static void sigmoidVector(double*a,size_t n){if(__builtin_cpu_supports("avx512f")){sigmoidVectorAVX512(a,n);return;}if(__builtin_cpu_supports("avx2")){sigmoidVectorAVX2(a,n);return;}for(size_t i=0;i<n;i++)a[i]=sigmoid(a[i]);}
static void fillData(Dataset&data){array<double,W1_SIZE>tw1{};array<double,HIDDEN>tw2{};for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)tw1[k*HIDDEN+j]=0.22*sin(0.17*(k+1)*(j+2));for(size_t j=0;j<HIDDEN;j++)tw2[j]=0.28*cos(0.31*(j+1));for(size_t i=0;i<data.rows;i++){double*x=data.rowData(i);for(size_t k=0;k<INPUTS;k++)x[k]=0.55*sin(0.013*(i+1)*(k+1))+0.35*cos(0.007*(i+3)*(k+2));double z3=0;for(size_t j=0;j<HIDDEN;j++){double z2=0;for(size_t k=0;k<INPUTS;k++)z2+=x[k]*tw1[k*HIDDEN+j];z3+=sigmoid(z2)*tw2[j];}data.y[i]=sigmoid(z3);}}
static void initWeights(Network&net){for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)net.w1[k*HIDDEN+j]=0.12*sin(0.43*(k+1)*(j+1));for(size_t j=0;j<HIDDEN;j++)net.w2[j]=0.15*cos(0.37*(j+1));net.velocityW1.fill(0);net.velocityW2.fill(0);}
__attribute__((target("avx512f,fma"))) static inline void processBlock(const Dataset&data,size_t base,const Network&net,Acc&acc){const size_t n=min(BLOCK,data.rows-base);alignas(64) double hidden[BLOCK*HIDDEN]={};alignas(64) double output[BLOCK];alignas(64) double delta3[BLOCK];
 const __m512d onev=_mm512_set1_pd(1.0);
 for(size_t b=0;b<n;b++){
   const double*x=data.rowData(base+b);double*h=hidden+b*HIDDEN;
   __m512d z0=_mm512_setzero_pd(),z1=_mm512_setzero_pd();
   for(size_t k=0;k<INPUTS;k++){const __m512d xv=_mm512_set1_pd(x[k]);const double*wr=net.w1.data()+k*HIDDEN;z0=_mm512_fmadd_pd(xv,_mm512_loadu_pd(wr),z0);z1=_mm512_fmadd_pd(xv,_mm512_loadu_pd(wr+8),z1);}
   _mm512_store_pd(h,z0);_mm512_store_pd(h+8,z1);
 }
 sigmoidVector(hidden,n*HIDDEN);
 const __m512d w20=_mm512_loadu_pd(net.w2.data()),w21=_mm512_loadu_pd(net.w2.data()+8);
 for(size_t b=0;b<n;b++){const double*h=hidden+b*HIDDEN;const __m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));output[b]=_mm512_reduce_add_pd(sum);}sigmoidVector(output,n);
 for(size_t b=0;b<n;b++){const double pred=output[b],actual=data.y[base+b],err=pred-actual;acc.sq+=err*err;if(actual!=0){const double p=abs(err/actual)*100;acc.pct+=p;acc.maxpct=max(acc.maxpct,p);}delta3[b]=err*pred*(1-pred);}
 __m512d gg20=_mm512_loadu_pd(acc.g2.data()),gg21=_mm512_loadu_pd(acc.g2.data()+8);
 for(size_t b=0;b<n;b++){double*h=hidden+b*HIDDEN;const __m512d d3v=_mm512_set1_pd(delta3[b]);const __m512d a0=_mm512_load_pd(h),a1=_mm512_load_pd(h+8);gg20=_mm512_fmadd_pd(a0,d3v,gg20);gg21=_mm512_fmadd_pd(a1,d3v,gg21);_mm512_store_pd(h,_mm512_mul_pd(_mm512_mul_pd(d3v,w20),_mm512_mul_pd(a0,_mm512_sub_pd(onev,a0))));_mm512_store_pd(h+8,_mm512_mul_pd(_mm512_mul_pd(d3v,w21),_mm512_mul_pd(a1,_mm512_sub_pd(onev,a1))));}
 _mm512_storeu_pd(acc.g2.data(),gg20);_mm512_storeu_pd(acc.g2.data()+8,gg21);
 for(size_t k=0;k<INPUTS;k++){
   double*g=acc.g1.data()+k*HIDDEN;__m512d a0=_mm512_setzero_pd(),a1=_mm512_setzero_pd(),b0=_mm512_setzero_pd(),b1=_mm512_setzero_pd();size_t r=0;
   for(;r+1<n;r+=2){const double*h0=hidden+r*HIDDEN,*h1=hidden+(r+1)*HIDDEN;const __m512d x0=_mm512_set1_pd(data.rowData(base+r)[k]),x1=_mm512_set1_pd(data.rowData(base+r+1)[k]);a0=_mm512_fmadd_pd(x0,_mm512_load_pd(h0),a0);a1=_mm512_fmadd_pd(x0,_mm512_load_pd(h0+8),a1);b0=_mm512_fmadd_pd(x1,_mm512_load_pd(h1),b0);b1=_mm512_fmadd_pd(x1,_mm512_load_pd(h1+8),b1);}
   if(r<n){const double*h=hidden+r*HIDDEN;const __m512d x=_mm512_set1_pd(data.rowData(base+r)[k]);a0=_mm512_fmadd_pd(x,_mm512_load_pd(h),a0);a1=_mm512_fmadd_pd(x,_mm512_load_pd(h+8),a1);}
   _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),_mm512_add_pd(a0,b0)));_mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),_mm512_add_pd(a1,b1)));
 }
}
static inline void applyUpdate(Network&net,const array<double,W1_SIZE>&g1,const array<double,HIDDEN>&g2,double lr,double momentum){for(size_t i=0;i<W1_SIZE;i++){net.velocityW1[i]=momentum*net.velocityW1[i]-lr*g1[i];net.w1[i]+=net.velocityW1[i];}for(size_t j=0;j<HIDDEN;j++){net.velocityW2[j]=momentum*net.velocityW2[j]-lr*g2[j];net.w2[j]+=net.velocityW2[j];}}
int main(int argc,char**argv){const size_t rows=argc>1?strtoull(argv[1],0,10):13853,updates=argc>2?strtoull(argv[2],0,10):10,reps=argc>3?strtoull(argv[3],0,10):1;const int threads=argc>4?atoi(argv[4]):4;const double lr=0.0001/static_cast<double>(rows),momentum=.75;Dataset data(rows);fillData(data);Network net;vector<Acc>accs(threads);array<double,W1_SIZE>g1{};array<double,HIDDEN>g2{};Metrics m;
 for(size_t rep=0;rep<reps;rep++){initWeights(net);auto start=chrono::steady_clock::now();
#pragma omp parallel num_threads(threads) shared(net,accs,g1,g2,m)
  {const int tid=omp_get_thread_num();for(size_t u=0;u<updates;u++){accs[tid].clear();
#pragma omp for schedule(static)
    for(long long block=0;block<static_cast<long long>((rows+BLOCK-1)/BLOCK);block++)processBlock(data,static_cast<size_t>(block)*BLOCK,net,accs[tid]);
#pragma omp single
    {g1.fill(0);g2.fill(0);double sq=0,pct=0,maxpct=0;for(int t=0;t<threads;t++){const Acc&a=accs[t];for(size_t i=0;i<W1_SIZE;i++)g1[i]+=a.g1[i];for(size_t j=0;j<HIDDEN;j++)g2[j]+=a.g2[j];sq+=a.sq;pct+=a.pct;maxpct=max(maxpct,a.maxpct);}m.cost=.5*sq;m.percentageError=pct/static_cast<double>(rows);m.maxPercentageError=maxpct;applyUpdate(net,g1,g2,lr,momentum);}
  }}
 auto end=chrono::steady_clock::now();double secs=chrono::duration<double>(end-start).count(),checksum=0;for(double v:net.w1)checksum+=v;for(double v:net.w2)checksum+=v;cout<<setprecision(17)<<"version=v5 rows="<<rows<<" updates="<<updates<<" rep="<<rep<<" threads="<<threads<<" seconds="<<secs<<" cost="<<m.cost<<" pct="<<m.percentageError<<" max="<<m.maxPercentageError<<" checksum="<<checksum<<"\n";}}
