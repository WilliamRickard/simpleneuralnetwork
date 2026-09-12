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
constexpr size_t INPUTS=11,HIDDEN=16,W1_SIZE=INPUTS*HIDDEN,BLOCK=16;
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
__attribute__((target("avx512f,fma"))) static inline void processBlock(const Dataset&data,size_t base,const Network&net,Acc&acc,bool detailed){
 const size_t n=min(BLOCK,data.rows-base);alignas(64) double hidden[BLOCK*HIDDEN]={};alignas(64) double output[BLOCK];alignas(64) double delta3[BLOCK]; const __m512d onev=_mm512_set1_pd(1.0);
 for(size_t b=0;b<n;b++){const double*x=data.rowData(base+b);double*h=hidden+b*HIDDEN;__m512d z0=_mm512_setzero_pd(),z1=_mm512_setzero_pd();for(size_t k=0;k<INPUTS;k++){const __m512d xv=_mm512_set1_pd(x[k]);const double*wr=net.w1.data()+k*HIDDEN;z0=_mm512_fmadd_pd(xv,_mm512_loadu_pd(wr),z0);z1=_mm512_fmadd_pd(xv,_mm512_loadu_pd(wr+8),z1);}_mm512_store_pd(h,z0);_mm512_store_pd(h+8,z1);}
 sigmoidVector(hidden,n*HIDDEN); const __m512d w20=_mm512_loadu_pd(net.w2.data()),w21=_mm512_loadu_pd(net.w2.data()+8);
 for(size_t b=0;b<n;b++){const double*h=hidden+b*HIDDEN;const __m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));output[b]=_mm512_reduce_add_pd(sum);} sigmoidVector(output,n);
 for(size_t b=0;b<n;b++){const double pred=output[b],actual=data.y[base+b],err=pred-actual;double p=0.0;if(actual!=0.0){p=abs(err/actual)*100.0;acc.pct+=p;}if(detailed){acc.sq+=err*err;acc.maxpct=max(acc.maxpct,p);}delta3[b]=err*pred*(1-pred);}
 __m512d gg20=_mm512_loadu_pd(acc.g2.data()),gg21=_mm512_loadu_pd(acc.g2.data()+8);
 __m512d ga0=_mm512_setzero_pd(), gb0=_mm512_setzero_pd();
 __m512d ga1=_mm512_setzero_pd(), gb1=_mm512_setzero_pd();
 __m512d ga2=_mm512_setzero_pd(), gb2=_mm512_setzero_pd();
 __m512d ga3=_mm512_setzero_pd(), gb3=_mm512_setzero_pd();
 __m512d ga4=_mm512_setzero_pd(), gb4=_mm512_setzero_pd();
 __m512d ga5=_mm512_setzero_pd(), gb5=_mm512_setzero_pd();
 __m512d ga6=_mm512_setzero_pd(), gb6=_mm512_setzero_pd();
 __m512d ga7=_mm512_setzero_pd(), gb7=_mm512_setzero_pd();
 __m512d ga8=_mm512_setzero_pd(), gb8=_mm512_setzero_pd();
 __m512d ga9=_mm512_setzero_pd(), gb9=_mm512_setzero_pd();
 __m512d ga10=_mm512_setzero_pd(), gb10=_mm512_setzero_pd();
 for(size_t r=0;r<n;r++){ const double*a=hidden+r*HIDDEN; const __m512d d3v=_mm512_set1_pd(delta3[r]); const __m512d a0=_mm512_load_pd(a),a1=_mm512_load_pd(a+8); gg20=_mm512_fmadd_pd(a0,d3v,gg20); gg21=_mm512_fmadd_pd(a1,d3v,gg21); const __m512d d0=_mm512_mul_pd(_mm512_mul_pd(d3v,w20),_mm512_mul_pd(a0,_mm512_sub_pd(onev,a0))); const __m512d d1=_mm512_mul_pd(_mm512_mul_pd(d3v,w21),_mm512_mul_pd(a1,_mm512_sub_pd(onev,a1))); const double*x=data.rowData(base+r);
 const __m512d x0=_mm512_set1_pd(x[0]); ga0=_mm512_fmadd_pd(x0,d0,ga0); gb0=_mm512_fmadd_pd(x0,d1,gb0);
 const __m512d x1=_mm512_set1_pd(x[1]); ga1=_mm512_fmadd_pd(x1,d0,ga1); gb1=_mm512_fmadd_pd(x1,d1,gb1);
 const __m512d x2=_mm512_set1_pd(x[2]); ga2=_mm512_fmadd_pd(x2,d0,ga2); gb2=_mm512_fmadd_pd(x2,d1,gb2);
 const __m512d x3=_mm512_set1_pd(x[3]); ga3=_mm512_fmadd_pd(x3,d0,ga3); gb3=_mm512_fmadd_pd(x3,d1,gb3);
 const __m512d x4=_mm512_set1_pd(x[4]); ga4=_mm512_fmadd_pd(x4,d0,ga4); gb4=_mm512_fmadd_pd(x4,d1,gb4);
 const __m512d x5=_mm512_set1_pd(x[5]); ga5=_mm512_fmadd_pd(x5,d0,ga5); gb5=_mm512_fmadd_pd(x5,d1,gb5);
 const __m512d x6=_mm512_set1_pd(x[6]); ga6=_mm512_fmadd_pd(x6,d0,ga6); gb6=_mm512_fmadd_pd(x6,d1,gb6);
 const __m512d x7=_mm512_set1_pd(x[7]); ga7=_mm512_fmadd_pd(x7,d0,ga7); gb7=_mm512_fmadd_pd(x7,d1,gb7);
 const __m512d x8=_mm512_set1_pd(x[8]); ga8=_mm512_fmadd_pd(x8,d0,ga8); gb8=_mm512_fmadd_pd(x8,d1,gb8);
 const __m512d x9=_mm512_set1_pd(x[9]); ga9=_mm512_fmadd_pd(x9,d0,ga9); gb9=_mm512_fmadd_pd(x9,d1,gb9);
 const __m512d x10=_mm512_set1_pd(x[10]); ga10=_mm512_fmadd_pd(x10,d0,ga10); gb10=_mm512_fmadd_pd(x10,d1,gb10);
 } _mm512_storeu_pd(acc.g2.data(),gg20);_mm512_storeu_pd(acc.g2.data()+8,gg21);
 {double*g=acc.g1.data()+0*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga0)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb0));}
 {double*g=acc.g1.data()+1*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga1)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb1));}
 {double*g=acc.g1.data()+2*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga2)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb2));}
 {double*g=acc.g1.data()+3*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga3)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb3));}
 {double*g=acc.g1.data()+4*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga4)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb4));}
 {double*g=acc.g1.data()+5*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga5)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb5));}
 {double*g=acc.g1.data()+6*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga6)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb6));}
 {double*g=acc.g1.data()+7*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga7)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb7));}
 {double*g=acc.g1.data()+8*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga8)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb8));}
 {double*g=acc.g1.data()+9*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga9)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb9));}
 {double*g=acc.g1.data()+10*HIDDEN; _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),ga10)); _mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),gb10));}
}
static inline void applyUpdate(Network&net,const array<double,W1_SIZE>&g1,const array<double,HIDDEN>&g2,double lr,double momentum){for(size_t i=0;i<W1_SIZE;i++){net.velocityW1[i]=momentum*net.velocityW1[i]-lr*g1[i];net.w1[i]+=net.velocityW1[i];}for(size_t j=0;j<HIDDEN;j++){net.velocityW2[j]=momentum*net.velocityW2[j]-lr*g2[j];net.w2[j]+=net.velocityW2[j];}}
int main(int argc,char**argv){const size_t rows=argc>1?strtoull(argv[1],0,10):13853,updates=argc>2?strtoull(argv[2],0,10):10,reps=argc>3?strtoull(argv[3],0,10):1;const int threads=argc>4?atoi(argv[4]):4;const double lr=0.0001/static_cast<double>(rows),momentum=.75;Dataset data(rows);fillData(data);Network net;vector<Acc>accs(threads);array<double,W1_SIZE>g1{};array<double,HIDDEN>g2{};Metrics m;
 for(size_t rep=0;rep<reps;rep++){initWeights(net);auto start=chrono::steady_clock::now();
#pragma omp parallel num_threads(threads) shared(net,accs,g1,g2,m)
  {const int tid=omp_get_thread_num();for(size_t u=0;u<updates;u++){accs[tid].clear();
#pragma omp for schedule(static)
    for(long long block=0;block<static_cast<long long>((rows+BLOCK-1)/BLOCK);block++)processBlock(data,static_cast<size_t>(block)*BLOCK,net,accs[tid],u+1==updates);
#pragma omp single
    {g1.fill(0);g2.fill(0);double sq=0,pct=0,maxpct=0;for(int t=0;t<threads;t++){const Acc&a=accs[t];for(size_t i=0;i<W1_SIZE;i++)g1[i]+=a.g1[i];for(size_t j=0;j<HIDDEN;j++)g2[j]+=a.g2[j];sq+=a.sq;pct+=a.pct;maxpct=max(maxpct,a.maxpct);}m.cost=.5*sq;m.percentageError=pct/static_cast<double>(rows);m.maxPercentageError=maxpct;applyUpdate(net,g1,g2,lr,momentum);}
  }}
 auto end=chrono::steady_clock::now();double secs=chrono::duration<double>(end-start).count(),checksum=0;for(double v:net.w1)checksum+=v;for(double v:net.w2)checksum+=v;cout<<setprecision(17)<<"version=v6 rows="<<rows<<" updates="<<updates<<" rep="<<rep<<" threads="<<threads<<" seconds="<<secs<<" cost="<<m.cost<<" pct="<<m.percentageError<<" max="<<m.maxPercentageError<<" checksum="<<checksum<<"\n";}}
