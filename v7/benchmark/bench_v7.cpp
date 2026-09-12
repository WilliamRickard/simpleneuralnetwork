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
constexpr size_t INPUTS=11,HIDDEN=16,W1_SIZE=INPUTS*HIDDEN,TILE=16;
struct Dataset{size_t rows;vector<double>x,y;explicit Dataset(size_t r):rows(r),x(r*INPUTS),y(r){}double*rowData(size_t r){return x.data()+r*INPUTS;}const double*rowData(size_t r)const{return x.data()+r*INPUTS;}};
struct Network{array<double,W1_SIZE>w1{};array<double,HIDDEN>w2{};array<double,W1_SIZE>v1{};array<double,HIDDEN>v2{};};
struct alignas(64) Acc{array<double,W1_SIZE>g1{};array<double,HIDDEN>g2{};double sq=0,pct=0,maxpct=0;void clear(){g1.fill(0);g2.fill(0);sq=pct=maxpct=0;}};
struct Metrics{double cost=0,pct=0,maxpct=0;};
static inline double sigmoid(double x){if(x < -700.0){double e=exp(x);return e/(1+e);}return 1.0/(1.0+exp(-x));}
extern "C" __m256d _ZGVdN4v_exp(__m256d);
extern "C" __m512d _ZGVeN8v_exp(__m512d);
__attribute__((target("avx2"))) static void sigAVX2(double*a,size_t n){const __m256d one=_mm256_set1_pd(1),zero=_mm256_setzero_pd(),lim=_mm256_set1_pd(-700);size_t i=0;for(;i+4<=n;i+=4){__m256d x=_mm256_loadu_pd(a+i);if(_mm256_movemask_pd(_mm256_cmp_pd(x,lim,_CMP_LT_OQ))){for(int q=0;q<4;q++)a[i+q]=sigmoid(a[i+q]);continue;}__m256d e=_ZGVdN4v_exp(_mm256_sub_pd(zero,x));_mm256_storeu_pd(a+i,_mm256_div_pd(one,_mm256_add_pd(one,e)));}for(;i<n;i++)a[i]=sigmoid(a[i]);}
__attribute__((target("avx512f"))) static void sigAVX512(double*a,size_t n){const __m512d one=_mm512_set1_pd(1),zero=_mm512_setzero_pd(),lim=_mm512_set1_pd(-700);size_t i=0;for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(a+i);if(_mm512_cmp_pd_mask(x,lim,_CMP_LT_OQ)){for(int q=0;q<8;q++)a[i+q]=sigmoid(a[i+q]);continue;}__m512d e=_ZGVeN8v_exp(_mm512_sub_pd(zero,x));_mm512_storeu_pd(a+i,_mm512_div_pd(one,_mm512_add_pd(one,e)));}for(;i<n;i++)a[i]=sigmoid(a[i]);}
static void sigvec(double*a,size_t n){if(__builtin_cpu_supports("avx512f")){sigAVX512(a,n);return;}if(__builtin_cpu_supports("avx2")){sigAVX2(a,n);return;}for(size_t i=0;i<n;i++)a[i]=sigmoid(a[i]);}
static void fillData(Dataset&d){array<double,W1_SIZE>tw1{};array<double,HIDDEN>tw2{};for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)tw1[k*HIDDEN+j]=.22*sin(.17*(k+1)*(j+2));for(size_t j=0;j<HIDDEN;j++)tw2[j]=.28*cos(.31*(j+1));for(size_t i=0;i<d.rows;i++){double*x=d.rowData(i);for(size_t k=0;k<INPUTS;k++)x[k]=.55*sin(.013*(i+1)*(k+1))+.35*cos(.007*(i+3)*(k+2));double z3=0;for(size_t j=0;j<HIDDEN;j++){double z2=0;for(size_t k=0;k<INPUTS;k++)z2+=x[k]*tw1[k*HIDDEN+j];z3+=sigmoid(z2)*tw2[j];}d.y[i]=sigmoid(z3);}}
static void init(Network&n){for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)n.w1[k*HIDDEN+j]=.12*sin(.43*(k+1)*(j+1));for(size_t j=0;j<HIDDEN;j++)n.w2[j]=.15*cos(.37*(j+1));n.v1.fill(0);n.v2.fill(0);}

#define DECL_G1 \
 __m512d ga0=_mm512_setzero_pd(),gb0=_mm512_setzero_pd(),ga1=_mm512_setzero_pd(),gb1=_mm512_setzero_pd(),ga2=_mm512_setzero_pd(),gb2=_mm512_setzero_pd(),ga3=_mm512_setzero_pd(),gb3=_mm512_setzero_pd(),ga4=_mm512_setzero_pd(),gb4=_mm512_setzero_pd(),ga5=_mm512_setzero_pd(),gb5=_mm512_setzero_pd(),ga6=_mm512_setzero_pd(),gb6=_mm512_setzero_pd(),ga7=_mm512_setzero_pd(),gb7=_mm512_setzero_pd(),ga8=_mm512_setzero_pd(),gb8=_mm512_setzero_pd(),ga9=_mm512_setzero_pd(),gb9=_mm512_setzero_pd(),ga10=_mm512_setzero_pd(),gb10=_mm512_setzero_pd();
#define ACC(K,A,B) do{__m512d xv=_mm512_set1_pd(x[K]);A=_mm512_fmadd_pd(xv,d0,A);B=_mm512_fmadd_pd(xv,d1,B);}while(0)
#define STORE(K,A,B) do{double*g=acc.g1.data()+K*HIDDEN;_mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),A));_mm512_storeu_pd(g+8,_mm512_add_pd(_mm512_loadu_pd(g+8),B));}while(0)

__attribute__((target("avx512f,fma"))) static inline void block_v6(const Dataset&data,size_t base,const Network&net,Acc&acc,bool detailed){
 const size_t n=min(TILE,data.rows-base);alignas(64) double hidden[TILE*HIDDEN]={};alignas(64) double out[TILE],d3[TILE];const __m512d one=_mm512_set1_pd(1.0);
 for(size_t r=0;r<n;r++){const double*x=data.rowData(base+r);__m512d z0=_mm512_setzero_pd(),z1=_mm512_setzero_pd();for(size_t k=0;k<INPUTS;k++){__m512d xv=_mm512_set1_pd(x[k]);const double*w=net.w1.data()+k*HIDDEN;z0=_mm512_fmadd_pd(xv,_mm512_loadu_pd(w),z0);z1=_mm512_fmadd_pd(xv,_mm512_loadu_pd(w+8),z1);}double*h=hidden+r*HIDDEN;_mm512_store_pd(h,z0);_mm512_store_pd(h+8,z1);}sigvec(hidden,n*HIDDEN);
 const __m512d w20=_mm512_loadu_pd(net.w2.data()),w21=_mm512_loadu_pd(net.w2.data()+8);for(size_t r=0;r<n;r++){double*h=hidden+r*HIDDEN;__m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));out[r]=_mm512_reduce_add_pd(sum);}sigvec(out,n);
 for(size_t r=0;r<n;r++){double pred=out[r],act=data.y[base+r],err=pred-act,p=0;if(act!=0){p=abs(err/act)*100;acc.pct+=p;}if(detailed){acc.sq+=err*err;acc.maxpct=max(acc.maxpct,p);}d3[r]=err*pred*(1-pred);}__m512d gg20=_mm512_loadu_pd(acc.g2.data()),gg21=_mm512_loadu_pd(acc.g2.data()+8);DECL_G1
 for(size_t r=0;r<n;r++){const double*a=hidden+r*HIDDEN;__m512d d=_mm512_set1_pd(d3[r]),a0=_mm512_load_pd(a),a1=_mm512_load_pd(a+8);gg20=_mm512_fmadd_pd(a0,d,gg20);gg21=_mm512_fmadd_pd(a1,d,gg21);__m512d d0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0)));__m512d d1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));const double*x=data.rowData(base+r);ACC(0,ga0,gb0);ACC(1,ga1,gb1);ACC(2,ga2,gb2);ACC(3,ga3,gb3);ACC(4,ga4,gb4);ACC(5,ga5,gb5);ACC(6,ga6,gb6);ACC(7,ga7,gb7);ACC(8,ga8,gb8);ACC(9,ga9,gb9);ACC(10,ga10,gb10);} _mm512_storeu_pd(acc.g2.data(),gg20);_mm512_storeu_pd(acc.g2.data()+8,gg21);STORE(0,ga0,gb0);STORE(1,ga1,gb1);STORE(2,ga2,gb2);STORE(3,ga3,gb3);STORE(4,ga4,gb4);STORE(5,ga5,gb5);STORE(6,ga6,gb6);STORE(7,ga7,gb7);STORE(8,ga8,gb8);STORE(9,ga9,gb9);STORE(10,ga10,gb10);
}
__attribute__((target("avx512f,avx512dq,fma"))) static inline void block_forward4(const Dataset&data,size_t base,const Network&net,Acc&acc,bool detailed){
 const size_t n=min(TILE,data.rows-base);alignas(64) double hidden[TILE*HIDDEN]={};alignas(64) double out[TILE],d3[TILE],pct[TILE];const __m512d one=_mm512_set1_pd(1.0),hundred=_mm512_set1_pd(100.0),signmask=_mm512_set1_pd(-0.0);
 size_t rr=0;for(;rr+3<n;rr+=4){const double*x0=data.rowData(base+rr),*x1=data.rowData(base+rr+1),*x2=data.rowData(base+rr+2),*x3=data.rowData(base+rr+3);__m512d a0=_mm512_setzero_pd(),a1=_mm512_setzero_pd(),b0=_mm512_setzero_pd(),b1=_mm512_setzero_pd(),c0=_mm512_setzero_pd(),c1=_mm512_setzero_pd(),e0=_mm512_setzero_pd(),e1=_mm512_setzero_pd();for(size_t k=0;k<INPUTS;k++){const double*w=net.w1.data()+k*HIDDEN;__m512d w0=_mm512_loadu_pd(w),w1=_mm512_loadu_pd(w+8),v0=_mm512_set1_pd(x0[k]),v1=_mm512_set1_pd(x1[k]),v2=_mm512_set1_pd(x2[k]),v3=_mm512_set1_pd(x3[k]);a0=_mm512_fmadd_pd(v0,w0,a0);a1=_mm512_fmadd_pd(v0,w1,a1);b0=_mm512_fmadd_pd(v1,w0,b0);b1=_mm512_fmadd_pd(v1,w1,b1);c0=_mm512_fmadd_pd(v2,w0,c0);c1=_mm512_fmadd_pd(v2,w1,c1);e0=_mm512_fmadd_pd(v3,w0,e0);e1=_mm512_fmadd_pd(v3,w1,e1);}double*h0=hidden+rr*HIDDEN,*h1=hidden+(rr+1)*HIDDEN,*h2=hidden+(rr+2)*HIDDEN,*h3=hidden+(rr+3)*HIDDEN;_mm512_store_pd(h0,a0);_mm512_store_pd(h0+8,a1);_mm512_store_pd(h1,b0);_mm512_store_pd(h1+8,b1);_mm512_store_pd(h2,c0);_mm512_store_pd(h2+8,c1);_mm512_store_pd(h3,e0);_mm512_store_pd(h3+8,e1);}for(;rr<n;rr++){const double*x=data.rowData(base+rr);__m512d z0=_mm512_setzero_pd(),z1=_mm512_setzero_pd();for(size_t k=0;k<INPUTS;k++){__m512d v=_mm512_set1_pd(x[k]);const double*w=net.w1.data()+k*HIDDEN;z0=_mm512_fmadd_pd(v,_mm512_loadu_pd(w),z0);z1=_mm512_fmadd_pd(v,_mm512_loadu_pd(w+8),z1);}double*h=hidden+rr*HIDDEN;_mm512_store_pd(h,z0);_mm512_store_pd(h+8,z1);}sigvec(hidden,n*HIDDEN);
 const __m512d w20=_mm512_loadu_pd(net.w2.data()),w21=_mm512_loadu_pd(net.w2.data()+8);for(size_t r=0;r<n;r++){double*h=hidden+r*HIDDEN;__m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));out[r]=_mm512_reduce_add_pd(sum);}sigvec(out,n);
 size_t r=0;for(;r+8<=n;r+=8){__m512d pred=_mm512_load_pd(out+r),act=_mm512_loadu_pd(data.y.data()+base+r),err=_mm512_sub_pd(pred,act),absErr=_mm512_andnot_pd(signmask,err);__m512d p=_mm512_mul_pd(_mm512_div_pd(absErr,act),hundred);__mmask8 zero=_mm512_cmp_pd_mask(act,_mm512_setzero_pd(),_CMP_EQ_OQ);p=_mm512_mask_mov_pd(p,zero,_mm512_setzero_pd());_mm512_store_pd(pct+r,p);_mm512_store_pd(d3+r,_mm512_mul_pd(_mm512_mul_pd(err,pred),_mm512_sub_pd(one,pred)));}for(;r<n;r++){double pred=out[r],act=data.y[base+r],err=pred-act;pct[r]=act?abs(err/act)*100:0;d3[r]=err*pred*(1-pred);}for(size_t q=0;q<n;q++){acc.pct+=pct[q];if(detailed){double err=out[q]-data.y[base+q];acc.sq+=err*err;acc.maxpct=max(acc.maxpct,pct[q]);}}
 __m512d gg20=_mm512_loadu_pd(acc.g2.data()),gg21=_mm512_loadu_pd(acc.g2.data()+8);DECL_G1
 for(size_t q=0;q<n;q++){const double*a=hidden+q*HIDDEN;__m512d d=_mm512_set1_pd(d3[q]),a0=_mm512_load_pd(a),a1=_mm512_load_pd(a+8);gg20=_mm512_fmadd_pd(a0,d,gg20);gg21=_mm512_fmadd_pd(a1,d,gg21);__m512d d0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0)));__m512d d1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));const double*x=data.rowData(base+q);ACC(0,ga0,gb0);ACC(1,ga1,gb1);ACC(2,ga2,gb2);ACC(3,ga3,gb3);ACC(4,ga4,gb4);ACC(5,ga5,gb5);ACC(6,ga6,gb6);ACC(7,ga7,gb7);ACC(8,ga8,gb8);ACC(9,ga9,gb9);ACC(10,ga10,gb10);} _mm512_storeu_pd(acc.g2.data(),gg20);_mm512_storeu_pd(acc.g2.data()+8,gg21);STORE(0,ga0,gb0);STORE(1,ga1,gb1);STORE(2,ga2,gb2);STORE(3,ga3,gb3);STORE(4,ga4,gb4);STORE(5,ga5,gb5);STORE(6,ga6,gb6);STORE(7,ga7,gb7);STORE(8,ga8,gb8);STORE(9,ga9,gb9);STORE(10,ga10,gb10);
}
using Kernel=void(*)(const Dataset&,size_t,const Network&,Acc&,bool);
static void update(Network&n,const array<double,W1_SIZE>&g1,const array<double,HIDDEN>&g2,double lr,double mom){for(size_t i=0;i<W1_SIZE;i++){n.v1[i]=mom*n.v1[i]-lr*g1[i];n.w1[i]+=n.v1[i];}for(size_t i=0;i<HIDDEN;i++){n.v2[i]=mom*n.v2[i]-lr*g2[i];n.w2[i]+=n.v2[i];}}
static double run(const Dataset&d,size_t updates,int threads,Kernel kernel,double&checksum,size_t stride=TILE){Network n;init(n);vector<Acc>as(threads);array<double,W1_SIZE>g1{};array<double,HIDDEN>g2{};double lr=.0001/d.rows,mom=.75;auto st=chrono::steady_clock::now();
#pragma omp parallel num_threads(threads) shared(n,as,g1,g2)
 {int tid=omp_get_thread_num();for(size_t u=0;u<updates;u++){as[tid].clear();
#pragma omp for schedule(static)
  for(long long b=0;b<(long long)((d.rows+stride-1)/stride);b++)kernel(d,(size_t)b*stride,n,as[tid],u+1==updates);
#pragma omp single
  {g1.fill(0);g2.fill(0);for(int t=0;t<threads;t++){for(size_t i=0;i<W1_SIZE;i++)g1[i]+=as[t].g1[i];for(size_t j=0;j<HIDDEN;j++)g2[j]+=as[t].g2[j];}update(n,g1,g2,lr,mom);}
 }}auto en=chrono::steady_clock::now();checksum=0;for(double v:n.w1)checksum+=v;for(double v:n.w2)checksum+=v;return chrono::duration<double>(en-st).count();}
int main(int argc,char**argv){
 size_t rows=argc>1?strtoull(argv[1],0,10):1000000,updates=argc>2?strtoull(argv[2],0,10):10,reps=argc>3?strtoull(argv[3],0,10):7; int threads=argc>4?atoi(argv[4]):1; Dataset d(rows); fillData(d);
 Kernel v7kernel=(threads>1 && rows<50000)?block_v6:block_forward4;
 for(size_t rep=0;rep<reps;rep++){
  double cv6=0,cv7=0,tv6=0,tv7=0;
  if(rep%2==0){tv6=run(d,updates,threads,block_v6,cv6);tv7=run(d,updates,threads,v7kernel,cv7);}else{tv7=run(d,updates,threads,v7kernel,cv7);tv6=run(d,updates,threads,block_v6,cv6);}
  cout<<setprecision(17)<<rows<<","<<updates<<","<<threads<<","<<rep<<","<<tv6<<","<<tv7<<","<<cv6<<","<<cv7<<"\n";
 }
}
