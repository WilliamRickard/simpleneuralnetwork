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

namespace v10_bench_v9 {
#include "../../v9/benchmark/bench_v9.cpp"
}
using namespace v10_bench_v9;

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void forward8_v10(const Dataset&data,size_t base,const Network&net,double*hidden){
 for(size_t rr=0;rr<TILE;rr+=8){
  const double*x0=data.rowData(base+rr),*x1=data.rowData(base+rr+1),*x2=data.rowData(base+rr+2),*x3=data.rowData(base+rr+3),*x4=data.rowData(base+rr+4),*x5=data.rowData(base+rr+5),*x6=data.rowData(base+rr+6),*x7=data.rowData(base+rr+7);
  __m512d a0=_mm512_setzero_pd(),a1=_mm512_setzero_pd(),b0=_mm512_setzero_pd(),b1=_mm512_setzero_pd(),c0=_mm512_setzero_pd(),c1=_mm512_setzero_pd(),d0v=_mm512_setzero_pd(),d1v=_mm512_setzero_pd();
  __m512d e0=_mm512_setzero_pd(),e1=_mm512_setzero_pd(),f0=_mm512_setzero_pd(),f1=_mm512_setzero_pd(),g0=_mm512_setzero_pd(),g1=_mm512_setzero_pd(),h0v=_mm512_setzero_pd(),h1v=_mm512_setzero_pd();
  for(size_t k=0;k<INPUTS;k++){const double*w=net.w1.data()+k*HIDDEN;const __m512d w0=_mm512_loadu_pd(w),w1=_mm512_loadu_pd(w+8);__m512d v=_mm512_set1_pd(x0[k]);a0=_mm512_fmadd_pd(v,w0,a0);a1=_mm512_fmadd_pd(v,w1,a1);v=_mm512_set1_pd(x1[k]);b0=_mm512_fmadd_pd(v,w0,b0);b1=_mm512_fmadd_pd(v,w1,b1);v=_mm512_set1_pd(x2[k]);c0=_mm512_fmadd_pd(v,w0,c0);c1=_mm512_fmadd_pd(v,w1,c1);v=_mm512_set1_pd(x3[k]);d0v=_mm512_fmadd_pd(v,w0,d0v);d1v=_mm512_fmadd_pd(v,w1,d1v);v=_mm512_set1_pd(x4[k]);e0=_mm512_fmadd_pd(v,w0,e0);e1=_mm512_fmadd_pd(v,w1,e1);v=_mm512_set1_pd(x5[k]);f0=_mm512_fmadd_pd(v,w0,f0);f1=_mm512_fmadd_pd(v,w1,f1);v=_mm512_set1_pd(x6[k]);g0=_mm512_fmadd_pd(v,w0,g0);g1=_mm512_fmadd_pd(v,w1,g1);v=_mm512_set1_pd(x7[k]);h0v=_mm512_fmadd_pd(v,w0,h0v);h1v=_mm512_fmadd_pd(v,w1,h1v);}
  double*ha=hidden+(rr+0)*HIDDEN,*hb=hidden+(rr+1)*HIDDEN,*hc=hidden+(rr+2)*HIDDEN,*hd=hidden+(rr+3)*HIDDEN,*he=hidden+(rr+4)*HIDDEN,*hf=hidden+(rr+5)*HIDDEN,*hg=hidden+(rr+6)*HIDDEN,*hh=hidden+(rr+7)*HIDDEN;
  _mm512_store_pd(ha,a0);_mm512_store_pd(ha+8,a1);_mm512_store_pd(hb,b0);_mm512_store_pd(hb+8,b1);_mm512_store_pd(hc,c0);_mm512_store_pd(hc+8,c1);_mm512_store_pd(hd,d0v);_mm512_store_pd(hd+8,d1v);_mm512_store_pd(he,e0);_mm512_store_pd(he+8,e1);_mm512_store_pd(hf,f0);_mm512_store_pd(hf+8,f1);_mm512_store_pd(hg,g0);_mm512_store_pd(hg+8,g1);_mm512_store_pd(hh,h0v);_mm512_store_pd(hh+8,h1v);
 }
}

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void block_v10_basic(const Dataset&data,size_t base,const Network&net,Acc&acc,bool detailed){
 alignas(64) double hidden[TILE*HIDDEN],out[TILE],d3[TILE],pct[TILE];const __m512d one=_mm512_set1_pd(1.0);forward8_v10(data,base,net,hidden);sigAVX512_unchecked(hidden,TILE*HIDDEN);
 const __m512d w20=_mm512_loadu_pd(net.w2.data()),w21=_mm512_loadu_pd(net.w2.data()+8);for(size_t r=0;r<TILE;r++){double*h=hidden+r*HIDDEN;__m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));out[r]=_mm512_reduce_add_pd(sum);}sigAVX512_unchecked(out,TILE);v9_metrics_and_delta(data,base,out,d3,pct,acc,detailed);
 __m512d gg20=_mm512_loadu_pd(acc.g2.data()),gg21=_mm512_loadu_pd(acc.g2.data()+8);DECL_G1
 for(size_t q=0;q<TILE;q++){const double*a=hidden+q*HIDDEN;__m512d d=_mm512_set1_pd(d3[q]),a0=_mm512_load_pd(a),a1=_mm512_load_pd(a+8);gg20=_mm512_fmadd_pd(a0,d,gg20);gg21=_mm512_fmadd_pd(a1,d,gg21);__m512d d0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0)));__m512d d1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));const double*x=data.rowData(base+q);ACC(0,ga0,gb0);ACC(1,ga1,gb1);ACC(2,ga2,gb2);ACC(3,ga3,gb3);ACC(4,ga4,gb4);ACC(5,ga5,gb5);ACC(6,ga6,gb6);ACC(7,ga7,gb7);ACC(8,ga8,gb8);ACC(9,ga9,gb9);ACC(10,ga10,gb10);} _mm512_storeu_pd(acc.g2.data(),gg20);_mm512_storeu_pd(acc.g2.data()+8,gg21);STORE(0,ga0,gb0);STORE(1,ga1,gb1);STORE(2,ga2,gb2);STORE(3,ga3,gb3);STORE(4,ga4,gb4);STORE(5,ga5,gb5);STORE(6,ga6,gb6);STORE(7,ga7,gb7);STORE(8,ga8,gb8);STORE(9,ga9,gb9);STORE(10,ga10,gb10);
}

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void block_v10_nopct(const Dataset&data,size_t base,const Network&net,Acc&acc,bool detailed){
 alignas(64) double hidden[TILE*HIDDEN],out[TILE],d3[TILE];const __m512d one=_mm512_set1_pd(1.0);forward8_v10(data,base,net,hidden);sigAVX512_unchecked(hidden,TILE*HIDDEN);
 const __m512d w20=_mm512_loadu_pd(net.w2.data()),w21=_mm512_loadu_pd(net.w2.data()+8);for(size_t r=0;r<TILE;r++){double*h=hidden+r*HIDDEN;__m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));out[r]=_mm512_reduce_add_pd(sum);}sigAVX512_unchecked(out,TILE);for(size_t r=0;r<TILE;r+=8){__m512d pred=_mm512_load_pd(out+r),act=_mm512_loadu_pd(data.y.data()+base+r),err=_mm512_sub_pd(pred,act);_mm512_store_pd(d3+r,_mm512_mul_pd(_mm512_mul_pd(err,pred),_mm512_sub_pd(one,pred)));}(void)detailed;
 __m512d gg20=_mm512_loadu_pd(acc.g2.data()),gg21=_mm512_loadu_pd(acc.g2.data()+8);DECL_G1
 for(size_t q=0;q<TILE;q++){const double*a=hidden+q*HIDDEN;__m512d d=_mm512_set1_pd(d3[q]),a0=_mm512_load_pd(a),a1=_mm512_load_pd(a+8);gg20=_mm512_fmadd_pd(a0,d,gg20);gg21=_mm512_fmadd_pd(a1,d,gg21);__m512d d0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0)));__m512d d1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));const double*x=data.rowData(base+q);ACC(0,ga0,gb0);ACC(1,ga1,gb1);ACC(2,ga2,gb2);ACC(3,ga3,gb3);ACC(4,ga4,gb4);ACC(5,ga5,gb5);ACC(6,ga6,gb6);ACC(7,ga7,gb7);ACC(8,ga8,gb8);ACC(9,ga9,gb9);ACC(10,ga10,gb10);} _mm512_storeu_pd(acc.g2.data(),gg20);_mm512_storeu_pd(acc.g2.data()+8,gg21);STORE(0,ga0,gb0);STORE(1,ga1,gb1);STORE(2,ga2,gb2);STORE(3,ga3,gb3);STORE(4,ga4,gb4);STORE(5,ga5,gb5);STORE(6,ga6,gb6);STORE(7,ga7,gb7);STORE(8,ga8,gb8);STORE(9,ga9,gb9);STORE(10,ga10,gb10);
}

__attribute__((target("avx512f,avx512dq,fma"))) static void range_v10_fast(const Dataset&d,size_t firstBlock,size_t lastBlock,const Network&n,Acc&a,bool detailed){const double pctLimit=3.9*static_cast<double>(d.rows);const size_t fullBlocks=d.rows/TILE,fullLast=min(lastBlock,fullBlocks);for(size_t b=firstBlock;b<fullLast;b++){if(detailed||a.pct<=pctLimit)block_v10_basic(d,b*TILE,n,a,detailed);else block_v10_nopct(d,b*TILE,n,a,false);}if(lastBlock>fullBlocks&&firstBlock<=fullBlocks&&fullBlocks*TILE<d.rows){if(detailed||a.pct<=pctLimit)block_v8_basic(d,fullBlocks*TILE,n,a,detailed);else block_v8_nopct(d,fullBlocks*TILE,n,a,false);}}
static double run_v10(const Dataset&d,size_t updates,int threads,double&checksum){Network n;init(n);vector<Acc>as(threads);array<double,W1_SIZE>g1{};array<double,HIDDEN>g2{};double lr=.0001/d.rows,mom=.75;const size_t blocks=(d.rows+TILE-1)/TILE;auto st=chrono::steady_clock::now();
#pragma omp parallel num_threads(threads) shared(n,as,g1,g2)
 {int tid=omp_get_thread_num();size_t q=blocks/(size_t)threads,rem=blocks%(size_t)threads;size_t first=(size_t)tid*q+min((size_t)tid,rem),count=q+((size_t)tid<rem?1:0),last=first+count;for(size_t u=0;u<updates;u++){as[tid].clear();const bool fast=v9UncheckedSafe(n);if(fast)range_v10_fast(d,first,last,n,as[tid],u+1==updates);else range_v8(d,first,last,n,as[tid],u+1==updates);
#pragma omp barrier
#pragma omp single
 {g1.fill(0);g2.fill(0);for(int t=0;t<threads;t++){for(size_t i=0;i<W1_SIZE;i++)g1[i]+=as[t].g1[i];for(size_t j=0;j<HIDDEN;j++)g2[j]+=as[t].g2[j];}update(n,g1,g2,lr,mom);}
 }}auto en=chrono::steady_clock::now();checksum=0;for(double v:n.w1)checksum+=v;for(double v:n.w2)checksum+=v;return chrono::duration<double>(en-st).count();}
static double run_v10_final(const Dataset&d,size_t updates,int threads,double&checksum){if(threads==4&&d.rows>=1000000)return run_v10(d,updates,threads,checksum);return run_v9_final(d,updates,threads,checksum);}
int main(int argc,char**argv){size_t rows=argc>1?strtoull(argv[1],0,10):1000000,updates=argc>2?strtoull(argv[2],0,10):100,reps=argc>3?strtoull(argv[3],0,10):5;int threads=argc>4?atoi(argv[4]):4;Dataset d(rows);fillData(d);prepareV9Bounds(d);for(size_t rep=0;rep<reps;rep++){double c9=0,c10=0,t9=0,t10=0;if(rep%2==0){t9=run_v9_final(d,updates,threads,c9);t10=run_v10_final(d,updates,threads,c10);}else{t10=run_v10_final(d,updates,threads,c10);t9=run_v9_final(d,updates,threads,c9);}cout<<setprecision(17)<<rows<<","<<updates<<","<<threads<<","<<rep<<","<<t9<<","<<t10<<","<<c9<<","<<c10<<"\n";}return 0;}
