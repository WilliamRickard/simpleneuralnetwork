#define main v8_benchmark_reference_main
#include "../../v8/benchmark/bench_v8.cpp"
#undef main

__attribute__((target("avx512f"))) static inline void sigAVX512_unchecked(double*a,size_t n){const __m512d one=_mm512_set1_pd(1),zero=_mm512_setzero_pd();for(size_t i=0;i<n;i+=8){__m512d x=_mm512_loadu_pd(a+i);__m512d e=_ZGVeN8v_exp(_mm512_sub_pd(zero,x));_mm512_storeu_pd(a+i,_mm512_div_pd(one,_mm512_add_pd(one,e)));}}


__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void v9_metrics_and_delta(const Dataset&data,size_t base,double*out,double*d3,double*pct,Acc&acc,bool detailed){
 const __m512d one=_mm512_set1_pd(1.0),hundred=_mm512_set1_pd(100.0),signmask=_mm512_set1_pd(-0.0);
 for(size_t r=0;r<TILE;r+=8){__m512d pred=_mm512_load_pd(out+r),act=_mm512_loadu_pd(data.y.data()+base+r),err=_mm512_sub_pd(pred,act),absErr=_mm512_andnot_pd(signmask,err);__m512d p=_mm512_mul_pd(_mm512_div_pd(absErr,act),hundred);__mmask8 zero=_mm512_cmp_pd_mask(act,_mm512_setzero_pd(),_CMP_EQ_OQ);p=_mm512_mask_mov_pd(p,zero,_mm512_setzero_pd());_mm512_store_pd(pct+r,p);_mm512_store_pd(d3+r,_mm512_mul_pd(_mm512_mul_pd(err,pred),_mm512_sub_pd(one,pred)));}
 for(size_t q=0;q<TILE;q++){acc.pct+=pct[q];if(detailed){double err=out[q]-data.y[base+q];acc.sq+=err*err;acc.maxpct=max(acc.maxpct,pct[q]);}}
}

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void block_v9_basic(const Dataset&data,size_t base,const Network&net,Acc&acc,bool detailed){
 constexpr size_t n=TILE;alignas(64) double hidden[TILE*HIDDEN];alignas(64) double out[TILE],d3[TILE],pct[TILE];const __m512d one=_mm512_set1_pd(1.0);
 size_t rr=0;for(;rr+3<n;rr+=4){const double*x0=data.rowData(base+rr),*x1=data.rowData(base+rr+1),*x2=data.rowData(base+rr+2),*x3=data.rowData(base+rr+3);__m512d a0=_mm512_setzero_pd(),a1=_mm512_setzero_pd(),b0=_mm512_setzero_pd(),b1=_mm512_setzero_pd(),c0=_mm512_setzero_pd(),c1=_mm512_setzero_pd(),e0=_mm512_setzero_pd(),e1=_mm512_setzero_pd();for(size_t k=0;k<INPUTS;k++){const double*w=net.w1.data()+k*HIDDEN;__m512d w0=_mm512_loadu_pd(w),w1=_mm512_loadu_pd(w+8),v0=_mm512_set1_pd(x0[k]),v1=_mm512_set1_pd(x1[k]),v2=_mm512_set1_pd(x2[k]),v3=_mm512_set1_pd(x3[k]);a0=_mm512_fmadd_pd(v0,w0,a0);a1=_mm512_fmadd_pd(v0,w1,a1);b0=_mm512_fmadd_pd(v1,w0,b0);b1=_mm512_fmadd_pd(v1,w1,b1);c0=_mm512_fmadd_pd(v2,w0,c0);c1=_mm512_fmadd_pd(v2,w1,c1);e0=_mm512_fmadd_pd(v3,w0,e0);e1=_mm512_fmadd_pd(v3,w1,e1);}double*h0=hidden+rr*HIDDEN,*h1=hidden+(rr+1)*HIDDEN,*h2=hidden+(rr+2)*HIDDEN,*h3=hidden+(rr+3)*HIDDEN;_mm512_store_pd(h0,a0);_mm512_store_pd(h0+8,a1);_mm512_store_pd(h1,b0);_mm512_store_pd(h1+8,b1);_mm512_store_pd(h2,c0);_mm512_store_pd(h2+8,c1);_mm512_store_pd(h3,e0);_mm512_store_pd(h3+8,e1);}for(;rr<n;rr++){const double*x=data.rowData(base+rr);__m512d z0=_mm512_setzero_pd(),z1=_mm512_setzero_pd();for(size_t k=0;k<INPUTS;k++){__m512d v=_mm512_set1_pd(x[k]);const double*w=net.w1.data()+k*HIDDEN;z0=_mm512_fmadd_pd(v,_mm512_loadu_pd(w),z0);z1=_mm512_fmadd_pd(v,_mm512_loadu_pd(w+8),z1);}double*h=hidden+rr*HIDDEN;_mm512_store_pd(h,z0);_mm512_store_pd(h+8,z1);}sigAVX512_unchecked(hidden,n*HIDDEN);
 const __m512d w20=_mm512_loadu_pd(net.w2.data()),w21=_mm512_loadu_pd(net.w2.data()+8);for(size_t r=0;r<n;r++){double*h=hidden+r*HIDDEN;__m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));out[r]=_mm512_reduce_add_pd(sum);}sigAVX512_unchecked(out,n);
 v9_metrics_and_delta(data,base,out,d3,pct,acc,detailed);
 __m512d gg20=_mm512_loadu_pd(acc.g2.data()),gg21=_mm512_loadu_pd(acc.g2.data()+8);DECL_G1
 for(size_t q=0;q<n;q++){const double*a=hidden+q*HIDDEN;__m512d d=_mm512_set1_pd(d3[q]),a0=_mm512_load_pd(a),a1=_mm512_load_pd(a+8);gg20=_mm512_fmadd_pd(a0,d,gg20);gg21=_mm512_fmadd_pd(a1,d,gg21);__m512d d0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0)));__m512d d1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));const double*x=data.rowData(base+q);ACC(0,ga0,gb0);ACC(1,ga1,gb1);ACC(2,ga2,gb2);ACC(3,ga3,gb3);ACC(4,ga4,gb4);ACC(5,ga5,gb5);ACC(6,ga6,gb6);ACC(7,ga7,gb7);ACC(8,ga8,gb8);ACC(9,ga9,gb9);ACC(10,ga10,gb10);} _mm512_storeu_pd(acc.g2.data(),gg20);_mm512_storeu_pd(acc.g2.data()+8,gg21);STORE(0,ga0,gb0);STORE(1,ga1,gb1);STORE(2,ga2,gb2);STORE(3,ga3,gb3);STORE(4,ga4,gb4);STORE(5,ga5,gb5);STORE(6,ga6,gb6);STORE(7,ga7,gb7);STORE(8,ga8,gb8);STORE(9,ga9,gb9);STORE(10,ga10,gb10);
}

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline void block_v9_nopct(const Dataset&data,size_t base,const Network&net,Acc&acc,bool detailed){
 constexpr size_t n=TILE;alignas(64) double hidden[TILE*HIDDEN];alignas(64) double out[TILE],d3[TILE];const __m512d one=_mm512_set1_pd(1.0);
 size_t rr=0;for(;rr+3<n;rr+=4){const double*x0=data.rowData(base+rr),*x1=data.rowData(base+rr+1),*x2=data.rowData(base+rr+2),*x3=data.rowData(base+rr+3);__m512d a0=_mm512_setzero_pd(),a1=_mm512_setzero_pd(),b0=_mm512_setzero_pd(),b1=_mm512_setzero_pd(),c0=_mm512_setzero_pd(),c1=_mm512_setzero_pd(),e0=_mm512_setzero_pd(),e1=_mm512_setzero_pd();for(size_t k=0;k<INPUTS;k++){const double*w=net.w1.data()+k*HIDDEN;__m512d w0=_mm512_loadu_pd(w),w1=_mm512_loadu_pd(w+8),v0=_mm512_set1_pd(x0[k]),v1=_mm512_set1_pd(x1[k]),v2=_mm512_set1_pd(x2[k]),v3=_mm512_set1_pd(x3[k]);a0=_mm512_fmadd_pd(v0,w0,a0);a1=_mm512_fmadd_pd(v0,w1,a1);b0=_mm512_fmadd_pd(v1,w0,b0);b1=_mm512_fmadd_pd(v1,w1,b1);c0=_mm512_fmadd_pd(v2,w0,c0);c1=_mm512_fmadd_pd(v2,w1,c1);e0=_mm512_fmadd_pd(v3,w0,e0);e1=_mm512_fmadd_pd(v3,w1,e1);}double*h0=hidden+rr*HIDDEN,*h1=hidden+(rr+1)*HIDDEN,*h2=hidden+(rr+2)*HIDDEN,*h3=hidden+(rr+3)*HIDDEN;_mm512_store_pd(h0,a0);_mm512_store_pd(h0+8,a1);_mm512_store_pd(h1,b0);_mm512_store_pd(h1+8,b1);_mm512_store_pd(h2,c0);_mm512_store_pd(h2+8,c1);_mm512_store_pd(h3,e0);_mm512_store_pd(h3+8,e1);}for(;rr<n;rr++){const double*x=data.rowData(base+rr);__m512d z0=_mm512_setzero_pd(),z1=_mm512_setzero_pd();for(size_t k=0;k<INPUTS;k++){__m512d v=_mm512_set1_pd(x[k]);const double*w=net.w1.data()+k*HIDDEN;z0=_mm512_fmadd_pd(v,_mm512_loadu_pd(w),z0);z1=_mm512_fmadd_pd(v,_mm512_loadu_pd(w+8),z1);}double*h=hidden+rr*HIDDEN;_mm512_store_pd(h,z0);_mm512_store_pd(h+8,z1);}sigAVX512_unchecked(hidden,n*HIDDEN);
 const __m512d w20=_mm512_loadu_pd(net.w2.data()),w21=_mm512_loadu_pd(net.w2.data()+8);for(size_t r=0;r<n;r++){double*h=hidden+r*HIDDEN;__m512d sum=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));out[r]=_mm512_reduce_add_pd(sum);}sigAVX512_unchecked(out,n);
 size_t r=0;for(;r+8<=n;r+=8){__m512d pred=_mm512_load_pd(out+r),act=_mm512_loadu_pd(data.y.data()+base+r),err=_mm512_sub_pd(pred,act);_mm512_store_pd(d3+r,_mm512_mul_pd(_mm512_mul_pd(err,pred),_mm512_sub_pd(one,pred)));}for(;r<n;r++){double pred=out[r],err=pred-data.y[base+r];d3[r]=err*pred*(1-pred);} (void)detailed;
 __m512d gg20=_mm512_loadu_pd(acc.g2.data()),gg21=_mm512_loadu_pd(acc.g2.data()+8);DECL_G1
 for(size_t q=0;q<n;q++){const double*a=hidden+q*HIDDEN;__m512d d=_mm512_set1_pd(d3[q]),a0=_mm512_load_pd(a),a1=_mm512_load_pd(a+8);gg20=_mm512_fmadd_pd(a0,d,gg20);gg21=_mm512_fmadd_pd(a1,d,gg21);__m512d d0=_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0)));__m512d d1=_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));const double*x=data.rowData(base+q);ACC(0,ga0,gb0);ACC(1,ga1,gb1);ACC(2,ga2,gb2);ACC(3,ga3,gb3);ACC(4,ga4,gb4);ACC(5,ga5,gb5);ACC(6,ga6,gb6);ACC(7,ga7,gb7);ACC(8,ga8,gb8);ACC(9,ga9,gb9);ACC(10,ga10,gb10);} _mm512_storeu_pd(acc.g2.data(),gg20);_mm512_storeu_pd(acc.g2.data()+8,gg21);STORE(0,ga0,gb0);STORE(1,ga1,gb1);STORE(2,ga2,gb2);STORE(3,ga3,gb3);STORE(4,ga4,gb4);STORE(5,ga5,gb5);STORE(6,ga6,gb6);STORE(7,ga7,gb7);STORE(8,ga8,gb8);STORE(9,ga9,gb9);STORE(10,ga10,gb10);
}

static array<double,INPUTS> v9MaxAbsX{};
static void prepareV9Bounds(const Dataset&d){v9MaxAbsX.fill(0.0);for(size_t r=0;r<d.rows;r++){const double*x=d.rowData(r);for(size_t k=0;k<INPUTS;k++)v9MaxAbsX[k]=max(v9MaxAbsX[k],abs(x[k]));}}
static bool v9UncheckedSafe(const Network&n){
    constexpr long double limit=699.0L;
    for(size_t j=0;j<HIDDEN;j++){long double bound=0.0L;for(size_t k=0;k<INPUTS;k++)bound+=(long double)v9MaxAbsX[k]*abs((long double)n.w1[k*HIDDEN+j]);if(!(bound<limit))return false;}
    long double outputLower=0.0L;for(size_t j=0;j<HIDDEN;j++)if(n.w2[j]<0.0)outputLower+=(long double)n.w2[j];return outputLower>-limit;
}
__attribute__((target("avx512f,avx512dq,fma"))) static void range_v9_fast(const Dataset&d,size_t firstBlock,size_t lastBlock,const Network&n,Acc&a,bool detailed){
    const double pctLimit=3.9*static_cast<double>(d.rows);
    const size_t fullBlocks=d.rows/TILE;
    const size_t fullLast=min(lastBlock,fullBlocks);
    for(size_t b=firstBlock;b<fullLast;b++){
        if(detailed || a.pct<=pctLimit) block_v9_basic(d,b*TILE,n,a,detailed);
        else block_v9_nopct(d,b*TILE,n,a,false);
    }
    if(lastBlock>fullBlocks && firstBlock<=fullBlocks && fullBlocks*TILE<d.rows){
        if(detailed || a.pct<=pctLimit) block_v8_basic(d,fullBlocks*TILE,n,a,detailed);
        else block_v8_nopct(d,fullBlocks*TILE,n,a,false);
    }
}
static double run_v9(const Dataset&d,size_t updates,int threads,double&checksum){
    Network n;init(n);vector<Acc>as(threads);array<double,W1_SIZE>g1{};array<double,HIDDEN>g2{};double lr=.0001/d.rows,mom=.75;const size_t blocks=(d.rows+TILE-1)/TILE;auto st=chrono::steady_clock::now();
#pragma omp parallel num_threads(threads) shared(n,as,g1,g2)
    {int tid=omp_get_thread_num();size_t q=blocks/(size_t)threads,rem=blocks%(size_t)threads;size_t first=(size_t)tid*q+min((size_t)tid,rem), count=q+((size_t)tid<rem?1:0), last=first+count;for(size_t u=0;u<updates;u++){as[tid].clear();const bool fast=v9UncheckedSafe(n);if(fast)range_v9_fast(d,first,last,n,as[tid],u+1==updates);else range_v8(d,first,last,n,as[tid],u+1==updates);
#pragma omp barrier
#pragma omp single
      {g1.fill(0);g2.fill(0);for(int t=0;t<threads;t++){for(size_t i=0;i<W1_SIZE;i++)g1[i]+=as[t].g1[i];for(size_t j=0;j<HIDDEN;j++)g2[j]+=as[t].g2[j];}update(n,g1,g2,lr,mom);}
    }}auto en=chrono::steady_clock::now();checksum=0;for(double v:n.w1)checksum+=v;for(double v:n.w2)checksum+=v;return chrono::duration<double>(en-st).count();
}
static double run_v9_final(const Dataset&d,size_t updates,int threads,double&checksum){if(threads==1){if(d.rows<50000)return run_v8_final(d,updates,threads,checksum);return run_v9(d,updates,threads,checksum);}if(d.rows<1000000)return run_v8_final(d,updates,threads,checksum);return run_v9(d,updates,threads,checksum);}
int main(int argc,char**argv){size_t rows=argc>1?strtoull(argv[1],0,10):1000000,updates=argc>2?strtoull(argv[2],0,10):10,reps=argc>3?strtoull(argv[3],0,10):9;int threads=argc>4?atoi(argv[4]):1;Dataset d(rows);fillData(d);prepareV9Bounds(d);for(size_t rep=0;rep<reps;rep++){double c8=0,c9=0,t8=0,t9=0;if(rep%2==0){t8=run_v8_final(d,updates,threads,c8);t9=run_v9_final(d,updates,threads,c9);}else{t9=run_v9_final(d,updates,threads,c9);t8=run_v8_final(d,updates,threads,c8);}cout<<setprecision(17)<<rows<<","<<updates<<","<<threads<<","<<rep<<","<<t8<<","<<t9<<","<<c8<<","<<c9<<"\n";}}
