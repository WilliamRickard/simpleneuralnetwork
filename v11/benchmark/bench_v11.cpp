#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;
constexpr size_t INPUTS=11,HIDDEN=16,W1_SIZE=INPUTS*HIDDEN;
struct Dataset { size_t rows; vector<double>x,y; explicit Dataset(size_t r):rows(r),x(r*INPUTS),y(r){} const double* row(size_t r)const{return x.data()+r*INPUTS;} };
struct SoA { size_t rows; array<vector<double>,INPUTS> x; explicit SoA(const Dataset&d):rows(d.rows){for(auto &v:x)v.resize(rows); for(size_t r=0;r<rows;r++)for(size_t k=0;k<INPUTS;k++)x[k][r]=d.x[r*INPUTS+k];} };
struct Network { array<double,W1_SIZE>w1{},v1{}; array<double,HIDDEN>w2{},v2{}; };
struct Grad { array<double,W1_SIZE>g1{}; array<double,HIDDEN>g2{}; void clear(){g1.fill(0);g2.fill(0);} };
static inline double sigmoid(double x){return 1.0/(1.0+exp(-x));}
extern "C" __m512d _ZGVeN8v_exp(__m512d);
__attribute__((target("avx512f"),always_inline)) static inline __m512d sig_exp(__m512d x){const __m512d one=_mm512_set1_pd(1),zero=_mm512_setzero_pd();return _mm512_div_pd(one,_mm512_add_pd(one,_ZGVeN8v_exp(_mm512_sub_pd(zero,x))));}
__attribute__((target("avx512f,fma"),always_inline)) static inline __m512d horner7(__m512d x,const double*c){__m512d y=_mm512_set1_pd(c[7]);for(int i=6;i>=0;i--)y=_mm512_fmadd_pd(y,x,_mm512_set1_pd(c[i]));return y;}
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d sig_poly(__m512d x){
 static const double c0[8]={0.5000536583248588,0.24905286710580854,0.003971220002518924,-0.027482918231841735,0.005064184209432201,0.0006650599914492094,-0.0002955219153516808,2.527875012219713e-05};
 static const double c1[8]={0.45272885944068675,0.39371907824193114,-0.12439369652570964,0.022214562641988605,-0.002407424657046711,0.00015757674997516407,-5.7471130746696645e-06,8.985262315466832e-08};
 const __m512d sign=_mm512_set1_pd(-0.0),four=_mm512_set1_pd(4.0),twelve=_mm512_set1_pd(12.0),one=_mm512_set1_pd(1.0);
 __m512d a=_mm512_andnot_pd(sign,x); __m512d p0=horner7(a,c0),p1=horner7(a,c1); __mmask8 m4=_mm512_cmp_pd_mask(a,four,_CMP_LE_OQ),m12=_mm512_cmp_pd_mask(a,twelve,_CMP_LE_OQ); __m512d p=_mm512_mask_blend_pd(m4,p1,p0); p=_mm512_mask_mov_pd(one,m12,p); __mmask8 neg=_mm512_cmp_pd_mask(x,_mm512_setzero_pd(),_CMP_LT_OQ); return _mm512_mask_sub_pd(p,neg,one,p);
}

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d horner9(__m512d x,const double*c){__m512d y=_mm512_set1_pd(c[9]);for(int i=8;i>=0;i--)y=_mm512_fmadd_pd(y,x,_mm512_set1_pd(c[i]));return y;}
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d sig_poly9(__m512d x){
 static const double c0[10]={4.99996280e-01,2.50107620e-01,-7.66383702e-04,-1.84967016e-02,-3.75485755e-03,5.55861726e-03,-1.87235914e-03,3.12411311e-04,-2.65294872e-05,9.00758371e-07};
 static const double c1[10]={3.05703511e-01,5.69493989e-01,-2.15449562e-01,4.90289331e-02,-7.35221983e-03,7.49416882e-04,-5.16857687e-05,2.31680826e-06,-6.10523825e-08,7.18768673e-10};
 const __m512d sign=_mm512_set1_pd(-0.0),four=_mm512_set1_pd(4.0),twelve=_mm512_set1_pd(12.0),one=_mm512_set1_pd(1.0);
 __m512d a=_mm512_andnot_pd(sign,x);__m512d p0=horner9(a,c0),p1=horner9(a,c1);__mmask8 m4=_mm512_cmp_pd_mask(a,four,_CMP_LE_OQ),m12=_mm512_cmp_pd_mask(a,twelve,_CMP_LE_OQ);__m512d p=_mm512_mask_blend_pd(m4,p1,p0);p=_mm512_mask_mov_pd(one,m12,p);__mmask8 neg=_mm512_cmp_pd_mask(x,_mm512_setzero_pd(),_CMP_LT_OQ);return _mm512_mask_sub_pd(p,neg,one,p);
}

__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d poly4c(__m512d x){
 const __m512d c4=_mm512_set1_pd(0.0041853293217438935),c3=_mm512_set1_pd(-0.024120015689114357),c2=_mm512_set1_pd(0.0011384450152525012),c1=_mm512_set1_pd(0.24984606999988493),c0=_mm512_set1_pd(0.5000049283417471);
 __m512d y=_mm512_fmadd_pd(c4,x,c3);y=_mm512_fmadd_pd(y,x,c2);y=_mm512_fmadd_pd(y,x,c1);return _mm512_fmadd_pd(y,x,c0);
}
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d poly5c(__m512d x){
 const __m512d c5=_mm512_set1_pd(0.0011067982467509862),c4=_mm512_set1_pd(0.0014183337048706388),c3=_mm512_set1_pd(-0.02166046710407002),c2=_mm512_set1_pd(0.00021611775455784382),c1=_mm512_set1_pd(0.24997782971952903),c0=_mm512_set1_pd(0.5000005365048115);
 __m512d y=_mm512_fmadd_pd(c5,x,c4);y=_mm512_fmadd_pd(y,x,c3);y=_mm512_fmadd_pd(y,x,c2);y=_mm512_fmadd_pd(y,x,c1);return _mm512_fmadd_pd(y,x,c0);
}
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d horner8(__m512d x,const double*c){__m512d y=_mm512_set1_pd(c[8]);for(int i=7;i>=0;i--)y=_mm512_fmadd_pd(y,x,_mm512_set1_pd(c[i]));return y;}
template<bool DEG5>
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d sig_adapt(__m512d x){
 static const double cm[9]={0.5000011358677788,0.24999835406259627,-0.00016539918722939355,-0.019899021588728794,-0.002045765106270335,0.0043622465752078555,-0.001373869927816452,0.0001903321352260143,-1.0315836568979578e-05};
 static const double ch[9]={0.35947614514228465,0.5002804028401375,-0.17657318540721206,0.03651949030026908,-0.004810100518711358,0.0004109220099530741,-2.213677938171726e-05,6.851190548062953e-07,-9.301037994539489e-09};
 const __m512d sign=_mm512_set1_pd(-0.0),one=_mm512_set1_pd(1.0),four=_mm512_set1_pd(4.0),twelve=_mm512_set1_pd(12.0);
 __m512d a=_mm512_andnot_pd(sign,x);__mmask8 m1=_mm512_cmp_pd_mask(a,one,_CMP_LE_OQ);__m512d p;
 if(m1==0xFF){p=DEG5?poly5c(a):poly4c(a);}else{__mmask8 m4=_mm512_cmp_pd_mask(a,four,_CMP_LE_OQ);if(m4==0xFF)p=horner8(a,cm);else{__m512d pm=horner8(a,cm),ph=horner8(a,ch);p=_mm512_mask_blend_pd(m4,ph,pm);__mmask8 m12=_mm512_cmp_pd_mask(a,twelve,_CMP_LE_OQ);p=_mm512_mask_mov_pd(one,m12,p);}}
 __mmask8 neg=_mm512_cmp_pd_mask(x,_mm512_setzero_pd(),_CMP_LT_OQ);return _mm512_mask_sub_pd(p,neg,one,p);
}
static void fillData(Dataset&d){array<double,W1_SIZE>tw1{};array<double,HIDDEN>tw2{};for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)tw1[k*HIDDEN+j]=.22*sin(.17*(k+1)*(j+2));for(size_t j=0;j<HIDDEN;j++)tw2[j]=.28*cos(.31*(j+1));for(size_t i=0;i<d.rows;i++){double*x=d.x.data()+i*INPUTS;for(size_t k=0;k<INPUTS;k++)x[k]=.55*sin(.013*(i+1)*(k+1))+.35*cos(.007*(i+3)*(k+2));double z3=0;for(size_t j=0;j<HIDDEN;j++){double z2=0;for(size_t k=0;k<INPUTS;k++)z2+=x[k]*tw1[k*HIDDEN+j];z3+=sigmoid(z2)*tw2[j];}d.y[i]=sigmoid(z3);}}
static void init(Network&n){for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)n.w1[k*HIDDEN+j]=.12*sin(.43*(k+1)*(j+1));for(size_t j=0;j<HIDDEN;j++)n.w2[j]=.15*cos(.37*(j+1));n.v1.fill(0);n.v2.fill(0);}
static void update(Network&n,const array<double,W1_SIZE>&g1,const array<double,HIDDEN>&g2,double lr,double mom){for(size_t i=0;i<W1_SIZE;i++){n.v1[i]=mom*n.v1[i]-lr*g1[i];n.w1[i]+=n.v1[i];}for(size_t j=0;j<HIDDEN;j++){n.v2[j]=mom*n.v2[j]-lr*g2[j];n.w2[j]+=n.v2[j];}}
// Baseline: v10-like 16-row tiles, 8-row forward, libmvec sigmoid, exact-ish tile accumulation.
template<int SIGMODE> __attribute__((target("avx512f,avx512dq,fma"))) static void baseline_range(const Dataset&d,size_t begin,size_t end,const Network&n,Grad&g){
 alignas(64) double hidden[16*16],out[16],d3[16]; const __m512d one=_mm512_set1_pd(1.0); size_t base=begin;
 for(;base+16<=end;base+=16){
  for(size_t rr=0;rr<16;rr+=8){const double*x[8]={d.row(base+rr),d.row(base+rr+1),d.row(base+rr+2),d.row(base+rr+3),d.row(base+rr+4),d.row(base+rr+5),d.row(base+rr+6),d.row(base+rr+7)}; __m512d z0[8],z1[8];for(int q=0;q<8;q++){z0[q]=_mm512_setzero_pd();z1[q]=_mm512_setzero_pd();}for(size_t k=0;k<INPUTS;k++){const double*w=n.w1.data()+k*HIDDEN;__m512d w0=_mm512_loadu_pd(w),w1=_mm512_loadu_pd(w+8);for(int q=0;q<8;q++){__m512d xv=_mm512_set1_pd(x[q][k]);z0[q]=_mm512_fmadd_pd(xv,w0,z0[q]);z1[q]=_mm512_fmadd_pd(xv,w1,z1[q]);}}for(int q=0;q<8;q++){_mm512_store_pd(hidden+(rr+q)*HIDDEN,z0[q]);_mm512_store_pd(hidden+(rr+q)*HIDDEN+8,z1[q]);}}
  for(size_t i=0;i<16*16;i+=8)_mm512_store_pd(hidden+i,SIGMODE==1?sig_poly(_mm512_load_pd(hidden+i)):(SIGMODE==2?sig_poly9(_mm512_load_pd(hidden+i)):(SIGMODE==3?sig_adapt<false>(_mm512_load_pd(hidden+i)):(SIGMODE==4?sig_adapt<true>(_mm512_load_pd(hidden+i)):sig_exp(_mm512_load_pd(hidden+i))))));
  __m512d w20=_mm512_loadu_pd(n.w2.data()),w21=_mm512_loadu_pd(n.w2.data()+8);for(int r=0;r<16;r++){double*h=hidden+r*HIDDEN;__m512d s=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));out[r]=_mm512_reduce_add_pd(s);}for(int r=0;r<16;r+=8){__m512d raw=_mm512_load_pd(out+r);__m512d s=SIGMODE==1?sig_poly(raw):(SIGMODE==2?sig_poly9(raw):(SIGMODE==3?sig_adapt<false>(raw):(SIGMODE==4?sig_adapt<true>(raw):sig_exp(raw))));_mm512_store_pd(out+r,s);__m512d y=_mm512_loadu_pd(d.y.data()+base+r),e=_mm512_sub_pd(s,y);_mm512_store_pd(d3+r,_mm512_mul_pd(_mm512_mul_pd(e,s),_mm512_sub_pd(one,s)));}
  __m512d gg2a=_mm512_setzero_pd(),gg2b=_mm512_setzero_pd(); __m512d ga[INPUTS][2];for(size_t k=0;k<INPUTS;k++){ga[k][0]=_mm512_setzero_pd();ga[k][1]=_mm512_setzero_pd();}
  for(int r=0;r<16;r++){double*h=hidden+r*HIDDEN;__m512d a0=_mm512_load_pd(h),a1=_mm512_load_pd(h+8),dv=_mm512_set1_pd(d3[r]);gg2a=_mm512_fmadd_pd(a0,dv,gg2a);gg2b=_mm512_fmadd_pd(a1,dv,gg2b);__m512d de0=_mm512_mul_pd(_mm512_mul_pd(dv,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0))),de1=_mm512_mul_pd(_mm512_mul_pd(dv,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));const double*x=d.row(base+r);for(size_t k=0;k<INPUTS;k++){__m512d xv=_mm512_set1_pd(x[k]);ga[k][0]=_mm512_fmadd_pd(xv,de0,ga[k][0]);ga[k][1]=_mm512_fmadd_pd(xv,de1,ga[k][1]);}}
  alignas(64) double tmp[16];_mm512_store_pd(tmp,gg2a);_mm512_store_pd(tmp+8,gg2b);for(int j=0;j<16;j++)g.g2[j]+=tmp[j];for(size_t k=0;k<INPUTS;k++){_mm512_store_pd(tmp,ga[k][0]);_mm512_store_pd(tmp+8,ga[k][1]);for(int j=0;j<16;j++)g.g1[k*HIDDEN+j]+=tmp[j];}
 }
}template<int SIGMODE> __attribute__((target("avx512f,avx512dq,fma"))) static void baseline4_range(const Dataset&d,size_t begin,size_t end,const Network&n,Grad&g){
 alignas(64) double hidden[16*16],out[16],d3[16]; const __m512d one=_mm512_set1_pd(1.0); size_t base=begin;
 for(;base+16<=end;base+=16){
  for(size_t rr=0;rr<16;rr+=4){const double*x[4]={d.row(base+rr),d.row(base+rr+1),d.row(base+rr+2),d.row(base+rr+3)}; __m512d z0[4],z1[4];for(int q=0;q<4;q++){z0[q]=_mm512_setzero_pd();z1[q]=_mm512_setzero_pd();}for(size_t k=0;k<INPUTS;k++){const double*w=n.w1.data()+k*HIDDEN;__m512d w0=_mm512_loadu_pd(w),w1=_mm512_loadu_pd(w+8);for(int q=0;q<4;q++){__m512d xv=_mm512_set1_pd(x[q][k]);z0[q]=_mm512_fmadd_pd(xv,w0,z0[q]);z1[q]=_mm512_fmadd_pd(xv,w1,z1[q]);}}for(int q=0;q<4;q++){_mm512_store_pd(hidden+(rr+q)*HIDDEN,z0[q]);_mm512_store_pd(hidden+(rr+q)*HIDDEN+8,z1[q]);}}
  for(size_t i=0;i<16*16;i+=8)_mm512_store_pd(hidden+i,SIGMODE==1?sig_poly(_mm512_load_pd(hidden+i)):(SIGMODE==2?sig_poly9(_mm512_load_pd(hidden+i)):(SIGMODE==3?sig_adapt<false>(_mm512_load_pd(hidden+i)):(SIGMODE==4?sig_adapt<true>(_mm512_load_pd(hidden+i)):sig_exp(_mm512_load_pd(hidden+i))))));
  __m512d w20=_mm512_loadu_pd(n.w2.data()),w21=_mm512_loadu_pd(n.w2.data()+8);for(int r=0;r<16;r++){double*h=hidden+r*HIDDEN;__m512d s=_mm512_fmadd_pd(_mm512_load_pd(h+8),w21,_mm512_mul_pd(_mm512_load_pd(h),w20));out[r]=_mm512_reduce_add_pd(s);}for(int r=0;r<16;r+=8){__m512d raw=_mm512_load_pd(out+r);__m512d s=SIGMODE==1?sig_poly(raw):(SIGMODE==2?sig_poly9(raw):(SIGMODE==3?sig_adapt<false>(raw):(SIGMODE==4?sig_adapt<true>(raw):sig_exp(raw))));_mm512_store_pd(out+r,s);__m512d y=_mm512_loadu_pd(d.y.data()+base+r),e=_mm512_sub_pd(s,y);_mm512_store_pd(d3+r,_mm512_mul_pd(_mm512_mul_pd(e,s),_mm512_sub_pd(one,s)));}
  __m512d gg2a=_mm512_setzero_pd(),gg2b=_mm512_setzero_pd(); __m512d ga[INPUTS][2];for(size_t k=0;k<INPUTS;k++){ga[k][0]=_mm512_setzero_pd();ga[k][1]=_mm512_setzero_pd();}
  for(int r=0;r<16;r++){double*h=hidden+r*HIDDEN;__m512d a0=_mm512_load_pd(h),a1=_mm512_load_pd(h+8),dv=_mm512_set1_pd(d3[r]);gg2a=_mm512_fmadd_pd(a0,dv,gg2a);gg2b=_mm512_fmadd_pd(a1,dv,gg2b);__m512d de0=_mm512_mul_pd(_mm512_mul_pd(dv,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0))),de1=_mm512_mul_pd(_mm512_mul_pd(dv,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));const double*x=d.row(base+r);for(size_t k=0;k<INPUTS;k++){__m512d xv=_mm512_set1_pd(x[k]);ga[k][0]=_mm512_fmadd_pd(xv,de0,ga[k][0]);ga[k][1]=_mm512_fmadd_pd(xv,de1,ga[k][1]);}}
  alignas(64) double tmp[16];_mm512_store_pd(tmp,gg2a);_mm512_store_pd(tmp+8,gg2b);for(int j=0;j<16;j++)g.g2[j]+=tmp[j];for(size_t k=0;k<INPUTS;k++){_mm512_store_pd(tmp,ga[k][0]);_mm512_store_pd(tmp+8,ga[k][1]);for(int j=0;j<16;j++)g.g1[k*HIDDEN+j]+=tmp[j];}
 }
}

template<int B,bool FASTSIG>
__attribute__((target("avx512f,avx512dq,fma"))) static void blocked_range(const Dataset&d,const SoA&s,size_t begin,size_t end,const Network&n,Grad&g){
 alignas(64) double hidden[HIDDEN*B],out[B],d3[B];const __m512d one=_mm512_set1_pd(1.0);size_t base=begin;
 for(;base+B<=end;base+=B){
  for(size_t j=0;j<HIDDEN;j++){double*hj=hidden+j*B;for(int r=0;r<B;r+=8){__m512d z=_mm512_setzero_pd();for(size_t k=0;k<INPUTS;k++)z=_mm512_fmadd_pd(_mm512_loadu_pd(s.x[k].data()+base+r),_mm512_set1_pd(n.w1[k*HIDDEN+j]),z);z=FASTSIG?sig_poly(z):sig_exp(z);_mm512_store_pd(hj+r,z);}}
  for(int r=0;r<B;r+=8){__m512d z=_mm512_setzero_pd();for(size_t j=0;j<HIDDEN;j++)z=_mm512_fmadd_pd(_mm512_load_pd(hidden+j*B+r),_mm512_set1_pd(n.w2[j]),z);z=FASTSIG?sig_poly(z):sig_exp(z);_mm512_store_pd(out+r,z);__m512d y=_mm512_loadu_pd(d.y.data()+base+r),e=_mm512_sub_pd(z,y);_mm512_store_pd(d3+r,_mm512_mul_pd(_mm512_mul_pd(e,z),_mm512_sub_pd(one,z)));}
  for(size_t j=0;j<HIDDEN;j++){
    __m512d wg2=_mm512_setzero_pd();__m512d wg1[INPUTS];for(size_t k=0;k<INPUTS;k++)wg1[k]=_mm512_setzero_pd();__m512d w2=_mm512_set1_pd(n.w2[j]);
    for(int r=0;r<B;r+=8){__m512d a=_mm512_load_pd(hidden+j*B+r),d3v=_mm512_load_pd(d3+r);wg2=_mm512_fmadd_pd(a,d3v,wg2);__m512d de=_mm512_mul_pd(_mm512_mul_pd(d3v,w2),_mm512_mul_pd(a,_mm512_sub_pd(one,a)));for(size_t k=0;k<INPUTS;k++)wg1[k]=_mm512_fmadd_pd(_mm512_loadu_pd(s.x[k].data()+base+r),de,wg1[k]);}
    g.g2[j]+=_mm512_reduce_add_pd(wg2);for(size_t k=0;k<INPUTS;k++)g.g1[k*HIDDEN+j]+=_mm512_reduce_add_pd(wg1[k]);
  }
 }
}

template<class F> static double run(const Dataset&d,const SoA&s,size_t updates,int threads,F func,Network&outnet){Network n;init(n);vector<Grad> gs(threads);array<double,W1_SIZE>g1{};array<double,HIDDEN>g2{};double lr=.0001/d.rows,mom=.75;auto st=chrono::steady_clock::now();for(size_t u=0;u<updates;u++){
 #pragma omp parallel num_threads(threads)
 {int tid=omp_get_thread_num();gs[tid].clear();size_t chunk=(d.rows/(size_t)threads/128)*128;size_t begin=(size_t)tid*chunk;size_t end=(tid==threads-1)?d.rows:begin+chunk;func(d,s,begin,end,n,gs[tid]);}
 g1.fill(0);g2.fill(0);for(int t=0;t<threads;t++){for(size_t i=0;i<W1_SIZE;i++)g1[i]+=gs[t].g1[i];for(size_t j=0;j<HIDDEN;j++)g2[j]+=gs[t].g2[j];}update(n,g1,g2,lr,mom);
 }outnet=n;return chrono::duration<double>(chrono::steady_clock::now()-st).count();}
static void metrics(const Dataset&d,const Network&n,double&rmse,double&maxAbs){long double ss=0;maxAbs=0;for(size_t r=0;r<d.rows;r++){const double*x=d.row(r);double z3=0;for(size_t j=0;j<HIDDEN;j++){double z2=0;for(size_t k=0;k<INPUTS;k++)z2+=x[k]*n.w1[k*HIDDEN+j];z3+=sigmoid(z2)*n.w2[j];}double e=sigmoid(z3)-d.y[r];ss+=e*e;maxAbs=max(maxAbs,abs(e));}rmse=sqrt((double)(ss/d.rows));}
int main(int argc,char**argv){size_t rows=argc>1?strtoull(argv[1],0,10):1000000,updates=argc>2?strtoull(argv[2],0,10):100,reps=argc>3?strtoull(argv[3],0,10):5;int threads=argc>4?atoi(argv[4]):4;Dataset d(rows);fillData(d);SoA s(d);cout<<setprecision(17);Network last0,last4;for(size_t rep=0;rep<reps;rep++){Network n0,n4;double t0,t4;if(rep%2==0){t0=run(d,s,updates,threads,[&](auto&d,auto&s,size_t b,size_t e,auto&n,auto&g){(void)s;if(threads==1) baseline4_range<0>(d,b,e,n,g); else baseline_range<0>(d,b,e,n,g);},n0);t4=run(d,s,updates,threads,[&](auto&d,auto&s,size_t b,size_t e,auto&n,auto&g){(void)s;if(threads==1) baseline4_range<3>(d,b,e,n,g); else baseline_range<3>(d,b,e,n,g);},n4);}else{t4=run(d,s,updates,threads,[&](auto&d,auto&s,size_t b,size_t e,auto&n,auto&g){(void)s;if(threads==1) baseline4_range<3>(d,b,e,n,g); else baseline_range<3>(d,b,e,n,g);},n4);t0=run(d,s,updates,threads,[&](auto&d,auto&s,size_t b,size_t e,auto&n,auto&g){(void)s;if(threads==1) baseline4_range<0>(d,b,e,n,g); else baseline_range<0>(d,b,e,n,g);},n0);}double maxW=0;for(size_t i=0;i<W1_SIZE;i++)maxW=max(maxW,abs(n0.w1[i]-n4.w1[i]));for(size_t j=0;j<HIDDEN;j++)maxW=max(maxW,abs(n0.w2[j]-n4.w2[j]));cout<<rows<<","<<updates<<","<<threads<<","<<rep<<","<<t0<<","<<t4<<","<<maxW<<"\n";last0=n0;last4=n4;}double rm0,mx0,rm4,mx4;metrics(d,last0,rm0,mx0);metrics(d,last4,rm4,mx4);cerr<<setprecision(17)<<"METRICS,"<<rows<<","<<updates<<","<<threads<<","<<rm0<<","<<rm4<<","<<mx0<<","<<mx4<<"\n";}
