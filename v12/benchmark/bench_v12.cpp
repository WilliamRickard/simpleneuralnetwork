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
constexpr int IN=11,H=16,W1=IN*H,TILE=16;
struct Data { size_t n; vector<double> xd,yd; vector<float> xf,yf; explicit Data(size_t n):n(n),xd(n*IN),yd(n),xf(n*IN),yf(n){} };
struct NetD { array<double,W1>w1{},v1{}; array<double,H>w2{},v2{}; };
struct NetF { array<float,W1>w1{},v1{}; array<float,H>w2{},v2{}; };
struct GradD { array<double,W1>g1{}; array<double,H>g2{}; void clear(){g1.fill(0);g2.fill(0);} };
struct GradF { alignas(64) array<float,W1>g1{}; alignas(64) array<float,H>g2{}; void clear(){g1.fill(0);g2.fill(0);} };
static inline double sigd(double x){return 1.0/(1.0+exp(-x));}
static inline float sigf_scalar(float x){if(fabsf(x)<=1.0f){const float c1=.24998101634651657f,c3=-.020677835421401624f,c5=.0017580292406143272f;float x2=x*x;return fmaf(x,fmaf(x2,fmaf(c5,x2,c3),c1),.5f);}return 1.0f/(1.0f+expf(-x));}
extern "C" __m512d _ZGVeN8v_exp(__m512d);
extern "C" __m512 _ZGVeN16v_expf(__m512);
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512d sigpd(__m512d x){
 const __m512d signless=_mm512_castsi512_pd(_mm512_set1_epi64(0x7fffffffffffffffLL)),lim=_mm512_set1_pd(1.0),half=_mm512_set1_pd(.5),one=_mm512_set1_pd(1),zero=_mm512_setzero_pd();
 const __m512d c1=_mm512_set1_pd(.24998101634651657),c3=_mm512_set1_pd(-.020677835421401624),c5=_mm512_set1_pd(.0017580292406143272);
 __m512d a=_mm512_and_pd(x,signless); if(_mm512_cmp_pd_mask(a,lim,_CMP_GT_OQ)){__m512d e=_ZGVeN8v_exp(_mm512_sub_pd(zero,x));return _mm512_div_pd(one,_mm512_add_pd(one,e));}
 __m512d x2=_mm512_mul_pd(x,x),q=_mm512_fmadd_pd(c5,x2,c3);q=_mm512_fmadd_pd(q,x2,c1);return _mm512_fmadd_pd(x,q,half);
}
__attribute__((target("avx512f,avx512dq,fma"),always_inline)) static inline __m512 sigps(__m512 x){
 const __m512 signless=_mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)),lim=_mm512_set1_ps(1.0f),half=_mm512_set1_ps(.5f),one=_mm512_set1_ps(1),zero=_mm512_setzero_ps();
 const __m512 c1=_mm512_set1_ps(.24998101634651657f),c3=_mm512_set1_ps(-.020677835421401624f),c5=_mm512_set1_ps(.0017580292406143272f);
 __m512 a=_mm512_and_ps(x,signless); if(_mm512_cmp_ps_mask(a,lim,_CMP_GT_OQ)){__m512 e=_ZGVeN16v_expf(_mm512_sub_ps(zero,x));return _mm512_div_ps(one,_mm512_add_ps(one,e));}
 __m512 x2=_mm512_mul_ps(x,x),q=_mm512_fmadd_ps(c5,x2,c3);q=_mm512_fmadd_ps(q,x2,c1);return _mm512_fmadd_ps(x,q,half);
}
static void fill(Data&d){array<double,W1>tw1{};array<double,H>tw2{};for(int k=0;k<IN;k++)for(int j=0;j<H;j++)tw1[k*H+j]=.22*sin(.17*(k+1)*(j+2));for(int j=0;j<H;j++)tw2[j]=.28*cos(.31*(j+1));for(size_t r=0;r<d.n;r++){double* x=d.xd.data()+r*IN;for(int k=0;k<IN;k++){x[k]=.55*sin(.013*(r+1)*(k+1))+.35*cos(.007*(r+3)*(k+2));d.xf[r*IN+k]=(float)x[k];}double z=0;for(int j=0;j<H;j++){double h=0;for(int k=0;k<IN;k++)h+=x[k]*tw1[k*H+j];z+=sigd(h)*tw2[j];}d.yd[r]=sigd(z);d.yf[r]=(float)d.yd[r];}}
static void init(NetD&n){for(int k=0;k<IN;k++)for(int j=0;j<H;j++)n.w1[k*H+j]=.12*sin(.43*(k+1)*(j+1));for(int j=0;j<H;j++)n.w2[j]=.15*cos(.37*(j+1));n.v1.fill(0);n.v2.fill(0);}
static void init(NetF&n){NetD d;init(d);for(int i=0;i<W1;i++)n.w1[i]=(float)d.w1[i];for(int j=0;j<H;j++)n.w2[j]=(float)d.w2[j];n.v1.fill(0);n.v2.fill(0);}
static void updD(NetD&n,const array<double,W1>&g1,const array<double,H>&g2,double lr){for(int i=0;i<W1;i++){n.v1[i]=.75*n.v1[i]-lr*g1[i];n.w1[i]+=n.v1[i];}for(int j=0;j<H;j++){n.v2[j]=.75*n.v2[j]-lr*g2[j];n.w2[j]+=n.v2[j];}}
static void updF(NetF&n,const array<float,W1>&g1,const array<float,H>&g2,float lr){for(int i=0;i<W1;i++){n.v1[i]=.75f*n.v1[i]-lr*g1[i];n.w1[i]+=n.v1[i];}for(int j=0;j<H;j++){n.v2[j]=.75f*n.v2[j]-lr*g2[j];n.w2[j]+=n.v2[j];}}
template<int ROWS> __attribute__((target("avx512f,avx512dq,fma"))) static void rangeD(const Data&d,size_t b,size_t e,const NetD&n,GradD&g){
 alignas(64) double h[TILE*H],out[TILE],d3[TILE],tmp[H];const __m512d one=_mm512_set1_pd(1);for(size_t base=b;base+TILE<=e;base+=TILE){
  for(int rr=0;rr<TILE;rr+=ROWS){__m512d lo[8],hi[8];for(int q=0;q<ROWS;q++){lo[q]=_mm512_setzero_pd();hi[q]=_mm512_setzero_pd();}for(int k=0;k<IN;k++){__m512d w0=_mm512_loadu_pd(n.w1.data()+k*H),w1=_mm512_loadu_pd(n.w1.data()+k*H+8);for(int q=0;q<ROWS;q++){__m512d x=_mm512_set1_pd(d.xd[(base+rr+q)*IN+k]);lo[q]=_mm512_fmadd_pd(x,w0,lo[q]);hi[q]=_mm512_fmadd_pd(x,w1,hi[q]);}}for(int q=0;q<ROWS;q++){_mm512_store_pd(h+(rr+q)*H,sigpd(lo[q]));_mm512_store_pd(h+(rr+q)*H+8,sigpd(hi[q]));}}
  __m512d w20=_mm512_loadu_pd(n.w2.data()),w21=_mm512_loadu_pd(n.w2.data()+8);for(int r=0;r<TILE;r++){double* a=h+r*H;__m512d s=_mm512_fmadd_pd(_mm512_load_pd(a+8),w21,_mm512_mul_pd(_mm512_load_pd(a),w20));out[r]=_mm512_reduce_add_pd(s);}for(int r=0;r<TILE;r+=8){__m512d z=sigpd(_mm512_load_pd(out+r));_mm512_store_pd(out+r,z);__m512d y=_mm512_loadu_pd(d.yd.data()+base+r),er=_mm512_sub_pd(z,y);_mm512_store_pd(d3+r,_mm512_mul_pd(_mm512_mul_pd(er,z),_mm512_sub_pd(one,z)));}
  __m512d gg20=_mm512_setzero_pd(),gg21=_mm512_setzero_pd(),ga[IN][2];for(int k=0;k<IN;k++){ga[k][0]=_mm512_setzero_pd();ga[k][1]=_mm512_setzero_pd();}for(int r=0;r<TILE;r++){__m512d a0=_mm512_load_pd(h+r*H),a1=_mm512_load_pd(h+r*H+8),dv=_mm512_set1_pd(d3[r]);gg20=_mm512_fmadd_pd(a0,dv,gg20);gg21=_mm512_fmadd_pd(a1,dv,gg21);__m512d de0=_mm512_mul_pd(_mm512_mul_pd(dv,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0))),de1=_mm512_mul_pd(_mm512_mul_pd(dv,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1)));for(int k=0;k<IN;k++){__m512d xv=_mm512_set1_pd(d.xd[(base+r)*IN+k]);ga[k][0]=_mm512_fmadd_pd(xv,de0,ga[k][0]);ga[k][1]=_mm512_fmadd_pd(xv,de1,ga[k][1]);}}
  _mm512_store_pd(tmp,gg20);_mm512_store_pd(tmp+8,gg21);for(int j=0;j<H;j++)g.g2[j]+=tmp[j];for(int k=0;k<IN;k++){_mm512_store_pd(tmp,ga[k][0]);_mm512_store_pd(tmp+8,ga[k][1]);for(int j=0;j<H;j++)g.g1[k*H+j]+=tmp[j];}
 }}
__attribute__((target("avx512f,avx512dq,fma"))) static void rangeMix(const Data&d,size_t b,size_t e,const array<float,W1>&w1,const array<float,H>&w2,GradF&g){
 alignas(64) float hidden[TILE*H],out[TILE],d3[TILE];const __m512 one=_mm512_set1_ps(1.0f);const __m512 w2v=_mm512_loadu_ps(w2.data());
 for(size_t base=b;base+TILE<=e;base+=TILE){
   for(int rr=0;rr<TILE;rr+=4){__m512 z0=_mm512_setzero_ps(),z1=_mm512_setzero_ps(),z2=_mm512_setzero_ps(),z3=_mm512_setzero_ps();const float*x0=d.xf.data()+(base+rr)*IN,*x1=d.xf.data()+(base+rr+1)*IN,*x2=d.xf.data()+(base+rr+2)*IN,*x3=d.xf.data()+(base+rr+3)*IN;for(int k=0;k<IN;k++){__m512 w=_mm512_loadu_ps(w1.data()+k*H);z0=_mm512_fmadd_ps(_mm512_set1_ps(x0[k]),w,z0);z1=_mm512_fmadd_ps(_mm512_set1_ps(x1[k]),w,z1);z2=_mm512_fmadd_ps(_mm512_set1_ps(x2[k]),w,z2);z3=_mm512_fmadd_ps(_mm512_set1_ps(x3[k]),w,z3);}_mm512_store_ps(hidden+(rr+0)*H,sigps(z0));_mm512_store_ps(hidden+(rr+1)*H,sigps(z1));_mm512_store_ps(hidden+(rr+2)*H,sigps(z2));_mm512_store_ps(hidden+(rr+3)*H,sigps(z3));}
   for(int r=0;r<TILE;r++){__m512 a=_mm512_load_ps(hidden+r*H);out[r]=sigf_scalar(_mm512_reduce_add_ps(_mm512_mul_ps(a,w2v)));float p=out[r];d3[r]=(p-d.yf[base+r])*p*(1.0f-p);}
   __m512 gg2=_mm512_setzero_ps(),ga[IN];for(int k=0;k<IN;k++)ga[k]=_mm512_setzero_ps();
   for(int r=0;r<TILE;r++){const float*x=d.xf.data()+(base+r)*IN;__m512 a=_mm512_load_ps(hidden+r*H),dv=_mm512_set1_ps(d3[r]);gg2=_mm512_fmadd_ps(a,dv,gg2);__m512 de=_mm512_mul_ps(_mm512_mul_ps(dv,w2v),_mm512_mul_ps(a,_mm512_sub_ps(one,a)));for(int k=0;k<IN;k++)ga[k]=_mm512_fmadd_ps(_mm512_set1_ps(x[k]),de,ga[k]);}
   _mm512_storeu_ps(g.g2.data(),_mm512_add_ps(_mm512_loadu_ps(g.g2.data()),gg2));for(int k=0;k<IN;k++)_mm512_storeu_ps(g.g1.data()+k*H,_mm512_add_ps(_mm512_loadu_ps(g.g1.data()+k*H),ga[k]));
 }
 for(size_t r=(e-((e-b)%TILE));r<e;r++){const float*x=d.xf.data()+r*IN;alignas(64) float z[H]={};for(int j=0;j<H;j++){float q=0;for(int k=0;k<IN;k++)q=fmaf(x[k],w1[k*H+j],q);z[j]=sigf_scalar(q);}float o=0;for(int j=0;j<H;j++)o=fmaf(z[j],w2[j],o);float p=sigf_scalar(o),dd=(p-d.yf[r])*p*(1-p);for(int j=0;j<H;j++){g.g2[j]+=z[j]*dd;float de=dd*w2[j]*z[j]*(1-z[j]);for(int k=0;k<IN;k++)g.g1[k*H+j]+=x[k]*de;}}
}
static double runD(const Data&d,int updates,int threads,NetD&out){NetD n;init(n);vector<GradD>gs(threads);array<double,W1>g1{};array<double,H>g2{};double lr=.0001/d.n;auto st=chrono::steady_clock::now();
#pragma omp parallel num_threads(threads) shared(n,gs,g1,g2)
 {int tid=omp_get_thread_num();size_t q=(d.n/TILE)/(size_t)threads,rem=(d.n/TILE)%(size_t)threads,first=(tid*q+min<size_t>(tid,rem))*TILE,last=first+(q+(tid<(int)rem))*TILE;for(int u=0;u<updates;u++){gs[tid].clear();if(threads==1)rangeD<4>(d,first,last,n,gs[tid]);else rangeD<8>(d,first,last,n,gs[tid]);
#pragma omp barrier
#pragma omp single
 {g1.fill(0);g2.fill(0);for(int t=0;t<threads;t++){for(int i=0;i<W1;i++)g1[i]+=gs[t].g1[i];for(int j=0;j<H;j++)g2[j]+=gs[t].g2[j];}updD(n,g1,g2,lr);}
 }}out=n;return chrono::duration<double>(chrono::steady_clock::now()-st).count();}
static double runMix(const Data&d,int updates,int threads,NetD&out){NetD n;init(n);vector<GradF>gs(threads);array<float,W1>w1f{};array<float,H>w2f{};array<double,W1>g1{};array<double,H>g2{};double lr=.0001/d.n;auto st=chrono::steady_clock::now();
#pragma omp parallel num_threads(threads) shared(n,gs,w1f,w2f,g1,g2)
 {int tid=omp_get_thread_num();size_t q=d.n/(size_t)threads,first=(size_t)tid*q,last=(tid==threads-1)?d.n:first+q;for(int u=0;u<updates;u++){
#pragma omp single
 {for(int i=0;i<W1;i++)w1f[i]=(float)n.w1[i];for(int j=0;j<H;j++)w2f[j]=(float)n.w2[j];}
  gs[tid].clear();rangeMix(d,first,last,w1f,w2f,gs[tid]);
#pragma omp barrier
#pragma omp single
 {g1.fill(0);g2.fill(0);for(int t=0;t<threads;t++){for(int i=0;i<W1;i++)g1[i]+=(double)gs[t].g1[i];for(int j=0;j<H;j++)g2[j]+=(double)gs[t].g2[j];}updD(n,g1,g2,lr);}
 }}out=n;return chrono::duration<double>(chrono::steady_clock::now()-st).count();}
static double runF(const Data&d,int updates,int threads,NetF&out){NetF n;init(n);vector<GradF>gs(threads);array<float,W1>g1{};array<float,H>g2{};float lr=(float)(.0001/d.n);auto st=chrono::steady_clock::now();
#pragma omp parallel num_threads(threads) shared(n,gs,g1,g2)
 {int tid=omp_get_thread_num();size_t q=d.n/(size_t)threads,first=(size_t)tid*q,last=(tid==threads-1)?d.n:first+q;for(int u=0;u<updates;u++){gs[tid].clear();rangeMix(d,first,last,n.w1,n.w2,gs[tid]);
#pragma omp barrier
#pragma omp single
 {g1.fill(0);g2.fill(0);for(int t=0;t<threads;t++){for(int i=0;i<W1;i++)g1[i]+=gs[t].g1[i];for(int j=0;j<H;j++)g2[j]+=gs[t].g2[j];}updF(n,g1,g2,lr);}
 }}out=n;return chrono::duration<double>(chrono::steady_clock::now()-st).count();}
static void metrics(const Data&d,const NetD&n,double&rmse,double&mx){long double ss=0;mx=0;for(size_t r=0;r<d.n;r++){double z=0;for(int j=0;j<H;j++){double h=0;for(int k=0;k<IN;k++)h+=d.xd[r*IN+k]*n.w1[k*H+j];z+=sigd(h)*n.w2[j];}double er=sigd(z)-d.yd[r];ss+=er*er;mx=max(mx,abs(er));}rmse=sqrt((double)(ss/d.n));}
static NetD tod(const NetF&f){NetD d;for(int i=0;i<W1;i++){d.w1[i]=f.w1[i];d.v1[i]=f.v1[i];}for(int j=0;j<H;j++){d.w2[j]=f.w2[j];d.v2[j]=f.v2[j];}return d;}
static double wdiff(const NetD&a,const NetD&b){double m=0;for(int i=0;i<W1;i++)m=max(m,abs(a.w1[i]-b.w1[i]));for(int j=0;j<H;j++)m=max(m,abs(a.w2[j]-b.w2[j]));return m;}
int main(int argc,char**argv){size_t rows=argc>1?strtoull(argv[1],0,10):100000;int updates=argc>2?atoi(argv[2]):100,threads=argc>3?atoi(argv[3]):1,reps=argc>4?atoi(argv[4]):3;Data d(rows);fill(d);cout<<setprecision(17);NetD ld,lm;NetF lf;for(int r=0;r<reps;r++){NetD nd,nm;NetF nf;double td,tm,tf;if(r%2==0){td=runD(d,updates,threads,nd);tm=runMix(d,updates,threads,nm);tf=runF(d,updates,threads,nf);}else{tf=runF(d,updates,threads,nf);tm=runMix(d,updates,threads,nm);td=runD(d,updates,threads,nd);}cout<<rows<<','<<updates<<','<<threads<<','<<r<<','<<td<<','<<tm<<','<<tf<<','<<wdiff(nd,nm)<<','<<wdiff(nd,tod(nf))<<'\n';ld=nd;lm=nm;lf=nf;}double rd,md,rm,mm,rf,mf;metrics(d,ld,rd,md);metrics(d,lm,rm,mm);NetD fd=tod(lf);metrics(d,fd,rf,mf);cerr<<"METRICS,"<<setprecision(17)<<rd<<','<<rm<<','<<rf<<','<<md<<','<<mm<<','<<mf<<','<<wdiff(ld,lm)<<','<<wdiff(ld,fd)<<'\n';}
