#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static const size_t INPUTS=11,HIDDEN=16,W1=INPUTS*HIDDEN,PARAMS=W1+HIDDEN;
static const size_t HIST_SHALLOW=10,HIST_DEEP=160,MAX_LS=20;
static const double ARMIJO=1e-4,CURV_EPS=1e-8,CLIP=28.0,FLOOR_RATIO=1e-18;
extern "C" __m256d _ZGVdN4v_exp(__m256d);
struct Data{size_t n;std::vector<double>x,y;};
struct Eval{double obj=0,pe=0;std::array<double,PARAMS>g{};};
struct Pair{std::array<double,PARAMS>s{},y{};double rho=0;};
struct Partial{double sse=0,pes=0;std::array<double,PARAMS>g{};std::array<double,PARAMS>gn{};};
static inline double sig(double z){return 1.0/(1.0+std::exp(-z));}
static inline __m256d sigv(__m256d z){const __m256d one=_mm256_set1_pd(1.0),zero=_mm256_setzero_pd();return _mm256_div_pd(one,_mm256_add_pd(one,_ZGVdN4v_exp(_mm256_sub_pd(zero,z))));}
static inline double hsum(__m256d v){__m128d lo=_mm256_castpd256_pd128(v),hi=_mm256_extractf128_pd(v,1);__m128d s=_mm_add_pd(lo,hi);s=_mm_hadd_pd(s,s);return _mm_cvtsd_f64(s);}
static inline double dot(const std::array<double,PARAMS>&a,const std::array<double,PARAMS>&b){double s=0;for(size_t i=0;i<PARAMS;i++)s+=a[i]*b[i];return s;}
static inline double norm(const std::array<double,PARAMS>&a){return std::sqrt(dot(a,a));}
static Data makeData(size_t n){Data d{n,std::vector<double>(n*INPUTS),std::vector<double>(n)};std::array<double,W1>gw1{};std::array<double,HIDDEN>gw2{};for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)gw1[k*HIDDEN+j]=.22*std::sin(.17*(k+1.0)*(j+2.0));for(size_t j=0;j<HIDDEN;j++)gw2[j]=.28*std::cos(.31*(j+1.0));for(size_t r=0;r<n;r++){double*x=&d.x[r*INPUTS];for(size_t k=0;k<INPUTS;k++)x[k]=.55*std::sin(.013*(r+1.0)*(k+1.0))+.35*std::cos(.007*(r+3.0)*(k+2.0));double out=0;for(size_t j=0;j<HIDDEN;j++){double z=0;for(size_t k=0;k<INPUTS;k++)z+=x[k]*gw1[k*HIDDEN+j];out+=sig(z)*gw2[j];}d.y[r]=sig(out);}return d;}
static std::array<double,PARAMS> initP(){std::array<double,PARAMS>p{};for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)p[k*HIDDEN+j]=.12*std::sin(.43*(k+1.0)*(j+1.0));for(size_t j=0;j<HIDDEN;j++)p[W1+j]=.15*std::cos(.37*(j+1.0));return p;}
static void evalSlice(const Data&d,size_t first,size_t last,const std::array<double,PARAMS>&p,Partial&o,bool wantGn){const __m256d one=_mm256_set1_pd(1.0),hundred=_mm256_set1_pd(100.0),signMask=_mm256_set1_pd(-0.0);__m256d gw1[W1],gw2[HIDDEN],gn1[W1],gn2[HIDDEN];for(size_t i=0;i<W1;i++){gw1[i]=_mm256_setzero_pd();if(wantGn)gn1[i]=_mm256_setzero_pd();}for(size_t j=0;j<HIDDEN;j++){gw2[j]=_mm256_setzero_pd();if(wantGn)gn2[j]=_mm256_setzero_pd();}__m256d sse=_mm256_setzero_pd(),pes=_mm256_setzero_pd();size_t r=first;for(;r+4<=last;r+=4){const double*x0=&d.x[(r+0)*INPUTS],*x1=&d.x[(r+1)*INPUTS],*x2=&d.x[(r+2)*INPUTS],*x3=&d.x[(r+3)*INPUTS];__m256d xv[INPUTS];for(size_t k=0;k<INPUTS;k++)xv[k]=_mm256_set_pd(x3[k],x2[k],x1[k],x0[k]);__m256d h[HIDDEN];for(size_t j=0;j<HIDDEN;j++){__m256d z=_mm256_setzero_pd();for(size_t k=0;k<INPUTS;k++)z=_mm256_fmadd_pd(xv[k],_mm256_set1_pd(p[k*HIDDEN+j]),z);h[j]=sigv(z);}__m256d zo=_mm256_setzero_pd();for(size_t j=0;j<HIDDEN;j++)zo=_mm256_fmadd_pd(h[j],_mm256_set1_pd(p[W1+j]),zo);__m256d pred=sigv(zo),actual=_mm256_loadu_pd(&d.y[r]),err=_mm256_sub_pd(pred,actual);__m256d od=_mm256_mul_pd(pred,_mm256_sub_pd(one,pred)),dout=_mm256_mul_pd(err,od);sse=_mm256_fmadd_pd(err,err,sse);__m256d abserr=_mm256_andnot_pd(signMask,err);pes=_mm256_add_pd(pes,_mm256_mul_pd(_mm256_div_pd(abserr,actual),hundred));for(size_t j=0;j<HIDDEN;j++){gw2[j]=_mm256_fmadd_pd(dout,h[j],gw2[j]);__m256d hp=_mm256_mul_pd(h[j],_mm256_sub_pd(one,h[j]));__m256d base=_mm256_mul_pd(_mm256_mul_pd(dout,_mm256_set1_pd(p[W1+j])),hp);for(size_t k=0;k<INPUTS;k++)gw1[k*HIDDEN+j]=_mm256_fmadd_pd(base,xv[k],gw1[k*HIDDEN+j]);if(wantGn){__m256d jo=_mm256_mul_pd(od,h[j]);gn2[j]=_mm256_fmadd_pd(jo,jo,gn2[j]);__m256d gb=_mm256_mul_pd(_mm256_mul_pd(od,_mm256_set1_pd(p[W1+j])),hp);for(size_t k=0;k<INPUTS;k++){__m256d q=_mm256_mul_pd(gb,xv[k]);gn1[k*HIDDEN+j]=_mm256_fmadd_pd(q,q,gn1[k*HIDDEN+j]);}}}}
o.sse+=hsum(sse);o.pes+=hsum(pes);for(size_t i=0;i<W1;i++){o.g[i]+=hsum(gw1[i]);if(wantGn)o.gn[i]+=hsum(gn1[i]);}for(size_t j=0;j<HIDDEN;j++){o.g[W1+j]+=hsum(gw2[j]);if(wantGn)o.gn[W1+j]+=hsum(gn2[j]);}for(;r<last;r++){const double*x=&d.x[r*INPUTS];double h[HIDDEN];for(size_t j=0;j<HIDDEN;j++){double z=0;for(size_t k=0;k<INPUTS;k++)z+=x[k]*p[k*HIDDEN+j];h[j]=sig(z);}double zo=0;for(size_t j=0;j<HIDDEN;j++)zo+=h[j]*p[W1+j];double pred=sig(zo),err=pred-d.y[r],od=pred*(1-pred),dout=err*od;o.sse+=err*err;o.pes+=std::abs(err/d.y[r])*100;for(size_t j=0;j<HIDDEN;j++){o.g[W1+j]+=dout*h[j];double hp=h[j]*(1-h[j]),base=dout*p[W1+j]*hp;for(size_t k=0;k<INPUTS;k++)o.g[k*HIDDEN+j]+=base*x[k];if(wantGn){double jo=od*h[j];o.gn[W1+j]+=jo*jo;double gb=od*p[W1+j]*hp;for(size_t k=0;k<INPUTS;k++){double q=gb*x[k];o.gn[k*HIDDEN+j]+=q*q;}}}}}
static std::vector<Partial>runSlices(const Data&d,const std::array<double,PARAMS>&p,int threads,bool gn){std::vector<Partial>parts(threads);
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
{int tid=omp_get_thread_num();size_t base=d.n/threads,rem=d.n%threads,off=(size_t)tid*base+std::min((size_t)tid,rem),cnt=base+((size_t)tid<rem?1:0);evalSlice(d,off,off+cnt,p,parts[tid],gn);}
#else
evalSlice(d,0,d.n,p,parts[0],gn);
#endif
return parts;}
static Eval evaluate(const Data&d,const std::array<double,PARAMS>&p,int threads){auto parts=runSlices(d,p,threads,false);long double sse=0,pes=0;Eval e;for(int t=0;t<threads;t++){sse+=parts[t].sse;pes+=parts[t].pes;for(size_t i=0;i<PARAMS;i++)e.g[i]+=parts[t].g[i];}double inv=1.0/d.n;e.obj=.5*(double)sse*inv;e.pe=(double)pes*inv;for(double&v:e.g)v*=inv;return e;}
static std::array<double,PARAMS>gnScale(const Data&d,const std::array<double,PARAMS>&p,int threads){auto parts=runSlices(d,p,threads,true);std::array<double,PARAMS>diag{},scale{};std::vector<double>ord(PARAMS);for(size_t i=0;i<PARAMS;i++){long double s=0;for(int t=0;t<threads;t++)s+=parts[t].gn[i];diag[i]=(double)(s/d.n);ord[i]=diag[i];}std::nth_element(ord.begin(),ord.begin()+PARAMS/2,ord.end());double med=ord[PARAMS/2];if(!(med>0)||!std::isfinite(med)){scale.fill(1);return scale;}double fl=med*FLOOR_RATIO,lo=1.0/CLIP;for(size_t i=0;i<PARAMS;i++){double sd=std::max(diag[i],fl),raw=std::sqrt(med/sd);scale[i]=std::min(CLIP,std::max(lo,raw));}return scale;}
static std::array<double,PARAMS>direction(const std::array<double,PARAMS>&g,const std::deque<Pair>&hist,const std::array<double,PARAMS>*scale){std::array<double,PARAMS>q=g,r{};std::vector<double>a(hist.size());for(size_t ii=hist.size();ii>0;--ii){size_t i=ii-1;a[i]=hist[i].rho*dot(hist[i].s,q);for(size_t k=0;k<PARAMS;k++)q[k]-=a[i]*hist[i].y[k];}double gamma=1;if(!hist.empty()){double yy=dot(hist.back().y,hist.back().y);if(yy>0)gamma=dot(hist.back().s,hist.back().y)/yy;}for(size_t k=0;k<PARAMS;k++)r[k]=q[k]*gamma*(scale?(*scale)[k]:1.0);for(size_t i=0;i<hist.size();i++){double b=hist[i].rho*dot(hist[i].y,r);for(size_t k=0;k<PARAMS;k++)r[k]+=hist[i].s[k]*(a[i]-b);}for(double&v:r)v=-v;return r;}
static bool acceptCurv(const std::array<double,PARAMS>&s,const std::array<double,PARAMS>&y,double sy){return sy>0&&sy>CURV_EPS*norm(s)*norm(y);}
static double gammaFromHist(const std::deque<Pair>&h){if(h.empty())return 1.0;double yy=dot(h.back().y,h.back().y),sy=dot(h.back().s,h.back().y);return (yy>0&&sy>0&&std::isfinite(yy)&&std::isfinite(sy))?sy/yy:1.0;}
static double gammaScaled(const std::deque<Pair>&h,const std::array<double,PARAMS>&scale,int mode){
    if(h.empty())return 1.0;
    const auto&s=h.back().s;
    const auto&y=h.back().y;
    if(mode==0)return gammaFromHist(h);
    double sy=dot(s,y),yDy=0,qyS=0,qy2=0,sDinvS=0;
    for(size_t i=0;i<PARAMS;i++){double q=scale[i],qy=q*y[i];yDy+=y[i]*qy;qyS+=qy*s[i];qy2+=qy*qy;if(q>0)sDinvS+=s[i]*s[i]/q;}
    double g=1.0;
    if(mode==1&&yDy>0)g=sy/yDy;
    else if(mode==2&&qy2>0)g=qyS/qy2;
    else if(mode==3&&yDy>0&&sDinvS>0)g=std::sqrt(sDinvS/yDy);
    if(!(g>0)||!std::isfinite(g))g=gammaFromHist(h);
    return g;
}
static int GAMMA_MODE=0;
static void initDense(const std::deque<Pair>&h,const std::array<double,PARAMS>&scale,std::vector<double>&H){H.assign(PARAMS*PARAMS,0.0);double gamma=gammaScaled(h,scale,GAMMA_MODE);for(size_t i=0;i<PARAMS;i++)H[i*PARAMS+i]=gamma*scale[i];}
static std::array<double,PARAMS> denseDir(const std::array<double,PARAMS>&g,const std::vector<double>&H){std::array<double,PARAMS>d{};for(size_t i=0;i<PARAMS;i++){double v=0.0;for(size_t j=0;j<PARAMS;j++)v+=H[i*PARAMS+j]*g[j];d[i]=-v;}return d;}
static bool updateDense(const Pair&p,std::vector<double>&H){double sy=dot(p.s,p.y);if(!acceptCurv(p.s,p.y,sy))return false;std::array<double,PARAMS>hy{};for(size_t i=0;i<PARAMS;i++){double v=0.0;for(size_t j=0;j<PARAMS;j++)v+=H[i*PARAMS+j]*p.y[j];hy[i]=v;}double yhy=dot(p.y,hy);if(!std::isfinite(yhy))return false;double a=(1.0+yhy/sy)/sy,b=1.0/sy;for(size_t i=0;i<PARAMS;i++)for(size_t j=0;j<PARAMS;j++)H[i*PARAMS+j]+=a*p.s[i]*p.s[j]-b*(hy[i]*p.s[j]+p.s[i]*hy[j]);return true;}
int main(int argc,char**argv){
    if(argc<6){std::cerr<<"usage rows target switch gamma_mode max_updates [threads]\n";return 2;}
    size_t rows=std::stoull(argv[1]);double target=std::stod(argv[2]),sw=std::stod(argv[3]);GAMMA_MODE=std::stoi(argv[4]);size_t maxU=std::stoull(argv[5]);int threads=argc>6?std::stoi(argv[6]):5;
    Data data=makeData(rows);auto p=initP();Eval cur=evaluate(data,p,threads);size_t evals=1,gnp=0,u=0;std::deque<Pair>hist;bool deep=false,have=false,dense=false;std::array<double,PARAMS>scale{};std::vector<double>H;bool r0225=false,r01=false,r002=false;
    while(cur.pe>target&&u<maxU){
        if(cur.pe<=.05)deep=true;
        if(deep&&!have){scale=gnScale(data,p,threads);have=true;gnp++;}
        if(have&&!dense){
            if(!r0225&&cur.pe<=.0225){scale=gnScale(data,p,threads);gnp++;r0225=true;}
            if(!r01&&target<=.001&&cur.pe<=.01){scale=gnScale(data,p,threads);gnp++;r01=true;}
            if(!r002&&target<=.0005&&cur.pe<=.002){scale=gnScale(data,p,threads);gnp++;r002=true;}
            if(cur.pe<=sw){initDense(hist,scale,H);dense=true;}
        }
        size_t lim=deep?HIST_DEEP:HIST_SHALLOW;std::array<double,PARAMS>d=dense?denseDir(cur.g,H):direction(cur.g,hist,have?&scale:nullptr);double dg=dot(cur.g,d);
        if(!(dg<0)&&!dense){for(size_t i=0;i<PARAMS;i++)d[i]=-cur.g[i];dg=-dot(cur.g,cur.g);hist.clear();}
        if(!(dg<0)&&dense){initDense(hist,scale,H);d=denseDir(cur.g,H);dg=dot(cur.g,d);if(!(dg<0)){for(size_t i=0;i<PARAMS;i++)d[i]=-cur.g[i];dg=-dot(cur.g,cur.g);}}
        bool ok=false;double step=1;std::array<double,PARAMS>cand{};Eval nxt;for(size_t ls=0;ls<MAX_LS;ls++){for(size_t i=0;i<PARAMS;i++)cand[i]=p[i]+step*d[i];nxt=evaluate(data,cand,threads);evals++;if(nxt.obj<=cur.obj+ARMIJO*step*dg){ok=true;break;}step*=.5;}if(!ok)break;
        Pair pair;for(size_t i=0;i<PARAMS;i++){pair.s[i]=cand[i]-p[i];pair.y[i]=nxt.g[i]-cur.g[i];}double sy=dot(pair.s,pair.y);
        if(acceptCurv(pair.s,pair.y,sy)){if(dense)updateDense(pair,H);else{if(hist.size()>=lim)hist.pop_front();pair.rho=1/sy;hist.push_back(pair);}}
        p=cand;cur=nxt;u++;
    }
    std::cout<<std::setprecision(15)<<"rows="<<rows<<" target="<<target<<" switch="<<sw<<" gamma="<<GAMMA_MODE<<" updates="<<u<<" evals="<<evals<<" gn="<<gnp<<" total="<<(evals+gnp)<<" pe="<<cur.pe<<" obj="<<cur.obj<<"\n";
}
