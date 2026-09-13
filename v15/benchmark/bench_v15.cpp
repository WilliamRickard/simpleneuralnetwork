#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <omp.h>
using namespace std;
constexpr int IN=11,H=16,P=192;
struct Data{size_t n;vector<double>x,y;explicit Data(size_t n):n(n),x(n*IN),y(n){}};
static inline double sig(double x){return 1.0/(1.0+exp(-x));}
static inline float sigf_(float x){return 1.0f/(1.0f+expf(-x));}
static vector<double> init(){vector<double>p(P);for(int k=0;k<IN;k++)for(int j=0;j<H;j++)p[k*H+j]=.12*sin(.43*(k+1)*(j+1));for(int j=0;j<H;j++)p[IN*H+j]=.15*cos(.37*(j+1));return p;}
static void fill(Data&d){array<double,IN*H>w1{};array<double,H>w2{};for(int k=0;k<IN;k++)for(int j=0;j<H;j++)w1[k*H+j]=.22*sin(.17*(k+1)*(j+2));for(int j=0;j<H;j++)w2[j]=.28*cos(.31*(j+1));for(size_t r=0;r<d.n;r++){double* x=&d.x[r*IN];for(int k=0;k<IN;k++)x[k]=.55*sin(.013*(r+1)*(k+1))+.35*cos(.007*(r+3)*(k+2));double z=0;for(int j=0;j<H;j++){double q=0;for(int k=0;k<IN;k++)q+=x[k]*w1[k*H+j];z+=sig(q)*w2[j];}d.y[r]=sig(z);}}
struct E{double f=0,pe=0;vector<double>g;};
static E eval_exact(const Data&d,const vector<double>&p,int threads){
    E e;e.g.assign(P,0);vector<array<double,P>> gs(threads);for(auto&a:gs)a.fill(0);vector<long double> ss(threads,0),ps(threads,0);const double*w2=&p[IN*H];
#pragma omp parallel num_threads(threads)
    {
        int tid=omp_get_thread_num();array<double,H>h{};auto &g=gs[tid];
#pragma omp for schedule(static)
        for(size_t r=0;r<d.n;r++){
            const double*x=&d.x[r*IN];
            for(int j=0;j<H;j++){double z=0;for(int k=0;k<IN;k++)z+=x[k]*p[k*H+j];h[j]=sig(z);}
            double z=0;for(int j=0;j<H;j++)z+=h[j]*w2[j];double o=sig(z),er=o-d.y[r],d3=er*o*(1-o);
            ss[tid]+=er*er;ps[tid]+=abs(er/d.y[r])*100.0;
            for(int j=0;j<H;j++){g[IN*H+j]+=h[j]*d3;double d2=d3*w2[j]*h[j]*(1-h[j]);for(int k=0;k<IN;k++)g[k*H+j]+=x[k]*d2;}
        }
    }
    long double ssum=0,psum=0;for(int t=0;t<threads;t++){ssum+=ss[t];psum+=ps[t];for(int i=0;i<P;i++)e.g[i]+=gs[t][i];}
    double inv=1.0/d.n;e.f=.5*(double)ssum*inv;e.pe=(double)psum*inv;for(double&v:e.g)v*=inv;return e;
}
static E eval_float_grad(const Data&d,const vector<double>&p,int threads){
    E e;e.g.assign(P,0);vector<array<float,P>> gs(threads);for(auto&a:gs)a.fill(0);vector<long double> ss(threads,0),ps(threads,0);array<float,P> pf{};for(int i=0;i<P;i++)pf[i]=(float)p[i];const float*w2=&pf[IN*H];
#pragma omp parallel num_threads(threads)
    {
        int tid=omp_get_thread_num();array<float,H>h{};auto &g=gs[tid];
#pragma omp for schedule(static)
        for(size_t r=0;r<d.n;r++){
            float x[IN];for(int k=0;k<IN;k++)x[k]=(float)d.x[r*IN+k];
            for(int j=0;j<H;j++){float z=0;for(int k=0;k<IN;k++)z=fmaf(x[k],pf[k*H+j],z);h[j]=sigf_(z);}
            float z=0;for(int j=0;j<H;j++)z=fmaf(h[j],w2[j],z);float o=sigf_(z),yy=(float)d.y[r],er=o-yy,d3=er*o*(1-o);double erd=(double)o-d.y[r];
            ss[tid]+=erd*erd;ps[tid]+=abs(erd/d.y[r])*100.0;
            for(int j=0;j<H;j++){g[IN*H+j]+=h[j]*d3;float d2=d3*w2[j]*h[j]*(1-h[j]);for(int k=0;k<IN;k++)g[k*H+j]+=x[k]*d2;}
        }
    }
    long double ssum=0,psum=0;for(int t=0;t<threads;t++){ssum+=ss[t];psum+=ps[t];for(int i=0;i<P;i++)e.g[i]+=(double)gs[t][i];}
    double inv=1.0/d.n;e.f=.5*(double)ssum*inv;e.pe=(double)psum*inv;for(double&v:e.g)v*=inv;return e;
}
static double dot(const vector<double>&a,const vector<double>&b){double s=0;for(size_t i=0;i<a.size();i++)s+=a[i]*b[i];return s;}
struct Pair{vector<double>s,y;double rho;};
struct R{double sec,pe;int exact_evals,proposal_evals;};
static R run(const Data&d,int threads,bool hybrid){
    vector<double>p=init();deque<Pair>hist;auto st=chrono::steady_clock::now();E cur=eval_exact(d,p,threads);int exact=1,prop=0;
    for(int it=0;it<20&&cur.pe>3.9;it++){
        E guide=hybrid?eval_float_grad(d,p,threads):cur;if(hybrid)prop++;
        vector<double>q=guide.g,alpha(hist.size());
        for(size_t ii=hist.size();ii>0;--ii){size_t i=ii-1;alpha[i]=hist[i].rho*dot(hist[i].s,q);for(size_t k=0;k<P;k++)q[k]-=alpha[i]*hist[i].y[k];}
        double gamma=1;if(!hist.empty()){auto&h=hist.back();double yy=dot(h.y,h.y);if(yy>0)gamma=dot(h.s,h.y)/yy;}
        vector<double>r=q;for(double&v:r)v*=gamma;for(size_t i=0;i<hist.size();i++){double b=hist[i].rho*dot(hist[i].y,r);for(size_t k=0;k<P;k++)r[k]+=hist[i].s[k]*(alpha[i]-b);}for(double&v:r)v=-v;
        double gd=dot(cur.g,r);if(gd>=0){r=cur.g;for(double&v:r)v=-v;gd=-dot(cur.g,cur.g);hist.clear();}
        double step=1;vector<double>pn(P);E ne;bool acc=false;
        for(int ls=0;ls<20;ls++){for(int k=0;k<P;k++)pn[k]=p[k]+step*r[k];ne=eval_exact(d,pn,threads);exact++;if(ne.f<=cur.f+1e-4*step*gd){acc=true;break;}step*=.5;}
        if(!acc) break;
        vector<double>s(P),y(P);for(int k=0;k<P;k++){s[k]=pn[k]-p[k];y[k]=ne.g[k]-cur.g[k];}double sy=dot(s,y);if(sy>1e-12){if(hist.size()==10)hist.pop_front();hist.push_back(Pair{s,y,1/sy});}p.swap(pn);cur=ne;
    }
    return {chrono::duration<double>(chrono::steady_clock::now()-st).count(),cur.pe,exact,prop};
}
int main(int argc,char**argv){
    const size_t n=argc>1?strtoull(argv[1],nullptr,10):1000000;
    const int reps=argc>2?atoi(argv[2]):7;
    Data d(n);fill(d);cout<<setprecision(17);
    for(int rep=0;rep<reps;rep++){
        R v14,v15;
        if(rep%2==0){v14=run(d,1,false);v15=run(d,4,false);}else{v15=run(d,4,false);v14=run(d,1,false);}
        cout<<n<<','<<rep<<','<<v14.sec<<','<<v15.sec<<','<<v14.pe<<','<<v15.pe<<','<<v14.exact_evals<<','<<v15.exact_evals<<'\n';
    }
    return 0;
}
