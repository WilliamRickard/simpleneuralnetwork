#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iostream>
#include <vector>
#include <iomanip>
using namespace std;
constexpr int IN=11,H=16,P=192;
struct Data{size_t n;vector<double>x,y;explicit Data(size_t n):n(n),x(n*IN),y(n){}};
static double sig(double x){return 1.0/(1.0+exp(-x));}
static vector<double> init(){vector<double>p(P);for(int k=0;k<IN;k++)for(int j=0;j<H;j++)p[k*H+j]=.12*sin(.43*(k+1)*(j+1));for(int j=0;j<H;j++)p[IN*H+j]=.15*cos(.37*(j+1));return p;}
static void fill(Data&d){array<double,IN*H>w1{};array<double,H>w2{};for(int k=0;k<IN;k++)for(int j=0;j<H;j++)w1[k*H+j]=.22*sin(.17*(k+1)*(j+2));for(int j=0;j<H;j++)w2[j]=.28*cos(.31*(j+1));for(size_t r=0;r<d.n;r++){double* x=&d.x[r*IN];for(int k=0;k<IN;k++)x[k]=.55*sin(.013*(r+1)*(k+1))+.35*cos(.007*(r+3)*(k+2));double z=0;for(int j=0;j<H;j++){double q=0;for(int k=0;k<IN;k++)q+=x[k]*w1[k*H+j];z+=sig(q)*w2[j];}d.y[r]=sig(z);}}
struct E{double f=0,pe=0;vector<double>g;};
static E eval(const Data&d,const vector<double>&p){E e;e.g.assign(P,0);array<double,H>h{};long double ss=0,ps=0;const double*w2=&p[IN*H];for(size_t r=0;r<d.n;r++){const double*x=&d.x[r*IN];for(int j=0;j<H;j++){double z=0;for(int k=0;k<IN;k++)z+=x[k]*p[k*H+j];h[j]=sig(z);}double z=0;for(int j=0;j<H;j++)z+=h[j]*w2[j];double o=sig(z),er=o-d.y[r],d3=er*o*(1-o);ss+=er*er;ps+=abs(er/d.y[r])*100.0;for(int j=0;j<H;j++){e.g[IN*H+j]+=h[j]*d3;double d2=d3*w2[j]*h[j]*(1-h[j]);for(int k=0;k<IN;k++)e.g[k*H+j]+=x[k]*d2;}}double inv=1.0/d.n;e.f=.5*(double)ss*inv;e.pe=(double)ps*inv;for(double&v:e.g)v*=inv;return e;}
static double dot(const vector<double>&a,const vector<double>&b){double s=0;for(size_t i=0;i<a.size();i++)s+=a[i]*b[i];return s;}
struct Pair{vector<double>s,y;double rho;};
int main(int argc,char**argv){size_t n=argc>1?strtoull(argv[1],0,10):20000;Data d(n);fill(d);vector<double>p=init();deque<Pair>hist;E cur=eval(d,p);int ev=1;cout<<ev<<','<<setprecision(17)<<cur.f<<','<<cur.pe<<'\n';for(int it=0;it<20&&cur.pe>3.9;it++){vector<double>q=cur.g,alpha(hist.size());for(size_t ii=hist.size();ii>0;--ii){size_t i=ii-1;alpha[i]=hist[i].rho*dot(hist[i].s,q);for(size_t k=0;k<P;k++)q[k]-=alpha[i]*hist[i].y[k];}double gamma=1;if(!hist.empty()){auto&h=hist.back();gamma=dot(h.s,h.y)/dot(h.y,h.y);}vector<double>r=q;for(double&v:r)v*=gamma;for(size_t i=0;i<hist.size();i++){double b=hist[i].rho*dot(hist[i].y,r);for(size_t k=0;k<P;k++)r[k]+=hist[i].s[k]*(alpha[i]-b);}for(double&v:r)v=-v;double gd=dot(cur.g,r);if(gd>=0){r=cur.g;for(double&v:r)v=-v;gd=-dot(cur.g,cur.g);hist.clear();}double step=1;vector<double>pn(P);E ne;for(int ls=0;ls<20;ls++){for(int k=0;k<P;k++)pn[k]=p[k]+step*r[k];ne=eval(d,pn);ev++;cout<<ev<<','<<ne.f<<','<<ne.pe<<'\n';if(ne.f<=cur.f+1e-4*step*gd)break;step*=.5;}vector<double>s(P),y(P);for(int k=0;k<P;k++){s[k]=pn[k]-p[k];y[k]=ne.g[k]-cur.g[k];}double sy=dot(s,y);if(sy>1e-12){if(hist.size()==10)hist.pop_front();hist.push_back(Pair{s,y,1/sy});}p.swap(pn);cur=ne;}return cur.pe<=3.9?0:1;}
