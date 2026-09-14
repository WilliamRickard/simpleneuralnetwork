#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <vector>
constexpr size_t N=192; using Vec=std::array<double,N>;
__attribute__((noinline)) static double dotS(const Vec&a,const Vec&b){double s=0;for(size_t i=0;i<N;i++)s+=a[i]*b[i];return s;}
static void dirRow(const Vec&g,const std::vector<double>&H,Vec&d){for(size_t i=0;i<N;i++){double v=0;for(size_t j=0;j<N;j++)v+=H[i*N+j]*g[j];d[i]=-v;}}
static bool updRow(const Vec&s,const Vec&y,std::vector<double>&H){double sy=dotS(s,y);Vec hy{};for(size_t i=0;i<N;i++){double v=0;for(size_t j=0;j<N;j++)v+=H[i*N+j]*y[j];hy[i]=v;}double yhy=dotS(y,hy),a=(1+yhy/sy)/sy,b=1/sy;for(size_t i=0;i<N;i++)for(size_t j=0;j<N;j++)H[i*N+j]+=a*s[i]*s[j]-b*(hy[i]*s[j]+s[i]*hy[j]);return true;}
static std::vector<double> toCol(const std::vector<double>&R){std::vector<double>C(N*N);for(size_t i=0;i<N;i++)for(size_t j=0;j<N;j++)C[j*N+i]=R[i*N+j];return C;}
static std::vector<double> toRow(const std::vector<double>&C){std::vector<double>R(N*N);for(size_t i=0;i<N;i++)for(size_t j=0;j<N;j++)R[i*N+j]=C[j*N+i];return R;}
__attribute__((target("avx512f,avx512dq"),optimize("fp-contract=off"))) static void matvecColExact(const std::vector<double>&C,const Vec&x,Vec&out,bool neg){
  for(size_t ib=0;ib<N;ib+=8){__m512d acc=_mm512_setzero_pd();for(size_t j=0;j<N;j++){__m512d h=_mm512_loadu_pd(C.data()+j*N+ib);__m512d prod=_mm512_mul_pd(h,_mm512_set1_pd(x[j]));acc=_mm512_add_pd(acc,prod);}if(neg)acc=_mm512_sub_pd(_mm512_setzero_pd(),acc);_mm512_storeu_pd(out.data()+ib,acc);}
}
__attribute__((target("avx512f,avx512dq"),optimize("fp-contract=off"))) static void dirCol(const Vec&g,const std::vector<double>&C,Vec&d){matvecColExact(C,g,d,true);}
__attribute__((target("avx512f,avx512dq"),optimize("fp-contract=off"))) static bool updCol(const Vec&s,const Vec&y,std::vector<double>&C){
  double sy=dotS(s,y);Vec hy{};matvecColExact(C,y,hy,false);double yhy=dotS(y,hy),a=(1+yhy/sy)/sy,b=1/sy;__m512d va=_mm512_set1_pd(a),vb=_mm512_set1_pd(b);
  for(size_t j=0;j<N;j++){double*col=C.data()+j*N;__m512d vsj=_mm512_set1_pd(s[j]),vhyj=_mm512_set1_pd(hy[j]);for(size_t ib=0;ib<N;ib+=8){__m512d si=_mm512_loadu_pd(s.data()+ib),hyi=_mm512_loadu_pd(hy.data()+ib),old=_mm512_loadu_pd(col+ib);__m512d asi=_mm512_mul_pd(va,si);__m512d term1=_mm512_mul_pd(asi,vsj);__m512d inner=_mm512_add_pd(_mm512_mul_pd(hyi,vsj),_mm512_mul_pd(si,vhyj));__m512d term2=_mm512_mul_pd(vb,inner);__m512d delta=_mm512_sub_pd(term1,term2);_mm512_storeu_pd(col+ib,_mm512_add_pd(old,delta));}}
  return true;
}
int main(){std::vector<double>R(N*N);Vec g{},s{},y{},dr{},dc{};for(size_t i=0;i<N;i++){g[i]=.01*std::sin(i+.2);s[i]=.02*std::cos(i+.3);y[i]=s[i]+.001*std::sin(.7*i);for(size_t j=0;j<N;j++)R[i*N+j]=(i==j?1.0:1e-5*std::sin(i+j));}auto C=toCol(R);
 dirRow(g,R,dr);dirCol(g,C,dc);size_t nd=0;double md=0;for(size_t i=0;i<N;i++){if(std::memcmp(&dr[i],&dc[i],8))nd++;md=std::max(md,std::abs(dr[i]-dc[i]));}std::cout<<"dir_neq="<<nd<<" max="<<std::setprecision(17)<<md<<"\n";
 updRow(s,y,R);updCol(s,y,C);auto RC=toRow(C);size_t nh=0;double mh=0;for(size_t i=0;i<R.size();i++){if(std::memcmp(&R[i],&RC[i],8))nh++;mh=std::max(mh,std::abs(R[i]-RC[i]));}std::cout<<"upd_neq="<<nh<<" max="<<mh<<"\n";
 // Repeat 100 alternating directions/updates and compare.
 for(int q=0;q<100;q++){dirRow(g,R,dr);dirCol(g,C,dc);for(size_t i=0;i<N;i++)g[i]+=.000001*dr[i];updRow(s,y,R);updCol(s,y,C);}RC=toRow(C);nd=0;nh=0;md=mh=0;dirRow(g,R,dr);dirCol(g,C,dc);for(size_t i=0;i<N;i++){if(std::memcmp(&dr[i],&dc[i],8))nd++;md=std::max(md,std::abs(dr[i]-dc[i]));}for(size_t i=0;i<R.size();i++){if(std::memcmp(&R[i],&RC[i],8))nh++;mh=std::max(mh,std::abs(R[i]-RC[i]));}std::cout<<"after100 dir_neq="<<nd<<" dirmax="<<md<<" upd_neq="<<nh<<" updmax="<<mh<<"\n";
 // timings fresh copies
 size_t reps=12000;volatile double sink=0;std::vector<double>R0(N*N);for(size_t i=0;i<N;i++)for(size_t j=0;j<N;j++)R0[i*N+j]=(i==j?1.0:1e-5*std::sin(i+j));auto C0=toCol(R0);auto X=R0;auto t=std::chrono::steady_clock::now();for(size_t r=0;r<reps;r++){dirRow(g,X,dr);sink+=dr[r%N];}double ds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();auto Y=C0;t=std::chrono::steady_clock::now();for(size_t r=0;r<reps;r++){dirCol(g,Y,dc);sink+=dc[r%N];}double dv=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();X=R0;t=std::chrono::steady_clock::now();for(size_t r=0;r<reps;r++){updRow(s,y,X);sink+=X[r%(N*N)];}double us=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();Y=C0;t=std::chrono::steady_clock::now();for(size_t r=0;r<reps;r++){updCol(s,y,Y);sink+=Y[r%(N*N)];}double uv=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::cout<<"dir_scalar_us="<<ds*1e6/reps<<" dir_col_us="<<dv*1e6/reps<<" dx="<<ds/dv<<" upd_scalar_us="<<us*1e6/reps<<" upd_col_us="<<uv*1e6/reps<<" ux="<<us/uv<<" combined_x="<<(ds+us)/(dv+uv)<<" sink="<<sink<<"\n";
}
