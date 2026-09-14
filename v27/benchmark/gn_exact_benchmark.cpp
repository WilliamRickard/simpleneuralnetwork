#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

constexpr size_t INPUTS=11,HIDDEN=16,WONE=176,PARAMS=192,ROWS=10000,BLOCK=8192;
using Param=array<double,PARAMS>;
using Scale=array<double,PARAMS>;
struct Data{vector<double>x;const double*row(size_t r)const{return x.data()+r*INPUTS;}};
struct Partial{array<long double,PARAMS>value{};};

static inline double sigmoid(double z){return 1.0/(1.0+exp(-z));}

static Data makeData(){
    Data d{vector<double>(ROWS*INPUTS)};
    for(size_t r=0;r<ROWS;r++)for(size_t k=0;k<INPUTS;k++)
        d.x[r*INPUTS+k]=.55*sin(.013*(r+1.0)*(k+1.0))+.35*cos(.007*(r+3.0)*(k+2.0));
    return d;
}

static Param makeParameters(){
    Param p{};
    for(size_t k=0;k<INPUTS;k++)for(size_t j=0;j<HIDDEN;j++)
        p[k*HIDDEN+j]=.12*sin(.43*(k+1.0)*(j+1.0));
    for(size_t j=0;j<HIDDEN;j++)p[WONE+j]=.15*cos(.37*(j+1.0));
    return p;
}

static void scalarAccumulate(const Data&data,const Param&p,Partial&out){
    double hidden[HIDDEN];
    for(size_t r=0;r<ROWS;r++){
        const double*x=data.row(r);
        for(size_t j=0;j<HIDDEN;j++){
            double raw=0.0;
            for(size_t k=0;k<INPUTS;k++)raw+=x[k]*p[k*HIDDEN+j];
            hidden[j]=sigmoid(raw);
        }
        double rawOutput=0.0;
        for(size_t j=0;j<HIDDEN;j++)rawOutput+=hidden[j]*p[WONE+j];
        const double prediction=sigmoid(rawOutput),od=prediction*(1.0-prediction);
        for(size_t j=0;j<HIDDEN;j++){
            const double oj=od*hidden[j];
            out.value[WONE+j]+=static_cast<long double>(oj)*oj;
            const double base=od*p[WONE+j]*hidden[j]*(1.0-hidden[j]);
            for(size_t k=0;k<INPUTS;k++){
                const double q=base*x[k];
                out.value[k*HIDDEN+j]+=static_cast<long double>(q)*q;
            }
        }
    }
}

__attribute__((target("avx512f,avx512dq"),optimize("fp-contract=off")))
static void rowExact(const double*x,const Param&p,double*out){
    alignas(64) double raw[HIDDEN],hidden[HIDDEN];
    __m512d low=_mm512_setzero_pd(),high=_mm512_setzero_pd();
    for(size_t k=0;k<INPUTS;k++){
        const __m512d xv=_mm512_set1_pd(x[k]);
        low=_mm512_add_pd(low,_mm512_mul_pd(xv,_mm512_loadu_pd(p.data()+k*HIDDEN)));
        high=_mm512_add_pd(high,_mm512_mul_pd(xv,_mm512_loadu_pd(p.data()+k*HIDDEN+8)));
    }
    _mm512_store_pd(raw,low);_mm512_store_pd(raw+8,high);
    for(size_t j=0;j<HIDDEN;j++)hidden[j]=sigmoid(raw[j]);
    double rawOutput=0.0;
    for(size_t j=0;j<HIDDEN;j++)rawOutput+=hidden[j]*p[WONE+j];
    const double prediction=sigmoid(rawOutput),od=prediction*(1.0-prediction);
    const __m512d one=_mm512_set1_pd(1.0),odv=_mm512_set1_pd(od);
    const __m512d hl=_mm512_load_pd(hidden),hh=_mm512_load_pd(hidden+8);
    const __m512d w2l=_mm512_loadu_pd(p.data()+WONE),w2h=_mm512_loadu_pd(p.data()+WONE+8);
    _mm512_storeu_pd(out+WONE,_mm512_mul_pd(odv,hl));
    _mm512_storeu_pd(out+WONE+8,_mm512_mul_pd(odv,hh));
    const __m512d bl=_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(odv,w2l),hl),_mm512_sub_pd(one,hl));
    const __m512d bh=_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(odv,w2h),hh),_mm512_sub_pd(one,hh));
    for(size_t k=0;k<INPUTS;k++){
        const __m512d xv=_mm512_set1_pd(x[k]);
        _mm512_storeu_pd(out+k*HIDDEN,_mm512_mul_pd(bl,xv));
        _mm512_storeu_pd(out+k*HIDDEN+8,_mm512_mul_pd(bh,xv));
    }
}

static void exactBlockAccumulate(const Data&data,const Param&p,Partial&out){
    vector<double>scratch(BLOCK*PARAMS);
    for(size_t first=0;first<ROWS;first+=BLOCK){
        const size_t count=min(BLOCK,ROWS-first);
#ifdef _OPENMP
#pragma omp parallel for num_threads(4) schedule(static)
#endif
        for(long long rr=0;rr<static_cast<long long>(count);rr++)
            rowExact(data.row(first+static_cast<size_t>(rr)),p,
                     scratch.data()+static_cast<size_t>(rr)*PARAMS);
        for(size_t rr=0;rr<count;rr++){
            const double*q=scratch.data()+rr*PARAMS;
            for(size_t j=0;j<HIDDEN;j++){
                out.value[WONE+j]+=static_cast<long double>(q[WONE+j])*q[WONE+j];
                for(size_t k=0;k<INPUTS;k++){
                    const double value=q[k*HIDDEN+j];
                    out.value[k*HIDDEN+j]+=static_cast<long double>(value)*value;
                }
            }
        }
    }
}

static Scale finish(const Partial&p){
    Scale diagonal{},scale{};
    vector<double>ordered(PARAMS);
    const long double inv=1.0L/static_cast<long double>(ROWS);
    for(size_t i=0;i<PARAMS;i++){diagonal[i]=static_cast<double>(p.value[i]*inv);ordered[i]=diagonal[i];}
    nth_element(ordered.begin(),ordered.begin()+PARAMS/2,ordered.end());
    const double ref=ordered[PARAMS/2],floorValue=ref*1e-18,minimumScale=1.0/28.0;
    for(size_t i=0;i<PARAMS;i++)scale[i]=min(28.0,max(minimumScale,pow(ref/max(diagonal[i],floorValue),0.5)));
    return scale;
}

int main(){
#if defined(__x86_64__) && defined(__GNUC__)
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512dq")){
        cout<<"v27 exact GN benchmark skipped: AVX-512 unavailable\n";
        return 0;
    }
#endif
    const Data data=makeData();
    const Param parameters=makeParameters();
    Partial scalar{},exact{};
    scalarAccumulate(data,parameters,scalar);
    exactBlockAccumulate(data,parameters,exact);
    const Scale a=finish(scalar),b=finish(exact);
    size_t differing=0;
    double maximum=0.0;
    for(size_t i=0;i<PARAMS;i++){
        if(memcmp(&a[i],&b[i],sizeof(double))!=0)differing++;
        maximum=max(maximum,abs(a[i]-b[i]));
    }
    cout<<setprecision(17)<<"scale_differing="<<differing<<" max_abs="<<maximum<<'\n';
    return differing==0?0:1;
}
