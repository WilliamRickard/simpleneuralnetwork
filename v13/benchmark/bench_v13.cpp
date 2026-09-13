#define main v12_benchmark_main
#include "../../v12/benchmark/bench_v12.cpp"
#undef main

#include <cstdint>
#include <cstring>

constexpr int V13_PAIRS=(IN+1)/2;

static inline uint16_t v13bf(float value){uint32_t bits;memcpy(&bits,&value,4);bits+=0x7fffu+((bits>>16)&1u);return (uint16_t)(bits>>16);}

struct V13Data {
    vector<uint32_t> pairs;
    explicit V13Data(const Data &d):pairs(d.n*V13_PAIRS){
        for(size_t row=0;row<d.n;row++)for(int p=0;p<V13_PAIRS;p++){
            int k=2*p;uint16_t lo=v13bf(d.xf[row*IN+k]),hi=k+1<IN?v13bf(d.xf[row*IN+k+1]):0;
            pairs[row*V13_PAIRS+p]=(uint32_t)lo|((uint32_t)hi<<16);
        }
    }
};

struct alignas(64) V13Weights { array<uint32_t,V13_PAIRS*H> w1{}; array<float,H>w2{}; };

static void packV13(const NetD &n,V13Weights &w){
    for(int p=0;p<V13_PAIRS;p++)for(int j=0;j<H;j++){
        int k=2*p;uint16_t lo=v13bf((float)n.w1[k*H+j]),hi=k+1<IN?v13bf((float)n.w1[(k+1)*H+j]):0;
        w.w1[p*H+j]=(uint32_t)lo|((uint32_t)hi<<16);
    }
    for(int j=0;j<H;j++)w.w2[j]=(float)n.w2[j];
}

template<int ROWS>
__attribute__((target("avx512f,avx512dq,avx512bf16,fma"))) static void rangeV13(const Data&d,const V13Data&bd,size_t b,size_t e,const V13Weights&w,GradF&g){
    alignas(64) float hidden[TILE*H],out[TILE],d3[TILE];const __m512 one=_mm512_set1_ps(1.0f),w2v=_mm512_load_ps(w.w2.data());
    for(size_t base=b;base+TILE<=e;base+=TILE){
        for(int rr=0;rr<TILE;rr+=ROWS){__m512 z[ROWS];for(int q=0;q<ROWS;q++)z[q]=_mm512_setzero_ps();for(int p=0;p<V13_PAIRS;p++){__m512bh wv=(__m512bh)_mm512_loadu_si512((const void*)(w.w1.data()+p*H));for(int q=0;q<ROWS;q++){__m512bh xv=(__m512bh)_mm512_set1_epi32((int)bd.pairs[(base+rr+q)*V13_PAIRS+p]);z[q]=_mm512_dpbf16_ps(z[q],xv,wv);}}for(int q=0;q<ROWS;q++)_mm512_store_ps(hidden+(rr+q)*H,sigps(z[q]));}
        for(int r=0;r<TILE;r++){__m512 a=_mm512_load_ps(hidden+r*H);out[r]=sigf_scalar(_mm512_reduce_add_ps(_mm512_mul_ps(a,w2v)));float p=out[r];d3[r]=(p-d.yf[base+r])*p*(1.0f-p);}
        __m512 gg2=_mm512_setzero_ps(),ga[IN];for(int k=0;k<IN;k++)ga[k]=_mm512_setzero_ps();
        for(int r=0;r<TILE;r++){const float*x=d.xf.data()+(base+r)*IN;__m512 a=_mm512_load_ps(hidden+r*H),dv=_mm512_set1_ps(d3[r]);gg2=_mm512_fmadd_ps(a,dv,gg2);__m512 de=_mm512_mul_ps(_mm512_mul_ps(dv,w2v),_mm512_mul_ps(a,_mm512_sub_ps(one,a)));for(int k=0;k<IN;k++)ga[k]=_mm512_fmadd_ps(_mm512_set1_ps(x[k]),de,ga[k]);}
        _mm512_storeu_ps(g.g2.data(),_mm512_add_ps(_mm512_loadu_ps(g.g2.data()),gg2));for(int k=0;k<IN;k++)_mm512_storeu_ps(g.g1.data()+k*H,_mm512_add_ps(_mm512_loadu_ps(g.g1.data()+k*H),ga[k]));
    }
}

static double runV13(const Data&d,const V13Data&bd,int updates,NetD&out){
    NetD n;init(n);GradF g;V13Weights w;array<double,W1>g1{};array<double,H>g2{};double lr=.0001/d.n;auto st=chrono::steady_clock::now();
    for(int u=0;u<updates;u++){packV13(n,w);g.clear();rangeV13<16>(d,bd,0,d.n,w,g);for(int i=0;i<W1;i++)g1[i]=(double)g.g1[i];for(int j=0;j<H;j++)g2[j]=(double)g.g2[j];updD(n,g1,g2,lr);}
    out=n;return chrono::duration<double>(chrono::steady_clock::now()-st).count();
}

int main(int argc,char**argv){size_t rows=argc>1?strtoull(argv[1],0,10):100000;int updates=argc>2?atoi(argv[2]):100,reps=argc>3?atoi(argv[3]):7;Data d(rows);fill(d);V13Data bd(d);cout<<setprecision(17);for(int r=0;r<reps;r++){NetD a,b;double ta,tb;if(r%2==0){ta=runMix(d,updates,1,a);tb=runV13(d,bd,updates,b);}else{tb=runV13(d,bd,updates,b);ta=runMix(d,updates,1,a);}double ra,ma,rb,mb;metrics(d,a,ra,ma);metrics(d,b,rb,mb);cout<<rows<<','<<updates<<",1,"<<r<<','<<ta<<','<<tb<<','<<wdiff(a,b)<<','<<ra<<','<<rb<<'\n';}}
