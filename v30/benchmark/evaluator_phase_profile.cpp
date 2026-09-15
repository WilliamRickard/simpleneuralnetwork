#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define SIMPLE_NN_V29_NO_MAIN
#include "../../v29/main.cpp"
#undef SIMPLE_NN_V29_NO_MAIN
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <x86intrin.h>

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)

struct V30EvaluatorPhaseCycles {
    uint64_t firstLinear=0;
    uint64_t hiddenSigmoid=0;
    uint64_t outputDot=0;
    uint64_t outputMetrics=0;
    uint64_t backprop=0;
};

static inline uint64_t ticksV30(){
    _mm_lfence();
    const uint64_t value=__rdtsc();
    _mm_lfence();
    return value;
}

#define V30P_DECLARE_W1(G) \
    __m512d g00=_mm512_loadu_pd((G)+0*HIDDEN_NODES), g01=_mm512_loadu_pd((G)+0*HIDDEN_NODES+8); \
    __m512d g10=_mm512_loadu_pd((G)+1*HIDDEN_NODES), g11=_mm512_loadu_pd((G)+1*HIDDEN_NODES+8); \
    __m512d g20=_mm512_loadu_pd((G)+2*HIDDEN_NODES), g21=_mm512_loadu_pd((G)+2*HIDDEN_NODES+8); \
    __m512d g30=_mm512_loadu_pd((G)+3*HIDDEN_NODES), g31=_mm512_loadu_pd((G)+3*HIDDEN_NODES+8); \
    __m512d g40=_mm512_loadu_pd((G)+4*HIDDEN_NODES), g41=_mm512_loadu_pd((G)+4*HIDDEN_NODES+8); \
    __m512d g50=_mm512_loadu_pd((G)+5*HIDDEN_NODES), g51=_mm512_loadu_pd((G)+5*HIDDEN_NODES+8); \
    __m512d g60=_mm512_loadu_pd((G)+6*HIDDEN_NODES), g61=_mm512_loadu_pd((G)+6*HIDDEN_NODES+8); \
    __m512d g70=_mm512_loadu_pd((G)+7*HIDDEN_NODES), g71=_mm512_loadu_pd((G)+7*HIDDEN_NODES+8); \
    __m512d g80=_mm512_loadu_pd((G)+8*HIDDEN_NODES), g81=_mm512_loadu_pd((G)+8*HIDDEN_NODES+8); \
    __m512d g90=_mm512_loadu_pd((G)+9*HIDDEN_NODES), g91=_mm512_loadu_pd((G)+9*HIDDEN_NODES+8); \
    __m512d g100=_mm512_loadu_pd((G)+10*HIDDEN_NODES), g101=_mm512_loadu_pd((G)+10*HIDDEN_NODES+8)

#define V30P_ACCUM_W1(K,G0,G1) do { \
    const __m512d xv=_mm512_set1_pd(xq[(K)]); \
    (G0)=_mm512_fmadd_pd(xv,hiddenDeltaLow,(G0)); \
    (G1)=_mm512_fmadd_pd(xv,hiddenDeltaHigh,(G1)); \
} while(false)

#define V30P_ACCUM_ALL_W1() do { \
    V30P_ACCUM_W1(0,g00,g01); V30P_ACCUM_W1(1,g10,g11); V30P_ACCUM_W1(2,g20,g21); \
    V30P_ACCUM_W1(3,g30,g31); V30P_ACCUM_W1(4,g40,g41); V30P_ACCUM_W1(5,g50,g51); \
    V30P_ACCUM_W1(6,g60,g61); V30P_ACCUM_W1(7,g70,g71); V30P_ACCUM_W1(8,g80,g81); \
    V30P_ACCUM_W1(9,g90,g91); V30P_ACCUM_W1(10,g100,g101); \
} while(false)

#define V30P_STORE_W1(G) do { \
    _mm512_storeu_pd((G)+0*HIDDEN_NODES,g00); _mm512_storeu_pd((G)+0*HIDDEN_NODES+8,g01); \
    _mm512_storeu_pd((G)+1*HIDDEN_NODES,g10); _mm512_storeu_pd((G)+1*HIDDEN_NODES+8,g11); \
    _mm512_storeu_pd((G)+2*HIDDEN_NODES,g20); _mm512_storeu_pd((G)+2*HIDDEN_NODES+8,g21); \
    _mm512_storeu_pd((G)+3*HIDDEN_NODES,g30); _mm512_storeu_pd((G)+3*HIDDEN_NODES+8,g31); \
    _mm512_storeu_pd((G)+4*HIDDEN_NODES,g40); _mm512_storeu_pd((G)+4*HIDDEN_NODES+8,g41); \
    _mm512_storeu_pd((G)+5*HIDDEN_NODES,g50); _mm512_storeu_pd((G)+5*HIDDEN_NODES+8,g51); \
    _mm512_storeu_pd((G)+6*HIDDEN_NODES,g60); _mm512_storeu_pd((G)+6*HIDDEN_NODES+8,g61); \
    _mm512_storeu_pd((G)+7*HIDDEN_NODES,g70); _mm512_storeu_pd((G)+7*HIDDEN_NODES+8,g71); \
    _mm512_storeu_pd((G)+8*HIDDEN_NODES,g80); _mm512_storeu_pd((G)+8*HIDDEN_NODES+8,g81); \
    _mm512_storeu_pd((G)+9*HIDDEN_NODES,g90); _mm512_storeu_pd((G)+9*HIDDEN_NODES+8,g91); \
    _mm512_storeu_pd((G)+10*HIDDEN_NODES,g100); _mm512_storeu_pd((G)+10*HIDDEN_NODES+8,g101); \
} while(false)

/** Duplicate v29 exactly, inserting cycle counters only at phase boundaries. */
__attribute__((target("avx512f,avx512dq,fma")))
static void evaluateSliceProfileV30(const Dataset&data,size_t firstRow,size_t lastRow,
                                    const Network&network,V15ThreadEvaluation&out,
                                    V30EvaluatorPhaseCycles&cycles){
    const __m512d one=_mm512_set1_pd(1.0);
    const __m512d hundred=_mm512_set1_pd(100.0);
    const __m512d zero=_mm512_setzero_pd();
    alignas(64) double hidden[V29_GROUP_ROWS*HIDDEN_NODES];
    alignas(64) double rawOutput[V29_GROUP_ROWS];
    alignas(64) double deltaThree[V29_GROUP_ROWS];

    size_t row=firstRow;
    while(row+V17_FORWARD_ROWS<=lastRow){
        const size_t fullTiles=(lastRow-row)/V17_FORWARD_ROWS;
        const size_t tiles=min(V29_GROUP_TILES,fullTiles);
        const size_t groupRows=tiles*V17_FORWARD_ROWS;

        for(size_t tile=0;tile<tiles;tile++){
            const size_t tileRow=row+tile*V17_FORWARD_ROWS;
            const double*x[V17_FORWARD_ROWS];
            __m512d rawLow[V17_FORWARD_ROWS],rawHigh[V17_FORWARD_ROWS];
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                x[q]=data.x.rowData(tileRow+q);
                rawLow[q]=zero;
                rawHigh[q]=zero;
            }
            uint64_t started=ticksV30();
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
                const __m512d weightsLow=_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES);
                const __m512d weightsHigh=_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES+8);
                for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                    const __m512d value=_mm512_set1_pd(x[q][k]);
                    rawLow[q]=_mm512_fmadd_pd(value,weightsLow,rawLow[q]);
                    rawHigh[q]=_mm512_fmadd_pd(value,weightsHigh,rawHigh[q]);
                }
            }
            cycles.firstLinear+=ticksV30()-started;

            double*tileHidden=hidden+tile*V17_FORWARD_ROWS*HIDDEN_NODES;
            started=ticksV30();
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                double*h=tileHidden+q*HIDDEN_NODES;
                _mm512_store_pd(h,sigmoidVectorV17(rawLow[q]));
                _mm512_store_pd(h+8,sigmoidVectorV17(rawHigh[q]));
            }
            cycles.hiddenSigmoid+=ticksV30()-started;
        }

        uint64_t started=ticksV30();
        {
            const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
            const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);
            for(size_t rr=0;rr<groupRows;rr++){
                const double*h=hidden+rr*HIDDEN_NODES;
                rawOutput[rr]=_mm512_reduce_add_pd(_mm512_add_pd(
                    _mm512_mul_pd(_mm512_load_pd(h),wTwoLow),
                    _mm512_mul_pd(_mm512_load_pd(h+8),wTwoHigh)));
            }
        }
        cycles.outputDot+=ticksV30()-started;

        started=ticksV30();
        for(size_t tile=0;tile<tiles;tile++){
            const size_t tileRow=row+tile*V17_FORWARD_ROWS;
            const __m512d prediction=sigmoidVectorV17(
                _mm512_load_pd(rawOutput+tile*V17_FORWARD_ROWS));
            const __m512d actual=_mm512_loadu_pd(data.y.data()+tileRow);
            const __m512d error=_mm512_sub_pd(prediction,actual);
            const __m512d delta=_mm512_mul_pd(
                _mm512_mul_pd(error,prediction),_mm512_sub_pd(one,prediction));
            _mm512_store_pd(deltaThree+tile*V17_FORWARD_ROWS,delta);
            out.squaredError+=_mm512_reduce_add_pd(_mm512_mul_pd(error,error));
            __m512d percentage=_mm512_mul_pd(
                _mm512_div_pd(_mm512_abs_pd(error),actual),hundred);
            const __mmask8 zeroActual=_mm512_cmp_pd_mask(actual,zero,_CMP_EQ_OQ);
            percentage=_mm512_mask_mov_pd(percentage,zeroActual,zero);
            out.percentageErrorSum+=_mm512_reduce_add_pd(percentage);
            alignas(64) double percentageValues[V17_FORWARD_ROWS];
            _mm512_store_pd(percentageValues,percentage);
            for(size_t q=0;q<V17_FORWARD_ROWS;q++)
                out.maxPercentageError=max(out.maxPercentageError,percentageValues[q]);
        }
        cycles.outputMetrics+=ticksV30()-started;

        started=ticksV30();
        double*gradient=out.gradient.data();
        __m512d gradientTwoLow=_mm512_loadu_pd(gradient+WONE_SIZE);
        __m512d gradientTwoHigh=_mm512_loadu_pd(gradient+WONE_SIZE+8);
        V30P_DECLARE_W1(gradient);
        const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
        const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);
        for(size_t rr=0;rr<groupRows;rr++){
            const double*h=hidden+rr*HIDDEN_NODES;
            const __m512d hiddenLow=_mm512_load_pd(h);
            const __m512d hiddenHigh=_mm512_load_pd(h+8);
            const __m512d d=_mm512_set1_pd(deltaThree[rr]);
            gradientTwoLow=_mm512_fmadd_pd(hiddenLow,d,gradientTwoLow);
            gradientTwoHigh=_mm512_fmadd_pd(hiddenHigh,d,gradientTwoHigh);
            const __m512d hiddenDeltaLow=_mm512_mul_pd(
                _mm512_mul_pd(d,wTwoLow),
                _mm512_mul_pd(hiddenLow,_mm512_sub_pd(one,hiddenLow)));
            const __m512d hiddenDeltaHigh=_mm512_mul_pd(
                _mm512_mul_pd(d,wTwoHigh),
                _mm512_mul_pd(hiddenHigh,_mm512_sub_pd(one,hiddenHigh)));
            const double*xq=data.x.rowData(row+rr);
            V30P_ACCUM_ALL_W1();
        }
        _mm512_storeu_pd(gradient+WONE_SIZE,gradientTwoLow);
        _mm512_storeu_pd(gradient+WONE_SIZE+8,gradientTwoHigh);
        V30P_STORE_W1(gradient);
        cycles.backprop+=ticksV30()-started;
        row+=groupRows;
    }
    if(row<lastRow)evaluateSliceV28(data,row,lastRow,network,out);
}

static Dataset phaseDataV30(size_t rows){
    Dataset data;
    data.x=Matrix(rows,NUMBER_OF_VARIABLES);
    data.y.resize(rows);
    for(size_t r=0;r<rows;r++){
        double*x=data.x.rowData(r);
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)
            x[k]=.61*sin(.0073*static_cast<double>((r+1)*(k+1)))
                +.23*cos(.0049*static_cast<double>((r+7)*(k+3)));
        data.y[r]=.35+.30*(.5+.5*sin(.0051*static_cast<double>(r+1)));
    }
    return data;
}

static Network phaseNetworkV30(){
    Network network;
    for(size_t i=0;i<network.wOne.size();i++)
        network.wOne[i]=.22*sin(.071*static_cast<double>(i+1));
    for(size_t i=0;i<network.wTwo.size();i++)
        network.wTwo[i]=.37*cos(.19*static_cast<double>(i+1));
    return network;
}

static bool exactPartialV30(const V15ThreadEvaluation&a,const V15ThreadEvaluation&b){
    return a.squaredError==b.squaredError&&
           a.percentageErrorSum==b.percentageErrorSum&&
           a.maxPercentageError==b.maxPercentageError&&
           memcmp(a.gradient.data(),b.gradient.data(),a.gradient.size()*sizeof(double))==0;
}

static uint64_t medianV30(vector<uint64_t> values){
    sort(values.begin(),values.end());
    return values[values.size()/2];
}

int main(){
    if(!supportsV16Kernel()){
        cerr<<"SKIP: AVX-512/libmvec production kernel unavailable\n";
        return 75;
    }
    cout<<setprecision(6)<<fixed;
    for(size_t rows: {size_t(5000),size_t(20000),size_t(50000),size_t(100000)}){
        Dataset data=phaseDataV30(rows);
        Network network=phaseNetworkV30();
        V15ThreadEvaluation expected,observed;
        V30EvaluatorPhaseCycles exactCycles;
        evaluateSliceV29(data,0,rows,network,expected);
        evaluateSliceProfileV30(data,0,rows,network,observed,exactCycles);
        if(!exactPartialV30(expected,observed)){
            cerr<<"FAIL: profiled evaluator changed exact result at rows="<<rows<<'\n';
            return 3;
        }

        vector<uint64_t> linear,sigmoid,dot,metrics,backprop;
        for(int repeat=0;repeat< nine;repeat++){}
        for(int repeat=0;repeat<9;repeat++){
            V15ThreadEvaluation output;
            V30EvaluatorPhaseCycles measured;
            evaluateSliceProfileV30(data,0,rows,network,output,measured);
            if(!exactPartialV30(expected,output))return 4;
            linear.push_back(measured.firstLinear);
            sigmoid.push_back(measured.hiddenSigmoid);
            dot.push_back(measured.outputDot);
            metrics.push_back(measured.outputMetrics);
            backprop.push_back(measured.backprop);
        }
        const uint64_t mLinear=medianV30(linear),mSigmoid=medianV30(sigmoid),mDot=medianV30(dot);
        const uint64_t mMetrics=medianV30(metrics),mBackprop=medianV30(backprop);
        const double total=static_cast<double>(mLinear+mSigmoid+mDot+mMetrics+mBackprop);
        cout<<"PHASE rows="<<rows
            <<" first_linear_pct="<<(100.0*mLinear/total)
            <<" hidden_sigmoid_pct="<<(100.0*mSigmoid/total)
            <<" output_dot_pct="<<(100.0*mDot/total)
            <<" output_metrics_pct="<<(100.0*mMetrics/total)
            <<" backprop_pct="<<(100.0*mBackprop/total)
            <<" measured_cycles="<<static_cast<uint64_t>(total)<<'\n';
    }
    return 0;
}

#undef V30P_DECLARE_W1
#undef V30P_ACCUM_W1
#undef V30P_ACCUM_ALL_W1
#undef V30P_STORE_W1

#else
int main(){return 75;}
#endif
