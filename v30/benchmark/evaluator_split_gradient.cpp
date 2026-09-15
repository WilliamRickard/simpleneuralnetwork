#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define SIMPLE_NN_V30_NO_MAIN
#include "../main.cpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <chrono>
#include <cstring>
#include <iomanip>
#include <thread>

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)

#define V30S_DECLARE_FIRST(G) \
    __m512d g00=_mm512_loadu_pd((G)+0*HIDDEN_NODES), g01=_mm512_loadu_pd((G)+0*HIDDEN_NODES+8); \
    __m512d g10=_mm512_loadu_pd((G)+1*HIDDEN_NODES), g11=_mm512_loadu_pd((G)+1*HIDDEN_NODES+8); \
    __m512d g20=_mm512_loadu_pd((G)+2*HIDDEN_NODES), g21=_mm512_loadu_pd((G)+2*HIDDEN_NODES+8); \
    __m512d g30=_mm512_loadu_pd((G)+3*HIDDEN_NODES), g31=_mm512_loadu_pd((G)+3*HIDDEN_NODES+8); \
    __m512d g40=_mm512_loadu_pd((G)+4*HIDDEN_NODES), g41=_mm512_loadu_pd((G)+4*HIDDEN_NODES+8)

#define V30S_STORE_FIRST(G) do { \
    _mm512_storeu_pd((G)+0*HIDDEN_NODES,g00); _mm512_storeu_pd((G)+0*HIDDEN_NODES+8,g01); \
    _mm512_storeu_pd((G)+1*HIDDEN_NODES,g10); _mm512_storeu_pd((G)+1*HIDDEN_NODES+8,g11); \
    _mm512_storeu_pd((G)+2*HIDDEN_NODES,g20); _mm512_storeu_pd((G)+2*HIDDEN_NODES+8,g21); \
    _mm512_storeu_pd((G)+3*HIDDEN_NODES,g30); _mm512_storeu_pd((G)+3*HIDDEN_NODES+8,g31); \
    _mm512_storeu_pd((G)+4*HIDDEN_NODES,g40); _mm512_storeu_pd((G)+4*HIDDEN_NODES+8,g41); \
} while(false)

#define V30S_DECLARE_SECOND(G) \
    __m512d g50=_mm512_loadu_pd((G)+5*HIDDEN_NODES), g51=_mm512_loadu_pd((G)+5*HIDDEN_NODES+8); \
    __m512d g60=_mm512_loadu_pd((G)+6*HIDDEN_NODES), g61=_mm512_loadu_pd((G)+6*HIDDEN_NODES+8); \
    __m512d g70=_mm512_loadu_pd((G)+7*HIDDEN_NODES), g71=_mm512_loadu_pd((G)+7*HIDDEN_NODES+8); \
    __m512d g80=_mm512_loadu_pd((G)+8*HIDDEN_NODES), g81=_mm512_loadu_pd((G)+8*HIDDEN_NODES+8); \
    __m512d g90=_mm512_loadu_pd((G)+9*HIDDEN_NODES), g91=_mm512_loadu_pd((G)+9*HIDDEN_NODES+8); \
    __m512d g100=_mm512_loadu_pd((G)+10*HIDDEN_NODES), g101=_mm512_loadu_pd((G)+10*HIDDEN_NODES+8)

#define V30S_STORE_SECOND(G) do { \
    _mm512_storeu_pd((G)+5*HIDDEN_NODES,g50); _mm512_storeu_pd((G)+5*HIDDEN_NODES+8,g51); \
    _mm512_storeu_pd((G)+6*HIDDEN_NODES,g60); _mm512_storeu_pd((G)+6*HIDDEN_NODES+8,g61); \
    _mm512_storeu_pd((G)+7*HIDDEN_NODES,g70); _mm512_storeu_pd((G)+7*HIDDEN_NODES+8,g71); \
    _mm512_storeu_pd((G)+8*HIDDEN_NODES,g80); _mm512_storeu_pd((G)+8*HIDDEN_NODES+8,g81); \
    _mm512_storeu_pd((G)+9*HIDDEN_NODES,g90); _mm512_storeu_pd((G)+9*HIDDEN_NODES+8,g91); \
    _mm512_storeu_pd((G)+10*HIDDEN_NODES,g100); _mm512_storeu_pd((G)+10*HIDDEN_NODES+8,g101); \
} while(false)

#define V30S_ACCUM(K,G0,G1) do { \
    const __m512d xv=_mm512_set1_pd(xq[(K)]); \
    (G0)=_mm512_fmadd_pd(xv,hiddenDeltaLow,(G0)); \
    (G1)=_mm512_fmadd_pd(xv,hiddenDeltaHigh,(G1)); \
} while(false)

/**
 * V29 arithmetic with the gradient state split into two row-ordered passes.
 * Each individual accumulator receives exactly the same FMA sequence as v29.
 */
__attribute__((target("avx512f,avx512dq,fma")))
static void evaluateSliceSplitGradientV30(const Dataset&data,size_t firstRow,size_t lastRow,
                                          const Network&network,V15ThreadEvaluation&out){
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

        /* Phases 1-3 are intentionally identical to v29. */
        for(size_t tile=0;tile<tiles;tile++){
            const size_t tileRow=row+tile*V17_FORWARD_ROWS;
            const double*x[V17_FORWARD_ROWS];
            __m512d rawLow[V17_FORWARD_ROWS],rawHigh[V17_FORWARD_ROWS];
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                x[q]=data.x.rowData(tileRow+q);
                rawLow[q]=zero;
                rawHigh[q]=zero;
            }
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
                const __m512d weightsLow=_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES);
                const __m512d weightsHigh=_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES+8);
                for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                    const __m512d value=_mm512_set1_pd(x[q][k]);
                    rawLow[q]=_mm512_fmadd_pd(value,weightsLow,rawLow[q]);
                    rawHigh[q]=_mm512_fmadd_pd(value,weightsHigh,rawHigh[q]);
                }
            }
            double*tileHidden=hidden+tile*V17_FORWARD_ROWS*HIDDEN_NODES;
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                double*h=tileHidden+q*HIDDEN_NODES;
                _mm512_store_pd(h,sigmoidVectorV17(rawLow[q]));
                _mm512_store_pd(h+8,sigmoidVectorV17(rawHigh[q]));
            }
        }

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

        for(size_t tile=0;tile<tiles;tile++){
            const size_t tileRow=row+tile*V17_FORWARD_ROWS;
            const __m512d prediction=sigmoidVectorV17(_mm512_load_pd(rawOutput+tile*V17_FORWARD_ROWS));
            const __m512d actual=_mm512_loadu_pd(data.y.data()+tileRow);
            const __m512d error=_mm512_sub_pd(prediction,actual);
            const __m512d delta=_mm512_mul_pd(_mm512_mul_pd(error,prediction),_mm512_sub_pd(one,prediction));
            _mm512_store_pd(deltaThree+tile*V17_FORWARD_ROWS,delta);
            out.squaredError+=_mm512_reduce_add_pd(_mm512_mul_pd(error,error));
            __m512d percentage=_mm512_mul_pd(_mm512_div_pd(_mm512_abs_pd(error),actual),hundred);
            const __mmask8 zeroActual=_mm512_cmp_pd_mask(actual,zero,_CMP_EQ_OQ);
            percentage=_mm512_mask_mov_pd(percentage,zeroActual,zero);
            out.percentageErrorSum+=_mm512_reduce_add_pd(percentage);
            alignas(64) double percentageValues[V17_FORWARD_ROWS];
            _mm512_store_pd(percentageValues,percentage);
            for(size_t q=0;q<V17_FORWARD_ROWS;q++)
                out.maxPercentageError=max(out.maxPercentageError,percentageValues[q]);
        }

        double*gradient=out.gradient.data();
        const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
        const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);

        /* Pass A: output gradients and first-layer inputs 0-4. */
        {
            __m512d gradientTwoLow=_mm512_loadu_pd(gradient+WONE_SIZE);
            __m512d gradientTwoHigh=_mm512_loadu_pd(gradient+WONE_SIZE+8);
            V30S_DECLARE_FIRST(gradient);
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
                V30S_ACCUM(0,g00,g01); V30S_ACCUM(1,g10,g11); V30S_ACCUM(2,g20,g21);
                V30S_ACCUM(3,g30,g31); V30S_ACCUM(4,g40,g41);
            }
            _mm512_storeu_pd(gradient+WONE_SIZE,gradientTwoLow);
            _mm512_storeu_pd(gradient+WONE_SIZE+8,gradientTwoHigh);
            V30S_STORE_FIRST(gradient);
        }

        /* Pass B: first-layer inputs 5-10 in the same row order. */
        {
            V30S_DECLARE_SECOND(gradient);
            for(size_t rr=0;rr<groupRows;rr++){
                const double*h=hidden+rr*HIDDEN_NODES;
                const __m512d hiddenLow=_mm512_load_pd(h);
                const __m512d hiddenHigh=_mm512_load_pd(h+8);
                const __m512d d=_mm512_set1_pd(deltaThree[rr]);
                const __m512d hiddenDeltaLow=_mm512_mul_pd(
                    _mm512_mul_pd(d,wTwoLow),
                    _mm512_mul_pd(hiddenLow,_mm512_sub_pd(one,hiddenLow)));
                const __m512d hiddenDeltaHigh=_mm512_mul_pd(
                    _mm512_mul_pd(d,wTwoHigh),
                    _mm512_mul_pd(hiddenHigh,_mm512_sub_pd(one,hiddenHigh)));
                const double*xq=data.x.rowData(row+rr);
                V30S_ACCUM(5,g50,g51); V30S_ACCUM(6,g60,g61); V30S_ACCUM(7,g70,g71);
                V30S_ACCUM(8,g80,g81); V30S_ACCUM(9,g90,g91); V30S_ACCUM(10,g100,g101);
            }
            V30S_STORE_SECOND(gradient);
        }
        row+=groupRows;
    }
    if(row<lastRow)evaluateSliceV28(data,row,lastRow,network,out);
}

static V14Evaluation evaluateSplitGradientV30(const Dataset&data,size_t startRow,size_t batchSize,
                                               Network&network,const vector<double>&parameters,
                                               size_t requestedThreads){
    unpackV14(parameters,network);
    const size_t threads=evaluationThreadsV25(requestedThreads,batchSize);
    vector<V15ThreadEvaluation>partials(threads);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(threads))
    {
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
        const size_t base=batchSize/threads,remainder=batchSize%threads;
        const size_t offset=tid*base+min(tid,remainder),count=base+(tid<remainder?1:0);
        evaluateSliceSplitGradientV30(data,startRow+offset,startRow+offset+count,network,partials[tid]);
    }
#else
    evaluateSliceSplitGradientV30(data,startRow,startRow+batchSize,network,partials[0]);
#endif
    V14Evaluation result;
    result.gradient.assign(V14_PARAMETER_COUNT,0.0);
    long double squaredError=0.0L,percentageErrorSum=0.0L;
    double maxPercentageError=0.0;
    for(size_t t=0;t<threads;t++){
        squaredError+=partials[t].squaredError;
        percentageErrorSum+=partials[t].percentageErrorSum;
        maxPercentageError=max(maxPercentageError,partials[t].maxPercentageError);
        for(size_t i=0;i<V14_PARAMETER_COUNT;i++)result.gradient[i]+=partials[t].gradient[i];
    }
    const double inv=1.0/static_cast<double>(batchSize);
    result.metrics.cost=.5*static_cast<double>(squaredError);
    result.metrics.percentageError=static_cast<double>(percentageErrorSum)*inv;
    result.metrics.maxPercentageError=maxPercentageError;
    result.objective=result.metrics.cost*inv;
    for(double&value:result.gradient)value*=inv;
    return result;
}

static Dataset makeSplitDataV30(size_t rows,unsigned seed){
    Dataset data;
    data.x=Matrix(rows,NUMBER_OF_VARIABLES);
    data.y.resize(rows);
    const double s=static_cast<double>(seed+1);
    for(size_t r=0;r<rows;r++){
        double sum=0.0;
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            const double value=.55*sin(.013*(r+1.0)*(k+1.0+s))+.35*cos(.007*(r+3.0+s)*(k+2.0));
            data.x.data[r*NUMBER_OF_VARIABLES+k]=value;
            sum+=value*(.03+.004*static_cast<double>(k));
        }
        const double base=.2+.6/(1.0+exp(-sum));
        data.y[r]=((r+seed*17u)%97u==0u)?0.0:base;
    }
    return data;
}

static Network makeSplitNetworkV30(unsigned seed){
    Network network;
    const double s=static_cast<double>(seed+1);
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)
        for(size_t j=0;j<HIDDEN_NODES;j++)
            network.wOne[k*HIDDEN_NODES+j]=.12*sin(.43*(k+1.0)*(j+1.0)+.017*s)
                +.03*cos(.11*(k+2.0+s)*(j+3.0));
    for(size_t j=0;j<HIDDEN_NODES;j++)network.wTwo[j]=.15*cos(.37*(j+1.0)+.023*s);
    return network;
}

static bool sameSplitEvaluationV30(const V14Evaluation&a,const V14Evaluation&b){
    return a.objective==b.objective&&a.metrics.cost==b.metrics.cost
        &&a.metrics.percentageError==b.metrics.percentageError
        &&a.metrics.maxPercentageError==b.metrics.maxPercentageError
        &&a.gradient.size()==b.gradient.size()
        &&memcmp(a.gradient.data(),b.gradient.data(),a.gradient.size()*sizeof(double))==0;
}

template<class F>
static double timeSplitV30(F&&function){
    const auto started=chrono::steady_clock::now();
    volatile double guard=function().objective;
    (void)guard;
    return chrono::duration<double>(chrono::steady_clock::now()-started).count();
}

static double medianSplitV30(vector<double> values){
    nth_element(values.begin(),values.begin()+values.size()/2,values.end());
    return values[values.size()/2];
}

int main(){
    if(!supportsV16Kernel())return 75;
    const size_t exactRows[]={1,7,8,9,127,133,999,1000,2048,3072,5000,10000,10003,13300,20005,50000,100000};
    size_t checks=0;
    for(unsigned seed=0;seed<5;seed++){
        Dataset data=makeSplitDataV30(100000,seed);
        Network network=makeSplitNetworkV30(seed);
        const vector<double>parameters=packV14(network);
        for(size_t rows:exactRows)for(size_t workers=1;workers<=5;workers++){
            Network n29=network,n30=network;
            const V14Evaluation a=evaluateV29(data,0,rows,n29,parameters,workers);
            const V14Evaluation b=evaluateSplitGradientV30(data,0,rows,n30,parameters,workers);
            if(!sameSplitEvaluationV30(a,b)){
                cerr<<"FAIL exactness seed="<<seed<<" rows="<<rows<<" workers="<<workers<<'\n';
                return 2;
            }
            checks++;
        }
    }
    cout<<"PASS exact_evaluations="<<checks<<"/425\n";

    Dataset data=makeSplitDataV30(100000,2);
    Network base=makeSplitNetworkV30(2),candidate=base;
    const vector<double>parameters=packV14(base);
    cout<<setprecision(12);
    for(size_t workers:vector<size_t>{1,2})for(size_t rows:vector<size_t>{5000,10000,20000,50000,100000}){
        vector<double>speedups;
        for(int pair=0;pair<15;pair++){
            this_thread::sleep_for(chrono::milliseconds(50));
            double a=0.0,b=0.0;
            if((pair&1)==0){
                a=timeSplitV30([&]{return evaluateV29(data,0,rows,base,parameters,workers);});
                b=timeSplitV30([&]{return evaluateSplitGradientV30(data,0,rows,candidate,parameters,workers);});
            }else{
                b=timeSplitV30([&]{return evaluateSplitGradientV30(data,0,rows,candidate,parameters,workers);});
                a=timeSplitV30([&]{return evaluateV29(data,0,rows,base,parameters,workers);});
            }
            speedups.push_back(a/b);
        }
        cout<<"SPLIT rows="<<rows<<" workers="<<workers
            <<" median_speedup="<<medianSplitV30(speedups)
            <<" wins="<<count_if(speedups.begin(),speedups.end(),[](double x){return x>1.0;})<<"/15\n";
    }
    return 0;
}

#undef V30S_DECLARE_FIRST
#undef V30S_STORE_FIRST
#undef V30S_DECLARE_SECOND
#undef V30S_STORE_SECOND
#undef V30S_ACCUM

#else
int main(){return 75;}
#endif
