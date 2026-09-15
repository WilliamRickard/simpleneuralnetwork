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

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>

constexpr size_t V30_GN_BASES_PER_ROW=2*HIDDEN_NODES;
constexpr size_t V30_GN_BLOCK_ROWS=8192;

/**
 * Prepare the 16 output Jacobians and 16 hidden Jacobian bases exactly as v27.
 * The 176 input-specific Jacobians are reconstructed during ordered reduction.
 */
__attribute__((target("avx512f,avx512dq"),optimize("fp-contract=off")))
static void gaussNewtonBasesRowV30(const double*x,const Network&network,double*out){
    alignas(64) double raw[HIDDEN_NODES],hidden[HIDDEN_NODES];
    __m512d low=_mm512_setzero_pd(),high=_mm512_setzero_pd();
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
        const __m512d value=_mm512_set1_pd(x[k]);
        low=_mm512_add_pd(low,_mm512_mul_pd(value,_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES)));
        high=_mm512_add_pd(high,_mm512_mul_pd(value,_mm512_loadu_pd(network.wOne.data()+k*HIDDEN_NODES+8)));
    }
    _mm512_store_pd(raw,low);
    _mm512_store_pd(raw+8,high);
    for(size_t j=0;j<HIDDEN_NODES;j++)hidden[j]=sigmoidV15(raw[j]);

    double rawOutput=0.0;
    for(size_t j=0;j<HIDDEN_NODES;j++)rawOutput+=hidden[j]*network.wTwo[j];
    const double prediction=sigmoidV15(rawOutput);
    const double outputDerivative=prediction*(1.0-prediction);

    const __m512d one=_mm512_set1_pd(1.0);
    const __m512d derivative=_mm512_set1_pd(outputDerivative);
    const __m512d hiddenLow=_mm512_load_pd(hidden);
    const __m512d hiddenHigh=_mm512_load_pd(hidden+8);
    const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
    const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);

    _mm512_storeu_pd(out,_mm512_mul_pd(derivative,hiddenLow));
    _mm512_storeu_pd(out+8,_mm512_mul_pd(derivative,hiddenHigh));
    const __m512d baseLow=_mm512_mul_pd(
        _mm512_mul_pd(_mm512_mul_pd(derivative,wTwoLow),hiddenLow),
        _mm512_sub_pd(one,hiddenLow));
    const __m512d baseHigh=_mm512_mul_pd(
        _mm512_mul_pd(_mm512_mul_pd(derivative,wTwoHigh),hiddenHigh),
        _mm512_sub_pd(one,hiddenHigh));
    _mm512_storeu_pd(out+HIDDEN_NODES,baseLow);
    _mm512_storeu_pd(out+HIDDEN_NODES+8,baseHigh);
}

/** Reduce precomputed bases in exactly v20's row, hidden-node and input order. */
static void accumulateBasesBlockV30(const Dataset&data,size_t startRow,size_t count,
                                    const double*scratch,V20DiagonalPartial&total){
    for(size_t rr=0;rr<count;rr++){
        const double*x=data.x.rowData(startRow+rr);
        const double*bases=scratch+rr*V30_GN_BASES_PER_ROW;
        for(size_t j=0;j<HIDDEN_NODES;j++){
            const double outputJacobian=bases[j];
            total.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
            const double hiddenJacobianBase=bases[HIDDEN_NODES+j];
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
                const double jacobian=hiddenJacobianBase*x[k];
                total.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
            }
        }
    }
}

/** Exact sub-50k replacement for v27 with 32 rather than 192 doubles of scratch per row. */
static V20DiagonalScale gaussNewtonScaleReducedScratchV30(
    const Dataset&data,size_t startRow,size_t batchSize,Network&network,
    const vector<double>&parameters,size_t requestedThreads){
    unpackV14(parameters,network);
    const size_t threads=evaluationThreadsV25(requestedThreads,batchSize);
    const size_t scratchRows=min(V30_GN_BLOCK_ROWS,batchSize);
    vector<double>scratch(scratchRows*V30_GN_BASES_PER_ROW);
    V20DiagonalPartial total;
    for(size_t offset=0;offset<batchSize;offset+=V30_GN_BLOCK_ROWS){
        const size_t count=min(V30_GN_BLOCK_ROWS,batchSize-offset);
#ifdef _OPENMP
#pragma omp parallel for num_threads(static_cast<int>(threads)) schedule(static)
#endif
        for(long long rr=0;rr<static_cast<long long>(count);rr++){
            const size_t local=static_cast<size_t>(rr);
            gaussNewtonBasesRowV30(data.x.rowData(startRow+offset+local),network,
                                  scratch.data()+local*V30_GN_BASES_PER_ROW);
        }
        accumulateBasesBlockV30(data,startRow+offset,count,scratch.data(),total);
    }
    return finishGaussNewtonScaleV27(total,batchSize);
}

/** Exact SIMD row calculation while retaining one v20 long-double partial per thread. */
static void accumulateGaussNewtonSliceSimdV30(const Dataset&data,size_t firstRow,size_t lastRow,
                                              const Network&network,V20DiagonalPartial&out){
    alignas(64) double bases[V30_GN_BASES_PER_ROW];
    for(size_t row=firstRow;row<lastRow;row++){
        const double*x=data.x.rowData(row);
        gaussNewtonBasesRowV30(x,network,bases);
        for(size_t j=0;j<HIDDEN_NODES;j++){
            const double outputJacobian=bases[j];
            out.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
            const double hiddenJacobianBase=bases[HIDDEN_NODES+j];
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
                const double jacobian=hiddenJacobianBase*x[k];
                out.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
            }
        }
    }
}

/** Exact 50k+ replacement for v20 preserving its thread partition and reduction order. */
static V20DiagonalScale gaussNewtonScaleDirectSimdV30(
    const Dataset&data,size_t startRow,size_t batchSize,Network&network,
    const vector<double>&parameters,size_t requestedThreads){
    unpackV14(parameters,network);
    const size_t threads=evaluationThreadsV15(requestedThreads,batchSize);
    vector<V20DiagonalPartial>partials(threads);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(threads))
    {
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
        const size_t base=batchSize/threads,remainder=batchSize%threads;
        const size_t offset=tid*base+min(tid,remainder),count=base+(tid<remainder?1:0);
        accumulateGaussNewtonSliceSimdV30(data,startRow+offset,startRow+offset+count,network,partials[tid]);
    }
#else
    accumulateGaussNewtonSliceSimdV30(data,startRow,startRow+batchSize,network,partials[0]);
#endif
    V20DiagonalScale diagonal{},scale{};
    vector<double>ordered(V14_PARAMETER_COUNT);
    const long double inv=1.0L/static_cast<long double>(batchSize);
    for(size_t i=0;i<V14_PARAMETER_COUNT;i++){
        long double sum=0.0L;
        for(size_t t=0;t<threads;t++)sum+=partials[t].values[i];
        diagonal[i]=static_cast<double>(sum*inv);
        ordered[i]=diagonal[i];
    }
    nth_element(ordered.begin(),ordered.begin()+V14_PARAMETER_COUNT/2,ordered.end());
    const double reference=ordered[V14_PARAMETER_COUNT/2];
    if(!(reference>0.0)||!isfinite(reference)){
        scale.fill(1.0);
        return scale;
    }
    const double floorValue=reference*V20_GN_FLOOR_RATIO;
    const double minimumScale=1.0/V20_GN_CLIP;
    for(size_t i=0;i<V14_PARAMETER_COUNT;i++){
        const double safeDiagonal=max(diagonal[i],floorValue);
        const double rawScale=pow(reference/safeDiagonal,V20_GN_EXPONENT);
        scale[i]=min(V20_GN_CLIP,max(minimumScale,rawScale));
    }
    return scale;
}

static Dataset gnDataV30(size_t rows){
    Dataset data;
    data.x=Matrix(rows,NUMBER_OF_VARIABLES);
    data.y.resize(rows,0.5);
    for(size_t r=0;r<rows;r++){
        double*x=data.x.rowData(r);
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)
            x[k]=.62*sin(.0091*static_cast<double>((r+1)*(k+1)))
                +.27*cos(.0067*static_cast<double>((r+5)*(k+2)))
                +.08*sin(.0013*static_cast<double>((r+11)*(k+4)));
    }
    return data;
}

static Network gnNetworkV30(unsigned state){
    Network network;
    const double s=static_cast<double>(state+1);
    for(size_t i=0;i<network.wOne.size();i++)
        network.wOne[i]=.21*sin(.071*static_cast<double>(i+1)+.13*s)
                       +.08*cos(.037*static_cast<double>(i+3)-.09*s);
    for(size_t i=0;i<network.wTwo.size();i++)
        network.wTwo[i]=.35*cos(.17*static_cast<double>(i+1)+.11*s)
                       +.06*sin(.053*static_cast<double>(i+2)*s);
    return network;
}

static bool sameScaleV30(const V20DiagonalScale&a,const V20DiagonalScale&b){
    return memcmp(a.data(),b.data(),sizeof(double)*V14_PARAMETER_COUNT)==0;
}

template<class F>
static double timeGnV30(F&&function){
    const auto start=chrono::steady_clock::now();
    volatile double guard=function()[0];
    (void)guard;
    return chrono::duration<double>(chrono::steady_clock::now()-start).count();
}

static double medianV30(vector<double> values){
    nth_element(values.begin(),values.begin()+values.size()/2,values.end());
    return values[values.size()/2];
}

int main(int argc,char**argv){
    if(!supportsV16Kernel()){
        cout<<"SKIP: AVX-512/libmvec production kernel unavailable\n";
        return 75;
    }
    const int pairs=argc>1?atoi(argv[1]):15;
    const size_t threads=argc>2?strtoull(argv[2],nullptr,10):2;
    if(pairs<=0||threads==0)return 2;

    const vector<size_t> rowCounts={5000,10000,20000,49992,50000,100000};
    cout<<setprecision(12);
    for(size_t rows:rowCounts){
        Dataset data=gnDataV30(rows);
        for(unsigned state=0;state<8;state++){
            Network baselineNetwork=gnNetworkV30(state),candidateNetwork=baselineNetwork;
            vector<double>parameters=packV14(baselineNetwork);
            const V20DiagonalScale baseline=rows<V27_GN_PARALLEL_ROW_LIMIT
                ?gaussNewtonScaleV27(data,0,rows,baselineNetwork,parameters,threads)
                :gaussNewtonScaleV20(data,0,rows,baselineNetwork,parameters,threads);
            const V20DiagonalScale candidate=rows<V27_GN_PARALLEL_ROW_LIMIT
                ?gaussNewtonScaleReducedScratchV30(data,0,rows,candidateNetwork,parameters,threads)
                :gaussNewtonScaleDirectSimdV30(data,0,rows,candidateNetwork,parameters,threads);
            if(!sameScaleV30(baseline,candidate)){
                cerr<<"FAIL exactness rows="<<rows<<" state="<<state<<'\n';
                for(size_t i=0;i<V14_PARAMETER_COUNT;i++)
                    if(baseline[i]!=candidate[i]){
                        cerr<<"first_diff="<<i<<" baseline="<<setprecision(17)<<baseline[i]
                            <<" candidate="<<candidate[i]<<'\n';
                        break;
                    }
                return 3;
            }
        }

        Network baselineNetwork=gnNetworkV30(3),candidateNetwork=baselineNetwork;
        const vector<double>parameters=packV14(baselineNetwork);
        vector<double> speedups;
        speedups.reserve(static_cast<size_t>(pairs));
        for(int pair=0;pair<pairs;pair++){
            double baselineSeconds=0.0,candidateSeconds=0.0;
            this_thread::sleep_for(chrono::milliseconds(50));
            if((pair&1)==0){
                baselineSeconds=timeGnV30([&]{
                    return rows<V27_GN_PARALLEL_ROW_LIMIT
                        ?gaussNewtonScaleV27(data,0,rows,baselineNetwork,parameters,threads)
                        :gaussNewtonScaleV20(data,0,rows,baselineNetwork,parameters,threads);
                });
                candidateSeconds=timeGnV30([&]{
                    return rows<V27_GN_PARALLEL_ROW_LIMIT
                        ?gaussNewtonScaleReducedScratchV30(data,0,rows,candidateNetwork,parameters,threads)
                        :gaussNewtonScaleDirectSimdV30(data,0,rows,candidateNetwork,parameters,threads);
                });
            }else{
                candidateSeconds=timeGnV30([&]{
                    return rows<V27_GN_PARALLEL_ROW_LIMIT
                        ?gaussNewtonScaleReducedScratchV30(data,0,rows,candidateNetwork,parameters,threads)
                        :gaussNewtonScaleDirectSimdV30(data,0,rows,candidateNetwork,parameters,threads);
                });
                baselineSeconds=timeGnV30([&]{
                    return rows<V27_GN_PARALLEL_ROW_LIMIT
                        ?gaussNewtonScaleV27(data,0,rows,baselineNetwork,parameters,threads)
                        :gaussNewtonScaleV20(data,0,rows,baselineNetwork,parameters,threads);
                });
            }
            speedups.push_back(baselineSeconds/candidateSeconds);
        }
        const size_t wins=count_if(speedups.begin(),speedups.end(),[](double x){return x>1.0;});
        cout<<"GN rows="<<rows<<" threads="<<threads<<" pairs="<<pairs
            <<" median_speedup="<<medianV30(speedups)<<" wins="<<wins<<'/'<<pairs
            <<" exact_states=8/8 mode="
            <<(rows<V27_GN_PARALLEL_ROW_LIMIT?"reduced_scratch":"direct_simd")<<'\n';
    }
    return 0;
}
