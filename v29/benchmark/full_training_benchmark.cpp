#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define SIMPLE_NN_V29_NO_MAIN
#include "../main.cpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>

static Network teacherNetworkV29(){
    Network network;
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)
        for(size_t j=0;j<HIDDEN_NODES;j++)
            network.wOne[k*HIDDEN_NODES+j]=
                .24*sin(.31*static_cast<double>((k+1)*(j+1)))
               +.07*cos(.17*static_cast<double>(k+2*j+3));
    for(size_t j=0;j<HIDDEN_NODES;j++)
        network.wTwo[j]=.42*cos(.29*static_cast<double>(j+1))
                       +.11*sin(.13*static_cast<double>(j+3));
    return network;
}

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
/** Generate targets with the same eight-row production forward arithmetic. */
__attribute__((target("avx512f,avx512dq,fma")))
static void fillTeacherTargetsV29(Dataset&data,const Network&teacher){
    const __m512d zero=_mm512_setzero_pd();
    size_t row=0;
    for(;row+V17_FORWARD_ROWS<=data.y.size();row+=V17_FORWARD_ROWS){
        const double*x[V17_FORWARD_ROWS];
        __m512d rawLow[V17_FORWARD_ROWS],rawHigh[V17_FORWARD_ROWS];
        for(size_t q=0;q<V17_FORWARD_ROWS;q++){
            x[q]=data.x.rowData(row+q);
            rawLow[q]=zero;
            rawHigh[q]=zero;
        }
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            const __m512d weightsLow=_mm512_loadu_pd(teacher.wOne.data()+k*HIDDEN_NODES);
            const __m512d weightsHigh=_mm512_loadu_pd(teacher.wOne.data()+k*HIDDEN_NODES+8);
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                const __m512d value=_mm512_set1_pd(x[q][k]);
                rawLow[q]=_mm512_fmadd_pd(value,weightsLow,rawLow[q]);
                rawHigh[q]=_mm512_fmadd_pd(value,weightsHigh,rawHigh[q]);
            }
        }

        alignas(64) double rawOutput[V17_FORWARD_ROWS];
        for(size_t q=0;q<V17_FORWARD_ROWS;q++){
            const __m512d hiddenLow=sigmoidVectorV17(rawLow[q]);
            const __m512d hiddenHigh=sigmoidVectorV17(rawHigh[q]);
            rawOutput[q]=_mm512_reduce_add_pd(_mm512_add_pd(
                _mm512_mul_pd(hiddenLow,_mm512_loadu_pd(teacher.wTwo.data())),
                _mm512_mul_pd(hiddenHigh,_mm512_loadu_pd(teacher.wTwo.data()+8))));
        }
        _mm512_storeu_pd(data.y.data()+row,
                         sigmoidVectorV17(_mm512_load_pd(rawOutput)));
    }

    /* The release matrix row counts are multiples of eight, but keep a correct tail. */
    for(;row<data.y.size();row++){
        const double*x=data.x.rowData(row);
        __m512d rawLow=zero,rawHigh=zero;
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            const __m512d value=_mm512_set1_pd(x[k]);
            rawLow=_mm512_fmadd_pd(value,
                _mm512_loadu_pd(teacher.wOne.data()+k*HIDDEN_NODES),rawLow);
            rawHigh=_mm512_fmadd_pd(value,
                _mm512_loadu_pd(teacher.wOne.data()+k*HIDDEN_NODES+8),rawHigh);
        }
        const __m512d hiddenLow=sigmoidVectorV17(rawLow);
        const __m512d hiddenHigh=sigmoidVectorV17(rawHigh);
        const double rawOutput=_mm512_reduce_add_pd(_mm512_add_pd(
            _mm512_mul_pd(hiddenLow,_mm512_loadu_pd(teacher.wTwo.data())),
            _mm512_mul_pd(hiddenHigh,_mm512_loadu_pd(teacher.wTwo.data()+8))));
        data.y[row]=sigmoidV15(rawOutput);
    }
}
#endif

static Dataset trainingDataV29(size_t rows){
    Dataset data;
    data.x=Matrix(rows,NUMBER_OF_VARIABLES);
    data.y.resize(rows);
    for(size_t r=0;r<rows;r++){
        double*x=data.x.rowData(r);
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)
            x[k]=.62*sin(.0091*static_cast<double>((r+1)*(k+1)))
                +.27*cos(.0067*static_cast<double>((r+5)*(k+2)))
                +.08*sin(.0013*static_cast<double>((r+11)*(k+4)));
    }
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    fillTeacherTargetsV29(data,teacherNetworkV29());
#else
    throw runtime_error("v29 full-training benchmark requires the accelerated production path");
#endif
    return data;
}

static Network initialNetworkV29(){
    Network network=teacherNetworkV29();
    for(size_t i=0;i<network.wOne.size();i++)
        network.wOne[i]+=.045*sin(.19*static_cast<double>(i+1))
                        +.012*cos(.071*static_cast<double>(i+3));
    for(size_t i=0;i<network.wTwo.size();i++)
        network.wTwo[i]+=.055*cos(.23*static_cast<double>(i+1));
    return network;
}

static bool sameResultV29(const TrainResult&a,const TrainResult&b,
                          const Network&na,const Network&nb){
    if(a.updates!=b.updates||a.reachedTarget!=b.reachedTarget)return false;
    if(a.metrics.cost!=b.metrics.cost||
       a.metrics.percentageError!=b.metrics.percentageError||
       a.metrics.maxPercentageError!=b.metrics.maxPercentageError)return false;
    const vector<double>pa=packV14(na),pb=packV14(nb);
    return pa.size()==pb.size()&&
           memcmp(pa.data(),pb.data(),pa.size()*sizeof(double))==0;
}

template<class F>
static double timedTrainingV29(F function){
    ostringstream sink;
    streambuf*old=cout.rdbuf(sink.rdbuf());
    const auto start=chrono::steady_clock::now();
    function();
    const auto finish=chrono::steady_clock::now();
    cout.rdbuf(old);
    std::remove("ybar.txt");
    return chrono::duration<double>(finish-start).count();
}

int main(int argc,char**argv){
    if(!supportsV16Kernel()){
        cout<<"SKIP: AVX-512/libmvec production kernel unavailable\n";
        return 0;
    }

    const size_t rows=argc>1?strtoull(argv[1],nullptr,10):20000;
    const double target=argc>2?atof(argv[2]):.001;
    const int pairs=argc>3?atoi(argv[3]):7;
    const size_t threads=argc>4?strtoull(argv[4],nullptr,10):4;
    if(rows==0||pairs<=0||threads==0)return 2;

    Dataset data=trainingDataV29(rows);
    const Network initial=initialNetworkV29();
    OptimiserConfig config;
    config.maxDescents=1000000;
    config.percentageErrorTarget=target;
    config.logEvery=0;
    config.threads=threads;

    /* Untimed equality gate on the exact production wrappers. */
    Network exact28=initial,exact29=initial;
    TrainResult result28,result29;
    timedTrainingV29([&]{result28=trainRangeV28(data,0,rows,0,exact28,config);});
    timedTrainingV29([&]{result29=trainRangeV29(data,0,rows,0,exact29,config);});
    if(!sameResultV29(result28,result29,exact28,exact29)){
        cerr<<"FAIL: v28/v29 full-training results differ\n";
        return 3;
    }
    if(!result28.reachedTarget||result28.updates==0){
        cerr<<"FAIL: benchmark problem did not exercise a non-trivial converged trajectory\n";
        return 4;
    }

    cout<<setprecision(12);
    for(int pair=0;pair<pairs;pair++){
        Network n28=initial,n29=initial;
        TrainResult r28,r29;
        double t28=0.0,t29=0.0;
        if((pair&1)==0){
            t28=timedTrainingV29([&]{r28=trainRangeV28(data,0,rows,0,n28,config);});
            t29=timedTrainingV29([&]{r29=trainRangeV29(data,0,rows,0,n29,config);});
        }else{
            t29=timedTrainingV29([&]{r29=trainRangeV29(data,0,rows,0,n29,config);});
            t28=timedTrainingV29([&]{r28=trainRangeV28(data,0,rows,0,n28,config);});
        }
        if(!sameResultV29(r28,r29,n28,n29)){
            cerr<<"FAIL: pair "<<pair<<" changed trajectory\n";
            return 5;
        }
        cout<<rows<<','<<target<<','<<threads<<','<<pair<<','
            <<r28.updates<<','<<t28<<','<<t29<<','<<(t28/t29)<<'\n';
    }
    return 0;
}
