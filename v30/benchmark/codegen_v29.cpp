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

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

static Network teacherNetworkV30Codegen(){
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
__attribute__((target("avx512f,avx512dq,fma")))
static void fillTeacherTargetsV30Codegen(Dataset&data,const Network&teacher){
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

static Dataset trainingDataV30Codegen(size_t rows){
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
    fillTeacherTargetsV30Codegen(data,teacherNetworkV30Codegen());
#else
    throw runtime_error("v30 codegen benchmark requires the accelerated production path");
#endif
    return data;
}

static Network initialNetworkV30Codegen(){
    Network network=teacherNetworkV30Codegen();
    for(size_t i=0;i<network.wOne.size();i++)
        network.wOne[i]+=.045*sin(.19*static_cast<double>(i+1))
                        +.012*cos(.071*static_cast<double>(i+3));
    for(size_t i=0;i<network.wTwo.size();i++)
        network.wTwo[i]+=.055*cos(.23*static_cast<double>(i+1));
    return network;
}

static void writeSignatureV30Codegen(const string&path,const TrainResult&result,const Network&network){
    ofstream output(path.c_str(),ios::binary);
    if(!output)throw runtime_error("Failed to open signature file: "+path);
    const uint64_t updates=static_cast<uint64_t>(result.updates);
    const unsigned char reached=result.reachedTarget?1:0;
    output.write(reinterpret_cast<const char*>(&updates),sizeof(updates));
    output.write(reinterpret_cast<const char*>(&reached),sizeof(reached));
    output.write(reinterpret_cast<const char*>(&result.metrics.cost),sizeof(double));
    output.write(reinterpret_cast<const char*>(&result.metrics.percentageError),sizeof(double));
    output.write(reinterpret_cast<const char*>(&result.metrics.maxPercentageError),sizeof(double));
    const vector<double>parameters=packV14(network);
    output.write(reinterpret_cast<const char*>(parameters.data()),
                 static_cast<streamsize>(parameters.size()*sizeof(double)));
}

int main(int argc,char**argv){
    if(!supportsV16Kernel())return 75;
    const size_t rows=argc>1?strtoull(argv[1],nullptr,10):20000;
    const double target=argc>2?atof(argv[2]):.0005;
    const size_t threads=argc>3?strtoull(argv[3],nullptr,10):2;
    const string signature=argc>4?argv[4]:"signature.bin";
    if(rows==0||threads==0)return 2;

    Dataset data=trainingDataV30Codegen(rows);
    Network network=initialNetworkV30Codegen();
    OptimiserConfig config;
    config.maxDescents=1000000;
    config.percentageErrorTarget=target;
    config.logEvery=0;
    config.threads=threads;

    ostringstream sink;
    streambuf*old=cout.rdbuf(sink.rdbuf());
    const auto start=chrono::steady_clock::now();
    const TrainResult result=trainRangeV29(data,0,rows,0,network,config);
    const auto finish=chrono::steady_clock::now();
    cout.rdbuf(old);
    std::remove("ybar.txt");
    if(!result.reachedTarget||result.updates==0)return 3;
    writeSignatureV30Codegen(signature,result,network);
    cout<<setprecision(12)<<chrono::duration<double>(finish-start).count()<<','
        <<result.updates<<'\n';
    return 0;
}
