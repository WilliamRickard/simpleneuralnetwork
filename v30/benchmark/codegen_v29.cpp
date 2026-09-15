#define main v29_benchmark_embedded_main
#include "../../v29/benchmark/full_training_benchmark.cpp"
#undef main

#include <cstdint>

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

    Dataset data=trainingDataV29(rows);
    Network network=initialNetworkV29();
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
    if(!result.reachedTarget||result.updates==0)return 3;
    writeSignatureV30Codegen(signature,result,network);
    cout<<setprecision(12)<<chrono::duration<double>(finish-start).count()<<','
        <<result.updates<<'\n';
    return 0;
}
