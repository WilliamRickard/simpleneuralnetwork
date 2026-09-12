#include<algorithm>
#include<array>
#include<chrono>
#include<cmath>
#include<cstdlib>
#include<iomanip>
#include<immintrin.h>
#include<iostream>
#include<vector>
#include<omp.h>
using namespace std;

constexpr size_t INPUTS=11,HIDDEN=16,W1_SIZE=INPUTS*HIDDEN,BLOCK=64;

struct Dataset {
    size_t rows;
    vector<double> x,y;
    explicit Dataset(size_t r) : rows(r),x(r*INPUTS),y(r) {}
    double *rowData(size_t r) { return x.data()+r*INPUTS; }
    const double *rowData(size_t r) const { return x.data()+r*INPUTS; }
};

struct Network {
    array<double,W1_SIZE> w1{};
    array<double,HIDDEN> w2{};
    array<double,W1_SIZE> velocityW1{};
    array<double,HIDDEN> velocityW2{};
};

struct alignas(64) Accumulator {
    array<double,W1_SIZE> gradientW1{};
    array<double,HIDDEN> gradientW2{};
    double squaredError=0.0;
    double percentageError=0.0;
    double maxPercentageError=0.0;

    void clear() {
        gradientW1.fill(0.0);
        gradientW2.fill(0.0);
        squaredError=0.0;
        percentageError=0.0;
        maxPercentageError=0.0;
    }
};

struct Metrics {
    double cost=0.0;
    double percentageError=0.0;
    double maxPercentageError=0.0;
};

static inline double sigmoid(double value) {
    if(value < -700.0) {
        const double exponent=exp(value);
        return exponent/(1.0+exponent);
    }
    return 1.0/(1.0+exp(-value));
}

extern "C" __m256d _ZGVdN4v_exp(__m256d);
extern "C" __m512d _ZGVeN8v_exp(__m512d);

__attribute__((target("avx2")))
static void sigmoidVectorAVX2(double *values,size_t count) {
    const __m256d one=_mm256_set1_pd(1.0);
    const __m256d zero=_mm256_setzero_pd();
    const __m256d lowerLimit=_mm256_set1_pd(-700.0);
    size_t i=0;
    for(;i+4<=count;i+=4) {
        const __m256d x=_mm256_loadu_pd(values+i);
        const __m256d extreme=_mm256_cmp_pd(x,lowerLimit,_CMP_LT_OQ);
        if(_mm256_movemask_pd(extreme)) {
            for(size_t j=0;j<4;j++) values[i+j]=sigmoid(values[i+j]);
            continue;
        }
        const __m256d exponent=_ZGVdN4v_exp(_mm256_sub_pd(zero,x));
        _mm256_storeu_pd(values+i,_mm256_div_pd(one,_mm256_add_pd(one,exponent)));
    }
    for(;i<count;i++) values[i]=sigmoid(values[i]);
}

__attribute__((target("avx512f")))
static void sigmoidVectorAVX512(double *values,size_t count) {
    const __m512d one=_mm512_set1_pd(1.0);
    const __m512d zero=_mm512_setzero_pd();
    const __m512d lowerLimit=_mm512_set1_pd(-700.0);
    size_t i=0;
    for(;i+8<=count;i+=8) {
        const __m512d x=_mm512_loadu_pd(values+i);
        const __mmask8 extreme=_mm512_cmp_pd_mask(x,lowerLimit,_CMP_LT_OQ);
        if(extreme) {
            for(size_t j=0;j<8;j++) values[i+j]=sigmoid(values[i+j]);
            continue;
        }
        const __m512d exponent=_ZGVeN8v_exp(_mm512_sub_pd(zero,x));
        _mm512_storeu_pd(values+i,_mm512_div_pd(one,_mm512_add_pd(one,exponent)));
    }
    for(;i<count;i++) values[i]=sigmoid(values[i]);
}

static void sigmoidVector(double *values,size_t count) {
    if(__builtin_cpu_supports("avx512f")) {
        sigmoidVectorAVX512(values,count);
        return;
    }
    if(__builtin_cpu_supports("avx2")) {
        sigmoidVectorAVX2(values,count);
        return;
    }
    for(size_t i=0;i<count;i++) values[i]=sigmoid(values[i]);
}

static void fillData(Dataset &data) {
    array<double,W1_SIZE> teacherW1{};
    array<double,HIDDEN> teacherW2{};
    for(size_t k=0;k<INPUTS;k++) {
        for(size_t j=0;j<HIDDEN;j++) teacherW1[k*HIDDEN+j]=0.22*sin(0.17*(k+1)*(j+2));
    }
    for(size_t j=0;j<HIDDEN;j++) teacherW2[j]=0.28*cos(0.31*(j+1));

    for(size_t i=0;i<data.rows;i++) {
        double *x=data.rowData(i);
        for(size_t k=0;k<INPUTS;k++) x[k]=0.55*sin(0.013*(i+1)*(k+1))+0.35*cos(0.007*(i+3)*(k+2));
        double zThree=0.0;
        for(size_t j=0;j<HIDDEN;j++) {
            double zTwo=0.0;
            for(size_t k=0;k<INPUTS;k++) zTwo+=x[k]*teacherW1[k*HIDDEN+j];
            zThree+=sigmoid(zTwo)*teacherW2[j];
        }
        data.y[i]=sigmoid(zThree);
    }
}

static void initialiseWeights(Network &network) {
    for(size_t k=0;k<INPUTS;k++) {
        for(size_t j=0;j<HIDDEN;j++) network.w1[k*HIDDEN+j]=0.12*sin(0.43*(k+1)*(j+1));
    }
    for(size_t j=0;j<HIDDEN;j++) network.w2[j]=0.15*cos(0.37*(j+1));
    network.velocityW1.fill(0.0);
    network.velocityW2.fill(0.0);
}

static void processBlock(const Dataset &data,size_t base,const Network &network,Accumulator &accumulator) {
    const size_t rows=min(BLOCK,data.rows-base);
    alignas(64) double hidden[BLOCK*HIDDEN]={};
    alignas(64) double output[BLOCK];
    alignas(64) double deltaThree[BLOCK];

    for(size_t row=0;row<rows;row++) {
        const double *x=data.rowData(base+row);
        double *hiddenRow=hidden+row*HIDDEN;
        for(size_t k=0;k<INPUTS;k++) {
            const double xValue=x[k];
            const double *weightRow=network.w1.data()+k*HIDDEN;
            for(size_t j=0;j<HIDDEN;j++) hiddenRow[j]+=xValue*weightRow[j];
        }
    }

    sigmoidVector(hidden,rows*HIDDEN);

    for(size_t row=0;row<rows;row++) {
        double *hiddenRow=hidden+row*HIDDEN;
        double zThree=0.0;
        for(size_t j=0;j<HIDDEN;j++) zThree+=hiddenRow[j]*network.w2[j];
        output[row]=zThree;
    }
    sigmoidVector(output,rows);

    for(size_t row=0;row<rows;row++) {
        const double prediction=output[row];
        const double actual=data.y[base+row];
        const double error=prediction-actual;
        accumulator.squaredError+=error*error;
        if(actual!=0.0) {
            const double percentage=abs(error/actual)*100.0;
            accumulator.percentageError+=percentage;
            accumulator.maxPercentageError=max(accumulator.maxPercentageError,percentage);
        }
        deltaThree[row]=error*prediction*(1.0-prediction);
    }

    for(size_t row=0;row<rows;row++) {
        double *hiddenRow=hidden+row*HIDDEN;
        const double delta=deltaThree[row];
        for(size_t j=0;j<HIDDEN;j++) {
            const double activation=hiddenRow[j];
            accumulator.gradientW2[j]+=activation*delta;
            hiddenRow[j]=delta*network.w2[j]*activation*(1.0-activation);
        }
        const double *x=data.rowData(base+row);
        for(size_t k=0;k<INPUTS;k++) {
            const double xValue=x[k];
            double *gradientRow=accumulator.gradientW1.data()+k*HIDDEN;
            for(size_t j=0;j<HIDDEN;j++) gradientRow[j]+=xValue*hiddenRow[j];
        }
    }
}

static void applyUpdate(Network &network,const array<double,W1_SIZE> &gradientW1,const array<double,HIDDEN> &gradientW2,double learningRate,double momentum) {
    for(size_t i=0;i<W1_SIZE;i++) {
        network.velocityW1[i]=momentum*network.velocityW1[i]-learningRate*gradientW1[i];
        network.w1[i]+=network.velocityW1[i];
    }
    for(size_t j=0;j<HIDDEN;j++) {
        network.velocityW2[j]=momentum*network.velocityW2[j]-learningRate*gradientW2[j];
        network.w2[j]+=network.velocityW2[j];
    }
}

int main(int argc,char **argv) {
    const size_t rows=argc>1?strtoull(argv[1],0,10):13853;
    const size_t updates=argc>2?strtoull(argv[2],0,10):10;
    const size_t repetitions=argc>3?strtoull(argv[3],0,10):1;
    const int threads=argc>4?atoi(argv[4]):4;
    const double learningRate=0.0001/static_cast<double>(rows);
    const double momentum=0.75;

    Dataset data(rows);
    fillData(data);
    Network network;
    vector<Accumulator> accumulators(threads);
    array<double,W1_SIZE> gradientW1{};
    array<double,HIDDEN> gradientW2{};
    Metrics metrics;

    for(size_t repetition=0;repetition<repetitions;repetition++) {
        initialiseWeights(network);
        const auto start=chrono::steady_clock::now();

#pragma omp parallel num_threads(threads) shared(network,accumulators,gradientW1,gradientW2,metrics)
        {
            const int thread=omp_get_thread_num();
            for(size_t update=0;update<updates;update++) {
                accumulators[thread].clear();

#pragma omp for schedule(static)
                for(long long block=0;block<static_cast<long long>((rows+BLOCK-1)/BLOCK);block++) {
                    processBlock(data,static_cast<size_t>(block)*BLOCK,network,accumulators[thread]);
                }

#pragma omp single
                {
                    gradientW1.fill(0.0);
                    gradientW2.fill(0.0);
                    double squaredError=0.0;
                    double percentageError=0.0;
                    double maxPercentageError=0.0;

                    for(int worker=0;worker<threads;worker++) {
                        const Accumulator &accumulator=accumulators[worker];
                        for(size_t i=0;i<W1_SIZE;i++) gradientW1[i]+=accumulator.gradientW1[i];
                        for(size_t j=0;j<HIDDEN;j++) gradientW2[j]+=accumulator.gradientW2[j];
                        squaredError+=accumulator.squaredError;
                        percentageError+=accumulator.percentageError;
                        maxPercentageError=max(maxPercentageError,accumulator.maxPercentageError);
                    }

                    metrics.cost=0.5*squaredError;
                    metrics.percentageError=percentageError/static_cast<double>(rows);
                    metrics.maxPercentageError=maxPercentageError;
                    applyUpdate(network,gradientW1,gradientW2,learningRate,momentum);
                }
            }
        }

        const auto end=chrono::steady_clock::now();
        const double seconds=chrono::duration<double>(end-start).count();
        double checksum=0.0;
        for(double value:network.w1) checksum+=value;
        for(double value:network.w2) checksum+=value;

        cout<<setprecision(17)
            <<"version=v4 rows="<<rows
            <<" updates="<<updates
            <<" rep="<<repetition
            <<" threads="<<threads
            <<" seconds="<<seconds
            <<" cost="<<metrics.cost
            <<" pct="<<metrics.percentageError
            <<" max="<<metrics.maxPercentageError
            <<" checksum="<<checksum
            <<'\n';
    }
}
