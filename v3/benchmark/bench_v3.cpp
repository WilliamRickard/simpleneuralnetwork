#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

constexpr size_t INPUTS = 11;
constexpr size_t HIDDEN = 16;
constexpr size_t W1_SIZE = INPUTS * HIDDEN;

struct Dataset {
    size_t rows;
    vector<double> x;
    vector<double> y;
    explicit Dataset(size_t r) : rows(r), x(r * INPUTS), y(r) {}
    double* rowData(size_t r) { return x.data() + r * INPUTS; }
    const double* rowData(size_t r) const { return x.data() + r * INPUTS; }
};

struct Network {
    array<double,W1_SIZE> w1{};
    array<double,HIDDEN> w2{};
    array<double,W1_SIZE> velocityW1{};
    array<double,HIDDEN> velocityW2{};
};

struct Gradients {
    array<double,W1_SIZE> w1{};
    array<double,HIDDEN> w2{};
    void clear() { w1.fill(0.0); w2.fill(0.0); }
};

struct Metrics { double cost=0.0, percentageError=0.0, maxPercentageError=0.0; };

static inline double sigmoid(double x) {
    if(x < -700.0) { const double e = exp(x); return e / (1.0 + e); }
    return 1.0 / (1.0 + exp(-x));
}

static void fillData(Dataset& data) {
    array<double,W1_SIZE> tw1{};
    array<double,HIDDEN> tw2{};
    for(size_t k=0;k<INPUTS;k++) for(size_t j=0;j<HIDDEN;j++) tw1[k*HIDDEN+j]=0.22*sin(0.17*(k+1)*(j+2));
    for(size_t j=0;j<HIDDEN;j++) tw2[j]=0.28*cos(0.31*(j+1));
    for(size_t i=0;i<data.rows;i++) {
        double* xr=data.rowData(i);
        for(size_t k=0;k<INPUTS;k++) xr[k]=0.55*sin(0.013*(i+1)*(k+1))+0.35*cos(0.007*(i+3)*(k+2));
        double z3=0.0;
        for(size_t j=0;j<HIDDEN;j++) {
            double z2=0.0;
            for(size_t k=0;k<INPUTS;k++) z2+=xr[k]*tw1[k*HIDDEN+j];
            z3+=sigmoid(z2)*tw2[j];
        }
        data.y[i]=sigmoid(z3);
    }
}

static void initWeights(Network& net) {
    for(size_t k=0;k<INPUTS;k++) for(size_t j=0;j<HIDDEN;j++) net.w1[k*HIDDEN+j]=0.12*sin(0.43*(k+1)*(j+1));
    for(size_t j=0;j<HIDDEN;j++) net.w2[j]=0.15*cos(0.37*(j+1));
    net.velocityW1.fill(0.0);
    net.velocityW2.fill(0.0);
}

static Metrics fusedGradientsAndMetrics(const Dataset& data, const Network& net, Gradients& gradients) {
    gradients.clear();
    double squaredError=0.0, percentageErrorSum=0.0, maxPercentageError=0.0;

    for(size_t i=0;i<data.rows;i++) {
        const double* x=data.rowData(i);
        array<double,HIDDEN> hidden{};
        array<double,HIDDEN> delta2{};

        for(size_t k=0;k<INPUTS;k++) {
            const double xv=x[k];
            const double* wr=net.w1.data()+k*HIDDEN;
            for(size_t j=0;j<HIDDEN;j++) hidden[j]+=xv*wr[j];
        }

        double z3=0.0;
        for(size_t j=0;j<HIDDEN;j++) {
            hidden[j]=sigmoid(hidden[j]);
            z3+=hidden[j]*net.w2[j];
        }
        const double prediction=sigmoid(z3);
        const double actual=data.y[i];
        const double error=prediction-actual;
        squaredError+=error*error;
        if(actual!=0.0) {
            const double percentage=abs(error/actual)*100.0;
            percentageErrorSum+=percentage;
            maxPercentageError=max(maxPercentageError,percentage);
        }

        const double delta3=error*prediction*(1.0-prediction);
        for(size_t j=0;j<HIDDEN;j++) {
            const double a=hidden[j];
            gradients.w2[j]+=a*delta3;
            delta2[j]=delta3*net.w2[j]*a*(1.0-a);
        }
        for(size_t k=0;k<INPUTS;k++) {
            const double xv=x[k];
            double* gr=gradients.w1.data()+k*HIDDEN;
            for(size_t j=0;j<HIDDEN;j++) gr[j]+=xv*delta2[j];
        }
    }

    Metrics m;
    m.cost=0.5*squaredError;
    m.percentageError=percentageErrorSum/static_cast<double>(data.rows);
    m.maxPercentageError=maxPercentageError;
    return m;
}

static inline void applyUpdate(Network& net,const Gradients& gradients,double lr,double momentum) {
    for(size_t i=0;i<W1_SIZE;i++) {
        net.velocityW1[i]=momentum*net.velocityW1[i]-lr*gradients.w1[i];
        net.w1[i]+=net.velocityW1[i];
    }
    for(size_t j=0;j<HIDDEN;j++) {
        net.velocityW2[j]=momentum*net.velocityW2[j]-lr*gradients.w2[j];
        net.w2[j]+=net.velocityW2[j];
    }
}

int main(int argc,char** argv) {
    const size_t rows=argc>1?strtoull(argv[1],nullptr,10):13853;
    const size_t updates=argc>2?strtoull(argv[2],nullptr,10):10;
    const size_t reps=argc>3?strtoull(argv[3],nullptr,10):1;
    const double lr=0.0001/static_cast<double>(rows), momentum=0.75;
    Dataset data(rows); fillData(data); Network net; Gradients gradients;
    for(size_t rep=0;rep<reps;rep++) {
        initWeights(net); Metrics m;
        auto start=chrono::steady_clock::now();
        for(size_t u=0;u<updates;u++) { m=fusedGradientsAndMetrics(data,net,gradients); applyUpdate(net,gradients,lr,momentum); }
        auto end=chrono::steady_clock::now();
        const double secs=chrono::duration<double>(end-start).count();
        double checksum=0.0; for(double v:net.w1) checksum+=v; for(double v:net.w2) checksum+=v;
        cout<<setprecision(17)<<"version=v3 rows="<<rows<<" updates="<<updates<<" rep="<<rep<<" seconds="<<secs<<" cost="<<m.cost<<" pct="<<m.percentageError<<" max="<<m.maxPercentageError<<" checksum="<<checksum<<"\n";
    }
    return 0;
}
