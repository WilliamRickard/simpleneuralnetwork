#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

struct Matrix {
    size_t rows, cols;
    vector<double> data;
    Matrix():rows(0),cols(0){}
    Matrix(size_t r,size_t c,double v=0.0):rows(r),cols(c),data(r*c,v){}
    double* rowData(size_t r){ return data.data()+r*cols; }
    const double* rowData(size_t r) const { return data.data()+r*cols; }
    void fill(double v){ std::fill(data.begin(),data.end(),v); }
};
struct Dataset { Matrix x; vector<double> y; };
struct Network { Matrix w1; vector<double> w2; Matrix velocityW1; vector<double> velocityW2; Network(size_t n,size_t s):w1(n,s),w2(s,0.0),velocityW1(n,s),velocityW2(s,0.0){} };
struct Buffers { Matrix hidden; vector<double> predictions; Matrix gradientW1; vector<double> gradientW2; vector<double> delta2; Buffers(size_t b,size_t n,size_t s):hidden(b,s),predictions(b,0.0),gradientW1(n,s),gradientW2(s,0.0),delta2(s,0.0){} };
struct Metrics { double cost=0.0, percentageError=0.0, maxPercentageError=0.0; };

static double sigmoid(double x){ if(x>=0.0){double e=exp(-x); return 1.0/(1.0+e);} double e=exp(x); return e/(1.0+e); }
static void fillData(Dataset& data) {
    const size_t rows=data.x.rows, inputs=data.x.cols, hidden=16;
    vector<double> tw1(inputs*hidden), tw2(hidden);
    for(size_t k=0;k<inputs;k++) for(size_t j=0;j<hidden;j++) tw1[k*hidden+j]=0.22*sin(0.17*(k+1)*(j+2));
    for(size_t j=0;j<hidden;j++) tw2[j]=0.28*cos(0.31*(j+1));
    for(size_t i=0;i<rows;i++){
        double* xr=data.x.rowData(i);
        for(size_t k=0;k<inputs;k++) xr[k]=0.55*sin(0.013*(i+1)*(k+1))+0.35*cos(0.007*(i+3)*(k+2));
        double z3=0.0;
        for(size_t j=0;j<hidden;j++){
            double z2=0.0;
            for(size_t k=0;k<inputs;k++) z2+=xr[k]*tw1[k*hidden+j];
            z3+=sigmoid(z2)*tw2[j];
        }
        data.y[i]=sigmoid(z3);
    }
}
static void initWeights(Network& net){ for(size_t k=0;k<net.w1.rows;k++) for(size_t j=0;j<net.w1.cols;j++) net.w1.data[k*net.w1.cols+j]=0.12*sin(0.43*(k+1)*(j+1)); for(size_t j=0;j<net.w2.size();j++) net.w2[j]=0.15*cos(0.37*(j+1)); }
static void forwardRange(const Dataset& data,const Network& net,Buffers& b){
    size_t n=net.w1.rows,s=net.w1.cols,rows=b.hidden.rows;
    for(size_t i=0;i<rows;i++){
        const double* x=data.x.rowData(i); double* h=b.hidden.rowData(i); fill(h,h+s,0.0);
        for(size_t k=0;k<n;k++){ const double xv=x[k]; const double* wr=net.w1.rowData(k); for(size_t j=0;j<s;j++) h[j]+=xv*wr[j]; }
        double z3=0.0; for(size_t j=0;j<s;j++){ h[j]=sigmoid(h[j]); z3+=h[j]*net.w2[j]; }
        b.predictions[i]=sigmoid(z3);
    }
}
static Metrics gradientsAndMetrics(const Dataset& data,const Network& net,Buffers& b){
    b.gradientW1.fill(0.0); fill(b.gradientW2.begin(),b.gradientW2.end(),0.0);
    const size_t n=net.w1.rows,s=net.w1.cols,rows=b.hidden.rows;
    double sq=0.0,pctsum=0.0,maxpct=0.0;
    for(size_t i=0;i<rows;i++){
        double actual=data.y[i], pred=b.predictions[i], error=pred-actual; sq+=error*error;
        if(actual!=0.0){ double p=abs(error/actual)*100.0; pctsum+=p; maxpct=max(maxpct,p); }
        double delta3=error*pred*(1.0-pred); const double* h=b.hidden.rowData(i);
        for(size_t j=0;j<s;j++){ double a=h[j]; b.gradientW2[j]+=a*delta3; b.delta2[j]=delta3*net.w2[j]*a*(1.0-a); }
        const double* x=data.x.rowData(i);
        for(size_t k=0;k<n;k++){ double xv=x[k]; double* gr=b.gradientW1.rowData(k); for(size_t j=0;j<s;j++) gr[j]+=xv*b.delta2[j]; }
    }
    Metrics m; m.cost=0.5*sq; m.percentageError=pctsum/static_cast<double>(rows); m.maxPercentageError=maxpct; return m;
}
static void applyUpdate(Network& net,const Buffers& b,double lr,double momentum){
    for(size_t i=0;i<net.w1.data.size();i++){ net.velocityW1.data[i]=momentum*net.velocityW1.data[i]-lr*b.gradientW1.data[i]; net.w1.data[i]+=net.velocityW1.data[i]; }
    for(size_t j=0;j<net.w2.size();j++){ net.velocityW2[j]=momentum*net.velocityW2[j]-lr*b.gradientW2[j]; net.w2[j]+=net.velocityW2[j]; }
}
int main(int argc,char** argv){
    const size_t rows=argc>1?strtoull(argv[1],nullptr,10):13853;
    const size_t updates=argc>2?strtoull(argv[2],nullptr,10):10;
    const size_t reps=argc>3?strtoull(argv[3],nullptr,10):1;
    const size_t n=11,s=16; const double lr=0.0001/static_cast<double>(rows), momentum=0.75;
    Dataset data{Matrix(rows,n),vector<double>(rows)}; fillData(data); Network net(n,s); Buffers b(rows,n,s);
    for(size_t rep=0;rep<reps;rep++){
        initWeights(net);
        fill(net.velocityW1.data.begin(),net.velocityW1.data.end(),0.0);
        fill(net.velocityW2.begin(),net.velocityW2.end(),0.0);
        Metrics m;
        auto start=chrono::steady_clock::now();
        for(size_t u=0;u<updates;u++){ forwardRange(data,net,b); m=gradientsAndMetrics(data,net,b); applyUpdate(net,b,lr,momentum); }
        auto end=chrono::steady_clock::now(); double secs=chrono::duration<double>(end-start).count(); double checksum=0.0; for(double v:net.w1.data) checksum+=v; for(double v:net.w2) checksum+=v;
        cout<<setprecision(17)<<"version=v2 rows="<<rows<<" updates="<<updates<<" rep="<<rep<<" seconds="<<secs<<" cost="<<m.cost<<" pct="<<m.percentageError<<" max="<<m.maxPercentageError<<" checksum="<<checksum<<"\n";
    }
    return 0;
}
