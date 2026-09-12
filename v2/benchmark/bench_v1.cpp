#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

using Matrix = vector<vector<double>>;

static double sigmoidScalar(double x) { return 1.0 / (1.0 + exp(-x)); }

static void fillData(Matrix& x, Matrix& y) {
    const size_t rows = x.size();
    const size_t inputs = x[0].size();
    const size_t hidden = 16;
    vector<vector<double>> tw1(inputs, vector<double>(hidden));
    vector<double> tw2(hidden);
    for (size_t k=0;k<inputs;k++) for (size_t j=0;j<hidden;j++) tw1[k][j] = 0.22*sin(0.17*(k+1)*(j+2));
    for (size_t j=0;j<hidden;j++) tw2[j] = 0.28*cos(0.31*(j+1));
    for (size_t i=0;i<rows;i++) {
        for (size_t k=0;k<inputs;k++) x[i][k] = 0.55*sin(0.013*(i+1)*(k+1)) + 0.35*cos(0.007*(i+3)*(k+2));
        double z3 = 0.0;
        for (size_t j=0;j<hidden;j++) {
            double z2 = 0.0;
            for (size_t k=0;k<inputs;k++) z2 += x[i][k]*tw1[k][j];
            z3 += sigmoidScalar(z2)*tw2[j];
        }
        y[i][0] = sigmoidScalar(z3);
    }
}

static void initWeights(Matrix& w1, Matrix& w2) {
    for (size_t k=0;k<w1.size();k++) for (size_t j=0;j<w1[0].size();j++) w1[k][j] = 0.12*sin(0.43*(k+1)*(j+1));
    for (size_t j=0;j<w2.size();j++) w2[j][0] = 0.15*cos(0.37*(j+1));
}

static void matrixmultiply(Matrix &c, Matrix &a, Matrix &b) {
    size_t m=a.size(), n=a[0].size(), p=b.size(), q=b[0].size();
    if (n != p) abort();
    for(size_t i=0;i<m;i++){
        for(size_t j=0;j<q;j++) c[i][j]=0.0;
        for(size_t t=0;t<n;t++){
            const double a_it=a[i][t];
            const vector<double>& b_row=b[t];
            for(size_t j=0;j<q;j++) c[i][j]+=a_it*b_row[j];
        }
    }
}
static void matrixadd(Matrix &c, Matrix &a, Matrix &b){ for(size_t i=0;i<a.size();i++) for(size_t j=0;j<a[0].size();j++) c[i][j]=a[i][j]+b[i][j]; }
static void matrixsubtract(Matrix &c, Matrix &a, Matrix &b){ for(size_t i=0;i<a.size();i++) for(size_t j=0;j<a[0].size();j++) c[i][j]=a[i][j]-b[i][j]; }
static void matrixScaler(Matrix &c,double a){ for(size_t i=0;i<c.size();i++) for(size_t j=0;j<c[0].size();j++) c[i][j]=a*c[i][j]; }
static void sigmoidAndPrime(Matrix &z, Matrix &a, Matrix &aprime){ for(size_t i=0;i<z.size();i++) for(size_t j=0;j<z[0].size();j++){ const double sig=1.0/(1.0+exp(-z[i][j])); a[i][j]=sig; aprime[i][j]=sig*(1.0-sig);} }
static void matrixTranspose(Matrix &b, Matrix &c){ for(size_t i=0;i<b.size();i++) for(size_t j=0;j<b[0].size();j++) b[i][j]=c[j][i]; }
static void hadamardproduct(Matrix &c, Matrix &a, Matrix &b){ for(size_t i=0;i<a.size();i++) for(size_t j=0;j<a[0].size();j++) c[i][j]=a[i][j]*b[i][j]; }
static void costfunc(double &costval, Matrix &a, Matrix &b){ double rr=0.0; for(size_t i=0;i<a.size();i++){ double k=a[i][0]-b[i][0]; rr+=k*k; } costval=0.5*rr; }
static void perError(size_t r,double &c,Matrix &a,Matrix &b){ double temp=0.0; for(size_t i=0;i<a.size();i++){ double diff=(a[i][0]==0.0)?0.0:abs((a[i][0]-b[i][0])/a[i][0])*100.0; temp+=diff;} c=temp/static_cast<double>(r); }
static void errordiff(Matrix &c,Matrix &a,Matrix &b){ for(size_t i=0;i<a.size();i++) c[i][0]=(a[i][0]==0.0)?0.0:(abs(a[i][0]-b[i][0])/a[i][0])*100.0; }
static void MaxCol(double &c, Matrix &a){ double mx=a[0][0]; for(size_t i=1;i<a.size();i++) if(a[i][0]>mx) mx=a[i][0]; c=mx; }

int main(int argc,char** argv){
    const size_t rows = argc>1 ? strtoull(argv[1],nullptr,10) : 13853;
    const size_t updates = argc>2 ? strtoull(argv[2],nullptr,10) : 10;
    const size_t reps = argc>3 ? strtoull(argv[3],nullptr,10) : 1;
    const size_t n=11, s=16;
    const double learningRate = 0.0001 / static_cast<double>(rows);
    const double momentum = 0.75;

    Matrix x(rows,vector<double>(n));
    Matrix y(rows,vector<double>(1));
    fillData(x,y);

    Matrix x_transpose(n,vector<double>(rows));
    Matrix y_bar(rows,vector<double>(1));
    Matrix yError(rows,vector<double>(1));
    Matrix w_one(n,vector<double>(s));
    Matrix djdw_one(n,vector<double>(s));
    Matrix deltawone(n,vector<double>(s));
    Matrix w_two(s,vector<double>(1));
    Matrix djdw_two(s,vector<double>(1));
    Matrix deltawtwo(s,vector<double>(1));
    Matrix w_twoTranspose(1,vector<double>(s));
    Matrix z_two(rows,vector<double>(s));
    Matrix z_twoprime(rows,vector<double>(s));
    Matrix a_two(rows,vector<double>(s));
    Matrix a_twoTranspose(s,vector<double>(rows));
    Matrix z_three(rows,vector<double>(1));
    Matrix z_threeprime(rows,vector<double>(1));
    Matrix delta_two(rows,vector<double>(s));
    Matrix delta_three(rows,vector<double>(1));
    matrixTranspose(x_transpose,x);

    for(size_t rep=0;rep<reps;rep++){
        initWeights(w_one,w_two);
        for(auto& r:deltawone) fill(r.begin(),r.end(),0.0);
        for(auto& r:deltawtwo) fill(r.begin(),r.end(),0.0);
        double cost=0.0, percentageError=0.0, maxdiff=0.0;
        auto start=chrono::steady_clock::now();
        for(size_t count=0;count<updates;count++){
            matrixmultiply(z_two,x,w_one);
            sigmoidAndPrime(z_two,a_two,z_twoprime);
            matrixTranspose(a_twoTranspose,a_two);
            matrixmultiply(z_three,a_two,w_two);
            sigmoidAndPrime(z_three,y_bar,z_threeprime);
            costfunc(cost,y_bar,y);
            perError(rows,percentageError,y,y_bar);
            errordiff(yError,y,y_bar);
            MaxCol(maxdiff,yError);
            matrixsubtract(delta_three,y_bar,y);
            hadamardproduct(delta_three,delta_three,z_threeprime);
            matrixmultiply(djdw_two,a_twoTranspose,delta_three);
            matrixTranspose(w_twoTranspose,w_two);
            matrixmultiply(delta_two,delta_three,w_twoTranspose);
            hadamardproduct(delta_two,delta_two,z_twoprime);
            matrixmultiply(djdw_one,x_transpose,delta_two);
            matrixScaler(djdw_one,learningRate);
            matrixScaler(djdw_two,learningRate);
            matrixScaler(deltawone,momentum);
            matrixScaler(deltawtwo,momentum);
            matrixsubtract(deltawone,deltawone,djdw_one);
            matrixsubtract(deltawtwo,deltawtwo,djdw_two);
            matrixadd(w_one,w_one,deltawone);
            matrixadd(w_two,w_two,deltawtwo);
        }
        auto end=chrono::steady_clock::now();
        double secs=chrono::duration<double>(end-start).count();
        double checksum=0.0;
        for(auto &r:w_one) for(double v:r) checksum+=v;
        for(auto &r:w_two) checksum+=r[0];
        cout<<setprecision(17)<<"version=v1 rows="<<rows<<" updates="<<updates<<" rep="<<rep<<" seconds="<<secs<<" cost="<<cost<<" pct="<<percentageError<<" max="<<maxdiff<<" checksum="<<checksum<<"\n";
    }
    return 0;
}
