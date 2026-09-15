#define SIMPLE_NN_V28_NO_MAIN
#include "../main.cpp"

#include <cstring>

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)

static Dataset makeV28BenchmarkData(size_t rows,unsigned seed){
    Dataset data;
    data.x=Matrix(rows,NUMBER_OF_VARIABLES);
    data.y.resize(rows);
    const double s=static_cast<double>(seed+1);
    for(size_t r=0;r<rows;r++){
        double sum=0.0;
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            const double value=.55*sin(.013*(r+1.0)*(k+1.0+s))
                              +.35*cos(.007*(r+3.0+s)*(k+2.0));
            data.x.data[r*NUMBER_OF_VARIABLES+k]=value;
            sum+=value*(.03+.004*static_cast<double>(k));
        }
        const double base=.2+.6/(1.0+exp(-sum));
        data.y[r]=((r+seed*17u)%97u==0u)?0.0:base;
    }
    return data;
}

static Network makeV28BenchmarkNetwork(unsigned seed){
    Network network;
    const double s=static_cast<double>(seed+1);
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++)
        for(size_t j=0;j<HIDDEN_NODES;j++)
            network.wOne[k*HIDDEN_NODES+j]=
                .12*sin(.43*(k+1.0)*(j+1.0)+.017*s)
               +.03*cos(.11*(k+2.0+s)*(j+3.0));
    for(size_t j=0;j<HIDDEN_NODES;j++)
        network.wTwo[j]=.15*cos(.37*(j+1.0)+.023*s);
    return network;
}

static bool sameV28Partial(const V15ThreadEvaluation&a,const V15ThreadEvaluation&b){
    return a.squaredError==b.squaredError
        &&a.percentageErrorSum==b.percentageErrorSum
        &&a.maxPercentageError==b.maxPercentageError
        &&memcmp(a.gradient.data(),b.gradient.data(),sizeof(double)*V14_PARAMETER_COUNT)==0;
}

int main(){
    if(!supportsV16Kernel()){
        cout<<"SKIP: AVX-512 production kernel unavailable on this host\n";
        return 0;
    }

    const size_t rowCounts[]={1,7,8,9,127,133,999,1000,2048,3072,5000,10000,10003,13300,20005,50000,100000};
    size_t cases=0;
    for(unsigned seed=0;seed<5;seed++){
        const size_t maxRows=100000;
        Dataset data=makeV28BenchmarkData(maxRows,seed);
        const Network network=makeV28BenchmarkNetwork(seed);

        for(size_t ri=0;ri<sizeof(rowCounts)/sizeof(rowCounts[0]);ri++){
            const size_t rows=rowCounts[ri];
            for(size_t workers=1;workers<=5;workers++){
                for(size_t tid=0;tid<workers;tid++){
                    const size_t base=rows/workers,remainder=rows%workers;
                    const size_t first=tid*base+min(tid,remainder);
                    const size_t count=base+(tid<remainder?1:0);
                    V15ThreadEvaluation baseline,candidate;
                    evaluateSliceV17(data,first,first+count,network,baseline);
                    evaluateSliceV28(data,first,first+count,network,candidate);
                    if(!sameV28Partial(baseline,candidate)){
                        cerr<<"Mismatch: seed="<<seed
                            <<" rows="<<rows
                            <<" workers="<<workers
                            <<" tid="<<tid<<'\n';
                        return 1;
                    }
                }
                cases++;
            }
        }
    }

    cout<<"PASS: "<<cases<<"/"<<cases
        <<" v17-v28 production-kernel cases are exact\n";
    return cases==425?0:2;
}

#else

int main(){
    cout<<"SKIP: build with GCC/glibc x86-64 and SIMPLE_NN_USE_LIBMVEC for the v28 exactness benchmark\n";
    return 0;
}

#endif
