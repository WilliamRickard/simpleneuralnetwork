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

#include <array>
#include <cstdio>
#include <iomanip>
#include <sstream>

/** Deterministic teacher used by the v29 full-training release matrix. */
static Network teacherNetworkV30(){
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
static void fillTeacherTargetsV30(Dataset&data,const Network&teacher){
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

static Dataset trainingDataV30(size_t rows){
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
    fillTeacherTargetsV30(data,teacherNetworkV30());
#else
    throw runtime_error("v30 profiler requires the accelerated production path");
#endif
    return data;
}

static Network initialNetworkV30(){
    Network network=teacherNetworkV30();
    for(size_t i=0;i<network.wOne.size();i++)
        network.wOne[i]+=.045*sin(.19*static_cast<double>(i+1))
                        +.012*cos(.071*static_cast<double>(i+3));
    for(size_t i=0;i<network.wTwo.size();i++)
        network.wTwo[i]+=.055*cos(.23*static_cast<double>(i+1));
    return network;
}

static bool sameResultV30(const TrainResult&a,const TrainResult&b,
                          const Network&na,const Network&nb){
    if(a.updates!=b.updates||a.reachedTarget!=b.reachedTarget)return false;
    if(a.metrics.cost!=b.metrics.cost||
       a.metrics.percentageError!=b.metrics.percentageError||
       a.metrics.maxPercentageError!=b.metrics.maxPercentageError)return false;
    const vector<double>pa=packV14(na),pb=packV14(nb);
    return pa.size()==pb.size()&&
           memcmp(pa.data(),pb.data(),pa.size()*sizeof(double))==0;
}

/** Wall-clock buckets for one exact v29 training trajectory. */
struct V30Profile {
    double total=0.0;
    double initialEvaluation=0.0;
    double deepState=0.0;
    double direction=0.0;
    double candidateAllocation=0.0;
    double lineCandidate=0.0;
    double lineEvaluation=0.0;
    double lineLogic=0.0;
    double curvature=0.0;
    double stateCommit=0.0;
    double confirmation=0.0;
    double finalProgress=0.0;
    double predictions=0.0;
    size_t evaluationCalls=0;
    size_t searchAttempts=0;
    size_t denseUpdates=0;
    size_t limitedMemoryUpdates=0;
    std::array<size_t,V14_MAX_LINE_SEARCH+1> attemptsPerAcceptedUpdate{};
};

using V30Clock=chrono::steady_clock;

static double elapsedV30(const V30Clock::time_point&start){
    return chrono::duration<double>(V30Clock::now()-start).count();
}

/** Profile the exact v29 Armijo loop without changing its floating-point expressions. */
static bool profiledArmijoV30(const Dataset&data,size_t startRow,size_t batchSize,Network&network,
                              const OptimiserConfig&config,const vector<double>&parameters,
                              const vector<double>&direction,double directional,double currentObjective,
                              bool denseReady,vector<double>&candidate,V14Evaluation&next,
                              V30Profile&profile,size_t&attempts){
    double step=1.0;
    attempts=0;
    for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++){
        attempts++;
        profile.searchAttempts++;

        auto started=V30Clock::now();
        for(size_t k=0;k<parameters.size();k++)
            candidate[k]=parameters[k]+step*direction[k];
        profile.lineCandidate+=elapsedV30(started);

        started=V30Clock::now();
        next=evaluateV29(data,startRow,batchSize,network,candidate,config.threads);
        profile.lineEvaluation+=elapsedV30(started);
        profile.evaluationCalls++;

        started=V30Clock::now();
        if(next.objective<=currentObjective+V14_ARMIJO*step*directional){
            profile.lineLogic+=elapsedV30(started);
            return true;
        }

        if(denseReady){
            const double denominator=2.0*(next.objective-currentObjective-step*directional);
            if(denominator>0.0&&isfinite(denominator)){
                const double quadratic=-directional*step*step/denominator;
                if(quadratic>0.0&&isfinite(quadratic)){
                    const double lower=V24_QUADRATIC_MIN_FRACTION*step;
                    const double upper=V24_QUADRATIC_MAX_FRACTION*step;
                    step=max(lower,min(upper,quadratic));
                    profile.lineLogic+=elapsedV30(started);
                    continue;
                }
            }
        }
        step*=0.5;
        profile.lineLogic+=elapsedV30(started);
    }
    return false;
}

/**
 * Duplicate v29's optimiser control flow solely to measure its phase costs.
 * The caller verifies the resulting trajectory against production trainRangeV29.
 */
static TrainResult profileTrainRangeV30(const Dataset&data,size_t startRow,size_t batchSize,
                                        Network&network,const OptimiserConfig&config,
                                        V30Profile&profile){
    const auto totalStarted=V30Clock::now();
    TrainResult result;
    deque<V14HistoryPair>history;
    vector<double>parameters=packV14(network);

    auto started=V30Clock::now();
    V14Evaluation current=evaluateV29(data,startRow,batchSize,network,parameters,config.threads);
    profile.initialEvaluation+=elapsedV30(started);
    profile.evaluationCalls++;

    bool deepStage=false,scaleReady=false,denseReady=false;
    unsigned refreshStage=0;
    V20DiagonalScale scale{};
    vector<double>denseInverse;
    if(config.logEvery!=0)printProgress(0,0,current.metrics);

    while(current.metrics.percentageError>config.percentageErrorTarget&&result.updates<config.maxDescents){
        started=V30Clock::now();
        prepareDeepStateV28(data,startRow,batchSize,network,parameters,config,
                            current.metrics.percentageError,history,deepStage,scaleReady,denseReady,
                            refreshStage,scale,denseInverse);
        profile.deepState+=elapsedV30(started);

        started=V30Clock::now();
        const size_t historyLimit=historyLimitV19(deepStage);
        vector<double>direction;
        double directional=0.0;
        buildDirectionV27(current.gradient,history,scale,scaleReady,denseReady,
                          denseInverse,direction,directional);
        profile.direction+=elapsedV30(started);

        started=V30Clock::now();
        vector<double>candidate(parameters.size());
        V14Evaluation next;
        profile.candidateAllocation+=elapsedV30(started);

        size_t attempts=0;
        const bool accepted=profiledArmijoV30(
            data,startRow,batchSize,network,config,parameters,direction,directional,current.objective,
            denseReady,candidate,next,profile,attempts);
        if(!accepted){
            unpackV14(parameters,network);
            break;
        }
        if(attempts<profile.attemptsPerAcceptedUpdate.size())
            profile.attemptsPerAcceptedUpdate[attempts]++;
        if(denseReady)profile.denseUpdates++;else profile.limitedMemoryUpdates++;

        started=V30Clock::now();
        updateCurvatureV27(parameters,candidate,current.gradient,next.gradient,historyLimit,
                           denseReady,history,denseInverse);
        profile.curvature+=elapsedV30(started);

        started=V30Clock::now();
        parameters.swap(candidate);
        current=std::move(next);
        result.updates++;
        profile.stateCommit+=elapsedV30(started);
        if(config.logEvery!=0&&result.updates%config.logEvery==0)
            printProgress(0,result.updates,current.metrics);
    }

    unpackV14(parameters,network);
    network.deltaWone.fill(0.0);
    network.deltaWtwo.fill(0.0);

    started=V30Clock::now();
    result.metrics=confirmMetricsV25(data,startRow,batchSize,network,config.threads);
    profile.confirmation+=elapsedV30(started);
    result.reachedTarget=result.metrics.percentageError<=config.percentageErrorTarget;

    started=V30Clock::now();
    if(config.logEvery!=0||result.reachedTarget)
        printProgress(0,result.updates,result.metrics);
    profile.finalProgress+=elapsedV30(started);

    started=V30Clock::now();
    writePredictions(data,startRow,batchSize,network,"v30-profile-ybar.txt");
    profile.predictions+=elapsedV30(started);
    std::remove("v30-profile-ybar.txt");

    profile.total=elapsedV30(totalStarted);
    return result;
}

static double percentageV30(double value,double total){
    return total==0.0?0.0:100.0*value/total;
}

int main(int argc,char**argv){
    if(!supportsV16Kernel()){
        cout<<"SKIP: AVX-512/libmvec production kernel unavailable\n";
        return 75;
    }
    const size_t rows=argc>1?strtoull(argv[1],nullptr,10):100000;
    const double target=argc>2?atof(argv[2]):.0005;
    const size_t threads=argc>3?strtoull(argv[3],nullptr,10):2;
    if(rows==0||threads==0)return 2;

    Dataset data=trainingDataV30(rows);
    const Network initial=initialNetworkV30();
    OptimiserConfig config;
    config.maxDescents=1000000;
    config.percentageErrorTarget=target;
    config.logEvery=0;
    config.threads=threads;

    Network productionNetwork=initial;
    TrainResult productionResult;
    ostringstream productionSink;
    streambuf*old=cout.rdbuf(productionSink.rdbuf());
    productionResult=trainRangeV29(data,0,rows,0,productionNetwork,config);
    cout.rdbuf(old);
    std::remove("ybar.txt");

    Network profiledNetwork=initial;
    V30Profile profile;
    TrainResult profiledResult;
    ostringstream profileSink;
    old=cout.rdbuf(profileSink.rdbuf());
    profiledResult=profileTrainRangeV30(data,0,rows,profiledNetwork,config,profile);
    cout.rdbuf(old);

    if(!sameResultV30(productionResult,profiledResult,productionNetwork,profiledNetwork)){
        cerr<<"FAIL: profiled v29 trajectory differs from production\n";
        return 3;
    }

    const double accounted=profile.initialEvaluation+profile.deepState+profile.direction+
        profile.candidateAllocation+profile.lineCandidate+profile.lineEvaluation+profile.lineLogic+
        profile.curvature+profile.stateCommit+profile.confirmation+profile.finalProgress+
        profile.predictions;
    const double other=max(0.0,profile.total-accounted);

    cout<<setprecision(12);
    cout<<"PROFILE rows="<<rows<<" target="<<target<<" threads="<<threads
        <<" updates="<<profiledResult.updates
        <<" eval_calls="<<profile.evaluationCalls
        <<" search_attempts="<<profile.searchAttempts
        <<" evals_per_update="<<(static_cast<double>(profile.searchAttempts)/profiledResult.updates)
        <<" dense_updates="<<profile.denseUpdates
        <<" lm_updates="<<profile.limitedMemoryUpdates<<'\n';

    const auto printBucket=[&](const char*name,double value){
        cout<<"BUCKET "<<name<<" seconds="<<value
            <<" pct="<<percentageV30(value,profile.total)<<'\n';
    };
    printBucket("initial_eval",profile.initialEvaluation);
    printBucket("deep_state",profile.deepState);
    printBucket("direction",profile.direction);
    printBucket("candidate_alloc",profile.candidateAllocation);
    printBucket("line_candidate",profile.lineCandidate);
    printBucket("line_eval",profile.lineEvaluation);
    printBucket("line_logic",profile.lineLogic);
    printBucket("curvature",profile.curvature);
    printBucket("state_commit",profile.stateCommit);
    printBucket("confirmation",profile.confirmation);
    printBucket("final_progress",profile.finalProgress);
    printBucket("predictions",profile.predictions);
    printBucket("other",other);
    cout<<"TOTAL seconds="<<profile.total<<'\n';

    for(size_t attempts=1;attempts<profile.attemptsPerAcceptedUpdate.size();attempts++)
        if(profile.attemptsPerAcceptedUpdate[attempts]!=0)
            cout<<"SEARCH attempts="<<attempts
                <<" accepted_updates="<<profile.attemptsPerAcceptedUpdate[attempts]<<'\n';
    return 0;
}
