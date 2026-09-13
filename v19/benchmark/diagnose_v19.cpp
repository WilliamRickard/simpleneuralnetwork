#define main v18_embedded_main
#include "../../v18/main.cpp"
#undef main

#include <cstdlib>
#include <limits>

namespace {

struct DeepRunConfig {
    size_t rows=20000;
    size_t historyLimit=10;
    double target=0.005;
    size_t maxUpdates=3000;
    size_t diagnosticsEvery=100;
    size_t threads=4;
};

struct CrossingCounts {
    size_t pe002=0;
    size_t pe001=0;
    size_t pe0005=0;
    size_t pe0001=0;
};

static double sigmoidScalar(double value) {
    return 1.0/(1.0+std::exp(-value));
}

static Dataset makeBenchmarkData(size_t rows) {
    Dataset data;
    data.x=Matrix(rows,NUMBER_OF_VARIABLES);
    data.y.resize(rows);
    std::array<double,WONE_SIZE> generatingW1{};
    std::array<double,HIDDEN_NODES> generatingW2{};
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
        for(size_t j=0;j<HIDDEN_NODES;j++) generatingW1[k*HIDDEN_NODES+j]=0.22*std::sin(0.17*static_cast<double>(k+1)*static_cast<double>(j+2));
    }
    for(size_t j=0;j<HIDDEN_NODES;j++) generatingW2[j]=0.28*std::cos(0.31*static_cast<double>(j+1));
    for(size_t row=0;row<rows;row++) {
        double*x=data.x.rowData(row);
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            x[k]=0.55*std::sin(0.013*static_cast<double>(row+1)*static_cast<double>(k+1))
                +0.35*std::cos(0.007*static_cast<double>(row+3)*static_cast<double>(k+2));
        }
        double outputRaw=0.0;
        for(size_t j=0;j<HIDDEN_NODES;j++) {
            double hiddenRaw=0.0;
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) hiddenRaw+=x[k]*generatingW1[k*HIDDEN_NODES+j];
            outputRaw+=sigmoidScalar(hiddenRaw)*generatingW2[j];
        }
        data.y[row]=sigmoidScalar(outputRaw);
    }
    return data;
}

static Network makeInitialNetwork() {
    Network network;
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
        for(size_t j=0;j<HIDDEN_NODES;j++) network.wOne[k*HIDDEN_NODES+j]=0.12*std::sin(0.43*static_cast<double>(k+1)*static_cast<double>(j+1));
    }
    for(size_t j=0;j<HIDDEN_NODES;j++) network.wTwo[j]=0.15*std::cos(0.37*static_cast<double>(j+1));
    network.deltaWone.fill(0.0);
    network.deltaWtwo.fill(0.0);
    return network;
}

static double norm2(const std::vector<double>&values) {
    return std::sqrt(dotV14(values,values));
}

static double normInf(const std::vector<double>&values) {
    double result=0.0;
    for(double value:values) result=std::max(result,std::abs(value));
    return result;
}

static double sliceNorm2(const std::vector<double>&values,size_t first,size_t last) {
    double sum=0.0;
    for(size_t i=first;i<last;i++) sum+=values[i]*values[i];
    return std::sqrt(sum);
}

static double safeCosine(double dot,double normA,double normB) {
    if(!(normA>0.0)||!(normB>0.0)) return 0.0;
    return dot/(normA*normB);
}

static double maxHistoryStepCosine(const std::deque<V14HistoryPair>&history,const std::vector<double>&stepVector,double stepNorm) {
    double maximum=0.0;
    if(!(stepNorm>0.0)) return maximum;
    for(const V14HistoryPair&pair:history) {
        const double pairNorm=norm2(pair.s);
        if(pairNorm>0.0) maximum=std::max(maximum,std::abs(dotV14(pair.s,stepVector)/(pairNorm*stepNorm)));
    }
    return maximum;
}

static void recordCrossings(CrossingCounts&crossings,double percentageError,size_t evaluations) {
    if(crossings.pe002==0&&percentageError<=0.02) crossings.pe002=evaluations;
    if(crossings.pe001==0&&percentageError<=0.01) crossings.pe001=evaluations;
    if(crossings.pe0005==0&&percentageError<=0.005) crossings.pe0005=evaluations;
    if(crossings.pe0001==0&&percentageError<=0.001) crossings.pe0001=evaluations;
}

static void printInitialDiagnostic(const DeepRunConfig&config,const V14Evaluation&current) {
    const double gradientNorm=norm2(current.gradient);
    std::cout<<std::setprecision(17)
             <<"DIAG rows="<<config.rows
             <<" history="<<config.historyLimit
             <<" update=0 objective="<<current.objective
             <<" pe="<<current.metrics.percentageError
             <<" grad_l2="<<gradientNorm
             <<" grad_inf="<<normInf(current.gradient)
             <<" grad_w1_l2="<<sliceNorm2(current.gradient,0,WONE_SIZE)
             <<" grad_w2_l2="<<sliceNorm2(current.gradient,WONE_SIZE,V14_PARAMETER_COUNT)
             <<'\n';
}

static void printStepDiagnostic(const DeepRunConfig&config,size_t updates,const V14Evaluation&current,
                                const std::vector<double>&direction,double directional,double step,size_t lineSearchEvaluations,
                                double sy,double sNorm,double yNorm,double maxStepCosine,bool curvatureAccepted,size_t historySize) {
    const double gradientNorm=norm2(current.gradient);
    const double directionNorm=norm2(direction);
    std::cout<<std::setprecision(17)
             <<"DIAG rows="<<config.rows
             <<" history="<<config.historyLimit
             <<" update="<<updates
             <<" objective="<<current.objective
             <<" pe="<<current.metrics.percentageError
             <<" grad_l2="<<gradientNorm
             <<" grad_inf="<<normInf(current.gradient)
             <<" grad_w1_l2="<<sliceNorm2(current.gradient,0,WONE_SIZE)
             <<" grad_w2_l2="<<sliceNorm2(current.gradient,WONE_SIZE,V14_PARAMETER_COUNT)
             <<" sy="<<sy
             <<" s_norm="<<sNorm
             <<" y_norm="<<yNorm
             <<" curvature_cos="<<safeCosine(sy,sNorm,yNorm)
             <<" step="<<step
             <<" ls_evals="<<lineSearchEvaluations
             <<" history_size="<<historySize
             <<" directional="<<directional
             <<" direction_cos="<<safeCosine(-directional,gradientNorm,directionNorm)
             <<" max_history_step_cos="<<maxStepCosine
             <<" curvature_accepted="<<(curvatureAccepted?1:0)
             <<'\n';
}

static int runHistorySweep(const DeepRunConfig&config) {
    Dataset data=makeBenchmarkData(config.rows);
    Network network=makeInitialNetwork();
    std::deque<V14HistoryPair>history;
    std::vector<double>parameters=packV14(network);
    V14Evaluation current=evaluateV17(data,0,config.rows,network,parameters,config.threads);
    size_t evaluations=1;
    size_t lineSearchEvaluations=0;
    size_t updates=0;
    size_t nonDescentResets=0;
    size_t rejectedCurvaturePairs=0;
    CrossingCounts crossings;
    recordCrossings(crossings,current.metrics.percentageError,evaluations);
    printInitialDiagnostic(config,current);
    const auto started=std::chrono::steady_clock::now();

    while(current.metrics.percentageError>config.target&&updates<config.maxUpdates) {
        std::vector<double>direction=directionV14(current.gradient,history);
        double directional=dotV14(current.gradient,direction);
        if(!(directional<0.0)) {
            direction=current.gradient;
            for(double&value:direction) value=-value;
            directional=-dotV14(current.gradient,current.gradient);
            history.clear();
            nonDescentResets++;
        }

        double step=1.0;
        bool accepted=false;
        std::vector<double>candidate(parameters.size());
        V14Evaluation next;
        size_t thisLineSearchEvaluations=0;
        for(size_t search=0;search<V14_MAX_LINE_SEARCH;search++) {
            for(size_t k=0;k<parameters.size();k++) candidate[k]=parameters[k]+step*direction[k];
            next=evaluateV17(data,0,config.rows,network,candidate,config.threads);
            evaluations++;
            lineSearchEvaluations++;
            thisLineSearchEvaluations++;
            if(next.objective<=current.objective+V14_ARMIJO*step*directional) {
                accepted=true;
                break;
            }
            step*=0.5;
        }
        if(!accepted) break;

        std::vector<double>s(parameters.size()),y(parameters.size());
        for(size_t k=0;k<parameters.size();k++) {
            s[k]=candidate[k]-parameters[k];
            y[k]=next.gradient[k]-current.gradient[k];
        }
        const double sy=dotV14(s,y);
        const double sNorm=norm2(s);
        const double yNorm=norm2(y);
        const double maxStepCosine=maxHistoryStepCosine(history,s,sNorm);
        const bool curvatureAccepted=acceptCurvatureV18(s,y,sy);
        if(curvatureAccepted) {
            if(history.size()==config.historyLimit) history.pop_front();
            V14HistoryPair pair;
            pair.s.swap(s);
            pair.y.swap(y);
            pair.rho=1.0/sy;
            history.push_back(std::move(pair));
        } else {
            rejectedCurvaturePairs++;
        }
        parameters.swap(candidate);
        current=std::move(next);
        updates++;
        recordCrossings(crossings,current.metrics.percentageError,evaluations);
        if(config.diagnosticsEvery!=0&&(updates%config.diagnosticsEvery==0||current.metrics.percentageError<=config.target)) {
            printStepDiagnostic(config,updates,current,direction,directional,step,thisLineSearchEvaluations,sy,sNorm,yNorm,maxStepCosine,curvatureAccepted,history.size());
        }
    }

    unpackV14(parameters,network);
    const Metrics exact=confirmMetricsV17(data,0,config.rows,network,config.threads);
    const auto finished=std::chrono::steady_clock::now();
    const double seconds=std::chrono::duration<double>(finished-started).count();
    const double exactObjective=exact.cost/static_cast<double>(config.rows);
    const bool reached=exact.percentageError<=config.target;
    std::cout<<std::setprecision(17)
             <<"SUMMARY rows="<<config.rows
             <<" history="<<config.historyLimit
             <<" target="<<config.target
             <<" reached="<<(reached?1:0)
             <<" updates="<<updates
             <<" evaluations="<<evaluations
             <<" line_search_evaluations="<<lineSearchEvaluations
             <<" final_confirmation_passes=1"
             <<" total_row_passes="<<(evaluations+1)
             <<" wall_seconds="<<seconds
             <<" final_pe="<<exact.percentageError
             <<" final_objective="<<exactObjective
             <<" eval_to_0.02="<<crossings.pe002
             <<" eval_to_0.01="<<crossings.pe001
             <<" eval_to_0.005="<<crossings.pe0005
             <<" eval_to_0.001="<<crossings.pe0001
             <<" non_descent_resets="<<nonDescentResets
             <<" rejected_curvature_pairs="<<rejectedCurvaturePairs
             <<'\n';
    return reached?0:2;
}

static size_t parseSize(const char*value,const char*name) {
    char*end=nullptr;
    const unsigned long long parsed=std::strtoull(value,&end,10);
    if(value==end||*end!='\0') throw std::runtime_error(std::string("Invalid ")+name);
    return static_cast<size_t>(parsed);
}

static double parseDouble(const char*value,const char*name) {
    char*end=nullptr;
    const double parsed=std::strtod(value,&end);
    if(value==end||*end!='\0'||!std::isfinite(parsed)) throw std::runtime_error(std::string("Invalid ")+name);
    return parsed;
}

} // namespace

int main(int argc,char**argv) {
    try {
        DeepRunConfig config;
        if(argc>1) config.rows=parseSize(argv[1],"rows");
        if(argc>2) config.historyLimit=parseSize(argv[2],"history");
        if(argc>3) config.target=parseDouble(argv[3],"target");
        if(argc>4) config.maxUpdates=parseSize(argv[4],"maxUpdates");
        if(argc>5) config.diagnosticsEvery=parseSize(argv[5],"diagnosticsEvery");
        if(config.rows==0||config.historyLimit==0||config.target<=0.0||config.maxUpdates==0) throw std::runtime_error("Arguments must be positive");
#ifndef _OPENMP
        config.threads=1;
#endif
        return runHistorySweep(config);
    } catch(const std::exception&error) {
        std::cerr<<"Error: "<<error.what()<<'\n';
        return 1;
    }
}
