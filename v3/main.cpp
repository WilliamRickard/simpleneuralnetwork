#include<iostream>
#include<algorithm>
#include<array>
#include<chrono>
#include<cmath>
#include<cstddef>
#include<ctime>
#include<exception>
#include<fstream>
#include<iomanip>
#include<random>
#include<stdexcept>
#include<string>
#include<vector>
#ifdef _OPENMP
#include<omp.h>
#endif
using namespace std;

/*
 * Simple Neural Network - v3
 * ==========================
 *
 * V3 keeps the same 11 -> 16 -> 1 sigmoid network and training equations as
 * the original program, while reducing the amount of memory moved during each
 * gradient descent. The main change from v2 is that forward propagation and
 * backpropagation are fused for each observation. Hidden activations are kept
 * in a small 16-value scratch array rather than stored for the full data set.
 *
 * The default build is single-threaded. If compiled with -fopenmp, training can
 * optionally split observations across threads. Each thread accumulates its own
 * gradient and the results are combined in a fixed thread order after the pass.
 */

constexpr size_t NUMBER_OF_VARIABLES = 11;
constexpr size_t HIDDEN_NODES = 16;
constexpr size_t WONE_SIZE = NUMBER_OF_VARIABLES * HIDDEN_NODES;

struct Matrix {
    size_t rows;
    size_t cols;
    vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(size_t r, size_t c, double value = 0.0) : rows(r), cols(c), data(r * c, value) {}

    double *rowData(size_t row) { return data.data() + row * cols; }
    const double *rowData(size_t row) const { return data.data() + row * cols; }
};

struct Dataset {
    Matrix x;
    vector<double> y;
};

struct Metrics {
    double cost = 0.0;
    double percentageError = 0.0;
    double maxPercentageError = 0.0;
};

struct Network {
    array<double,WONE_SIZE> wOne{};
    array<double,HIDDEN_NODES> wTwo{};
    array<double,WONE_SIZE> deltaWone{};
    array<double,HIDDEN_NODES> deltaWtwo{};
};

struct Gradients {
    array<double,WONE_SIZE> dJdWone{};
    array<double,HIDDEN_NODES> dJdWtwo{};

    void clear() {
        dJdWone.fill(0.0);
        dJdWtwo.fill(0.0);
    }
};

struct alignas(64) ThreadAccumulator {
    array<double,WONE_SIZE> dJdWone{};
    array<double,HIDDEN_NODES> dJdWtwo{};
    double squaredError = 0.0;
    double percentageErrorSum = 0.0;
    double maxPercentageError = 0.0;

    void clear() {
        dJdWone.fill(0.0);
        dJdWtwo.fill(0.0);
        squaredError = 0.0;
        percentageErrorSum = 0.0;
        maxPercentageError = 0.0;
    }
};

struct OptimiserConfig {
    size_t maxDescents = 1000000;
    size_t logEvery = 1000;
    size_t threads = 1;
    double percentageErrorTarget = 3.9;
    double learningRate = 0.0001;
    double momentum = 0.75;
};

struct TrainResult {
    Metrics metrics;
    size_t updates = 0;
    bool reachedTarget = false;
};

double sigmoid(double value);
Matrix readMatrix(const string &path, size_t rows, size_t cols);
vector<double> readVector(const string &path, size_t count);
void writeWeights(const Network &network);
void writePredictions(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, const string &path);
Dataset loadDataset(size_t rows);
void loadWeights(Network &network);
void setmatrixrandom(Network &network, double rangeWone, double rangeWtwo, mt19937 &generator);
double predictRow(const double *xRow, const Network &network);
Metrics calculateMetrics(const Dataset &data, size_t startRow, size_t rowCount, const Network &network);
Metrics calculateGradientsAndMetrics(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, Gradients &gradients, size_t threads);
void applyMomentumUpdate(Network &network, const Gradients &gradients, double learningRate, double momentum);
void printProgress(size_t batchIndex, size_t updates, const Metrics &metrics);
TrainResult trainRange(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config);
void batchOnlineRun(size_t iterations, size_t exampleSize, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator);
void offlineRun(size_t rowCount, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator);
void testRun(size_t rowCount);

int main() {
    try {
        //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // Run mode
        bool batchOnline = false;
        bool offline     = false;
        bool test        = true;

        //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // Overall setup. The network dimensions are fixed at 11 -> 16 -> 1 so
        // the compiler can optimise the small inner loops more aggressively.
        double rangeWone             = 4.0;
        double rangeWtwo             = 4.0;
        bool randomiseWeights        = false;
        double percentageErrorTarget = 3.9;
        double learningRate          = 0.0001;
        double momentum              = 0.75;

        //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // Batch setup
        size_t iterations       = 1;
        size_t exampleSize      = 13853;
        size_t numberOfDescents = 1000000;
        size_t times            = 1;
        size_t logEvery         = 1000;
        size_t trainingThreads  = 1;

        //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // Offline / testing setup
        size_t offlineRows = 10000;
        size_t testRows    = 13853;

        // trainingThreads = 1 gives the deterministic single-threaded path.
        // Compile with -fopenmp and increase this value to parallelise rows.
#ifndef _OPENMP
        if(trainingThreads > 1) {
            throw runtime_error("trainingThreads > 1 requires compiling with -fopenmp");
        }
#endif

        mt19937 generator(static_cast<unsigned int>(time(nullptr)));
        const auto startTime = chrono::steady_clock::now();

        if(batchOnline == true) {
            for(size_t i=0;i<times;i++) {
                OptimiserConfig optimiser;
                optimiser.maxDescents = numberOfDescents;
                optimiser.logEvery = logEvery;
                optimiser.threads = trainingThreads;
                optimiser.percentageErrorTarget = percentageErrorTarget;
                optimiser.learningRate = learningRate;
                optimiser.momentum = momentum;
                batchOnlineRun(iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
                numberOfDescents = numberOfDescents + numberOfDescents;
            }
        }

        if(offline == true) {
            OptimiserConfig optimiser;
            optimiser.maxDescents = numberOfDescents;
            optimiser.logEvery = logEvery;
            optimiser.threads = trainingThreads;
            optimiser.percentageErrorTarget = percentageErrorTarget;
            optimiser.learningRate = 1.0;
            optimiser.momentum = 0.0;
            offlineRun(offlineRows,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
        }

        if(test == true) {
            testRun(testRows);
        }

        const auto endTime = chrono::steady_clock::now();
        const chrono::duration<double> elapsed = endTime - startTime;
        cout << "Elapsed time = " << elapsed.count() << " seconds\n";
        return 0;
    }
    catch(const exception &error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

/* Numerically stable sigmoid, preserving the mathematical function used by v2. */
double sigmoid(double value) {
    // The direct form is faster for the normal range. Only very large negative
    // values need the alternative form to avoid overflow in exp(-value).
    if(value < -700.0) {
        const double exponent = exp(value);
        return exponent / (1.0 + exponent);
    }
    return 1.0 / (1.0 + exp(-value));
}

Matrix readMatrix(const string &path, size_t rows, size_t cols) {
    ifstream input(path.c_str());
    if(!input) {
        throw runtime_error("Failed to open file: " + path);
    }

    Matrix matrix(rows,cols);
    for(size_t i=0;i<matrix.data.size();i++) {
        if(!(input >> matrix.data[i])) {
            throw runtime_error("Insufficient or invalid data in file: " + path);
        }
    }
    return matrix;
}

vector<double> readVector(const string &path, size_t count) {
    ifstream input(path.c_str());
    if(!input) {
        throw runtime_error("Failed to open file: " + path);
    }

    vector<double> values(count,0.0);
    for(size_t i=0;i<count;i++) {
        if(!(input >> values[i])) {
            throw runtime_error("Insufficient or invalid data in file: " + path);
        }
    }
    return values;
}

Dataset loadDataset(size_t rows) {
    Dataset data;
    data.x = readMatrix("InputVariables.txt",rows,NUMBER_OF_VARIABLES);
    data.y = readVector("OutputVariables.txt",rows);
    return data;
}

void loadWeights(Network &network) {
    Matrix wOne = readMatrix("wone.txt",NUMBER_OF_VARIABLES,HIDDEN_NODES);
    vector<double> wTwo = readVector("wtwo.txt",HIDDEN_NODES);
    copy(wOne.data.begin(),wOne.data.end(),network.wOne.begin());
    copy(wTwo.begin(),wTwo.end(),network.wTwo.begin());
}

void writeWeights(const Network &network) {
    ofstream wOneOutput("wone.txt");
    ofstream wTwoOutput("wtwo.txt");
    if(!wOneOutput || !wTwoOutput) {
        throw runtime_error("Failed to open weight output files");
    }

    wOneOutput << setprecision(10);
    for(size_t i=0;i<NUMBER_OF_VARIABLES;i++) {
        for(size_t j=0;j<HIDDEN_NODES;j++) {
            if(j != 0) {
                wOneOutput << ' ';
            }
            wOneOutput << network.wOne[i * HIDDEN_NODES + j];
        }
        wOneOutput << '\n';
    }

    wTwoOutput << setprecision(10);
    for(size_t j=0;j<HIDDEN_NODES;j++) {
        wTwoOutput << network.wTwo[j] << '\n';
    }
}

void setmatrixrandom(Network &network, double rangeWone, double rangeWtwo, mt19937 &generator) {
    uniform_real_distribution<double> distributionWone(-rangeWone,rangeWone);
    uniform_real_distribution<double> distributionWtwo(-rangeWtwo,rangeWtwo);
    for(size_t i=0;i<WONE_SIZE;i++) {
        network.wOne[i] = distributionWone(generator);
    }
    for(size_t j=0;j<HIDDEN_NODES;j++) {
        network.wTwo[j] = distributionWtwo(generator);
    }
}

/*
 * Predict one observation. Test mode and final ybar output use this function so
 * they need only 16 hidden values regardless of the number of observations.
 */
double predictRow(const double *xRow, const Network &network) {
    array<double,HIDDEN_NODES> hidden{};
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
        const double xValue = xRow[k];
        const double *weightRow = network.wOne.data() + k * HIDDEN_NODES;
        for(size_t j=0;j<HIDDEN_NODES;j++) {
            hidden[j] += xValue * weightRow[j];
        }
    }

    double zThree = 0.0;
    for(size_t j=0;j<HIDDEN_NODES;j++) {
        hidden[j] = sigmoid(hidden[j]);
        zThree += hidden[j] * network.wTwo[j];
    }
    return sigmoid(zThree);
}

Metrics calculateMetrics(const Dataset &data, size_t startRow, size_t rowCount, const Network &network) {
    double squaredError = 0.0;
    double percentageErrorSum = 0.0;
    double maxPercentageError = 0.0;

    for(size_t i=0;i<rowCount;i++) {
        const size_t dataRow = startRow + i;
        const double prediction = predictRow(data.x.rowData(dataRow),network);
        const double actual = data.y[dataRow];
        const double error = prediction - actual;
        squaredError += error * error;
        if(actual != 0.0) {
            const double percentage = abs(error / actual) * 100.0;
            percentageErrorSum += percentage;
            maxPercentageError = max(maxPercentageError,percentage);
        }
    }

    Metrics metrics;
    metrics.cost = 0.5 * squaredError;
    metrics.percentageError = percentageErrorSum / static_cast<double>(rowCount);
    metrics.maxPercentageError = maxPercentageError;
    return metrics;
}

/*
 * Fused single-threaded training pass. Forward propagation, diagnostics and
 * backpropagation are completed for one observation before moving to the next.
 * The hidden layer and deltaTwo therefore occupy only 16 doubles each.
 */
static Metrics calculateGradientsSingleThread(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, Gradients &gradients) {
    gradients.clear();
    double squaredError = 0.0;
    double percentageErrorSum = 0.0;
    double maxPercentageError = 0.0;

    for(size_t i=0;i<rowCount;i++) {
        const size_t dataRow = startRow + i;
        const double *xRow = data.x.rowData(dataRow);
        array<double,HIDDEN_NODES> hidden{};
        array<double,HIDDEN_NODES> deltaTwo{};

        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const double xValue = xRow[k];
            const double *weightRow = network.wOne.data() + k * HIDDEN_NODES;
            for(size_t j=0;j<HIDDEN_NODES;j++) {
                hidden[j] += xValue * weightRow[j];
            }
        }

        double zThree = 0.0;
        for(size_t j=0;j<HIDDEN_NODES;j++) {
            hidden[j] = sigmoid(hidden[j]);
            zThree += hidden[j] * network.wTwo[j];
        }

        const double prediction = sigmoid(zThree);
        const double actual = data.y[dataRow];
        const double error = prediction - actual;
        squaredError += error * error;
        if(actual != 0.0) {
            const double percentage = abs(error / actual) * 100.0;
            percentageErrorSum += percentage;
            maxPercentageError = max(maxPercentageError,percentage);
        }

        const double deltaThree = error * prediction * (1.0 - prediction);
        for(size_t j=0;j<HIDDEN_NODES;j++) {
            const double activation = hidden[j];
            gradients.dJdWtwo[j] += activation * deltaThree;
            deltaTwo[j] = deltaThree * network.wTwo[j] * activation * (1.0 - activation);
        }

        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const double xValue = xRow[k];
            double *gradientRow = gradients.dJdWone.data() + k * HIDDEN_NODES;
            for(size_t j=0;j<HIDDEN_NODES;j++) {
                gradientRow[j] += xValue * deltaTwo[j];
            }
        }
    }

    Metrics metrics;
    metrics.cost = 0.5 * squaredError;
    metrics.percentageError = percentageErrorSum / static_cast<double>(rowCount);
    metrics.maxPercentageError = maxPercentageError;
    return metrics;
}

#ifdef _OPENMP
/*
 * Optional parallel form of the same pass. Each worker owns a complete local
 * gradient, so the hot loop contains no locks or atomics. Local gradients are
 * combined afterwards in increasing thread order for reproducible reduction at
 * a fixed thread count.
 */
static Metrics calculateGradientsParallel(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, Gradients &gradients, size_t threads) {
    vector<ThreadAccumulator> accumulators(threads);

#pragma omp parallel num_threads(static_cast<int>(threads))
    {
        const size_t thread = static_cast<size_t>(omp_get_thread_num());
        ThreadAccumulator &accumulator = accumulators[thread];
        accumulator.clear();

#pragma omp for schedule(static)
        for(long long ii=0;ii<static_cast<long long>(rowCount);ii++) {
            const size_t dataRow = startRow + static_cast<size_t>(ii);
            const double *xRow = data.x.rowData(dataRow);
            array<double,HIDDEN_NODES> hidden{};
            array<double,HIDDEN_NODES> deltaTwo{};

            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
                const double xValue = xRow[k];
                const double *weightRow = network.wOne.data() + k * HIDDEN_NODES;
                for(size_t j=0;j<HIDDEN_NODES;j++) {
                    hidden[j] += xValue * weightRow[j];
                }
            }

            double zThree = 0.0;
            for(size_t j=0;j<HIDDEN_NODES;j++) {
                hidden[j] = sigmoid(hidden[j]);
                zThree += hidden[j] * network.wTwo[j];
            }

            const double prediction = sigmoid(zThree);
            const double actual = data.y[dataRow];
            const double error = prediction - actual;
            accumulator.squaredError += error * error;
            if(actual != 0.0) {
                const double percentage = abs(error / actual) * 100.0;
                accumulator.percentageErrorSum += percentage;
                accumulator.maxPercentageError = max(accumulator.maxPercentageError,percentage);
            }

            const double deltaThree = error * prediction * (1.0 - prediction);
            for(size_t j=0;j<HIDDEN_NODES;j++) {
                const double activation = hidden[j];
                accumulator.dJdWtwo[j] += activation * deltaThree;
                deltaTwo[j] = deltaThree * network.wTwo[j] * activation * (1.0 - activation);
            }

            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
                const double xValue = xRow[k];
                double *gradientRow = accumulator.dJdWone.data() + k * HIDDEN_NODES;
                for(size_t j=0;j<HIDDEN_NODES;j++) {
                    gradientRow[j] += xValue * deltaTwo[j];
                }
            }
        }
    }

    gradients.clear();
    double squaredError = 0.0;
    double percentageErrorSum = 0.0;
    double maxPercentageError = 0.0;
    for(size_t thread=0;thread<threads;thread++) {
        const ThreadAccumulator &accumulator = accumulators[thread];
        for(size_t i=0;i<WONE_SIZE;i++) {
            gradients.dJdWone[i] += accumulator.dJdWone[i];
        }
        for(size_t j=0;j<HIDDEN_NODES;j++) {
            gradients.dJdWtwo[j] += accumulator.dJdWtwo[j];
        }
        squaredError += accumulator.squaredError;
        percentageErrorSum += accumulator.percentageErrorSum;
        maxPercentageError = max(maxPercentageError,accumulator.maxPercentageError);
    }

    Metrics metrics;
    metrics.cost = 0.5 * squaredError;
    metrics.percentageError = percentageErrorSum / static_cast<double>(rowCount);
    metrics.maxPercentageError = maxPercentageError;
    return metrics;
}
#endif

Metrics calculateGradientsAndMetrics(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, Gradients &gradients, size_t threads) {
#ifdef _OPENMP
    if(threads > 1) {
        return calculateGradientsParallel(data,startRow,rowCount,network,gradients,threads);
    }
#else
    (void)threads;
#endif
    return calculateGradientsSingleThread(data,startRow,rowCount,network,gradients);
}

void applyMomentumUpdate(Network &network, const Gradients &gradients, double learningRate, double momentum) {
    for(size_t i=0;i<WONE_SIZE;i++) {
        network.deltaWone[i] = momentum * network.deltaWone[i] - learningRate * gradients.dJdWone[i];
        network.wOne[i] += network.deltaWone[i];
    }
    for(size_t j=0;j<HIDDEN_NODES;j++) {
        network.deltaWtwo[j] = momentum * network.deltaWtwo[j] - learningRate * gradients.dJdWtwo[j];
        network.wTwo[j] += network.deltaWtwo[j];
    }
}

void printProgress(size_t batchIndex, size_t updates, const Metrics &metrics) {
    cout << "Batch " << batchIndex << " | updates = " << updates << " | cost = " << metrics.cost << " | percentage error = " << metrics.percentageError << " | max error = " << metrics.maxPercentageError << '\n';
}

TrainResult trainRange(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config) {
    Gradients gradients;
    TrainResult result;

    while(true) {
        result.metrics = calculateGradientsAndMetrics(data,startRow,batchSize,network,gradients,config.threads);
        const bool targetReached = result.metrics.percentageError <= config.percentageErrorTarget;
        const bool updateLimitReached = result.updates >= config.maxDescents;
        const bool logNow = config.logEvery != 0 && result.updates % config.logEvery == 0;

        if(logNow || targetReached || updateLimitReached) {
            printProgress(batchIndex,result.updates,result.metrics);
        }
        if(targetReached || updateLimitReached) {
            result.reachedTarget = targetReached;
            break;
        }

        applyMomentumUpdate(network,gradients,config.learningRate,config.momentum);
        result.updates++;
    }

    // Predictions are generated once after training rather than stored during
    // every descent. This keeps training memory independent of hidden width x rows.
    writePredictions(data,startRow,batchSize,network,"ybar.txt");
    return result;
}

void writePredictions(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, const string &path) {
    ofstream output(path.c_str());
    if(!output) {
        throw runtime_error("Failed to open output file: " + path);
    }
    output << setprecision(10);
    for(size_t i=0;i<rowCount;i++) {
        output << predictRow(data.x.rowData(startRow + i),network) << '\n';
    }
}

void batchOnlineRun(size_t iterations, size_t exampleSize, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator) {
    const size_t totalRows = iterations * exampleSize;
    Dataset data = loadDataset(totalRows);
    Network network;

    if(randomiseWeights == true) {
        setmatrixrandom(network,rangeWone,rangeWtwo,generator);
    }
    else {
        loadWeights(network);
    }

    for(size_t batch=0;batch<iterations;batch++) {
        const size_t startRow = batch * exampleSize;
        const TrainResult result = trainRange(data,startRow,exampleSize,batch,network,optimiser);
        writeWeights(network);
        if(result.reachedTarget == false) {
            cout << "Batch " << batch << " reached the maximum number of descents before the target.\n";
        }
    }
}

void offlineRun(size_t rowCount, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator) {
    Dataset data = loadDataset(rowCount);
    Network network;

    if(randomiseWeights == true) {
        setmatrixrandom(network,rangeWone,rangeWtwo,generator);
    }
    else {
        loadWeights(network);
    }

    const TrainResult result = trainRange(data,0,rowCount,0,network,optimiser);
    writeWeights(network);
    if(result.reachedTarget == false) {
        cout << "Offline training reached the maximum number of descents before the target.\n";
    }
}

void testRun(size_t rowCount) {
    Dataset data = loadDataset(rowCount);
    Network network;
    loadWeights(network);
    const Metrics metrics = calculateMetrics(data,0,rowCount,network);
    cout << "Test cost = " << metrics.cost << " | percentage error = " << metrics.percentageError << " | max error = " << metrics.maxPercentageError << '\n';
    writePredictions(data,0,rowCount,network,"ybar.txt");
}
