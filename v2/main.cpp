#include<iostream>
#include<algorithm>
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
using namespace std;

/*
 * Simple Neural Network - v2
 * ==========================
 *
 * This file is an optimised companion to the original 2017 main.cpp. The
 * original file remains unchanged at the repository root. V2 keeps the same
 * network mathematics and text-file formats while reducing allocations,
 * temporary matrices, memory copies and console I/O.
 *
 * The network remains:
 *
 *     inputs -> W1 -> sigmoid hidden layer -> W2 -> sigmoid output
 *
 * V2 deliberately does not add biases or change the activation, loss or
 * momentum equations. The aim is to improve the original implementation while
 * keeping it easy to read from top to bottom in one source file.
 */

struct Matrix {
    size_t rows;
    size_t cols;
    vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(size_t r, size_t c, double value = 0.0) : rows(r), cols(c), data(r * c, value) {}

    double *rowData(size_t row) { return data.data() + row * cols; }
    const double *rowData(size_t row) const { return data.data() + row * cols; }
    void fill(double value) { std::fill(data.begin(), data.end(), value); }
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
    Matrix wOne;
    vector<double> wTwo;
    Matrix deltaWone;
    vector<double> deltaWtwo;

    Network(size_t inputs, size_t hidden) : wOne(inputs, hidden), wTwo(hidden, 0.0), deltaWone(inputs, hidden), deltaWtwo(hidden, 0.0) {}
};

struct TrainingBuffers {
    Matrix aTwo;
    vector<double> yBar;
    Matrix dJdWone;
    vector<double> dJdWtwo;
    vector<double> deltaTwo;

    TrainingBuffers(size_t examples, size_t inputs, size_t hidden) : aTwo(examples, hidden), yBar(examples, 0.0), dJdWone(inputs, hidden), dJdWtwo(hidden, 0.0), deltaTwo(hidden, 0.0) {}
};

struct OptimiserConfig {
    size_t maxDescents = 1000000;
    size_t logEvery = 1000;
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
void writeMatrix(const Matrix &matrix, const string &path);
void writeVector(const vector<double> &values, const string &path);
Dataset loadDataset(size_t rows, size_t inputCount);
void loadWeights(Network &network);
void saveWeights(const Network &network);
void setmatrixrandom(Network &network, double rangeWone, double rangeWtwo, mt19937 &generator);
void forwardRange(const Dataset &data, size_t startRow, const Network &network, TrainingBuffers &buffers);
Metrics calculateMetrics(const Dataset &data, size_t startRow, const vector<double> &predictions);
Metrics calculateGradientsAndMetrics(const Dataset &data, size_t startRow, const Network &network, TrainingBuffers &buffers);
void applyMomentumUpdate(Network &network, const TrainingBuffers &buffers, double learningRate, double momentum);
void printProgress(size_t batchIndex, size_t updates, const Metrics &metrics);
TrainResult trainRange(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config);
void batchOnlineRun(size_t inputCount, size_t hiddenCount, size_t iterations, size_t exampleSize, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator);
void offlineRun(size_t inputCount, size_t rowCount, size_t hiddenCount, size_t maxDescents, double percentageErrorTarget, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator);
void testRun(size_t inputCount, size_t rowCount, size_t hiddenCount);

int main() {
    try {
        //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // Run mode
        bool batchOnline = false;
        bool offline     = false;
        bool test        = true;

        //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // Overall setup. These values match the original checked-in configuration.
        size_t numberOfVariables       = 11;
        size_t nodesInFirstHiddenLayer = 16;
        double rangeWone               = 4.0;
        double rangeWtwo               = 4.0;
        bool randomiseWeights          = false;
        double percentageErrorTarget   = 3.9;
        double learningRate            = 0.0001;
        double momentum                = 0.75;

        //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // Batch setup
        size_t iterations       = 1;
        size_t exampleSize      = 13853;
        size_t numberOfDescents = 1000000;
        size_t times            = 1;
        size_t logEvery         = 1000;

        //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // Offline / testing setup
        size_t offlineRows = 10000;
        size_t testRows    = 13853;

        // Time-based seeding preserves the original behaviour. Replace this with
        // a fixed integer when reproducible experiments are preferred.
        mt19937 generator(static_cast<unsigned int>(time(nullptr)));
        const auto startTime = chrono::steady_clock::now();

        if(batchOnline == true) {
            for(size_t i=0;i<times;i++) {
                OptimiserConfig optimiser;
                optimiser.maxDescents = numberOfDescents;
                optimiser.logEvery = logEvery;
                optimiser.percentageErrorTarget = percentageErrorTarget;
                optimiser.learningRate = learningRate;
                optimiser.momentum = momentum;
                batchOnlineRun(numberOfVariables,nodesInFirstHiddenLayer,iterations,exampleSize,optimiser,rangeWone,rangeWtwo,randomiseWeights,generator);
                numberOfDescents = numberOfDescents + numberOfDescents;
            }
        }

        if(offline == true) {
            offlineRun(numberOfVariables,offlineRows,nodesInFirstHiddenLayer,numberOfDescents,percentageErrorTarget,rangeWone,rangeWtwo,randomiseWeights,generator);
        }

        if(test == true) {
            testRun(numberOfVariables,testRows,nodesInFirstHiddenLayer);
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

/*
 * Numerically stable sigmoid. This is algebraically identical to the original
 * 1 / (1 + exp(-x)), while avoiding overflow in exp() for a large negative x.
 */
double sigmoid(double value) {
    if(value >= 0.0) {
        const double exponent = exp(-value);
        return 1.0 / (1.0 + exponent);
    }
    const double exponent = exp(value);
    return exponent / (1.0 + exponent);
}

/* Read a fixed-size whitespace-delimited matrix. */
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

/* Read a fixed number of scalar values. */
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

/* Write a matrix using the same whitespace-delimited layout as the original. */
void writeMatrix(const Matrix &matrix, const string &path) {
    ofstream output(path.c_str());
    if(!output) {
        throw runtime_error("Failed to open output file: " + path);
    }

    output << setprecision(10);
    for(size_t i=0;i<matrix.rows;i++) {
        const double *row = matrix.rowData(i);
        for(size_t j=0;j<matrix.cols;j++) {
            if(j != 0) {
                output << ' ';
            }
            output << row[j];
        }
        // '\n' avoids the forced flush performed by std::endl.
        output << '\n';
    }
}

/* Write one value per line, matching wtwo.txt and ybar.txt in the original. */
void writeVector(const vector<double> &values, const string &path) {
    ofstream output(path.c_str());
    if(!output) {
        throw runtime_error("Failed to open output file: " + path);
    }

    output << setprecision(10);
    for(size_t i=0;i<values.size();i++) {
        output << values[i] << '\n';
    }
}

Dataset loadDataset(size_t rows, size_t inputCount) {
    Dataset data;
    data.x = readMatrix("InputVariables.txt",rows,inputCount);
    data.y = readVector("OutputVariables.txt",rows);
    return data;
}

void loadWeights(Network &network) {
    network.wOne = readMatrix("wone.txt",network.wOne.rows,network.wOne.cols);
    network.wTwo = readVector("wtwo.txt",network.wTwo.size());
}

void saveWeights(const Network &network) {
    writeMatrix(network.wOne,"wone.txt");
    writeVector(network.wTwo,"wtwo.txt");
}

/*
 * Random weight initialisation remains configurable over the same -range to
 * +range interval as the original. This is not a hot path, so the standard
 * random-number facilities are used for clarity and uniform sampling.
 */
void setmatrixrandom(Network &network, double rangeWone, double rangeWtwo, mt19937 &generator) {
    uniform_real_distribution<double> distributionWone(-rangeWone,rangeWone);
    uniform_real_distribution<double> distributionWtwo(-rangeWtwo,rangeWtwo);

    for(size_t i=0;i<network.wOne.data.size();i++) {
        network.wOne.data[i] = distributionWone(generator);
    }
    for(size_t i=0;i<network.wTwo.size();i++) {
        network.wTwo[i] = distributionWtwo(generator);
    }
}

/*
 * Forward propagation over one contiguous range of observations.
 *
 * Original equations:
 *     z2   = X * W1
 *     a2   = sigmoid(z2)
 *     z3   = a2 * W2
 *     yBar = sigmoid(z3)
 *
 * V2 writes the activated hidden values directly into aTwo. Separate z2 and z3
 * matrices are unnecessary because they are not needed after activation.
 */
void forwardRange(const Dataset &data, size_t startRow, const Network &network, TrainingBuffers &buffers) {
    const size_t inputCount = network.wOne.rows;
    const size_t hiddenCount = network.wOne.cols;
    const size_t batchSize = buffers.aTwo.rows;

    for(size_t i=0;i<batchSize;i++) {
        const size_t dataRow = startRow + i;
        const double *xRow = data.x.rowData(dataRow);
        double *hiddenRow = buffers.aTwo.rowData(i);
        fill(hiddenRow,hiddenRow + hiddenCount,0.0);

        // Hidden nodes are the inner loop so both W1 and aTwo are traversed
        // contiguously in memory.
        for(size_t k=0;k<inputCount;k++) {
            const double xValue = xRow[k];
            const double *weightRow = network.wOne.rowData(k);
            for(size_t j=0;j<hiddenCount;j++) {
                hiddenRow[j] += xValue * weightRow[j];
            }
        }

        double zThree = 0.0;
        for(size_t j=0;j<hiddenCount;j++) {
            hiddenRow[j] = sigmoid(hiddenRow[j]);
            zThree += hiddenRow[j] * network.wTwo[j];
        }
        buffers.yBar[i] = sigmoid(zThree);
    }
}

/* Calculate the same cost and percentage-error diagnostics used by v1. */
Metrics calculateMetrics(const Dataset &data, size_t startRow, const vector<double> &predictions) {
    double squaredError = 0.0;
    double percentageErrorSum = 0.0;
    double maxPercentageError = 0.0;

    for(size_t i=0;i<predictions.size();i++) {
        const double actual = data.y[startRow + i];
        const double error = predictions[i] - actual;
        squaredError += error * error;

        // Preserve the original behaviour: a zero target contributes zero
        // percentage error and remains in the denominator of the mean.
        if(actual != 0.0) {
            const double percentage = abs(error / actual) * 100.0;
            percentageErrorSum += percentage;
            maxPercentageError = max(maxPercentageError,percentage);
        }
    }

    Metrics metrics;
    metrics.cost = 0.5 * squaredError;
    metrics.percentageError = percentageErrorSum / static_cast<double>(predictions.size());
    metrics.maxPercentageError = maxPercentageError;
    return metrics;
}

/*
 * Backpropagation and diagnostics are calculated together in one pass.
 *
 * The original implementation created transposes and full intermediate
 * matrices. The same equations can be applied directly:
 *
 *     delta3_i   = (yBar_i - y_i) * yBar_i * (1 - yBar_i)
 *     dJdW2_j   += a2_ij * delta3_i
 *     delta2_ij  = delta3_i * W2_j * a2_ij * (1 - a2_ij)
 *     dJdW1_kj  += X_ik * delta2_ij
 *
 * Only one hidden-sized deltaTwo scratch vector is required for the current
 * observation. X transpose, aTwo transpose, W2 transpose and z-prime matrices
 * are therefore removed from the hot training path.
 */
Metrics calculateGradientsAndMetrics(const Dataset &data, size_t startRow, const Network &network, TrainingBuffers &buffers) {
    buffers.dJdWone.fill(0.0);
    fill(buffers.dJdWtwo.begin(),buffers.dJdWtwo.end(),0.0);

    const size_t inputCount = network.wOne.rows;
    const size_t hiddenCount = network.wOne.cols;
    const size_t batchSize = buffers.aTwo.rows;
    double squaredError = 0.0;
    double percentageErrorSum = 0.0;
    double maxPercentageError = 0.0;

    for(size_t i=0;i<batchSize;i++) {
        const size_t dataRow = startRow + i;
        const double actual = data.y[dataRow];
        const double prediction = buffers.yBar[i];
        const double error = prediction - actual;
        squaredError += error * error;

        if(actual != 0.0) {
            const double percentage = abs(error / actual) * 100.0;
            percentageErrorSum += percentage;
            maxPercentageError = max(maxPercentageError,percentage);
        }

        const double deltaThree = error * prediction * (1.0 - prediction);
        const double *hiddenRow = buffers.aTwo.rowData(i);

        for(size_t j=0;j<hiddenCount;j++) {
            const double activation = hiddenRow[j];
            buffers.dJdWtwo[j] += activation * deltaThree;
            buffers.deltaTwo[j] = deltaThree * network.wTwo[j] * activation * (1.0 - activation);
        }

        const double *xRow = data.x.rowData(dataRow);
        for(size_t k=0;k<inputCount;k++) {
            const double xValue = xRow[k];
            double *gradientRow = buffers.dJdWone.rowData(k);
            for(size_t j=0;j<hiddenCount;j++) {
                gradientRow[j] += xValue * buffers.deltaTwo[j];
            }
        }
    }

    Metrics metrics;
    metrics.cost = 0.5 * squaredError;
    metrics.percentageError = percentageErrorSum / static_cast<double>(batchSize);
    metrics.maxPercentageError = maxPercentageError;
    return metrics;
}

/*
 * Classical momentum, matching the original batchOnlineRun equation:
 *
 *     deltaW = momentum * deltaW - learningRate * gradient
 *     W      = W + deltaW
 *
 * V1 expressed this as several matrix-wide scale/subtract/add passes. V2 fuses
 * those operations into one pass over each weight array.
 */
void applyMomentumUpdate(Network &network, const TrainingBuffers &buffers, double learningRate, double momentum) {
    for(size_t i=0;i<network.wOne.data.size();i++) {
        network.deltaWone.data[i] = momentum * network.deltaWone.data[i] - learningRate * buffers.dJdWone.data[i];
        network.wOne.data[i] += network.deltaWone.data[i];
    }

    for(size_t i=0;i<network.wTwo.size();i++) {
        network.deltaWtwo[i] = momentum * network.deltaWtwo[i] - learningRate * buffers.dJdWtwo[i];
        network.wTwo[i] += network.deltaWtwo[i];
    }
}

void printProgress(size_t batchIndex, size_t updates, const Metrics &metrics) {
    cout << "Batch " << batchIndex << " | updates = " << updates << " | cost = " << metrics.cost << " | percentage error = " << metrics.percentageError << " | max error = " << metrics.maxPercentageError << '\n';
}

/*
 * Train one sequential batch until either the percentage-error target is met or
 * the maximum number of descents is reached. Scratch buffers are allocated once
 * and reused for every update.
 */
TrainResult trainRange(const Dataset &data, size_t startRow, size_t batchSize, size_t batchIndex, Network &network, const OptimiserConfig &config) {
    TrainingBuffers buffers(batchSize,network.wOne.rows,network.wOne.cols);
    TrainResult result;

    while(true) {
        forwardRange(data,startRow,network,buffers);
        result.metrics = calculateGradientsAndMetrics(data,startRow,network,buffers);

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

        applyMomentumUpdate(network,buffers,config.learningRate,config.momentum);
        result.updates++;
    }

    // As in v1, ybar.txt contains predictions for the most recently processed
    // batch. The final forward pass already contains those values.
    writeVector(buffers.yBar,"ybar.txt");
    return result;
}

/*
 * Optimised version of the original batchOnlineRun(). Data are loaded once and
 * each sequential batch is represented by a starting row rather than copied
 * into separate x/y matrices. Momentum persists between batches, matching v1.
 */
void batchOnlineRun(size_t inputCount, size_t hiddenCount, size_t iterations, size_t exampleSize, const OptimiserConfig &optimiser, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator) {
    const size_t totalRows = iterations * exampleSize;
    Dataset data = loadDataset(totalRows,inputCount);
    Network network(inputCount,hiddenCount);

    if(randomiseWeights == true) {
        setmatrixrandom(network,rangeWone,rangeWtwo,generator);
    }
    else {
        loadWeights(network);
    }

    for(size_t batch=0;batch<iterations;batch++) {
        const size_t startRow = batch * exampleSize;
        const TrainResult result = trainRange(data,startRow,exampleSize,batch,network,optimiser);
        saveWeights(network);

        if(result.reachedTarget == false) {
            cout << "Batch " << batch << " reached the maximum number of descents before the target.\n";
        }
    }
}

/*
 * Optimised version of offlineRun(). The original offline path subtracts the
 * full gradient directly, which is equivalent to learningRate=1 and momentum=0.
 */
void offlineRun(size_t inputCount, size_t rowCount, size_t hiddenCount, size_t maxDescents, double percentageErrorTarget, double rangeWone, double rangeWtwo, bool randomiseWeights, mt19937 &generator) {
    Dataset data = loadDataset(rowCount,inputCount);
    Network network(inputCount,hiddenCount);

    if(randomiseWeights == true) {
        setmatrixrandom(network,rangeWone,rangeWtwo,generator);
    }
    else {
        loadWeights(network);
    }

    OptimiserConfig optimiser;
    optimiser.maxDescents = maxDescents;
    optimiser.percentageErrorTarget = percentageErrorTarget;
    optimiser.learningRate = 1.0;
    optimiser.momentum = 0.0;

    const TrainResult result = trainRange(data,0,rowCount,0,network,optimiser);
    saveWeights(network);

    if(result.reachedTarget == false) {
        cout << "Offline training reached the maximum number of descents before the target.\n";
    }
}

/*
 * Optimised version of testRun(). Detailed predictions remain in ybar.txt;
 * printing every matrix is omitted because console formatting can dominate the
 * runtime for large data sets.
 */
void testRun(size_t inputCount, size_t rowCount, size_t hiddenCount) {
    Dataset data = loadDataset(rowCount,inputCount);
    Network network(inputCount,hiddenCount);
    loadWeights(network);

    TrainingBuffers buffers(rowCount,inputCount,hiddenCount);
    forwardRange(data,0,network,buffers);
    const Metrics metrics = calculateMetrics(data,0,buffers.yBar);

    cout << "Test cost = " << metrics.cost << " | percentage error = " << metrics.percentageError << " | max error = " << metrics.maxPercentageError << '\n';
    writeVector(buffers.yBar,"ybar.txt");
}
