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
#if defined(__x86_64__) && defined(__GNUC__)
#include<immintrin.h>
#endif
#include<random>
#include<stdexcept>
#include<string>
#include<vector>
#ifdef _OPENMP
#include<omp.h>
#endif
using namespace std;

/*
 * Simple Neural Network - v5
 * ==========================
 *
 * V5 keeps the same 11 -> 16 -> 1 sigmoid network and training equations as
 * v4. It retains v4's blocked/vector-sigmoid design, then specialises the two
 * hottest arithmetic stages for the fixed 11 x 16 architecture. Runtime-dispatched
 * AVX2/FMA and AVX-512/FMA kernels keep hidden values and gradient accumulators in
 * vector registers across each block. The portable scalar v4 kernel remains the
 * fallback on unsupported CPUs.
 *
 * Defining SIMPLE_NN_USE_LIBMVEC still enables v4's vector exp() backend on
 * glibc/x86 systems. Arithmetic-kernel dispatch is independent of that option.
 */

constexpr size_t NUMBER_OF_VARIABLES = 11;
constexpr size_t HIDDEN_NODES = 16;
constexpr size_t WONE_SIZE = NUMBER_OF_VARIABLES * HIDDEN_NODES;
constexpr size_t BLOCK_SIZE = 64;

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

        // trainingThreads = 1 gives the single-threaded blocked path.
        // Compile with -fopenmp and increase this value to keep a persistent
        // worker team across descents.
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

#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
extern "C" __m256d _ZGVdN4v_exp(__m256d);
extern "C" __m512d _ZGVeN8v_exp(__m512d);

__attribute__((target("avx2")))
static void sigmoidVectorAVX2(double *values, size_t count) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d lowerLimit = _mm256_set1_pd(-700.0);
    size_t i = 0;
    for(;i + 4 <= count;i += 4) {
        const __m256d x = _mm256_loadu_pd(values + i);
        const __m256d extreme = _mm256_cmp_pd(x,lowerLimit,_CMP_LT_OQ);
        if(_mm256_movemask_pd(extreme) != 0) {
            for(size_t j=0;j<4;j++) values[i + j] = sigmoid(values[i + j]);
            continue;
        }
        const __m256d exponent = _ZGVdN4v_exp(_mm256_sub_pd(zero,x));
        _mm256_storeu_pd(values + i,_mm256_div_pd(one,_mm256_add_pd(one,exponent)));
    }
    for(;i<count;i++) values[i] = sigmoid(values[i]);
}

__attribute__((target("avx512f")))
static void sigmoidVectorAVX512(double *values, size_t count) {
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d zero = _mm512_setzero_pd();
    const __m512d lowerLimit = _mm512_set1_pd(-700.0);
    size_t i = 0;
    for(;i + 8 <= count;i += 8) {
        const __m512d x = _mm512_loadu_pd(values + i);
        const __mmask8 extreme = _mm512_cmp_pd_mask(x,lowerLimit,_CMP_LT_OQ);
        if(extreme != 0) {
            for(size_t j=0;j<8;j++) values[i + j] = sigmoid(values[i + j]);
            continue;
        }
        const __m512d exponent = _ZGVeN8v_exp(_mm512_sub_pd(zero,x));
        _mm512_storeu_pd(values + i,_mm512_div_pd(one,_mm512_add_pd(one,exponent)));
    }
    for(;i<count;i++) values[i] = sigmoid(values[i]);
}
#endif

static void sigmoidVector(double *values, size_t count) {
#if defined(SIMPLE_NN_USE_LIBMVEC) && defined(__GLIBC__) && defined(__x86_64__) && defined(__GNUC__)
    if(__builtin_cpu_supports("avx512f")) {
        sigmoidVectorAVX512(values,count);
        return;
    }
    if(__builtin_cpu_supports("avx2")) {
        sigmoidVectorAVX2(values,count);
        return;
    }
#endif
    for(size_t i=0;i<count;i++) values[i] = sigmoid(values[i]);
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
 * Process one block with the portable v4-style arithmetic path.
 */
static void processTrainingBlockScalar(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, ThreadAccumulator &accumulator) {
    array<double,BLOCK_SIZE * HIDDEN_NODES> hidden{};
    array<double,BLOCK_SIZE> output{};
    array<double,BLOCK_SIZE> deltaThree{};

    for(size_t i=0;i<rowCount;i++) {
        const double *xRow = data.x.rowData(startRow + i);
        double *hiddenRow = hidden.data() + i * HIDDEN_NODES;
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const double xValue = xRow[k];
            const double *weightRow = network.wOne.data() + k * HIDDEN_NODES;
            for(size_t j=0;j<HIDDEN_NODES;j++) hiddenRow[j] += xValue * weightRow[j];
        }
    }

    sigmoidVector(hidden.data(),rowCount * HIDDEN_NODES);

    for(size_t i=0;i<rowCount;i++) {
        const double *hiddenRow = hidden.data() + i * HIDDEN_NODES;
        double zThree = 0.0;
        for(size_t j=0;j<HIDDEN_NODES;j++) zThree += hiddenRow[j] * network.wTwo[j];
        output[i] = zThree;
    }
    sigmoidVector(output.data(),rowCount);

    for(size_t i=0;i<rowCount;i++) {
        const double prediction = output[i];
        const double actual = data.y[startRow + i];
        const double error = prediction - actual;
        accumulator.squaredError += error * error;
        if(actual != 0.0) {
            const double percentage = abs(error / actual) * 100.0;
            accumulator.percentageErrorSum += percentage;
            accumulator.maxPercentageError = max(accumulator.maxPercentageError,percentage);
        }
        deltaThree[i] = error * prediction * (1.0 - prediction);
    }

    for(size_t i=0;i<rowCount;i++) {
        double *hiddenRow = hidden.data() + i * HIDDEN_NODES;
        const double delta = deltaThree[i];
        for(size_t j=0;j<HIDDEN_NODES;j++) {
            const double activation = hiddenRow[j];
            accumulator.dJdWtwo[j] += activation * delta;
            hiddenRow[j] = delta * network.wTwo[j] * activation * (1.0 - activation);
        }
        const double *xRow = data.x.rowData(startRow + i);
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const double xValue = xRow[k];
            double *gradientRow = accumulator.dJdWone.data() + k * HIDDEN_NODES;
            for(size_t j=0;j<HIDDEN_NODES;j++) gradientRow[j] += xValue * hiddenRow[j];
        }
    }
}

#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2")))
static inline double horizontalSum256(__m256d value) {
    const __m128d low = _mm256_castpd256_pd128(value);
    const __m128d high = _mm256_extractf128_pd(value,1);
    const __m128d sum = _mm_add_pd(low,high);
    const __m128d swapped = _mm_shuffle_pd(sum,sum,1);
    return _mm_cvtsd_f64(_mm_add_sd(sum,swapped));
}

__attribute__((target("avx2,fma")))
static void processTrainingBlockAVX2(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, ThreadAccumulator &accumulator) {
    alignas(64) array<double,BLOCK_SIZE * HIDDEN_NODES> hidden{};
    alignas(64) array<double,BLOCK_SIZE> output{};
    alignas(64) array<double,BLOCK_SIZE> deltaThree{};
    const __m256d one = _mm256_set1_pd(1.0);

    for(size_t i=0;i<rowCount;i++) {
        const double *xRow = data.x.rowData(startRow + i);
        __m256d h0 = _mm256_setzero_pd(), h1 = _mm256_setzero_pd();
        __m256d h2 = _mm256_setzero_pd(), h3 = _mm256_setzero_pd();
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const __m256d x = _mm256_set1_pd(xRow[k]);
            const double *w = network.wOne.data() + k * HIDDEN_NODES;
            h0 = _mm256_fmadd_pd(x,_mm256_loadu_pd(w),h0);
            h1 = _mm256_fmadd_pd(x,_mm256_loadu_pd(w + 4),h1);
            h2 = _mm256_fmadd_pd(x,_mm256_loadu_pd(w + 8),h2);
            h3 = _mm256_fmadd_pd(x,_mm256_loadu_pd(w + 12),h3);
        }
        double *h = hidden.data() + i * HIDDEN_NODES;
        _mm256_storeu_pd(h,h0); _mm256_storeu_pd(h + 4,h1);
        _mm256_storeu_pd(h + 8,h2); _mm256_storeu_pd(h + 12,h3);
    }

    sigmoidVector(hidden.data(),rowCount * HIDDEN_NODES);

    const __m256d w20 = _mm256_loadu_pd(network.wTwo.data());
    const __m256d w21 = _mm256_loadu_pd(network.wTwo.data() + 4);
    const __m256d w22 = _mm256_loadu_pd(network.wTwo.data() + 8);
    const __m256d w23 = _mm256_loadu_pd(network.wTwo.data() + 12);
    for(size_t i=0;i<rowCount;i++) {
        const double *h = hidden.data() + i * HIDDEN_NODES;
        __m256d sum = _mm256_mul_pd(_mm256_loadu_pd(h),w20);
        sum = _mm256_fmadd_pd(_mm256_loadu_pd(h + 4),w21,sum);
        sum = _mm256_fmadd_pd(_mm256_loadu_pd(h + 8),w22,sum);
        sum = _mm256_fmadd_pd(_mm256_loadu_pd(h + 12),w23,sum);
        output[i] = horizontalSum256(sum);
    }
    sigmoidVector(output.data(),rowCount);

    for(size_t i=0;i<rowCount;i++) {
        const double prediction = output[i];
        const double actual = data.y[startRow + i];
        const double error = prediction - actual;
        accumulator.squaredError += error * error;
        if(actual != 0.0) {
            const double percentage = abs(error / actual) * 100.0;
            accumulator.percentageErrorSum += percentage;
            accumulator.maxPercentageError = max(accumulator.maxPercentageError,percentage);
        }
        deltaThree[i] = error * prediction * (1.0 - prediction);
    }

    __m256d g20 = _mm256_loadu_pd(accumulator.dJdWtwo.data());
    __m256d g21 = _mm256_loadu_pd(accumulator.dJdWtwo.data() + 4);
    __m256d g22 = _mm256_loadu_pd(accumulator.dJdWtwo.data() + 8);
    __m256d g23 = _mm256_loadu_pd(accumulator.dJdWtwo.data() + 12);
    for(size_t i=0;i<rowCount;i++) {
        double *h = hidden.data() + i * HIDDEN_NODES;
        const __m256d d = _mm256_set1_pd(deltaThree[i]);
        __m256d a0 = _mm256_loadu_pd(h), a1 = _mm256_loadu_pd(h + 4);
        __m256d a2 = _mm256_loadu_pd(h + 8), a3 = _mm256_loadu_pd(h + 12);
        g20 = _mm256_fmadd_pd(a0,d,g20); g21 = _mm256_fmadd_pd(a1,d,g21);
        g22 = _mm256_fmadd_pd(a2,d,g22); g23 = _mm256_fmadd_pd(a3,d,g23);
        _mm256_storeu_pd(h,_mm256_mul_pd(_mm256_mul_pd(d,w20),_mm256_mul_pd(a0,_mm256_sub_pd(one,a0))));
        _mm256_storeu_pd(h + 4,_mm256_mul_pd(_mm256_mul_pd(d,w21),_mm256_mul_pd(a1,_mm256_sub_pd(one,a1))));
        _mm256_storeu_pd(h + 8,_mm256_mul_pd(_mm256_mul_pd(d,w22),_mm256_mul_pd(a2,_mm256_sub_pd(one,a2))));
        _mm256_storeu_pd(h + 12,_mm256_mul_pd(_mm256_mul_pd(d,w23),_mm256_mul_pd(a3,_mm256_sub_pd(one,a3))));
    }
    _mm256_storeu_pd(accumulator.dJdWtwo.data(),g20); _mm256_storeu_pd(accumulator.dJdWtwo.data() + 4,g21);
    _mm256_storeu_pd(accumulator.dJdWtwo.data() + 8,g22); _mm256_storeu_pd(accumulator.dJdWtwo.data() + 12,g23);

    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
        double *g = accumulator.dJdWone.data() + k * HIDDEN_NODES;
        __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd(), a2 = _mm256_setzero_pd(), a3 = _mm256_setzero_pd();
        __m256d b0 = _mm256_setzero_pd(), b1 = _mm256_setzero_pd(), b2 = _mm256_setzero_pd(), b3 = _mm256_setzero_pd();
        size_t i = 0;
        for(;i + 1<rowCount;i += 2) {
            const double *h0 = hidden.data() + i * HIDDEN_NODES;
            const double *h1 = hidden.data() + (i + 1) * HIDDEN_NODES;
            const __m256d x0 = _mm256_set1_pd(data.x.rowData(startRow + i)[k]);
            const __m256d x1 = _mm256_set1_pd(data.x.rowData(startRow + i + 1)[k]);
            a0 = _mm256_fmadd_pd(x0,_mm256_loadu_pd(h0),a0); a1 = _mm256_fmadd_pd(x0,_mm256_loadu_pd(h0 + 4),a1);
            a2 = _mm256_fmadd_pd(x0,_mm256_loadu_pd(h0 + 8),a2); a3 = _mm256_fmadd_pd(x0,_mm256_loadu_pd(h0 + 12),a3);
            b0 = _mm256_fmadd_pd(x1,_mm256_loadu_pd(h1),b0); b1 = _mm256_fmadd_pd(x1,_mm256_loadu_pd(h1 + 4),b1);
            b2 = _mm256_fmadd_pd(x1,_mm256_loadu_pd(h1 + 8),b2); b3 = _mm256_fmadd_pd(x1,_mm256_loadu_pd(h1 + 12),b3);
        }
        if(i<rowCount) {
            const double *h = hidden.data() + i * HIDDEN_NODES;
            const __m256d x = _mm256_set1_pd(data.x.rowData(startRow + i)[k]);
            a0 = _mm256_fmadd_pd(x,_mm256_loadu_pd(h),a0); a1 = _mm256_fmadd_pd(x,_mm256_loadu_pd(h + 4),a1);
            a2 = _mm256_fmadd_pd(x,_mm256_loadu_pd(h + 8),a2); a3 = _mm256_fmadd_pd(x,_mm256_loadu_pd(h + 12),a3);
        }
        _mm256_storeu_pd(g,_mm256_add_pd(_mm256_loadu_pd(g),_mm256_add_pd(a0,b0)));
        _mm256_storeu_pd(g + 4,_mm256_add_pd(_mm256_loadu_pd(g + 4),_mm256_add_pd(a1,b1)));
        _mm256_storeu_pd(g + 8,_mm256_add_pd(_mm256_loadu_pd(g + 8),_mm256_add_pd(a2,b2)));
        _mm256_storeu_pd(g + 12,_mm256_add_pd(_mm256_loadu_pd(g + 12),_mm256_add_pd(a3,b3)));
    }
}

__attribute__((target("avx512f,fma")))
static void processTrainingBlockAVX512(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, ThreadAccumulator &accumulator) {
    alignas(64) array<double,BLOCK_SIZE * HIDDEN_NODES> hidden{};
    alignas(64) array<double,BLOCK_SIZE> output{};
    alignas(64) array<double,BLOCK_SIZE> deltaThree{};
    const __m512d one = _mm512_set1_pd(1.0);

    for(size_t i=0;i<rowCount;i++) {
        const double *xRow = data.x.rowData(startRow + i);
        __m512d h0 = _mm512_setzero_pd(), h1 = _mm512_setzero_pd();
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
            const __m512d x = _mm512_set1_pd(xRow[k]);
            const double *w = network.wOne.data() + k * HIDDEN_NODES;
            h0 = _mm512_fmadd_pd(x,_mm512_loadu_pd(w),h0);
            h1 = _mm512_fmadd_pd(x,_mm512_loadu_pd(w + 8),h1);
        }
        double *h = hidden.data() + i * HIDDEN_NODES;
        _mm512_storeu_pd(h,h0); _mm512_storeu_pd(h + 8,h1);
    }

    sigmoidVector(hidden.data(),rowCount * HIDDEN_NODES);

    const __m512d w20 = _mm512_loadu_pd(network.wTwo.data());
    const __m512d w21 = _mm512_loadu_pd(network.wTwo.data() + 8);
    for(size_t i=0;i<rowCount;i++) {
        const double *h = hidden.data() + i * HIDDEN_NODES;
        const __m512d sum = _mm512_fmadd_pd(_mm512_loadu_pd(h + 8),w21,_mm512_mul_pd(_mm512_loadu_pd(h),w20));
        output[i] = _mm512_reduce_add_pd(sum);
    }
    sigmoidVector(output.data(),rowCount);

    for(size_t i=0;i<rowCount;i++) {
        const double prediction = output[i];
        const double actual = data.y[startRow + i];
        const double error = prediction - actual;
        accumulator.squaredError += error * error;
        if(actual != 0.0) {
            const double percentage = abs(error / actual) * 100.0;
            accumulator.percentageErrorSum += percentage;
            accumulator.maxPercentageError = max(accumulator.maxPercentageError,percentage);
        }
        deltaThree[i] = error * prediction * (1.0 - prediction);
    }

    __m512d g20 = _mm512_loadu_pd(accumulator.dJdWtwo.data());
    __m512d g21 = _mm512_loadu_pd(accumulator.dJdWtwo.data() + 8);
    for(size_t i=0;i<rowCount;i++) {
        double *h = hidden.data() + i * HIDDEN_NODES;
        const __m512d d = _mm512_set1_pd(deltaThree[i]);
        const __m512d a0 = _mm512_loadu_pd(h), a1 = _mm512_loadu_pd(h + 8);
        g20 = _mm512_fmadd_pd(a0,d,g20); g21 = _mm512_fmadd_pd(a1,d,g21);
        _mm512_storeu_pd(h,_mm512_mul_pd(_mm512_mul_pd(d,w20),_mm512_mul_pd(a0,_mm512_sub_pd(one,a0))));
        _mm512_storeu_pd(h + 8,_mm512_mul_pd(_mm512_mul_pd(d,w21),_mm512_mul_pd(a1,_mm512_sub_pd(one,a1))));
    }
    _mm512_storeu_pd(accumulator.dJdWtwo.data(),g20); _mm512_storeu_pd(accumulator.dJdWtwo.data() + 8,g21);

    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++) {
        double *g = accumulator.dJdWone.data() + k * HIDDEN_NODES;
        __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
        __m512d b0 = _mm512_setzero_pd(), b1 = _mm512_setzero_pd();
        size_t i = 0;
        for(;i + 1<rowCount;i += 2) {
            const double *h0 = hidden.data() + i * HIDDEN_NODES;
            const double *h1 = hidden.data() + (i + 1) * HIDDEN_NODES;
            const __m512d x0 = _mm512_set1_pd(data.x.rowData(startRow + i)[k]);
            const __m512d x1 = _mm512_set1_pd(data.x.rowData(startRow + i + 1)[k]);
            a0 = _mm512_fmadd_pd(x0,_mm512_loadu_pd(h0),a0); a1 = _mm512_fmadd_pd(x0,_mm512_loadu_pd(h0 + 8),a1);
            b0 = _mm512_fmadd_pd(x1,_mm512_loadu_pd(h1),b0); b1 = _mm512_fmadd_pd(x1,_mm512_loadu_pd(h1 + 8),b1);
        }
        if(i<rowCount) {
            const double *h = hidden.data() + i * HIDDEN_NODES;
            const __m512d x = _mm512_set1_pd(data.x.rowData(startRow + i)[k]);
            a0 = _mm512_fmadd_pd(x,_mm512_loadu_pd(h),a0); a1 = _mm512_fmadd_pd(x,_mm512_loadu_pd(h + 8),a1);
        }
        _mm512_storeu_pd(g,_mm512_add_pd(_mm512_loadu_pd(g),_mm512_add_pd(a0,b0)));
        _mm512_storeu_pd(g + 8,_mm512_add_pd(_mm512_loadu_pd(g + 8),_mm512_add_pd(a1,b1)));
    }
}
#endif

using TrainingBlockKernel = void (*)(const Dataset &, size_t, size_t, const Network &, ThreadAccumulator &);

static TrainingBlockKernel selectTrainingBlockKernel() {
#if defined(__x86_64__) && defined(__GNUC__)
    if(__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) return processTrainingBlockAVX512;
    if(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return processTrainingBlockAVX2;
#endif
    return processTrainingBlockScalar;
}

static TrainingBlockKernel trainingBlockKernel() {
    static const TrainingBlockKernel kernel = selectTrainingBlockKernel();
    return kernel;
}

static Metrics combineAccumulators(const vector<ThreadAccumulator> &accumulators, Gradients &gradients, size_t rowCount) {
    gradients.clear();
    double squaredError = 0.0;
    double percentageErrorSum = 0.0;
    double maxPercentageError = 0.0;
    for(size_t thread=0;thread<accumulators.size();thread++) {
        const ThreadAccumulator &accumulator = accumulators[thread];
        for(size_t i=0;i<WONE_SIZE;i++) gradients.dJdWone[i] += accumulator.dJdWone[i];
        for(size_t j=0;j<HIDDEN_NODES;j++) gradients.dJdWtwo[j] += accumulator.dJdWtwo[j];
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

static Metrics calculateGradientsBlocked(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, Gradients &gradients) {
    vector<ThreadAccumulator> accumulators(1);
    ThreadAccumulator &accumulator = accumulators[0];
    accumulator.clear();
    const TrainingBlockKernel kernel = trainingBlockKernel();
    for(size_t offset=0;offset<rowCount;offset+=BLOCK_SIZE) {
        const size_t rowsInBlock = min(BLOCK_SIZE,rowCount - offset);
        kernel(data,startRow + offset,rowsInBlock,network,accumulator);
    }
    return combineAccumulators(accumulators,gradients,rowCount);
}

Metrics calculateGradientsAndMetrics(const Dataset &data, size_t startRow, size_t rowCount, const Network &network, Gradients &gradients, size_t threads) {
    (void)threads;
    return calculateGradientsBlocked(data,startRow,rowCount,network,gradients);
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

#ifdef _OPENMP
    if(config.threads > 1) {
        vector<ThreadAccumulator> accumulators(config.threads);
        bool stop = false;
        const size_t blockCount = (batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const TrainingBlockKernel kernel = trainingBlockKernel();

#pragma omp parallel num_threads(static_cast<int>(config.threads)) shared(stop,result,gradients,network,accumulators)
        {
            const size_t thread = static_cast<size_t>(omp_get_thread_num());
            while(true) {
                accumulators[thread].clear();

#pragma omp for schedule(static)
                for(long long block=0;block<static_cast<long long>(blockCount);block++) {
                    const size_t offset = static_cast<size_t>(block) * BLOCK_SIZE;
                    const size_t rowsInBlock = min(BLOCK_SIZE,batchSize - offset);
                    kernel(data,startRow + offset,rowsInBlock,network,accumulators[thread]);
                }

#pragma omp single
                {
                    result.metrics = combineAccumulators(accumulators,gradients,batchSize);
                    const bool targetReached = result.metrics.percentageError <= config.percentageErrorTarget;
                    const bool updateLimitReached = result.updates >= config.maxDescents;
                    const bool logNow = config.logEvery != 0 && result.updates % config.logEvery == 0;
                    if(logNow || targetReached || updateLimitReached) printProgress(batchIndex,result.updates,result.metrics);
                    stop = targetReached || updateLimitReached;
                    if(stop) result.reachedTarget = targetReached;
                    else {
                        applyMomentumUpdate(network,gradients,config.learningRate,config.momentum);
                        result.updates++;
                    }
                }
                if(stop) break;
            }
        }
    }
    else
#endif
    {
        while(true) {
            result.metrics = calculateGradientsBlocked(data,startRow,batchSize,network,gradients);
            const bool targetReached = result.metrics.percentageError <= config.percentageErrorTarget;
            const bool updateLimitReached = result.updates >= config.maxDescents;
            const bool logNow = config.logEvery != 0 && result.updates % config.logEvery == 0;
            if(logNow || targetReached || updateLimitReached) printProgress(batchIndex,result.updates,result.metrics);
            if(targetReached || updateLimitReached) {
                result.reachedTarget = targetReached;
                break;
            }
            applyMomentumUpdate(network,gradients,config.learningRate,config.momentum);
            result.updates++;
        }
    }

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
