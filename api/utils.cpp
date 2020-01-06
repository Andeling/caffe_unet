#include <vector>
#include "caffe/caffe.hpp"

template<typename T>
void copyBlock(
    caffe::Blob<T> const &in, caffe::Blob<T> &out, std::vector<int> shape,
    std::vector<int> inPos, std::vector<int> outPos,
    bool padMirror) {

CHECK_EQ(in.shape().size(), out.shape().size())
    << "Input and output blobs must have same dimensionality";
CHECK_EQ(inPos.size(), out.shape().size())
    << "Input position dimensionality must match blob dimensionality";
CHECK_EQ(outPos.size(), out.shape().size())
    << "Output position dimensionality must match blob dimensionality";
CHECK_EQ(shape.size(), out.shape().size())
    << "block shape dimensionality must match blob dimensionality";

std::vector<int> inShape(in.shape());
std::vector<int> outShape(out.shape());

int nBlobDims = inShape.size();

// Intersect block to crop with output blob
int nElements = 1;
for (int d = 0; d < nBlobDims; ++d) {
    if (outPos[d] < 0) {
    inPos[d] += -outPos[d];
    shape[d] -= -outPos[d];
    if (shape[d] <= 0) return;
    outPos[d] = 0;
    }
    if (outPos[d] + shape[d] > outShape[d]) {
    shape[d] = outShape[d] - outPos[d];
    if (shape[d] <= 0) return;
    }
    nElements *= shape[d];
}

T const *inPtr = in.cpu_data();
T *outPtr = out.mutable_cpu_data();

bool fullInput = true, fullOutput = true;
for (int d = 0; d < nBlobDims && fullInput; ++d)
    fullInput &= inPos[d] == 0 && shape[d] == inShape[d];
for (int d = 0; d < nBlobDims && fullOutput; ++d)
    fullOutput &= outPos[d] == 0 && shape[d] == outShape[d];
if (fullInput && fullOutput) {
    std::memcpy(outPtr, inPtr, nElements * sizeof(T));
    return;
}

std::vector<int> stridesIn(nBlobDims, 1);
std::vector<int> stridesOut(nBlobDims, 1);
for (int d = nBlobDims - 2; d >= 0; --d) {
    stridesIn[d] = stridesIn[d + 1] * inShape[d + 1];
    stridesOut[d] = stridesOut[d + 1] * outShape[d + 1];
}

if (fullInput) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < nElements; ++i) {
    T *outP = outPtr;
    int tmp = i;
    for (int d = nBlobDims - 1; d >= 0; --d) {
        outP += (outPos[d] + (tmp % shape[d])) * stridesOut[d];
        tmp /= shape[d];
    }
    *outP = inPtr[i];
    }
}
else {

    if (padMirror) {

    // Precompute lookup-table for input positions
    std::vector< std::vector<int> > rdPos(nBlobDims);
    for (int d = 0; d < nBlobDims; ++d) {
        rdPos[d].resize(shape[d]);
        for (int i = 0; i < shape[d]; ++i) {
        int p = inPos[d] + i;
        if (p < 0 || p >= inShape[d]) {
            if (p < 0) p = -p;
            int n = p / (inShape[d] - 1);
            if (n % 2 == 0) p = p - n * (inShape[d] - 1);
            else p = (n + 1) * (inShape[d] - 1) - p;
        }
        rdPos[d][i] = p;
        }
    }

    if (fullOutput) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < nElements; ++i)
        {
        T const *inP = inPtr;
        int tmp = i;
        for (int d = nBlobDims - 1; d >= 0; --d) {
            int x = tmp % shape[d];
            tmp /= shape[d];
            inP += rdPos[d][x] * stridesIn[d];
        }
        outPtr[i] = *inP;
        }
    }
    else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < nElements; ++i)
        {
        T const *inP = inPtr;
        T *outP = outPtr;
        int tmp = i;
        for (int d = nBlobDims - 1; d >= 0; --d) {
            int x = tmp % shape[d];
            tmp /= shape[d];
            inP += rdPos[d][x] * stridesIn[d];
            outP += (outPos[d] + x) * stridesOut[d];
        }
        *outP = *inP;
        }
    } // else (fullOutput)
    } // if (padMirror)
    else {

    if (fullOutput) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < nElements; ++i) {
        T const *inP = inPtr;
        int tmp = i;
        int d = nBlobDims - 1;
        for (; d >= 0; --d) {
            int offs = tmp % shape[d];
            tmp /= shape[d];
            int p = offs + inPos[d];
            if (p < 0 || p >= inShape[d]) break;
            inP += p * stridesIn[d];
        }
        outPtr[i] = (d < 0) ? *inP : 0;
        }
    }
    else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < nElements; ++i) {
        T const *inP = inPtr;
        T *outP = outPtr;
        int tmp = i;
        bool valid = true;
        for (int d = nBlobDims - 1; d >= 0; --d) {
            int offs = tmp % shape[d];
            tmp /= shape[d];
            int p = offs + inPos[d];
            valid &= p >= 0 && p < inShape[d];
            inP += p * stridesIn[d];
            outP += (offs + outPos[d]) * stridesOut[d];
        }
        *outP = valid ? *inP : 0;
        }
    }
    }
}
}

template
void copyBlock<float>(
    caffe::Blob<float> const &in, caffe::Blob<float> &out, std::vector<int> shape,
    std::vector<int> inPos, std::vector<int> outPos,
    bool padMirror);