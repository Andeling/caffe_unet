#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/f1_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/connected_component_labeling.hpp"
#include "caffe/util/hungarian_algorithm.hpp"

namespace caffe {

template <typename Dtype>
void F1Layer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ =
      this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  background_label_ = this->layer_param_.f1_param().background_label();
  distance_mode_ = this->layer_param_.f1_param().distance_mode();
  distance_threshold_ = this->layer_param_.f1_param().distance_threshold();
}

template <typename Dtype>
void F1Layer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.iou_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if( bottom.size() == 3) {
    CHECK_EQ(outer_num_ * inner_num_, bottom[2]->count())
        << "Number of weights must match number of predictions; "
        << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
        << "weight count (number of weights) must be N*H*W, "
        << "with positive float values";
  }
  // Per-class F1 is a vector; 1 axes.
  vector<int> top_shape(1);
  top_shape[0] = bottom[0]->shape(label_axis_) - 1;
  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void F1Layer<Dtype>::doSoftmax(
    Blob<Dtype> &softmaxBlob, Dtype const *scores) {
  int nClasses = softmaxBlob.shape(label_axis_);
  int nPixels = softmaxBlob.count() / nClasses;
  Dtype *sm = softmaxBlob.mutable_cpu_data();
  Dtype *expSum = new Dtype[nPixels];
  std::memset(expSum, 0, nPixels * sizeof(Dtype));
  for (int c = 0; c < nClasses; ++c) {
    for (int j = 0; j < nPixels; j++) {
      sm[c * nPixels + j] = std::exp(scores[c * nPixels + j]);
      expSum[j] += sm[c * nPixels + j];
    }
  }
  for (int c = 0; c < nClasses; ++c)
      for (int j = 0; j < nPixels; j++) sm[c * nPixels + j] /= expSum[j];
  delete[] expSum;
}

template<typename Dtype>
void F1Layer<Dtype>::doSegmentation(
    Blob<Dtype> &segmentationBlob, Dtype const *scores, int nClasses) {
  Dtype *seg = segmentationBlob.mutable_cpu_data();
  int nPixels = segmentationBlob.count();
  for (int j = 0; j < nPixels; j++) {
    seg[j] = Dtype(0);
    Dtype maxScore = scores[j];
    for (int c = 1; c < nClasses; ++c) {
      if (scores[c * nPixels + j] > maxScore) {
        maxScore = scores[c * nPixels + j];
        seg[j] = Dtype(c);
      }
    }
  }
}

template <typename Dtype>
void F1Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);

  // nObj is the number of ALL ground truth objects per class
  std::vector<int> nObj(num_labels - 1, 0);
  // nPred is the number of ALL predicted objects per class
  std::vector<int> nPred(num_labels - 1, 0);
  // nTP is the number of ALL true positives per class
  std::vector<int> nTP(num_labels - 1, 0);

  // Sequentially process all samples
  for (int i = 0; i < outer_num_; ++i) {

    std::vector<int> sliceShape(bottom[1]->shape());
    for (int d = 0; d <= label_axis_; ++d) sliceShape[d] = 1;
    Dtype const *scores = bottom[0]->cpu_data() + i * dim;
    Dtype const *labels = bottom[1]->cpu_data() + i * inner_num_;
    Dtype const *weights = (bottom.size() > 2) ?
        (bottom[2]->cpu_data() + i * inner_num_) : NULL;

    // gtLabels and predLabels are the labels for ground truth and predicted
    // objects in the distance matrix.
    //   I.e. gtLabels.size() == distanceMatrix.size()      (#rows)    and
    //        predLabels.size() == distanceMatrix[0].size() (#columns)
    std::vector<int> gtLabels, predLabels;

    // The distance matrix only contains candidates within the user-defined
    // distance threshold, all other objects are already removed
    // to speed up the Hungarian Algorithm
    std::vector< std::vector<double> > distanceMatrix;

    // Segmentation
    Blob<Dtype> predictedLabelsBlob(sliceShape);
    Dtype const *predictedLabels = predictedLabelsBlob.cpu_data();
    doSegmentation(predictedLabelsBlob, scores, num_labels);

    // Connected component labeling
    Blob<int> gtInstanceBlob(sliceShape);
    int const *gtInstances = gtInstanceBlob.cpu_data();
    std::vector<int> nObjects(
        connectedComponentLabeling(
            &gtInstanceBlob, labels, weights, label_axis_,
            Dtype(background_label_), has_ignore_label_,
            Dtype(ignore_label_)));
    Blob<int> predInstanceBlob(sliceShape);
    int const *predictedInstances = predInstanceBlob.cpu_data();
    std::vector<int> nPredictedObjects(
        connectedComponentLabeling(
            &predInstanceBlob, predictedLabels, (Dtype*)NULL,
            label_axis_, Dtype(background_label_)));

    int nRows = nObjects[0];
    int nCols = nPredictedObjects[0];

    if (nRows == 0 || nCols == 0) continue;

    if (distance_mode_ == F1Parameter::IOU) {

      // Compute confusion matrix
      int *confusion = new int[nRows * nCols];

      // Store size in pixels and class label per GT and predicted object
      int *gtSizes = new int[nRows];
      int *gtLbl = new int[nRows];
      int *predSizes = new int[nCols];
      int *predLbl = new int[nCols];

      std::memset(confusion, 0, nRows * nCols * sizeof(int));
      std::memset(gtSizes, 0, nRows * sizeof(int));
      std::memset(predSizes, 0, nCols * sizeof(int));
      int lastGtInstanceSeen = 0, lastPredInstanceSeen = 0;
      for (int j = 0; j < inner_num_; ++j) {
        if (gtInstances[j] > 0 && predictedInstances[j] > 0)
            ++confusion[(gtInstances[j] - 1) * nCols +
                        (predictedInstances[j] - 1)];
        if (gtInstances[j] > 0) {
          ++gtSizes[gtInstances[j] - 1];
          if (gtInstances[j] > lastGtInstanceSeen) {
            lastGtInstanceSeen = gtInstances[j];
            ++nObj[labels[j] - ((labels[j] > background_label_) ? 1 : 0)];
            gtLbl[gtInstances[j] - 1] = labels[j];
          }
        }
        if (predictedInstances[j] > 0) {
          ++predSizes[predictedInstances[j] - 1];
          if (predictedInstances[j] > lastPredInstanceSeen) {
            lastPredInstanceSeen = predictedInstances[j];
            ++nPred[predictedLabels[j] -
                    ((predictedLabels[j] > background_label_) ? 1 : 0)];
            predLbl[predictedInstances[j] - 1] = predictedLabels[j];
          }
        }
      }

      // Compute IoU matrix
      double *iou = new double[nRows * nCols];
      for (int r = 0; r < nRows; ++r) {
        for (int c = 0; c < nCols; ++c) {
          iou[r * nCols + c] = static_cast<double>(confusion[r * nCols + c]) /
              (static_cast<double>(gtSizes[r] + predSizes[c] -
                                   confusion[r * nCols + c]));
        }
      }

      // Free helper arrays
      delete[] confusion;
      delete[] gtSizes;
      delete[] predSizes;

      // Search for rows and columns that contain matching candidates with
      // (1 - IoU) <= distance_threshold_
      std::vector<int> gtCandidateMap;
      for (int r = 0; r < nRows; ++r) {
        int c = 0;
        for (; c < nCols && (1.0 - iou[r * nCols + c]) > distance_threshold_;
             ++c);
        if (c < nCols) gtCandidateMap.push_back(r);
      }
      std::vector<int> predCandidateMap;
      for (int c = 0; c < nCols; ++c) {
        int r = 0;
        for (; r < nRows && (1.0 - iou[r * nCols + c]) > distance_threshold_;
             ++r);
        if (r < nRows) predCandidateMap.push_back(c);
      }

      if (gtCandidateMap.size() == 0 || predCandidateMap.size() == 0) continue;

      // Construct distance matrix from matching candidates
      gtLabels.resize(gtCandidateMap.size());
      predLabels.resize(predCandidateMap.size());
      distanceMatrix.resize(
          gtCandidateMap.size(), std::vector<double>(predCandidateMap.size()));
      for (size_t rIdx = 0; rIdx < gtCandidateMap.size(); ++rIdx)
          gtLabels[rIdx] = gtLbl[gtCandidateMap[rIdx]];
      delete[] gtLbl;
      for (size_t cIdx = 0; cIdx < predCandidateMap.size(); ++cIdx)
          predLabels[cIdx] = predLbl[predCandidateMap[cIdx]];
      delete[] predLbl;
      for (size_t rIdx = 0; rIdx < gtCandidateMap.size(); ++rIdx) {
        int r = gtCandidateMap[rIdx];
        for (size_t cIdx = 0; cIdx < predCandidateMap.size(); ++cIdx) {
          int c = predCandidateMap[cIdx];
          distanceMatrix[rIdx][cIdx] = 1.0 - iou[r * nCols + c];
        }
      }
      delete[] iou;
    }
    else {

      // Softmax scores
      std::vector<int> labelSliceShape(bottom[0]->shape());
      for (int d = 0; d < label_axis_; ++d) labelSliceShape[d] = 1;
      Blob<Dtype> softmaxBlob(labelSliceShape);
      Dtype const *smScores = softmaxBlob.cpu_data();
      doSoftmax(softmaxBlob, scores);

      int nDims = bottom[0]->shape().size() - label_axis_ - 1;

      // Detection
      double *gtPositions = new double[nDims * nRows];
      double *gtSizes = new double[nRows];
      int *gtLbl = new int[nRows];
      double *predPositions = new double[nDims * nCols];
      double *predSizes = new double[nCols];
      int *predLbl = new int[nCols];

      std::memset(gtPositions, 0, nDims * nRows * sizeof(double));
      std::memset(gtSizes, 0, nRows * sizeof(double));
      std::memset(predPositions, 0, nDims * nCols * sizeof(double));
      std::memset(predSizes, 0, nCols * sizeof(double));
      int lastGtInstanceSeen = 0, lastPredInstanceSeen = 0;
      int *pos = new int[nDims];
      for (int j = 0; j < inner_num_; ++j) {
        if (gtInstances[j] == 0 && predictedInstances[j] == 0) continue;
        int tmp = j;
        for (int d = nDims - 1; d >= 0; --d) {
          pos[d] = tmp % sliceShape[d + label_axis_ + 1];
          tmp /= sliceShape[d + label_axis_ + 1];
        }
        if (gtInstances[j] > 0) {
          for (int d = 0; d < nDims; ++d)
              gtPositions[(gtInstances[j] - 1) * nDims + d] += pos[d];
          ++gtSizes[gtInstances[j] - 1];
          if (gtInstances[j] > lastGtInstanceSeen) {
            lastGtInstanceSeen = gtInstances[j];
            ++nObj[labels[j] - ((labels[j] > background_label_) ? 1 : 0)];
            gtLbl[gtInstances[j] - 1] = labels[j];
          }
        }
        if (predictedInstances[j] > 0) {
          for (int d = 0; d < nDims; ++d)
              predPositions[(predictedInstances[j] - 1) * nDims + d] +=
                  smScores[static_cast<int>(predictedLabels[j]) *
                           inner_num_ + j] * pos[d];
          predSizes[predictedInstances[j] - 1] +=
              smScores[static_cast<int>(predictedLabels[j]) * inner_num_ + j];
          if (predictedInstances[j] > lastPredInstanceSeen) {
            lastPredInstanceSeen = predictedInstances[j];
            ++nPred[predictedLabels[j] -
                    ((predictedLabels[j] > background_label_) ? 1 : 0)];
            predLbl[predictedInstances[j] - 1] = predictedLabels[j];
          }
        }
      }
      delete[] pos;

      // Normalize positions
      for (int r = 0; r < nRows; ++r)
          for (int d = 0; d < nDims; ++d)
              gtPositions[r * nDims + d] /= gtSizes[r];
      delete[] gtSizes;
      for (int c = 0; c < nCols; ++c)
          for (int d = 0; d < nDims; ++d)
              predPositions[c * nDims + d] /= predSizes[c];
      delete[] predSizes;

      // Compute distance matrix
      double *pdist = new double[nRows * nCols];
      for (int r = 0; r < nRows; ++r) {
        for (int c = 0; c < nCols; ++c) {
          double sqrSum = 0.0;
          for (int d = 0; d < nDims; ++d)
              sqrSum +=
                  (gtPositions[r * nDims + d] - predPositions[c * nDims + d]) *
                  (gtPositions[r * nDims + d] - predPositions[c * nDims + d]);
          pdist[r * nCols + c] = std::sqrt(sqrSum);
        }
      }
      delete[] gtPositions;
      delete[] predPositions;

      // Search for rows and columns that contain matching candidates with
      // distance <= distance_threshold_
      std::vector<int> gtCandidateMap;
      for (int r = 0; r < nRows; ++r) {
        int c = 0;
        for (; c < nCols && pdist[r * nCols + c] > distance_threshold_; ++c);
        if (c < nCols) gtCandidateMap.push_back(r);
      }
      std::vector<int> predCandidateMap;
      for (int c = 0; c < nCols; ++c) {
        int r = 0;
        for (; r < nRows && pdist[r * nCols + c] > distance_threshold_; ++r);
        if (r < nRows) predCandidateMap.push_back(c);
      }

      if (gtCandidateMap.size() == 0 || predCandidateMap.size() == 0) continue;

      // Construct distance matrix from matching candidates
      gtLabels.resize(gtCandidateMap.size());
      predLabels.resize(predCandidateMap.size());
      distanceMatrix.resize(
          gtCandidateMap.size(), std::vector<double>(predCandidateMap.size()));
      for (size_t rIdx = 0; rIdx < gtCandidateMap.size(); ++rIdx)
          gtLabels[rIdx] = gtLbl[gtCandidateMap[rIdx]];
      delete[] gtLbl;
      for (size_t cIdx = 0; cIdx < predCandidateMap.size(); ++cIdx)
          predLabels[cIdx] = predLbl[predCandidateMap[cIdx]];
      delete[] predLbl;
      for (size_t rIdx = 0; rIdx < gtCandidateMap.size(); ++rIdx) {
        int r = gtCandidateMap[rIdx];
        for (size_t cIdx = 0; cIdx < predCandidateMap.size(); ++cIdx) {
          int c = predCandidateMap[cIdx];
          distanceMatrix[rIdx][cIdx] = pdist[r * nCols + c];
        }
      }
      delete[] pdist;
    }

    // Apply Hungarian method
    std::vector<int> assignment;
    HungarianAlgorithm ha;
    ha.Solve(distanceMatrix, assignment);

    // Compute per-class TP
    for (size_t j = 0; j < assignment.size(); ++j)
        if (gtLabels[j] == predLabels[assignment[j]])
            ++nTP[gtLabels[j] - ((gtLabels[j] > background_label_) ? 1 : 0)];
  }

  std::stringstream nTPStream, nPredStream, nObjStream;
  for (int c = 0; c < num_labels - 1; ++c) {

    nTPStream << nTP[c] << " ";
    nPredStream << nPred[c] << " ";
    nObjStream << nObj[c] << " ";

    // Compute precision and recall
    double prec = (nPred[c] != 0) ?
        (static_cast<double>(nTP[c]) / static_cast<double>(nPred[c])) : 0.0;
    double rec = (nObj[c] != 0) ?
        (static_cast<double>(nTP[c]) / static_cast<double>(nObj[c])) : 0.0;

    // Compute F1 score
    top[0]->mutable_cpu_data()[c] =
        (prec + rec > 0.0) ? (2.0 * prec * rec / (prec + rec)) : 0.0;
  }
  LOG(INFO) << "    " << this->layer_param_.name()
            << " - Per class true positive count: " << nTPStream.str();
  LOG(INFO) << "    " << this->layer_param_.name()
            << " - Per class prediction count: " << nPredStream.str();
  LOG(INFO) << "    " << this->layer_param_.name()
            << " - Per class object count: " << nObjStream.str();
}

// #ifdef CPU_ONLY
// STUB_GPU(F1Layer);
// #endif

INSTANTIATE_CLASS(F1Layer);
REGISTER_LAYER_CLASS(F1);

}  // namespace caffe
