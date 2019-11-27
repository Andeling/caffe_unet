#ifndef CAFFE_F1_LAYER_HPP_
#define CAFFE_F1_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief Computes the F1-measure for a detection task.
 */
template <typename Dtype>
class F1Layer : public Layer<Dtype> {
 public:
  /**
   * @param param provides F1Parameter f1_param,
   *     with F1Layer options:
   *   - background_class   The background class label (\b optional, default 0).
   *   - axis               The label axis (\b optional, default 1).
   *   - distance_mode      One of EUCLIDEAN or IOU. (\b optional, default IOU)
   *                        EUCLIDEAN - Euclidean distance between CoG of
   *                        GT segment and CoG predicted segment (weighted with
   *                        softmax score),
   *                        IOU - (1 - Intersection over Union) between
   *                        groundtruth segment and predicted segment (argmax
   *                        of scores)
   *   - distance_threshold First the distance matrix between all predictions
   *                        and ground truth objects is computed, then the
   *                        Hungarian algorithm is applied to get an optimal
   *                        assignment. Finally the threshold is applied to
   *                        remove matches with distance exceeding the given
   *                        threshold.
   */
  explicit F1Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "F1"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  // The top blob will contain the intersections over unions for all foreground
  // classes as a vector
  virtual inline int ExactNumTopBlobs() const { return 1; }

private:

  void doSoftmax(Blob<Dtype> &softmaxBlob, Dtype const *scores);
  void doSegmentation(Blob<Dtype> &segmentationBlob, Dtype const *scores,
                      int nClasses);

protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$, a Blob with values in
   *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
   *      the @f$ K = CHW @f$ classes. Each @f$ x_n @f$ is mapped to a predicted
   *      label @f$ \hat{l}_n @f$ given by its maximal index:
   *      @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels @f$ l @f$, an integer-valued Blob with values
   *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
   *      indicating the correct class label among the @f$ K @f$ classes
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the weights @f$ w @f$, a float-valued Blob with values
   *      @f$ l_n \in (0, 1) @f$
   *      indicating the per-pixel loss-weight. Pixels with weight zero are
   *      ignored in the F1 computation. All other pixels will be included
   *      with weight 1. For F1 computation the weights are just an
   *      alternative way of encoding an ignore class.
   * @param top output Blob vector (length 1)
   *   -# @f$ (C-1) @f$
   *      the computed per class F1-measure: @f$
   *        F1_c = 2 \cdot \frac{\mathrm{precision}_c \cdot \mathrm{recall}_c}{\mathrm{precision}_c + \mathrm{recall}_c}
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- F1Layer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int label_axis_, outer_num_, inner_num_;

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// The label of the background class that will not be considered
  int background_label_;
  /// The used distance mode (one of EUCLIDEAN or IOU)
  F1Parameter::DistanceMode distance_mode_;
  /// Matches with distances exceeding this threshold are treated as misses
  Dtype distance_threshold_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
