#ifndef CAFFE_IOU_LAYER_HPP_
#define CAFFE_IOU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief Computes the Intersection over Union for a semantic
 *        segmentation task.
 */
template <typename Dtype>
class IoULayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides IoUParameter iou_param,
   *     with IoULayer options:
   *   - background_class (\b optional, default 0).
   *   - axis The label axis (\b optional, default 1).
   */
  explicit IoULayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "IoU"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  // The top blob will contain the intersections over unions for all foreground
  // classes as a vector
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2-3)
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
   *      ignored in the IoU computation. All other pixels will be included
   *      with weight 1. For IoU computation the weights are just an
   *      alternative way of encoding an ignore class.
   * @param top output Blob vector (length 1)
   *   -# @f$ (C-1) @f$
   *      the computed per class IoU: @f$
   *        IoU_c = \frac{\sum_{n=1}^N (1 - \delta\{ w_n \}) \cdot \delta\{ \hat{l}_n - l_n \} \cdot \delta\{ c - l_n \}}{\sum_{n=1}^N (1 - \delta\{ w_n \}) \cdot \delta\{ l_n - c \} + \sum_{n=1}^N (1 - \delta\{ w_n \}) \cdot \delta\{ \hat{l}_n - c \}}
   *      @f$, where @f$
   *      \delta\{x} = \left\{
   *         \begin{array}{lr}
   *            1 & x = 0 \\
   *            0 & \mbox{otherwise}
   *         \end{array} \right.
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- IoULayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int label_axis_, outer_num_, inner_num_;

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// The label of the background class that will not be considered
  int background_label_;
  /// Keeps counts of the number of samples per class.
  Blob<Dtype> intersection_buffer_, union_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
