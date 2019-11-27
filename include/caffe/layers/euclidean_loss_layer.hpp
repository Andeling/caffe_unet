#ifndef CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
#define CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the (weighted) Euclidean (L2) loss @f[
 *          E = \frac{1}{2 C \sum_{n=1}^N \sum_{\vec{x} \in \Omega}
 *          w_n(\vec{x})} \sum_{n=1}^N \sum_{\vec{x} \in \Omega} w_n(\vec{x})
 *          \left\| \hat{y}_n(\vec{x}) - y_n(\vec{x}) \right\|_2^2 @f]
 *   for real-valued regression tasks.
 *
 * @param bottom input Blob vector (length 2 or 3)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 *   -# OPTIONAL: @f$ (N \times 1 \times H \times W) @f$
 *      the weights @f$ w \in [0, +\infty]@f$. If not given
 *      @f$ \forall n \in \left\{1, \ldots, N \right\}, \forall \vec{x} \in
 *      \Omega: w_n(\vec{x}) := 1 @f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Euclidean loss: @f$ E = \frac{1}{2 C \sum_{n=1}^N
 *          \sum_{\vec{x} \in \Omega} w_n(\vec{x})} \sum\limits_{n=1}^N
 *          \sum\limits_{\vec{x} \in \Omega} w_n(\vec{x}) \left\|
 *          \hat{y}_n(\vec{x}) - y_n(\vec{x}) \right\|_2^2 @f$
 *
 * This can be used for least-squares regression tasks.  An InnerProductLayer
 * input to a EuclideanLossLayer exactly formulates a linear least squares
 * regression problem. With non-zero weight decay the problem becomes one of
 * ridge regression -- see src/caffe/test/test_gradient_based_solver.cpp for a concrete
 * example wherein we check that the gradients computed for a Net with exactly
 * this structure match hand-computed gradient formulas for ridge regression.
 *
 * (Note: Caffe, and SGD in general, is certainly \b not the best way to solve
 * linear least squares problems! We use it only as an instructive example.)
 */
template <typename Dtype>
class EuclideanLossLayer : public LossLayer<Dtype> {
 public:
  explicit EuclideanLossLayer(const LayerParameter& param)
          : LossLayer<Dtype>(param), diff_(), weightedDiff_(), weightSum_(0) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EuclideanLoss"; }
  /**
   * Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index < 2;
  }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the (weighted) Euclidean error gradient w.r.t. the inputs.
   *
   * Unlike other children of LossLayer, EuclideanLossLayer \b can compute
   * gradients with respect to the label inputs bottom[1] (but still only will
   * if propagate_down[1] is set, due to being produced by learnable parameters
   * or if force_backward is set). In fact, this layer is "commutative" -- the
   * result is the same regardless of the order of the two bottoms.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2 or 3)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}}\left( n, c, \vec{x} \right) =
   *            \frac{1}{C \sum_{n=1}^N
   *            \sum_{\vec{x} \in \Omega} w_n(\vec{x})} w_n(\vec{x}) \left(
   *            \hat{y}_{n,c}(\vec{x}) - y_{n,c}(\vec{x}) \right)
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$; Backward fills their diff with gradients @f$
   *        \frac{\partial E}{\partial y}\left( n, c, \vec{x} \right) =
   *            \frac{1}{C \sum_{n=1}^N
   *            \sum_{\vec{x} \in \Omega} w_n(\vec{x})} w_n(\vec{x}) \left(
   *            y_{n,c}(\vec{x}) - \hat{y}_{n,c}(\vec{x}) \right)
   *      @f$ if propagate_down[1]
   *   -# OPTIONAL: @f$ (N \times 1 \times H \times W) @f$
   *      the weights @f$w@f$
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> weightedDiff_;
  Dtype weightSum_;

};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
