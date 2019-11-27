#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  if (bottom.size() > 2) {
    CHECK_EQ(bottom[2]->shape(1), 1) << "Weights may only contain one channel.";
    CHECK_EQ(bottom[0]->count(2), bottom[2]->count(2))
        << "Weights must have the same spatial shape as the input blobs.";
  }
  diff_.ReshapeLike(*bottom[0]);
  if (bottom.size() > 2) weightedDiff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype normalizer, dot;
  if (bottom.size() == 2) {
    dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
    normalizer = bottom[0]->num();
  }
  else {
    for (int n = 0; n < bottom[0]->shape(0); ++n) {
      for (int c = 0; c < bottom[0]->shape(1); ++c) {
        caffe_mul(
            bottom[0]->count(2),
            diff_.cpu_data() + n * diff_.count(1) + c * diff_.count(2),
            bottom[2]->cpu_data() + n * bottom[2]->count(2),
            weightedDiff_.mutable_cpu_data() + n * diff_.count(1) +
            c * diff_.count(2));
      }
    }
    weightSum_ = caffe_cpu_asum(bottom[2]->count(), bottom[2]->cpu_data());
    dot = caffe_cpu_dot(count, diff_.cpu_data(), weightedDiff_.cpu_data());
    normalizer = bottom[0]->shape(1) * std::max(Dtype(1), weightSum_);
  }
  top[0]->mutable_cpu_data()[0] = dot / Dtype(2) / normalizer;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() > 2 && propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to weight inputs.";
  }
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      if (bottom.size() == 2) {
        const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
        caffe_cpu_axpby(
            bottom[i]->count(),              // count
            alpha,                           // alpha
            diff_.cpu_data(),                // a
            Dtype(0),                        // beta
            bottom[i]->mutable_cpu_diff());  // b
      }
      else {
        const Dtype alpha =
            sign * top[0]->cpu_diff()[0] / bottom[i]->shape(1) /
            std::max(Dtype(1), weightSum_);
        caffe_cpu_axpby(
            bottom[i]->count(),
            alpha,
            weightedDiff_.cpu_data(),
            Dtype(0),
            bottom[i]->mutable_cpu_diff());
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
