#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());

  Dtype normalizer, dot;
  if (bottom.size() == 2) {
    caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
    normalizer = bottom[0]->num();
  }
  else {
    for (int n = 0; n < bottom[0]->shape(0); ++n) {
      for (int c = 0; c < bottom[0]->shape(1); ++c) {
        caffe_gpu_mul(
            bottom[0]->count(2),
            diff_.gpu_data() + n * diff_.count(1) + c * diff_.count(2),
            bottom[2]->gpu_data() + n * bottom[2]->count(2),
            weightedDiff_.mutable_gpu_data() + n * diff_.count(1) +
            c * diff_.count(2));
      }
    }
    caffe_gpu_asum(bottom[2]->count(), bottom[2]->cpu_data(), &weightSum_);
    caffe_gpu_dot(count, diff_.gpu_data(), weightedDiff_.gpu_data(), &dot);
    normalizer = bottom[0]->shape(1) * std::max(Dtype(1), weightSum_);
  }
  top[0]->mutable_cpu_data()[0] = dot / Dtype(2) / normalizer;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      if (bottom.size() == 2) {
        const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
        caffe_gpu_axpby(
            bottom[i]->count(),              // count
            alpha,                           // alpha
            diff_.gpu_data(),                // a
            Dtype(0),                        // beta
            bottom[i]->mutable_gpu_diff());  // b
      }
      else{
        const Dtype alpha =
            sign * top[0]->cpu_diff()[0] / bottom[i]->shape(1) /
            std::max(Dtype(1), weightSum_);
        caffe_gpu_axpby(
            bottom[i]->count(),
            alpha,
            weightedDiff_.gpu_data(),
            Dtype(0),
            bottom[i]->mutable_gpu_diff());
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
