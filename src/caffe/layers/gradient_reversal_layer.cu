#include <vector>

#include "caffe/layers/gradient_reversal_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GradientReversalLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_copy(
      bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void GradientReversalLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(-1.0), bottom_diff);
    caffe_gpu_mul(bottom[0]->count(), top_diff, bottom_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GradientReversalLayer);

}
