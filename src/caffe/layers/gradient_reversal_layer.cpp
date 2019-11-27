#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gradient_reversal_layer.hpp"

namespace caffe {

template <typename Dtype>
void GradientReversalLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void GradientReversalLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_copy(
      bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void GradientReversalLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(-1.0), bottom_diff);
    caffe_mul(bottom[0]->count(), top_diff, bottom_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientReversalLayer);
#endif

INSTANTIATE_CLASS(GradientReversalLayer);
REGISTER_LAYER_CLASS(GradientReversal);

}  // namespace caffe
