#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/util/vector_helper.hpp"

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  VLOG(1) << "  " << this->layer_param_.name()
          << " - Reshaping top[0] to "
          << toString(bottom[0]->shape()) << std::endl;
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
