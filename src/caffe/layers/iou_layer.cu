#include <vector>

#include "caffe/layers/iou_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void IoUForwardGPU(
    const int nthreads, const Dtype* score, const Dtype* label,
    const Dtype* weights, Dtype* intersectionData, Dtype* unionData,
    const int num, const int dim, const int spatial_dim,
    const int num_labels, const bool has_ignore_label_,
    const int ignore_label_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if ((has_ignore_label_ && label_value == ignore_label_) ||
        (weights != NULL && weights[n * spatial_dim + s] == Dtype(0))) return;

    Dtype sc = score[n * dim + s];
    int predicted_label = 0;
    for (int k = 1; k < num_labels; k++) {
      if (score[n * dim + k * spatial_dim + s] > sc) {
        sc = score[n * dim + k * spatial_dim + s];
        predicted_label = k;
      }
    }
    if (label_value == predicted_label)
        intersectionData[label_value * nthreads + index] = 1;
    unionData[label_value * nthreads + index] = 1;
    unionData[predicted_label * nthreads + index] = 1;
  }
}

template <typename Dtype>
void IoULayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* score = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything, we use it here to avoid having
  // to allocate new GPU memory to accumulate intermediate results.
  Dtype* intersectionData = bottom[0]->mutable_gpu_diff();
  union_buffer_.ReshapeLike(*bottom[0]);
  Dtype* unionData = union_buffer_.mutable_gpu_data();

  caffe_gpu_set(bottom[0]->count(), Dtype(0), intersectionData);
  caffe_gpu_set(union_buffer_.count(), Dtype(0), unionData);

  // NOLINT_NEXT_LINE(whitespace/operators)
  if (bottom.size() == 2)
      IoUForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(
              nthreads, score, label, NULL, intersectionData, unionData,
              outer_num_, dim, inner_num_, num_labels, has_ignore_label_,
              ignore_label_);
  else
      IoUForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(
              nthreads, score, label, bottom[2]->gpu_data(), intersectionData,
              unionData, outer_num_, dim, inner_num_, num_labels,
              has_ignore_label_, ignore_label_);

  // get per-class IoU
  Dtype* per_class_iou = top[0]->mutable_cpu_data();
  vector<int> buf_shape(0);
  Blob<Dtype> buf(buf_shape);
  Dtype* bufData = buf.mutable_cpu_data();
  std::stringstream intersectionStream, unionStream;
  for (int l = 0; l < num_labels; l++) {
    if (l == background_label_) continue;
    caffe_gpu_asum(nthreads, intersectionData + l*nthreads,
                   per_class_iou + l - ((l > background_label_) ? 1 : 0));
    intersectionStream << per_class_iou[l - ((l > background_label_) ? 1 : 0)]
                       << " ";
    caffe_gpu_asum(nthreads, unionData + l*nthreads, bufData);
    unionStream << *bufData << " ";
    if (*bufData > 0) {
      per_class_iou[l - ((l > background_label_) ? 1 : 0)] /= *bufData;
    } else {
      per_class_iou[l - ((l > background_label_) ? 1 : 0)] = 0;
    }
  }
  LOG(INFO) << "    " << this->layer_param_.name()
            << " - Per class intersection: " << intersectionStream.str();
  LOG(INFO) << "    " << this->layer_param_.name() << " - Per class union: "
            << unionStream.str();
  // Clear scratch memory to prevent interfering with backward (see #6202).
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
}


template <typename Dtype>
void IoULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {  NOT_IMPLEMENTED;  }
}

INSTANTIATE_LAYER_GPU_FUNCS(IoULayer);
}  // namespace caffe
