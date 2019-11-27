#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/iou_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void IoULayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ =
      this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  background_label_ = this->layer_param_.iou_param().background_label();
}

template <typename Dtype>
void IoULayer<Dtype>::Reshape(
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
  // Per-class IoU is a vector; 1 axes.
  vector<int> top_shape(1);
  top_shape[0] = bottom[0]->shape(label_axis_) - 1;
  top[0]->Reshape(top_shape);
  intersection_buffer_.Reshape(top_shape);
  union_buffer_.Reshape(top_shape);
}

template <typename Dtype>
void IoULayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* score = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* weights = (bottom.size() > 2) ? bottom[2]->cpu_data() : NULL;
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  caffe_set(intersection_buffer_.count(), Dtype(0),
            intersection_buffer_.mutable_cpu_data());
  caffe_set(union_buffer_.count(), Dtype(0),
            union_buffer_.mutable_cpu_data());
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());

  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if ((has_ignore_label_ && label_value == ignore_label_) ||
          (weights != NULL && weights[i * inner_num_ + j] == 0))
          continue;
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      Dtype s = score[i * dim + j];
      int predicted_label = 0;
      for (int k = 1; k < num_labels; ++k) {
        if (score[i * dim + k * inner_num_ + j] > s) {
          s = score[i * dim + k * inner_num_ + j];
          predicted_label = k;
        }
      }
      if (label_value != background_label_ && label_value == predicted_label) {
        intersection_buffer_.mutable_cpu_data()[
            label_value - ((label_value > background_label_) ? 1 : 0)]++;
        union_buffer_.mutable_cpu_data()[
            label_value - ((label_value > background_label_) ? 1 : 0)]++;
      }
      else {
        if (label_value != background_label_)
            union_buffer_.mutable_cpu_data()[
                label_value -
                ((label_value > background_label_) ? 1 : 0)]++;
        if (predicted_label != background_label_)
            union_buffer_.mutable_cpu_data()[
                predicted_label -
                ((predicted_label > background_label_) ? 1 : 0)]++;
      }
    }
  }
  std::stringstream intersectionStream, unionStream;
  for (int k = 0; k < num_labels - 1; ++k)
  {
    top[0]->mutable_cpu_data()[k] = (union_buffer_.cpu_data()[k] > 0) ?
        (intersection_buffer_.cpu_data()[k] / union_buffer_.cpu_data()[k]) : 0;
    intersectionStream << intersection_buffer_.cpu_data()[k] << " ";
    unionStream << union_buffer_.cpu_data()[k] << " ";
  }
  LOG(INFO) << "    " << this->layer_param_.name()
            << " - Per class intersection: " << intersectionStream.str();
  LOG(INFO) << "    " << this->layer_param_.name() << " - Per class union: "
            << unionStream.str();
}

#ifdef CPU_ONLY
STUB_GPU(IoULayer);
#endif

INSTANTIATE_CLASS(IoULayer);
REGISTER_LAYER_CLASS(IoU);

}  // namespace caffe
