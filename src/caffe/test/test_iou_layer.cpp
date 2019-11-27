#include <cfloat>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/iou_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

#include <cstring>

namespace caffe {

template <typename TypeParam>
class IoULayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  IoULayerTest()
      : blob_bottom_score_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_bottom_weights_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    vector<int> shape(2);
    shape[0] = nPixels_;
    shape[1] = nClasses_;
    blob_bottom_score_->Reshape(shape);
    shape[1] = 1;
    shape.resize(1);
    blob_bottom_label_->Reshape(shape);
    blob_bottom_weights_->Reshape(shape);
    FillBottoms();

    blob_bottom_vec_.push_back(blob_bottom_score_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual void FillBottoms() {
    // fill the probability values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_score_);
    filler.Fill(this->blob_bottom_weights_);
    Dtype* weights_data = blob_bottom_weights_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_weights_->count(); ++i)
        weights_data[i] = (weights_data[i] > 0) ? weights_data[i] : Dtype(0);

    Dtype labelData[] = {
        0, 0, 0, 0, 8, 0, 0, 5, 0, 5,
        0, 0, 1, 0, 0, 0, 6, 5, 5, 5,
        0, 1, 1, 2, 2, 0, 6, 6, 0, 0,
        0, 1, 0, 0, 2, 0, 6, 6, 0, 2,
        0, 0, 0, 0, 0, 7, 7, 0, 0, 2,
        8, 8, 3, 3, 7, 4, 7, 0, 8, 0,
        0, 0, 3, 3, 7, 7, 7, 0, 0, 0,
        1, 0, 0, 7, 7, 7, 4, 0, 9, 9,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 6, 6, 6, 6, 6, 6, 0, 0 };
    std::memcpy(blob_bottom_label_->mutable_cpu_data(),
                labelData, nPixels_ * sizeof(Dtype));

    int predictedLabels[] = {
        7, 7, 0, 0, 0, 0, 0, 5, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 5, 5, 5,
        0, 1, 1, 0, 0, 0, 6, 6, 0, 0,
        0, 1, 1, 2, 2, 0, 6, 6, 0, 0,
        0, 0, 0, 2, 2, 0, 6, 6, 0, 0,
        8, 8, 3, 3, 7, 4, 7, 0, 8, 0,
        0, 0, 3, 3, 7, 7, 7, 0, 0, 0,
        1, 0, 0, 7, 7, 7, 4, 9, 9, 9,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 4, 4, 4, 4, 4, 4, 4, 0, 0 };

    // Modify scores for desired output
    Dtype *scores = this->blob_bottom_score_->mutable_cpu_data();
    for (int i = 0; i < nPixels_; ++i) {
      Dtype maxScore = scores[i * nClasses_];
      for (int c = 1; c < nClasses_; ++c)
          if (scores[i * nClasses_ + c] > maxScore)
              maxScore = scores[i * nClasses_ + c];
      scores[i * nClasses_ + predictedLabels[i]] = maxScore + 1.0;
    }
  }

  virtual ~IoULayerTest() {
    delete blob_bottom_score_;
    delete blob_bottom_label_;
    delete blob_top_;
  }
  static int const nPixels_ = 100;
  static int const nClasses_ = 10;
  Blob<Dtype>* const blob_bottom_score_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_weights_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(IoULayerTest, TestDtypesAndDevices);

TYPED_TEST(IoULayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  IoULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->nClasses_ - 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(IoULayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  IoULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // repeat the forward
  for (int iter = 0; iter < 3; iter++) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_NEAR(0.625,      this->blob_top_->data_at(0, 0, 0, 0), 1e-4);
    EXPECT_NEAR(0.125,      this->blob_top_->data_at(1, 0, 0, 0), 1e-4);
    EXPECT_NEAR(1,          this->blob_top_->data_at(2, 0, 0, 0), 1e-4);
    EXPECT_NEAR(2.0 / 9.0,  this->blob_top_->data_at(3, 0, 0, 0), 1e-4);
    EXPECT_NEAR(0.8,        this->blob_top_->data_at(4, 0, 0, 0), 1e-4);
    EXPECT_NEAR(4.0 / 13.0, this->blob_top_->data_at(5, 0, 0, 0), 1e-4);
    EXPECT_NEAR(2.0 / 3.0,  this->blob_top_->data_at(6, 0, 0, 0), 1e-4);
    EXPECT_NEAR(0.75,       this->blob_top_->data_at(7, 0, 0, 0), 1e-4);
    EXPECT_NEAR(2.0 / 3.0,  this->blob_top_->data_at(8, 0, 0, 0), 1e-4);
  }
}

TYPED_TEST(IoULayerTest, TestForwardWithSpatialAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  this->blob_bottom_score_->Reshape(2, 10, 4, 5);
  vector<int> label_shape(3);
  label_shape[0] = 2; label_shape[1] = 4; label_shape[2] = 5;
  this->blob_bottom_label_->Reshape(label_shape);
  this->FillBottoms();
  layer_param.mutable_iou_param()->set_axis(1);
  IoULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // repeat the forward
  for (int iter = 0; iter < 3; iter++) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    int intersectionCount[this->nClasses_ - 1];
    int unionCount[this->nClasses_ - 1];
    for (int i = 0; i < this->nClasses_ - 1; ++i) {
      intersectionCount[i] = 0;
      unionCount[i] = 0;
    }
    for (int n = 0; n < this->blob_bottom_score_->num(); ++n) {
      for (int h = 0; h < this->blob_bottom_score_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_score_->width(); ++w) {
          int label = static_cast<int>(
              this->blob_bottom_label_->data_at(n, h, w, 0));
          Dtype max_value = this->blob_bottom_score_->data_at(n, 0, h, w);
          int predicted_label = 0;
          for (int j = 1; j < this->nClasses_; ++j) {
            if (this->blob_bottom_score_->data_at(n, j, h, w) > max_value) {
              max_value = this->blob_bottom_score_->data_at(n, j, h, w);
              predicted_label = j;
            }
          }
          if (label != 0 && label == predicted_label) {
            ++intersectionCount[label - 1];
            ++unionCount[label - 1];
          }
          else {
            if (label != 0) ++unionCount[label - 1];
            if (predicted_label != 0) ++unionCount[predicted_label - 1];
          }
        }
      }
    }
    for (int i = 0; i < this->nClasses_ - 1; ++i) {
      Dtype iou = (unionCount[i] != 0) ?
          ((Dtype)intersectionCount[i] / (Dtype)unionCount[i]) : Dtype(0);
      EXPECT_NEAR(this->blob_top_->data_at(i, 0, 0, 0), iou, 1e-7);
    }
  }
}

TYPED_TEST(IoULayerTest, TestForwardWithIgnore) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  int ignore_label = -1;
  layer_param.mutable_loss_param()->set_ignore_label(ignore_label);
  IoULayer<Dtype> layer(layer_param);
  this->blob_bottom_label_->mutable_cpu_data()[2] = ignore_label;
  this->blob_bottom_label_->mutable_cpu_data()[5] = ignore_label;
  this->blob_bottom_label_->mutable_cpu_data()[32] = ignore_label;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // repeat the forward
  for (int iter = 0; iter < 3; iter++) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    int intersectionCount[this->nClasses_ - 1];
    int unionCount[this->nClasses_ - 1];
    for (int i = 0; i < this->nClasses_ - 1; ++i) {
      intersectionCount[i] = 0;
      unionCount[i] = 0;
    }
    for (int i = 0; i < this->nPixels_; ++i) {
      int label = static_cast<int>(
          this->blob_bottom_label_->data_at(i, 0, 0, 0));
      if (label == ignore_label) continue;
      Dtype max_value = this->blob_bottom_score_->data_at(i, 0, 0, 0);
      int predicted_label = 0;
      for (int j = 1; j < this->nClasses_; ++j) {
        if (this->blob_bottom_score_->data_at(i, j, 0, 0) > max_value) {
          max_value = this->blob_bottom_score_->data_at(i, j, 0, 0);
          predicted_label = j;
        }
      }
      if (label != 0 && label == predicted_label) {
        ++intersectionCount[label - 1];
        ++unionCount[label - 1];
      }
      else {
        if (label != 0) ++unionCount[label - 1];
        if (predicted_label != 0) ++unionCount[predicted_label - 1];
      }
    }
    for (int i = 0; i < this->nClasses_ - 1; ++i) {
      Dtype iou = (unionCount[i] != 0) ?
          ((Dtype)intersectionCount[i] / (Dtype)unionCount[i]) : Dtype(0);
      EXPECT_NEAR(this->blob_top_->data_at(i, 0, 0, 0), iou, 1e-7);
    }
  }
}

TYPED_TEST(IoULayerTest, TestForwardWithBackgroundClass) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  int background_label = 2;
  layer_param.mutable_iou_param()->set_background_label(background_label);
  IoULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // repeat the forward
  for (int iter = 0; iter < 3; iter++) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    int intersectionCount[this->nClasses_ - 1];
    int unionCount[this->nClasses_ - 1];
    for (int i = 0; i < this->nClasses_ - 1; ++i) {
      intersectionCount[i] = 0;
      unionCount[i] = 0;
    }
    for (int i = 0; i < this->nPixels_; ++i) {
      int label = static_cast<int>(
          this->blob_bottom_label_->data_at(i, 0, 0, 0));
      Dtype max_value = this->blob_bottom_score_->data_at(i, 0, 0, 0);
      int predicted_label = 0;
      for (int j = 1; j < this->nClasses_; ++j) {
        if (this->blob_bottom_score_->data_at(i, j, 0, 0) > max_value) {
          max_value = this->blob_bottom_score_->data_at(i, j, 0, 0);
          predicted_label = j;
        }
      }
      if (label != background_label && label == predicted_label) {
        ++intersectionCount[label - ((label > background_label) ? 1 : 0)];
        ++unionCount[label - ((label > background_label) ? 1 : 0)];
      }
      else {
        if (label != background_label)
            ++unionCount[label - ((label > background_label) ? 1 : 0)];
        if (predicted_label != background_label)
            ++unionCount[predicted_label -
                         ((predicted_label > background_label) ? 1 : 0)];
      }
    }
    for (int i = 0; i < this->nClasses_ - 1; ++i) {
      Dtype iou = (unionCount[i] != 0) ?
          ((Dtype)intersectionCount[i] / (Dtype)unionCount[i]) : Dtype(0);
      EXPECT_NEAR(this->blob_top_->data_at(i, 0, 0, 0), iou, 1e-7);
    }
  }
}

TYPED_TEST(IoULayerTest, TestForwardWithWeights) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_weights_);
  LayerParameter layer_param;
  IoULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // repeat the forward
  for (int iter = 0; iter < 3; iter++) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    int intersectionCount[this->nClasses_ - 1];
    int unionCount[this->nClasses_ - 1];
    for (int i = 0; i < this->nClasses_ - 1; ++i) {
      intersectionCount[i] = 0;
      unionCount[i] = 0;
    }
    for (int i = 0; i < this->nPixels_; ++i) {
      int label = static_cast<int>(
          this->blob_bottom_label_->data_at(i, 0, 0, 0));
      Dtype weight = this->blob_bottom_weights_->data_at(i, 0, 0, 0);
      if (weight == Dtype(0)) continue;
      Dtype max_value = this->blob_bottom_score_->data_at(i, 0, 0, 0);
      int predicted_label = 0;
      for (int j = 1; j < this->nClasses_; ++j) {
        if (this->blob_bottom_score_->data_at(i, j, 0, 0) > max_value) {
          max_value = this->blob_bottom_score_->data_at(i, j, 0, 0);
          predicted_label = j;
        }
      }
      if (label != 0 && label == predicted_label) {
        ++intersectionCount[label - 1];
        ++unionCount[label - 1];
      }
      else {
        if (label != 0) ++unionCount[label - 1];
        if (predicted_label != 0) ++unionCount[predicted_label - 1];
      }
    }
    for (int i = 0; i < this->nClasses_ - 1; ++i) {
      Dtype iou = (unionCount[i] != 0) ?
          ((Dtype)intersectionCount[i] / (Dtype)unionCount[i]) : Dtype(0);
      EXPECT_NEAR(this->blob_top_->data_at(i, 0, 0, 0), iou, 1e-7);
    }
  }
}

}  // namespace caffe
