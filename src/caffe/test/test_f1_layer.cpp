#include <cfloat>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/f1_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

#include <cstring>

namespace caffe {

template <typename TypeParam>
class F1LayerTest : public CPUDeviceTest<TypeParam> {
  typedef TypeParam Dtype;

 protected:
  F1LayerTest()
      : blob_bottom_score_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_bottom_weights_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    vector<int> shape(4);
    shape[0] = 1;
    shape[1] = nClasses_;
    shape[2] = nPixels_ / 10;
    shape[3] = nPixels_ / 10;
    blob_bottom_score_->Reshape(shape);
    shape[1] = 1;
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
    Dtype *weights_data = this->blob_bottom_weights_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_weights_->count(); ++i) {
      weights_data[i] = (weights_data[i] > 0) ? weights_data[i] : Dtype(0);
    }

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
      Dtype maxScore = scores[i];
      for (int c = 1; c < nClasses_; ++c)
          if (scores[c * nPixels_ + i] > maxScore)
              maxScore = scores[c * nPixels_ + i];
      scores[predictedLabels[i] * nPixels_ + i] = maxScore + 1.0;
    }
  }

  virtual ~F1LayerTest() {
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

TYPED_TEST_CASE(F1LayerTest, TestDtypes);

TYPED_TEST(F1LayerTest, TestSetup) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  F1Layer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->nClasses_ - 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(F1LayerTest, TestForward) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  F1Layer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_NEAR(1,         this->blob_top_->data_at(0, 0, 0, 0), 1e-4);
  EXPECT_NEAR(0,         this->blob_top_->data_at(1, 0, 0, 0), 1e-4);
  EXPECT_NEAR(1,         this->blob_top_->data_at(2, 0, 0, 0), 1e-4);
  EXPECT_NEAR(0.8,       this->blob_top_->data_at(3, 0, 0, 0), 1e-4);
  EXPECT_NEAR(1,         this->blob_top_->data_at(4, 0, 0, 0), 1e-4);
  EXPECT_NEAR(2.0 / 3.0, this->blob_top_->data_at(5, 0, 0, 0), 1e-4);
  EXPECT_NEAR(2.0 / 3.0, this->blob_top_->data_at(6, 0, 0, 0), 1e-4);
  EXPECT_NEAR(0.8,       this->blob_top_->data_at(7, 0, 0, 0), 1e-4);
  EXPECT_NEAR(1,         this->blob_top_->data_at(8, 0, 0, 0), 1e-4);
}

TYPED_TEST(F1LayerTest, TestForwardDetectionMode) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  layer_param.mutable_f1_param()->set_distance_mode(F1Parameter::EUCLIDEAN);
  layer_param.mutable_f1_param()->set_distance_threshold(1.5);
  F1Layer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_NEAR(1,         this->blob_top_->data_at(0, 0, 0, 0), 1e-4);
  EXPECT_NEAR(2.0 / 3.0, this->blob_top_->data_at(1, 0, 0, 0), 1e-4);
  EXPECT_NEAR(1,         this->blob_top_->data_at(2, 0, 0, 0), 1e-4);
  EXPECT_NEAR(0.8,       this->blob_top_->data_at(3, 0, 0, 0), 1e-4);
  EXPECT_NEAR(1,         this->blob_top_->data_at(4, 0, 0, 0), 1e-4);
  EXPECT_NEAR(2.0 / 3.0, this->blob_top_->data_at(5, 0, 0, 0), 1e-4);
  EXPECT_NEAR(2.0 / 3.0, this->blob_top_->data_at(6, 0, 0, 0), 1e-4);
  EXPECT_NEAR(0.8,       this->blob_top_->data_at(7, 0, 0, 0), 1e-4);
  EXPECT_NEAR(1,         this->blob_top_->data_at(8, 0, 0, 0), 1e-4);
}

}  // namespace caffe
