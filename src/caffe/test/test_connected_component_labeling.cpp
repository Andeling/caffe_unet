#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/util/connected_component_labeling.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ConnectedComponentLabelingTest : public CPUDeviceTest<TypeParam> {
  typedef TypeParam Dtype;

 protected:
  ConnectedComponentLabelingTest()
      : blob_labels_(new Blob<Dtype>()), blob_weights_(new Blob<Dtype>()),
        blob_instancelabels_(new Blob<int>()) {}

  virtual ~ConnectedComponentLabelingTest() {
    delete blob_labels_;
    delete blob_weights_;
    delete blob_instancelabels_;
  }
  Blob<Dtype> *const blob_labels_;
  Blob<Dtype> *const blob_weights_;
  Blob<int> *const blob_instancelabels_;
};

TYPED_TEST_CASE(ConnectedComponentLabelingTest, TestDtypes);

TYPED_TEST(ConnectedComponentLabelingTest, TestBinary2D) {
  typedef TypeParam Dtype;

  Dtype labels[] = {
      0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
      1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 0, 1, 1, 0, 0, 1, 0,
      0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
      1, 0, 0, 0, 0, 0, 0, 1, 0, 1,
      1, 0, 1, 0, 0, 1, 0, 1, 1, 1,
      1, 1, 1, 0, 1, 1, 1, 1, 0, 0 };

  int instancelabels[] = {
      0, 0, 1, 0, 0, 0, 2, 0, 0, 0,
      1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 0, 3, 3, 0, 0, 4, 0,
      0, 0, 1, 0, 3, 0, 3, 3, 0, 0,
      0, 0, 0, 0, 3, 3, 3, 0, 0, 5,
      6, 0, 0, 0, 0, 0, 0, 5, 0, 5,
      6, 0, 6, 0, 0, 5, 0, 5, 5, 5,
      6, 6, 6, 0, 5, 5, 5, 5, 0, 0 };

  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 8;
  shape[3] = 10;
  this->blob_labels_->Reshape(shape);
  std::memcpy(this->blob_labels_->mutable_cpu_data(),
              labels, 80 * sizeof(Dtype));
  this->blob_instancelabels_->Reshape(shape);

  std::vector<int> nInstances(
      connectedComponentLabeling(
          this->blob_instancelabels_, this->blob_labels_->cpu_data()));
  EXPECT_EQ(1, nInstances.size());
  EXPECT_EQ(6, nInstances[0]);
  for (int i = 0; i < 80; ++i)
      EXPECT_EQ(instancelabels[i], this->blob_instancelabels_->cpu_data()[i]);
}

TYPED_TEST(ConnectedComponentLabelingTest, TestMultilabel2D) {
  typedef TypeParam Dtype;

  Dtype labels[] = {
      0,  0,  1,  0,  0,  0,  1,  0,  0,  0,
      1,  0,  1,  1,  0,  0,  0,  0,  0,  0,
      1,  1,  3,  0,  3,  3,  0,  0,  1,  0,
      0,  0,  3,  0,  1,  0,  1,  1,  0,  0,
      0,  0,  0,  0,  1,  1,  1,  0,  0,  1,
      1,  0,  0,  0,  0,  0,  0,  2,  0,  1,
      1,  0,  7,  0,  0,  1,  0,  2,  1,  1,
      1,  1,  1,  0,  1,  1,  2,  2,  0,  0 };

  int instancelabels[] = {
      0,  0,  1,  0,  0,  0,  2,  0,  0,  0,
      3,  0,  1,  1,  0,  0,  0,  0,  0,  0,
      3,  3,  4,  0,  5,  5,  0,  0,  6,  0,
      0,  0,  4,  0,  7,  0,  7,  7,  0,  0,
      0,  0,  0,  0,  7,  7,  7,  0,  0,  8,
      9,  0,  0,  0,  0,  0,  0, 10,  0,  8,
      9,  0, 11,  0,  0, 12,  0, 10,  8,  8,
      9,  9,  9,  0, 12, 12, 10, 10,  0,  0 };

  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 8;
  shape[3] = 10;
  this->blob_labels_->Reshape(shape);
  std::memcpy(this->blob_labels_->mutable_cpu_data(),
              labels, 80 * sizeof(Dtype));
  this->blob_instancelabels_->Reshape(shape);

  std::vector<int> nInstances(
      connectedComponentLabeling(
          this->blob_instancelabels_, this->blob_labels_->cpu_data()));
  EXPECT_EQ(1, nInstances.size());
  EXPECT_EQ(12, nInstances[0]);
  for (int i = 0; i < 80; ++i)
      EXPECT_EQ(instancelabels[i], this->blob_instancelabels_->cpu_data()[i]);
}

TYPED_TEST(ConnectedComponentLabelingTest, TestMultilabel2DWithIgnore) {
  typedef TypeParam Dtype;

  Dtype labels[] = {
      0,  0,  1,  0,  0,  0,  1,  0,  0,  0,
      1,  0,  1,  1,  0,  0,  0,  0,  0,  0,
      1,  1,  3,  0,  3,  3,  0,  0,  1,  0,
      0,  0,  3,  0,  1,  0,  1,  1,  0,  0,
      0,  0,  0,  0,  1,  1,  1,  0,  0,  1,
      1,  0,  0,  0,  0,  0,  0,  2,  0,  1,
      1,  0,  7,  0,  0,  1,  0,  2,  1,  1,
      1,  1,  1,  0,  1,  1,  2,  2,  0,  0 };

  int instancelabels[] = {
      0,  0,  1,  0,  0,  0,  2,  0,  0,  0,
      3,  0,  1,  1,  0,  0,  0,  0,  0,  0,
      3,  3,  4,  0,  5,  5,  0,  0,  6,  0,
      0,  0,  4,  0,  7,  0,  7,  7,  0,  0,
      0,  0,  0,  0,  7,  7,  7,  0,  0,  8,
      9,  0,  0,  0,  0,  0,  0,  0,  0,  8,
      9,  0, 10,  0,  0, 11,  0,  0,  8,  8,
      9,  9,  9,  0, 11, 11,  0,  0,  0,  0 };

  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 8;
  shape[3] = 10;
  this->blob_labels_->Reshape(shape);
  std::memcpy(this->blob_labels_->mutable_cpu_data(),
              labels, 80 * sizeof(Dtype));
  this->blob_instancelabels_->Reshape(shape);

  std::vector<int> nInstances(
      connectedComponentLabeling(
          this->blob_instancelabels_, this->blob_labels_->cpu_data(),
          (Dtype*)NULL, 1, Dtype(0), true, Dtype(2)));
  EXPECT_EQ(1, nInstances.size());
  EXPECT_EQ(11, nInstances[0]);
  for (int i = 0; i < 80; ++i)
      EXPECT_EQ(instancelabels[i], this->blob_instancelabels_->cpu_data()[i]);
}

TYPED_TEST(ConnectedComponentLabelingTest, TestMultilabel2DWithWeights) {
  typedef TypeParam Dtype;

  Dtype labels[] = {
      0,  0,  1,  0,  0,  0,  1,  0,  0,  0,
      1,  0,  1,  1,  0,  0,  0,  0,  0,  0,
      1,  1,  3,  0,  3,  3,  0,  0,  1,  0,
      0,  0,  3,  0,  1,  0,  1,  1,  0,  0,
      0,  0,  0,  0,  1,  1,  1,  0,  0,  1,
      1,  0,  0,  0,  0,  0,  0,  2,  0,  1,
      1,  0,  7,  0,  0,  1,  0,  2,  1,  1,
      1,  1,  1,  0,  1,  1,  2,  2,  0,  0 };

  Dtype weights[] = {
      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  1,  1,  1,  0,  0,  0,  0,  1,  1,
      1,  1,  1,  1,  0,  0,  0,  0,  1,  1,
      1,  1,  1,  1,  0,  0,  0,  0,  1,  1,
      1,  1,  1,  1,  0,  0,  0,  0,  1,  1,
      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  1,  1,  1,  1,  1,  1,  1,  1,  1 };

  int instancelabels[] = {
      0,  0,  1,  0,  0,  0,  2,  0,  0,  0,
      3,  0,  1,  1,  0,  0,  0,  0,  0,  0,
      3,  3,  4,  0,  0,  0,  0,  0,  5,  0,
      0,  0,  4,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  6,
      7,  0,  0,  0,  0,  0,  0,  0,  0,  6,
      7,  0,  8,  0,  0,  9,  0, 10,  6,  6,
      7,  7,  7,  0,  9,  9, 10, 10,  0,  0 };

  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 8;
  shape[3] = 10;
  this->blob_labels_->Reshape(shape);
  std::memcpy(this->blob_labels_->mutable_cpu_data(),
              labels, 80 * sizeof(Dtype));
  this->blob_weights_->Reshape(shape);
  std::memcpy(this->blob_weights_->mutable_cpu_data(),
              weights, 80 * sizeof(Dtype));
  this->blob_instancelabels_->Reshape(shape);

  std::vector<int> nInstances(
      connectedComponentLabeling(
          this->blob_instancelabels_, this->blob_labels_->cpu_data(),
          this->blob_weights_->cpu_data()));
  EXPECT_EQ(1, nInstances.size());
  EXPECT_EQ(10, nInstances[0]);
  for (int i = 0; i < 80; ++i)
      EXPECT_EQ(instancelabels[i], this->blob_instancelabels_->cpu_data()[i]);
}

TYPED_TEST(ConnectedComponentLabelingTest,
           TestMultilabel2DSpecialBackgroundLabel) {
  typedef TypeParam Dtype;

  Dtype labels[] = {
      4,  4,  1,  4,  4,  4,  1,  4,  4,  4,
      1,  4,  1,  1,  4,  4,  4,  4,  4,  4,
      1,  1,  3,  4,  3,  3,  4,  4,  1,  4,
      4,  4,  3,  4,  1,  4,  1,  1,  4,  4,
      4,  4,  4,  4,  1,  1,  1,  4,  4,  1,
      1,  4,  4,  4,  4,  4,  4,  2,  4,  1,
      1,  4,  7,  4,  4,  1,  4,  2,  1,  1,
      1,  1,  1,  4,  1,  1,  2,  2,  4,  4 };

  int instancelabels[] = {
      0,  0,  1,  0,  0,  0,  2,  0,  0,  0,
      3,  0,  1,  1,  0,  0,  0,  0,  0,  0,
      3,  3,  4,  0,  5,  5,  0,  0,  6,  0,
      0,  0,  4,  0,  7,  0,  7,  7,  0,  0,
      0,  0,  0,  0,  7,  7,  7,  0,  0,  8,
      9,  0,  0,  0,  0,  0,  0, 10,  0,  8,
      9,  0, 11,  0,  0, 12,  0, 10,  8,  8,
      9,  9,  9,  0, 12, 12, 10, 10,  0,  0 };

  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 8;
  shape[3] = 10;
  this->blob_labels_->Reshape(shape);
  std::memcpy(this->blob_labels_->mutable_cpu_data(),
              labels, 80 * sizeof(Dtype));
  this->blob_instancelabels_->Reshape(shape);

  std::vector<int> nInstances(
      connectedComponentLabeling(
          this->blob_instancelabels_, this->blob_labels_->cpu_data(),
          (Dtype*)NULL, 1, Dtype(4)));
  EXPECT_EQ(1, nInstances.size());
  EXPECT_EQ(12, nInstances[0]);
  for (int i = 0; i < 80; ++i)
      EXPECT_EQ(instancelabels[i], this->blob_instancelabels_->cpu_data()[i]);
}

TYPED_TEST(ConnectedComponentLabelingTest,
           TestMultilabel2DSpecialLabelAxis) {
  typedef TypeParam Dtype;

  Dtype labels[] = {
      0,  0,  1,  0,  0,  0,  1,  0,  0,  0,
      1,  0,  1,  1,  0,  0,  0,  0,  0,  0,
      1,  1,  3,  0,  3,  3,  0,  0,  1,  0,
      0,  0,  3,  0,  1,  0,  1,  1,  0,  0,
      0,  0,  0,  0,  1,  1,  1,  0,  0,  1,
      1,  0,  0,  0,  0,  0,  0,  2,  0,  1,
      1,  0,  7,  0,  0,  1,  0,  2,  1,  1,
      1,  1,  1,  0,  1,  1,  2,  2,  0,  0 };

  int instancelabels[] = {
      0,  0,  1,  0,  0,  0,  2,  0,  0,  0,
      3,  0,  1,  1,  0,  0,  0,  0,  0,  0,
      3,  3,  4,  0,  5,  5,  0,  0,  6,  0,
      0,  0,  4,  0,  7,  0,  7,  7,  0,  0,
      0,  0,  0,  0,  7,  7,  7,  0,  0,  8,
      9,  0,  0,  0,  0,  0,  0, 10,  0,  8,
      9,  0, 11,  0,  0, 12,  0, 10,  8,  8,
      9,  9,  9,  0, 12, 12, 10, 10,  0,  0 };

  vector<int> shape(3);
  shape[0] = 1;
  shape[1] = 8;
  shape[2] = 10;
  this->blob_labels_->Reshape(shape);
  std::memcpy(this->blob_labels_->mutable_cpu_data(),
              labels, 80 * sizeof(Dtype));
  this->blob_instancelabels_->Reshape(shape);

  std::vector<int> nInstances(
      connectedComponentLabeling(
          this->blob_instancelabels_, this->blob_labels_->cpu_data(),
          (Dtype*)NULL, 0));
  EXPECT_EQ(1, nInstances.size());
  EXPECT_EQ(12, nInstances[0]);
  for (int i = 0; i < 80; ++i)
      EXPECT_EQ(instancelabels[i], this->blob_instancelabels_->cpu_data()[i]);
}

TYPED_TEST(ConnectedComponentLabelingTest,
           TestMultilabel2DSpecialLabelAxis2) {
  typedef TypeParam Dtype;

  Dtype labels[] = {
      0,  0,  1,  0,  0,  0,  1,  0,  0,  0,
      1,  0,  1,  1,  0,  0,  0,  0,  0,  0,
      1,  1,  3,  0,  3,  3,  0,  0,  1,  0,
      0,  0,  3,  0,  1,  0,  1,  1,  0,  0,
      0,  0,  0,  0,  1,  1,  1,  0,  0,  1,
      1,  0,  0,  0,  0,  0,  0,  2,  0,  1,
      1,  0,  7,  0,  0,  1,  0,  2,  1,  1,
      1,  1,  1,  0,  1,  1,  2,  2,  0,  0 };

  int instancelabels[] = {
      0,  0,  1,  0,  0,  0,  2,  0,  0,  0,
      1,  0,  2,  2,  0,  0,  0,  0,  0,  0,
      1,  1,  2,  0,  3,  3,  0,  0,  4,  0,
      0,  0,  1,  0,  2,  0,  3,  3,  0,  0,
      0,  0,  0,  0,  1,  1,  1,  0,  0,  2,
      1,  0,  0,  0,  0,  0,  0,  2,  0,  3,
      1,  0,  2,  0,  0,  3,  0,  4,  5,  5,
      1,  1,  1,  0,  2,  2,  3,  3,  0,  0 };

  vector<int> shape(4);
  shape[0] = 2;
  shape[1] = 4;
  shape[2] = 1;
  shape[3] = 10;
  this->blob_labels_->Reshape(shape);
  std::memcpy(this->blob_labels_->mutable_cpu_data(),
              labels, 80 * sizeof(Dtype));
  this->blob_instancelabels_->Reshape(shape);

  std::vector<int> nInstances(
      connectedComponentLabeling(
          this->blob_instancelabels_, this->blob_labels_->cpu_data(),
          (Dtype*)NULL, 2));
  EXPECT_EQ(8, nInstances.size());
  EXPECT_EQ(2, nInstances[0]);
  EXPECT_EQ(2, nInstances[1]);
  EXPECT_EQ(4, nInstances[2]);
  EXPECT_EQ(3, nInstances[3]);
  EXPECT_EQ(2, nInstances[4]);
  EXPECT_EQ(3, nInstances[5]);
  EXPECT_EQ(5, nInstances[6]);
  EXPECT_EQ(3, nInstances[7]);
  for (int i = 0; i < 80; ++i)
      EXPECT_EQ(instancelabels[i], this->blob_instancelabels_->cpu_data()[i]);
}

TYPED_TEST(ConnectedComponentLabelingTest, TestBinary3D) {
  typedef TypeParam Dtype;

  Dtype labels[] = {
      0, 0, 1, 0, 0,
      0, 1, 0, 0, 0,
      1, 0, 1, 1, 0,
      0, 0, 0, 0, 0,

      1, 1, 1, 0, 1,
      1, 0, 0, 1, 0,
      0, 0, 1, 0, 1,
      0, 1, 1, 0, 0,

      0, 0, 0, 0, 1,
      1, 1, 0, 0, 1,
      1, 0, 0, 0, 0,
      0, 0, 1, 0, 1,

      1, 0, 1, 0, 0,
      1, 0, 1, 1, 1,
      1, 1, 1, 0, 1,
      1, 1, 1, 0, 0 };

  int instancelabels[] = {
      0, 0, 1, 0, 0,
      0, 2, 0, 0, 0,
      3, 0, 1, 1, 0,
      0, 0, 0, 0, 0,

      1, 1, 1, 0, 1,
      1, 0, 0, 4, 0,
      0, 0, 1, 0, 5,
      0, 1, 1, 0, 0,

      0, 0, 0, 0, 1,
      1, 1, 0, 0, 1,
      1, 0, 0, 0, 0,
      0, 0, 1, 0, 6,

      1, 0, 1, 0, 0,
      1, 0, 1, 1, 1,
      1, 1, 1, 0, 1,
      1, 1, 1, 0, 0 };

  vector<int> shape(5);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 4;
  shape[3] = 4;
  shape[4] = 5;
  this->blob_labels_->Reshape(shape);
  std::memcpy(this->blob_labels_->mutable_cpu_data(),
              labels, 80 * sizeof(Dtype));
  this->blob_instancelabels_->Reshape(shape);

  std::vector<int> nInstances(
      connectedComponentLabeling(
          this->blob_instancelabels_, this->blob_labels_->cpu_data()));
  EXPECT_EQ(1, nInstances.size());
  EXPECT_EQ(6, nInstances[0]);
  for (int i = 0; i < 80; ++i)
      EXPECT_EQ(instancelabels[i], this->blob_instancelabels_->cpu_data()[i]);
}

TYPED_TEST(ConnectedComponentLabelingTest, TestMultilabel3D) {
  typedef TypeParam Dtype;

  Dtype labels[] = {
      0, 0, 1, 0, 0,
      0, 1, 0, 0, 0,
      1, 0, 2, 2, 0,
      0, 0, 0, 0, 0,

      1, 1, 1, 0, 1,
      1, 0, 0, 1, 0,
      0, 0, 4, 0, 1,
      0, 1, 1, 0, 0,

      0, 0, 0, 0, 1,
      1, 2, 0, 0, 1,
      1, 0, 0, 0, 0,
      0, 0, 4, 0, 1,

      1, 0, 1, 0, 0,
      1, 0, 1, 1, 1,
      1, 1, 4, 0, 1,
      1, 1, 4, 0, 0 };

  int instancelabels[] = {
      0, 0, 1, 0, 0,
      0, 2, 0, 0, 0,
      3, 0, 4, 4, 0,
      0, 0, 0, 0, 0,

      1, 1, 1, 0, 5,
      1, 0, 0, 6, 0,
      0, 0, 7, 0, 8,
      0, 9, 9, 0, 0,

      0, 0, 0, 0, 5,
      1, 10, 0, 0, 5,
      1, 0, 0, 0, 0,
      0, 0, 11, 0, 12,

      1, 0, 5, 0, 0,
      1, 0, 5, 5, 5,
      1, 1, 11, 0, 5,
      1, 1, 11, 0, 0 };

  vector<int> shape(5);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 4;
  shape[3] = 4;
  shape[4] = 5;
  this->blob_labels_->Reshape(shape);
  std::memcpy(this->blob_labels_->mutable_cpu_data(),
              labels, 80 * sizeof(Dtype));
  this->blob_instancelabels_->Reshape(shape);

  std::vector<int> nInstances(
      connectedComponentLabeling(
          this->blob_instancelabels_, this->blob_labels_->cpu_data()));
  EXPECT_EQ(1, nInstances.size());
  EXPECT_EQ(12, nInstances[0]);
  for (int i = 0; i < 80; ++i)
      EXPECT_EQ(instancelabels[i], this->blob_instancelabels_->cpu_data()[i]);
}

}
