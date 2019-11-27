#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/util/tiled_predict.hpp"
#include "caffe/test/test_caffe_main.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

  template<typename Dtype>
  class TiledPredictHelpersTest : public ::testing::Test {
  protected:
    TiledPredictHelpersTest()
            : ::testing::Test() {
    }

    virtual void SetUp() {
    }

    virtual ~TiledPredictHelpersTest() {
    }

  };

  TYPED_TEST_CASE(TiledPredictHelpersTest, TestDtypes);

  TYPED_TEST(TiledPredictHelpersTest, TestRotate2DInplaceNoRotation) {

    // This should be a noop
    std::vector<int> shape;
    shape.push_back(2);
    shape.push_back(3);
    shape.push_back(4);
    shape.push_back(5);
    Blob<TypeParam> in(shape);
    Blob<TypeParam> *wantedOutPtr = &in;
    TypeParam const *wantedOutDataPtr = in.cpu_data();
    Blob<TypeParam> wantedOut(shape);
    for (int i = 0; i < in.count(); ++i)
    {
      in.mutable_cpu_data()[i] = i;
      wantedOut.mutable_cpu_data()[i] = i;
    }

    // Test with usual parameters
    rotate90(in, in, 0);
    ASSERT_EQ(&in, wantedOutPtr);
    ASSERT_EQ(in.cpu_data(), wantedOutDataPtr);
    ASSERT_EQ(in.shape(), wantedOut.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], in.cpu_data()[i]);

    // Positive modulo variant
    rotate90(in, in, 4);
    ASSERT_EQ(&in, wantedOutPtr);
    ASSERT_EQ(in.cpu_data(), wantedOutDataPtr);
    ASSERT_EQ(in.shape(), wantedOut.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], in.cpu_data()[i]);

    // Negative modulo variant
    rotate90(in, in, -8);
    ASSERT_EQ(&in, wantedOutPtr);
    ASSERT_EQ(in.cpu_data(), wantedOutDataPtr);
    ASSERT_EQ(in.shape(), wantedOut.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], in.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestRotate2DNoRotation) {

    // This should just copy the blob
    std::vector<int> shape;
    shape.push_back(2);
    shape.push_back(3);
    shape.push_back(4);
    shape.push_back(5);
    Blob<TypeParam> in(shape);
    Blob<TypeParam> out;
    Blob<TypeParam> wantedOut(shape);
    for (int i = 0; i < in.count(); ++i)
    {
      in.mutable_cpu_data()[i] = i;
      wantedOut.mutable_cpu_data()[i] = i;
    }

    // Test with usual parameters
    rotate90(in, out, 0);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Positive modulo variant
    rotate90(in, out, 4);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Negative modulo variant
    rotate90(in, out, -8);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestRotate2DRotation90) {

    TypeParam inData[120];
    for (int i = 0; i < 120; ++i) inData[i] = i;

    TypeParam outData[] = {

        15, 10,  5,  0,   16, 11,  6,  1,   17, 12,  7,  2,   18, 13,  8,  3,
        19, 14,  9,  4,

        35, 30, 25, 20,   36, 31, 26, 21,   37, 32, 27, 22,   38, 33, 28, 23,
        39, 34, 29, 24,

        55, 50, 45, 40,   56, 51, 46, 41,   57, 52, 47, 42,   58, 53, 48, 43,
        59, 54, 49, 44,

        75, 70, 65, 60,   76, 71, 66, 61,   77, 72, 67, 62,   78, 73, 68, 63,
        79, 74, 69, 64,

        95, 90, 85, 80,   96, 91, 86, 81,   97, 92, 87, 82,   98, 93, 88, 83,
        99, 94, 89, 84,

        115, 110, 105, 100,    116, 111, 106, 101,    117, 112, 107, 102,
        118, 113, 108, 103,    119, 114, 109, 104

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(5);
    outShape.push_back(4);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    // Test with usual parameters
    rotate90(in, out, 1);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Positive modulo variant
    rotate90(in, out, 5);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Negative modulo variant
    rotate90(in, out, -7);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestRotate2DRotation180) {

    TypeParam inData[120];
    for (int i = 0; i < 120; ++i) inData[i] = i;

    TypeParam outData[] = {

        19, 18, 17, 16, 15,   14, 13, 12, 11, 10,    9,  8,  7,  6,  5,
         4,  3,  2,  1,  0,

        39, 38, 37, 36, 35,   34, 33, 32, 31, 30,   29, 28, 27, 26, 25,
        24, 23, 22, 21, 20,

        59, 58, 57, 56, 55,   54, 53, 52, 51, 50,   49, 48, 47, 46, 45,
        44, 43, 42, 41, 40,

        79, 78, 77, 76, 75,   74, 73, 72, 71, 70,   69, 68, 67, 66, 65,
        64, 63, 62, 61, 60,

        99, 98, 97, 96, 95,   94, 93, 92, 91, 90,   89, 88, 87, 86, 85,
        84, 83, 82, 81, 80,

        119, 118, 117, 116, 115,   114, 113, 112, 111, 110,
        109, 108, 107, 106, 105,   104, 103, 102, 101, 100

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    // Test with usual parameters
    rotate90(in, out, 2);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Positive modulo variant
    rotate90(in, out, 6);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Negative modulo variant
    rotate90(in, out, -6);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestRotate2DRotation270) {

    TypeParam inData[120];
    for (int i = 0; i < 120; ++i) inData[i] = i;

    TypeParam outData[] = {

        4,  9, 14, 19,   3,  8, 13, 18,   2,  7, 12, 17,   1,  6, 11, 16,
        0,  5, 10, 15,

        24, 29, 34, 39,   23, 28, 33, 38,   22, 27, 32, 37,   21, 26, 31, 36,
        20, 25, 30, 35,

        44, 49, 54, 59,   43, 48, 53, 58,   42, 47, 52, 57,   41, 46, 51, 56,
        40, 45, 50, 55,

        64, 69, 74, 79,   63, 68, 73, 78,   62, 67, 72, 77,   61, 66, 71, 76,
        60, 65, 70, 75,

        84, 89, 94, 99,   83, 88, 93, 98,   82, 87, 92, 97,   81, 86, 91, 96,
        80, 85, 90, 95,

        104, 109, 114, 119,   103, 108, 113, 118,   102, 107, 112, 117,
        101, 106, 111, 116,   100, 105, 110, 115

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(5);
    outShape.push_back(4);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    // Test with usual parameters
    rotate90(in, out, 3);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Positive modulo variant
    rotate90(in, out, 7);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Negative modulo variant
    rotate90(in, out, -5);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestRotate2DInplaceRotation90) {

    TypeParam inData[120];
    for (int i = 0; i < 120; ++i) inData[i] = i;

    TypeParam outData[] = {

        15, 10,  5,  0,   16, 11,  6,  1,   17, 12,  7,  2,   18, 13,  8,  3,
        19, 14,  9,  4,

        35, 30, 25, 20,   36, 31, 26, 21,   37, 32, 27, 22,   38, 33, 28, 23,
        39, 34, 29, 24,

        55, 50, 45, 40,   56, 51, 46, 41,   57, 52, 47, 42,   58, 53, 48, 43,
        59, 54, 49, 44,

        75, 70, 65, 60,   76, 71, 66, 61,   77, 72, 67, 62,   78, 73, 68, 63,
        79, 74, 69, 64,

        95, 90, 85, 80,   96, 91, 86, 81,   97, 92, 87, 82,   98, 93, 88, 83,
        99, 94, 89, 84,

        115, 110, 105, 100,   116, 111, 106, 101,   117, 112, 107, 102,
        118, 113, 108, 103,   119, 114, 109, 104

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(5);
    outShape.push_back(4);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> *wantedOutPtr = &in;

    // Test with usual parameters
    Blob<TypeParam> &out = in;
    rotate90(in, out, 1);
    ASSERT_EQ(&out, wantedOutPtr);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestRotate2DInplaceRotation180) {

    TypeParam inData[120];
    for (int i = 0; i < 120; ++i) inData[i] = i;

    TypeParam outData[] = {

        19, 18, 17, 16, 15,   14, 13, 12, 11, 10,    9,  8,  7,  6,  5,
         4,  3,  2,  1,  0,

        39, 38, 37, 36, 35,   34, 33, 32, 31, 30,   29, 28, 27, 26, 25,
        24, 23, 22, 21, 20,

        59, 58, 57, 56, 55,   54, 53, 52, 51, 50,   49, 48, 47, 46, 45,
        44, 43, 42, 41, 40,

        79, 78, 77, 76, 75,   74, 73, 72, 71, 70,   69, 68, 67, 66, 65,
        64, 63, 62, 61, 60,

        99, 98, 97, 96, 95,   94, 93, 92, 91, 90,   89, 88, 87, 86, 85,
        84, 83, 82, 81, 80,

        119, 118, 117, 116, 115,   114, 113, 112, 111, 110,
        109, 108, 107, 106, 105,   104, 103, 102, 101, 100

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> *wantedOutPtr = &in;

    // Test with usual parameters
    Blob<TypeParam> &out = in;
    rotate90(in, out, 2);
    ASSERT_EQ(&out, wantedOutPtr);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestRotate2DInplaceRotation270) {

    TypeParam inData[120];
    for (int i = 0; i < 120; ++i) inData[i] = i;

    TypeParam outData[] = {

        4,  9, 14, 19,   3,  8, 13, 18,   2,  7, 12, 17,   1,  6, 11, 16,
        0,  5, 10, 15,

        24, 29, 34, 39,   23, 28, 33, 38,   22, 27, 32, 37,   21, 26, 31, 36,
        20, 25, 30, 35,

        44, 49, 54, 59,   43, 48, 53, 58,   42, 47, 52, 57,   41, 46, 51, 56,
        40, 45, 50, 55,

        64, 69, 74, 79,   63, 68, 73, 78,   62, 67, 72, 77,   61, 66, 71, 76,
        60, 65, 70, 75,

        84, 89, 94, 99,   83, 88, 93, 98,   82, 87, 92, 97,   81, 86, 91, 96,
        80, 85, 90, 95,

        104, 109, 114, 119,   103, 108, 113, 118,   102, 107, 112, 117,
        101, 106, 111, 116,   100, 105, 110, 115

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(5);
    outShape.push_back(4);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> *wantedOutPtr = &in;

    // Test with usual parameters
    Blob<TypeParam> &out = in;
    rotate90(in, out, 3);
    ASSERT_EQ(&out, wantedOutPtr);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, Test3DInplaceNoRotation) {

    // This should be a noop
    std::vector<int> shape;
    shape.push_back(2);
    shape.push_back(2);
    shape.push_back(3);
    shape.push_back(4);
    shape.push_back(5);
    Blob<TypeParam> in(shape);
    Blob<TypeParam> *wantedOutPtr = &in;
    TypeParam const *wantedOutDataPtr = in.cpu_data();
    Blob<TypeParam> wantedOut(shape);
    for (int i = 0; i < in.count(); ++i)
    {
      in.mutable_cpu_data()[i] = i;
      wantedOut.mutable_cpu_data()[i] = i;
    }

    // Test with usual parameters
    rotate90(in, in, 0);
    ASSERT_EQ(&in, wantedOutPtr);
    ASSERT_EQ(in.cpu_data(), wantedOutDataPtr);
    ASSERT_EQ(in.shape(), wantedOut.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], in.cpu_data()[i]);

    // Positive modulo variant
    rotate90(in, in, 4);
    ASSERT_EQ(&in, wantedOutPtr);
    ASSERT_EQ(in.cpu_data(), wantedOutDataPtr);
    ASSERT_EQ(in.shape(), wantedOut.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], in.cpu_data()[i]);

    // Negative modulo variant
    rotate90(in, in, -8);
    ASSERT_EQ(&in, wantedOutPtr);
    ASSERT_EQ(in.cpu_data(), wantedOutDataPtr);
    ASSERT_EQ(in.shape(), wantedOut.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], in.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, Test3DNoRotation) {

    // This should just copy the blob
    std::vector<int> shape;
    shape.push_back(2);
    shape.push_back(2);
    shape.push_back(3);
    shape.push_back(4);
    shape.push_back(5);
    Blob<TypeParam> in(shape);
    Blob<TypeParam> out;
    Blob<TypeParam> wantedOut(shape);
    for (int i = 0; i < in.count(); ++i)
    {
      in.mutable_cpu_data()[i] = i;
      wantedOut.mutable_cpu_data()[i] = i;
    }

    // Test with usual parameters
    rotate90(in, out, 0);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Positive modulo variant
    rotate90(in, out, 4);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Negative modulo variant
    rotate90(in, out, -8);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, Test3DRotation90) {

    TypeParam inData[240];
    for (int i = 0; i < 120; ++i) inData[i] = i;
    for (int i = 0; i < 120; ++i) inData[120 + i] = i;
    inData[189] = 1001;

    TypeParam outData[] = {

        15, 10,  5,  0,   16, 11,  6,  1,   17, 12,  7,  2,   18, 13,  8,  3,
        19, 14,  9,  4,

        35, 30, 25, 20,   36, 31, 26, 21,   37, 32, 27, 22,   38, 33, 28, 23,
        39, 34, 29, 24,

        55, 50, 45, 40,   56, 51, 46, 41,   57, 52, 47, 42,   58, 53, 48, 43,
        59, 54, 49, 44,

        75, 70, 65, 60,   76, 71, 66, 61,   77, 72, 67, 62,   78, 73, 68, 63,
        79, 74, 69, 64,

        95, 90, 85, 80,   96, 91, 86, 81,   97, 92, 87, 82,   98, 93, 88, 83,
        99, 94, 89, 84,

        115, 110, 105, 100,   116, 111, 106, 101,   117, 112, 107, 102,
        118, 113, 108, 103,   119, 114, 109, 104,

        15, 10,  5,  0,   16, 11,  6,  1,   17, 12,  7,  2,   18, 13,  8,  3,
        19, 14,  9,  4,

        35, 30, 25, 20,   36, 31, 26, 21,   37, 32, 27, 22,   38, 33, 28, 23,
        39, 34, 29, 24,

        55, 50, 45, 40,   56, 51, 46, 41,   57, 52, 47, 42,   58, 53, 48, 43,
        59, 54, 49, 44,

        75, 70, 65, 60,   76, 71, 66, 61,   77, 72, 67, 62,   78, 73, 68, 63,
        79, 74, 1001, 64,

        95, 90, 85, 80,   96, 91, 86, 81,   97, 92, 87, 82,   98, 93, 88, 83,
        99, 94, 89, 84,

        115, 110, 105, 100,   116, 111, 106, 101,   117, 112, 107, 102,
        118, 113, 108, 103,   119, 114, 109, 104

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(5);
    outShape.push_back(4);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    // Test with usual parameters
    rotate90(in, out, 1);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Positive modulo variant
    rotate90(in, out, 5);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Negative modulo variant
    rotate90(in, out, -7);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, Test3DRotation180) {

    TypeParam inData[240];
    for (int i = 0; i < 120; ++i) inData[i] = i;
    for (int i = 0; i < 120; ++i) inData[120 + i] = i;
    inData[189] = 1001;

    TypeParam outData[] = {

        19, 18, 17, 16, 15,   14, 13, 12, 11, 10,    9,  8,  7,  6,  5,
         4,  3,  2,  1,  0,

        39, 38, 37, 36, 35,   34, 33, 32, 31, 30,   29, 28, 27, 26, 25,
        24, 23, 22, 21, 20,

        59, 58, 57, 56, 55,   54, 53, 52, 51, 50,   49, 48, 47, 46, 45,
        44, 43, 42, 41, 40,

        79, 78, 77, 76, 75,   74, 73, 72, 71, 70,   69, 68, 67, 66, 65,
        64, 63, 62, 61, 60,

        99, 98, 97, 96, 95,   94, 93, 92, 91, 90,   89, 88, 87, 86, 85,
        84, 83, 82, 81, 80,

        119, 118, 117, 116, 115,   114, 113, 112, 111, 110,
        109, 108, 107, 106, 105,   104, 103, 102, 101, 100,

        19, 18, 17, 16, 15,   14, 13, 12, 11, 10,    9,  8,  7,  6,  5,
         4,  3,  2,  1,  0,

        39, 38, 37, 36, 35,   34, 33, 32, 31, 30,   29, 28, 27, 26, 25,
        24, 23, 22, 21, 20,

        59, 58, 57, 56, 55,   54, 53, 52, 51, 50,   49, 48, 47, 46, 45,
        44, 43, 42, 41, 40,

        79, 78, 77, 76, 75,   74, 73, 72, 71, 70,   1001, 68, 67, 66, 65,
        64, 63, 62, 61, 60,

        99, 98, 97, 96, 95,   94, 93, 92, 91, 90,   89, 88, 87, 86, 85,
        84, 83, 82, 81, 80,

        119, 118, 117, 116, 115,   114, 113, 112, 111, 110,
        109, 108, 107, 106, 105,   104, 103, 102, 101, 100

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    // Test with usual parameters
    rotate90(in, out, 2);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Positive modulo variant
    rotate90(in, out, 6);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Negative modulo variant
    rotate90(in, out, -6);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, Test3DRotation270) {

    TypeParam inData[240];
    for (int i = 0; i < 120; ++i) inData[i] = i;
    for (int i = 0; i < 120; ++i) inData[120 + i] = i;
    inData[189] = 1001;

    TypeParam outData[] = {

        4,  9, 14, 19,   3,  8, 13, 18,   2,  7, 12, 17,   1,  6, 11, 16,
        0,  5, 10, 15,

        24, 29, 34, 39,   23, 28, 33, 38,   22, 27, 32, 37,   21, 26, 31, 36,
        20, 25, 30, 35,

        44, 49, 54, 59,   43, 48, 53, 58,   42, 47, 52, 57,   41, 46, 51, 56,
        40, 45, 50, 55,

        64, 69, 74, 79,   63, 68, 73, 78,   62, 67, 72, 77,   61, 66, 71, 76,
        60, 65, 70, 75,

        84, 89, 94, 99,   83, 88, 93, 98,   82, 87, 92, 97,   81, 86, 91, 96,
        80, 85, 90, 95,

        104, 109, 114, 119,   103, 108, 113, 118,   102, 107, 112, 117,
        101, 106, 111, 116,   100, 105, 110, 115,

        4,  9, 14, 19,   3,  8, 13, 18,   2,  7, 12, 17,   1,  6, 11, 16,
        0,  5, 10, 15,

        24, 29, 34, 39,   23, 28, 33, 38,   22, 27, 32, 37,   21, 26, 31, 36,
        20, 25, 30, 35,

        44, 49, 54, 59,   43, 48, 53, 58,   42, 47, 52, 57,   41, 46, 51, 56,
        40, 45, 50, 55,

        64, 1001, 74, 79,   63, 68, 73, 78,   62, 67, 72, 77,   61, 66, 71, 76,
        60, 65, 70, 75,

        84, 89, 94, 99,   83, 88, 93, 98,   82, 87, 92, 97,   81, 86, 91, 96,
        80, 85, 90, 95,

        104, 109, 114, 119,   103, 108, 113, 118,   102, 107, 112, 117,
        101, 106, 111, 116,   100, 105, 110, 115

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(5);
    outShape.push_back(4);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    // Test with usual parameters
    rotate90(in, out, 3);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Positive modulo variant
    rotate90(in, out, 7);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);

    // Negative modulo variant
    rotate90(in, out, -5);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, Test3DInplaceRotation90) {

    TypeParam inData[240];
    for (int i = 0; i < 120; ++i) inData[i] = i;
    for (int i = 0; i < 120; ++i) inData[120 + i] = i;
    inData[189] = 1001;

    TypeParam outData[] = {

        15, 10,  5,  0,   16, 11,  6,  1,   17, 12,  7,  2,   18, 13,  8,  3,
        19, 14,  9,  4,

        35, 30, 25, 20,   36, 31, 26, 21,   37, 32, 27, 22,   38, 33, 28, 23,
        39, 34, 29, 24,

        55, 50, 45, 40,   56, 51, 46, 41,   57, 52, 47, 42,   58, 53, 48, 43,
        59, 54, 49, 44,

        75, 70, 65, 60,   76, 71, 66, 61,   77, 72, 67, 62,   78, 73, 68, 63,
        79, 74, 69, 64,

        95, 90, 85, 80,   96, 91, 86, 81,   97, 92, 87, 82,   98, 93, 88, 83,
        99, 94, 89, 84,

        115, 110, 105, 100,   116, 111, 106, 101,   117, 112, 107, 102,
        118, 113, 108, 103,   119, 114, 109, 104,

        15, 10,  5,  0,   16, 11,  6,  1,   17, 12,  7,  2,   18, 13,  8,  3,
        19, 14,  9,  4,

        35, 30, 25, 20,   36, 31, 26, 21,   37, 32, 27, 22,   38, 33, 28, 23,
        39, 34, 29, 24,

        55, 50, 45, 40,   56, 51, 46, 41,   57, 52, 47, 42,   58, 53, 48, 43,
        59, 54, 49, 44,

        75, 70, 65, 60,   76, 71, 66, 61,   77, 72, 67, 62,   78, 73, 68, 63,
        79, 74, 1001, 64,

        95, 90, 85, 80,   96, 91, 86, 81,   97, 92, 87, 82,   98, 93, 88, 83,
        99, 94, 89, 84,

        115, 110, 105, 100,   116, 111, 106, 101,   117, 112, 107, 102,
        118, 113, 108, 103,   119, 114, 109, 104

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(5);
    outShape.push_back(4);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> *wantedOutPtr = &in;

    // Test with usual parameters
    Blob<TypeParam> &out = in;
    rotate90(in, out, 1);
    ASSERT_EQ(&out, wantedOutPtr);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, Test3DInplaceRotation180) {

    TypeParam inData[240];
    for (int i = 0; i < 120; ++i) inData[i] = i;
    for (int i = 0; i < 120; ++i) inData[120 + i] = i;
    inData[189] = 1001;

    TypeParam outData[] = {

        19, 18, 17, 16, 15,   14, 13, 12, 11, 10,    9,  8,  7,  6,  5,
         4,  3,  2,  1,  0,

        39, 38, 37, 36, 35,   34, 33, 32, 31, 30,   29, 28, 27, 26, 25,
        24, 23, 22, 21, 20,

        59, 58, 57, 56, 55,   54, 53, 52, 51, 50,   49, 48, 47, 46, 45,
        44, 43, 42, 41, 40,

        79, 78, 77, 76, 75,   74, 73, 72, 71, 70,   69, 68, 67, 66, 65,
        64, 63, 62, 61, 60,

        99, 98, 97, 96, 95,   94, 93, 92, 91, 90,   89, 88, 87, 86, 85,
        84, 83, 82, 81, 80,

        119, 118, 117, 116, 115,   114, 113, 112, 111, 110,
        109, 108, 107, 106, 105,   104, 103, 102, 101, 100,

        19, 18, 17, 16, 15,   14, 13, 12, 11, 10,    9,  8,  7,  6,  5,
         4,  3,  2,  1,  0,

        39, 38, 37, 36, 35,   34, 33, 32, 31, 30,   29, 28, 27, 26, 25,
        24, 23, 22, 21, 20,

        59, 58, 57, 56, 55,   54, 53, 52, 51, 50,   49, 48, 47, 46, 45,
        44, 43, 42, 41, 40,

        79, 78, 77, 76, 75,   74, 73, 72, 71, 70,   1001, 68, 67, 66, 65,
        64, 63, 62, 61, 60,

        99, 98, 97, 96, 95,   94, 93, 92, 91, 90,   89, 88, 87, 86, 85,
        84, 83, 82, 81, 80,

        119, 118, 117, 116, 115,   114, 113, 112, 111, 110,
        109, 108, 107, 106, 105,   104, 103, 102, 101, 100

     };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> *wantedOutPtr = &in;

    // Test with usual parameters
    Blob<TypeParam> &out = in;
    rotate90(in, out, 2);
    ASSERT_EQ(&out, wantedOutPtr);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, Test3DInplaceRotation270) {

    TypeParam inData[240];
    for (int i = 0; i < 120; ++i) inData[i] = i;
    for (int i = 0; i < 120; ++i) inData[120 + i] = i;
    inData[189] = 1001;

    TypeParam outData[] = {

        4,  9, 14, 19,   3,  8, 13, 18,   2,  7, 12, 17,   1,  6, 11, 16,
        0,  5, 10, 15,

        24, 29, 34, 39,   23, 28, 33, 38,   22, 27, 32, 37,   21, 26, 31, 36,
        20, 25, 30, 35,

        44, 49, 54, 59,   43, 48, 53, 58,   42, 47, 52, 57,   41, 46, 51, 56,
        40, 45, 50, 55,

        64, 69, 74, 79,   63, 68, 73, 78,   62, 67, 72, 77,   61, 66, 71, 76,
        60, 65, 70, 75,

        84, 89, 94, 99,   83, 88, 93, 98,   82, 87, 92, 97,   81, 86, 91, 96,
        80, 85, 90, 95,

        104, 109, 114, 119,   103, 108, 113, 118,   102, 107, 112, 117,
        101, 106, 111, 116,   100, 105, 110, 115,

        4,  9, 14, 19,   3,  8, 13, 18,   2,  7, 12, 17,   1,  6, 11, 16,
        0,  5, 10, 15,

        24, 29, 34, 39,   23, 28, 33, 38,   22, 27, 32, 37,   21, 26, 31, 36,
        20, 25, 30, 35,

        44, 49, 54, 59,   43, 48, 53, 58,   42, 47, 52, 57,   41, 46, 51, 56,
        40, 45, 50, 55,


        64, 1001, 74, 79,   63, 68, 73, 78,   62, 67, 72, 77,   61, 66, 71, 76,
        60, 65, 70, 75,

        84, 89, 94, 99,   83, 88, 93, 98,   82, 87, 92, 97,   81, 86, 91, 96,
        80, 85, 90, 95,

        104, 109, 114, 119,   103, 108, 113, 118,   102, 107, 112, 117,
        101, 106, 111, 116,   100, 105, 110, 115

    };

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(5);
    outShape.push_back(4);
    Blob<TypeParam> wantedOut(outShape);
    std::memcpy(wantedOut.mutable_cpu_data(), outData,
                wantedOut.count() * sizeof(TypeParam));

    Blob<TypeParam> *wantedOutPtr = &in;

    // Test with usual parameters
    Blob<TypeParam> &out = in;
    rotate90(in, out, 3);
    ASSERT_EQ(&out, wantedOutPtr);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < out.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip2DInplaceNoFlip) {

    // This should be a noop
    std::vector<int> shape;
    shape.push_back(2);
    shape.push_back(3);
    shape.push_back(4);
    shape.push_back(5);
    Blob<TypeParam> in(shape);
    Blob<TypeParam> *wantedOutPtr = &in;
    TypeParam const *wantedOutDataPtr = in.cpu_data();
    Blob<TypeParam> wantedOut(shape);
    for (int i = 0; i < in.count(); ++i)
    {
      in.mutable_cpu_data()[i] = i;
      wantedOut.mutable_cpu_data()[i] = i;
    }

    flip(in, in, std::vector<bool>(2, false));
    ASSERT_EQ(&in, wantedOutPtr);
    ASSERT_EQ(in.cpu_data(), wantedOutDataPtr);
    ASSERT_EQ(in.shape(), wantedOut.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], in.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip2DNoFlip) {

    // This should just copy the blob
    std::vector<int> shape;
    shape.push_back(2);
    shape.push_back(3);
    shape.push_back(4);
    shape.push_back(5);
    Blob<TypeParam> in(shape);
    Blob<TypeParam> wantedOut(shape);
    for (int i = 0; i < in.count(); ++i)
    {
      in.mutable_cpu_data()[i] = i;
      wantedOut.mutable_cpu_data()[i] = i;
    }

    Blob<TypeParam> out;

    flip(in, out, std::vector<bool>(2, false));
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip2DFlipX) {

    TypeParam inData[120];
    for (int i = 0; i < 120; ++i) inData[i] = i;

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    Blob<TypeParam> wantedOut(outShape);
    for (int n = 0; n < outShape[0]; ++n) {
      for (int c = 0; c < outShape[1]; ++c) {
        TypeParam *sliceStart =
            wantedOut.mutable_cpu_data() +
            (n * outShape[1] + c) * outShape[2] * outShape[3];
        for (int y = 0; y < outShape[2]; ++y) {
          for (int x = 0; x < outShape[3]; ++x) {
            sliceStart[y * outShape[3] + x] =
                ((n * outShape[1] + c) * outShape[2] + y) * outShape[3] +
                (outShape[3] - x - 1);
          }
        }
      }
    }

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    std::vector<bool> flipConfig(2, false);
    flipConfig[1] = true;
    flip(in, out, flipConfig);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip2DFlipY) {

    TypeParam inData[120];
    for (int i = 0; i < 120; ++i) inData[i] = i;

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    Blob<TypeParam> wantedOut(outShape);
    for (int n = 0; n < outShape[0]; ++n) {
      for (int c = 0; c < outShape[1]; ++c) {
        TypeParam *sliceStart =
            wantedOut.mutable_cpu_data() +
            (n * outShape[1] + c) * outShape[2] * outShape[3];
        for (int y = 0; y < outShape[2]; ++y) {
          for (int x = 0; x < outShape[3]; ++x) {
            sliceStart[y * outShape[3] + x] =
                ((n * outShape[1] + c) * outShape[2] + (outShape[2] - y - 1)) *
                outShape[3] + x;
          }
        }
      }
    }

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    std::vector<bool> flipConfig(2, false);
    flipConfig[0] = true;
    flip(in, out, flipConfig);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip2DFlipXY) {

    TypeParam inData[120];
    for (int i = 0; i < 120; ++i) inData[i] = i;

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    Blob<TypeParam> wantedOut(outShape);
    for (int n = 0; n < outShape[0]; ++n) {
      for (int c = 0; c < outShape[1]; ++c) {
        TypeParam *sliceStart =
            wantedOut.mutable_cpu_data() +
            (n * outShape[1] + c) * outShape[2] * outShape[3];
        for (int y = 0; y < outShape[2]; ++y) {
          for (int x = 0; x < outShape[3]; ++x) {
            sliceStart[y * outShape[3] + x] =
                ((n * outShape[1] + c) * outShape[2] + (outShape[2] - y - 1)) *
                outShape[3] + (outShape[3] - x - 1);
          }
        }
      }
    }

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    std::vector<bool> flipConfig(2, true);
    flip(in, out, flipConfig);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip3DInplaceNoFlip) {

    // This should be a noop
    std::vector<int> shape;
    shape.push_back(2);
    shape.push_back(3);
    shape.push_back(4);
    shape.push_back(5);
    shape.push_back(6);
    Blob<TypeParam> in(shape);
    Blob<TypeParam> *wantedOutPtr = &in;
    TypeParam const *wantedOutDataPtr = in.cpu_data();
    Blob<TypeParam> wantedOut(shape);
    for (int i = 0; i < in.count(); ++i)
    {
      in.mutable_cpu_data()[i] = i;
      wantedOut.mutable_cpu_data()[i] = i;
    }

    flip(in, in, std::vector<bool>(3, false));
    ASSERT_EQ(&in, wantedOutPtr);
    ASSERT_EQ(in.cpu_data(), wantedOutDataPtr);
    ASSERT_EQ(in.shape(), wantedOut.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], in.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip3DNoFlip) {

    // This should just copy the blob
    std::vector<int> shape;
    shape.push_back(2);
    shape.push_back(3);
    shape.push_back(4);
    shape.push_back(5);
    shape.push_back(6);
    Blob<TypeParam> in(shape);
    Blob<TypeParam> wantedOut(shape);
    for (int i = 0; i < in.count(); ++i)
    {
      in.mutable_cpu_data()[i] = i;
      wantedOut.mutable_cpu_data()[i] = i;
    }

    Blob<TypeParam> out;

    flip(in, out, std::vector<bool>(3, false));
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip3DFlipX) {

    TypeParam inData[2 * 3 * 4 * 5 * 6];
    for (int i = 0; i < 2 * 3 * 4 * 5 * 6; ++i) inData[i] = i;

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    outShape.push_back(6);
    Blob<TypeParam> wantedOut(outShape);
    for (int n = 0; n < outShape[0]; ++n) {
      for (int c = 0; c < outShape[1]; ++c) {
        TypeParam *blockStart =
            wantedOut.mutable_cpu_data() +
            (n * outShape[1] + c) * outShape[2] * outShape[3] * outShape[4];
        for (int z = 0; z < outShape[2]; ++z) {
          for (int y = 0; y < outShape[3]; ++y) {
            for (int x = 0; x < outShape[4]; ++x) {
              blockStart[(z * outShape[3] + y) * outShape[4] + x] =
                  (((n * outShape[1] + c) * outShape[2] + z) * outShape[3] +
                   y) * outShape[4] + (outShape[4] - x - 1);
            }
          }
        }
      }
    }

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    inShape.push_back(6);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    std::vector<bool> flipConfig(3, false);
    flipConfig[2] = true;
    flip(in, out, flipConfig);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip3DFlipY) {

    TypeParam inData[2 * 3 * 4 * 5 * 6];
    for (int i = 0; i < 2 * 3 * 4 * 5 * 6; ++i) inData[i] = i;

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    outShape.push_back(6);
    Blob<TypeParam> wantedOut(outShape);
    for (int n = 0; n < outShape[0]; ++n) {
      for (int c = 0; c < outShape[1]; ++c) {
        TypeParam *blockStart =
            wantedOut.mutable_cpu_data() +
            (n * outShape[1] + c) * outShape[2] * outShape[3] * outShape[4];
        for (int z = 0; z < outShape[2]; ++z) {
          for (int y = 0; y < outShape[3]; ++y) {
            for (int x = 0; x < outShape[4]; ++x) {
              blockStart[(z * outShape[3] + y) * outShape[4] + x] =
                  (((n * outShape[1] + c) * outShape[2] + z) * outShape[3] +
                   (outShape[3] - y - 1)) * outShape[4] + x;
            }
          }
        }
      }
    }

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    inShape.push_back(6);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    std::vector<bool> flipConfig(3, false);
    flipConfig[1] = true;
    flip(in, out, flipConfig);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip3DFlipZ) {

    TypeParam inData[2 * 3 * 4 * 5 * 6];
    for (int i = 0; i < 2 * 3 * 4 * 5 * 6; ++i) inData[i] = i;

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    outShape.push_back(6);
    Blob<TypeParam> wantedOut(outShape);
    for (int n = 0; n < outShape[0]; ++n) {
      for (int c = 0; c < outShape[1]; ++c) {
        TypeParam *blockStart =
            wantedOut.mutable_cpu_data() +
            (n * outShape[1] + c) * outShape[2] * outShape[3] * outShape[4];
        for (int z = 0; z < outShape[2]; ++z) {
          for (int y = 0; y < outShape[3]; ++y) {
            for (int x = 0; x < outShape[4]; ++x) {
              blockStart[(z * outShape[3] + y) * outShape[4] + x] =
                  (((n * outShape[1] + c) * outShape[2] +
                    (outShape[2] - z - 1)) * outShape[3] + y) * outShape[4] + x;
            }
          }
        }
      }
    }

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    inShape.push_back(6);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    std::vector<bool> flipConfig(3, false);
    flipConfig[0] = true;
    flip(in, out, flipConfig);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip3DFlipXY) {

    TypeParam inData[2 * 3 * 4 * 5 * 6];
    for (int i = 0; i < 2 * 3 * 4 * 5 * 6; ++i) inData[i] = i;

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    outShape.push_back(6);
    Blob<TypeParam> wantedOut(outShape);
    for (int n = 0; n < outShape[0]; ++n) {
      for (int c = 0; c < outShape[1]; ++c) {
        TypeParam *blockStart =
            wantedOut.mutable_cpu_data() +
            (n * outShape[1] + c) * outShape[2] * outShape[3] * outShape[4];
        for (int z = 0; z < outShape[2]; ++z) {
          for (int y = 0; y < outShape[3]; ++y) {
            for (int x = 0; x < outShape[4]; ++x) {
              blockStart[(z * outShape[3] + y) * outShape[4] + x] =
                  (((n * outShape[1] + c) * outShape[2] + z) * outShape[3] +
                   (outShape[3] - y - 1)) * outShape[4] + (outShape[4] - x - 1);
            }
          }
        }
      }
    }

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    inShape.push_back(6);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    std::vector<bool> flipConfig(3, true);
    flipConfig[0] = false;
    flip(in, out, flipConfig);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip3DFlipXZ) {

    TypeParam inData[2 * 3 * 4 * 5 * 6];
    for (int i = 0; i < 2 * 3 * 4 * 5 * 6; ++i) inData[i] = i;

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    outShape.push_back(6);
    Blob<TypeParam> wantedOut(outShape);
    for (int n = 0; n < outShape[0]; ++n) {
      for (int c = 0; c < outShape[1]; ++c) {
        TypeParam *blockStart =
            wantedOut.mutable_cpu_data() +
            (n * outShape[1] + c) * outShape[2] * outShape[3] * outShape[4];
        for (int z = 0; z < outShape[2]; ++z) {
          for (int y = 0; y < outShape[3]; ++y) {
            for (int x = 0; x < outShape[4]; ++x) {
              blockStart[(z * outShape[3] + y) * outShape[4] + x] =
                  (((n * outShape[1] + c) * outShape[2] +
                    (outShape[2] - z - 1)) * outShape[3] + y) * outShape[4] +
                   (outShape[4] - x - 1);
            }
          }
        }
      }
    }

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    inShape.push_back(6);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    std::vector<bool> flipConfig(3, true);
    flipConfig[1] = false;
    flip(in, out, flipConfig);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip3DFlipYZ) {

    TypeParam inData[2 * 3 * 4 * 5 * 6];
    for (int i = 0; i < 2 * 3 * 4 * 5 * 6; ++i) inData[i] = i;

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    outShape.push_back(6);
    Blob<TypeParam> wantedOut(outShape);
    for (int n = 0; n < outShape[0]; ++n) {
      for (int c = 0; c < outShape[1]; ++c) {
        TypeParam *blockStart =
            wantedOut.mutable_cpu_data() +
            (n * outShape[1] + c) * outShape[2] * outShape[3] * outShape[4];
        for (int z = 0; z < outShape[2]; ++z) {
          for (int y = 0; y < outShape[3]; ++y) {
            for (int x = 0; x < outShape[4]; ++x) {
              blockStart[(z * outShape[3] + y) * outShape[4] + x] =
                  (((n * outShape[1] + c) * outShape[2] +
                    (outShape[2] - z - 1)) * outShape[3] +
                   (outShape[3] - y - 1)) * outShape[4] + x;
            }
          }
        }
      }
    }

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    inShape.push_back(6);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    std::vector<bool> flipConfig(3, true);
    flipConfig[2] = false;
    flip(in, out, flipConfig);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestFlip3DFlipXYZ) {

    TypeParam inData[2 * 3 * 4 * 5 * 6];
    for (int i = 0; i < 2 * 3 * 4 * 5 * 6; ++i) inData[i] = i;

    std::vector<int> outShape;
    outShape.push_back(2);
    outShape.push_back(3);
    outShape.push_back(4);
    outShape.push_back(5);
    outShape.push_back(6);
    Blob<TypeParam> wantedOut(outShape);
    for (int n = 0; n < outShape[0]; ++n) {
      for (int c = 0; c < outShape[1]; ++c) {
        TypeParam *blockStart =
            wantedOut.mutable_cpu_data() +
            (n * outShape[1] + c) * outShape[2] * outShape[3] * outShape[4];
        for (int z = 0; z < outShape[2]; ++z) {
          for (int y = 0; y < outShape[3]; ++y) {
            for (int x = 0; x < outShape[4]; ++x) {
              blockStart[(z * outShape[3] + y) * outShape[4] + x] =
                  (((n * outShape[1] + c) * outShape[2] +
                    (outShape[2] - z - 1)) * outShape[3] +
                   (outShape[3] - y - 1)) * outShape[4] + (outShape[4] - x - 1);
            }
          }
        }
      }
    }

    std::vector<int> inShape;
    inShape.push_back(2);
    inShape.push_back(3);
    inShape.push_back(4);
    inShape.push_back(5);
    inShape.push_back(6);
    Blob<TypeParam> in(inShape);
    std::memcpy(in.mutable_cpu_data(), inData, in.count() * sizeof(TypeParam));

    Blob<TypeParam> out;

    std::vector<bool> flipConfig(3, true);
    flip(in, out, flipConfig);
    ASSERT_EQ(wantedOut.shape(), out.shape());
    for (int i = 0; i < in.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], out.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestCopyBlock1DFull) {

    Blob<TypeParam> src(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i) src.mutable_cpu_data()[i] = i;
    Blob<TypeParam> *wantedOut = &src;

    Blob<TypeParam> dst(std::vector<int>(1, 10));

    copyBlock(src, dst, src.shape(), std::vector<int>(1, 0),
              std::vector<int>(1, 0), true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut->cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestCopyBlock1DPartialInternal) {

    Blob<TypeParam> src(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i) src.mutable_cpu_data()[i] = i;

    Blob<TypeParam> dst(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i) dst.mutable_cpu_data()[i] = 2 * i;

    Blob<TypeParam> wantedOut(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i)
        wantedOut.mutable_cpu_data()[i] = (i > 2 && i < 8) ? (i - 2) : 2 * i;

    copyBlock(src, dst, std::vector<int>(1, 5), std::vector<int>(1, 1),
              std::vector<int>(1, 3), true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock1DPartialInternalDifferentShapes) {

    Blob<TypeParam> src(std::vector<int>(1, 7));
    for (int i = 0; i < 7; ++i) src.mutable_cpu_data()[i] = i;

    Blob<TypeParam> dst(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i) dst.mutable_cpu_data()[i] = 2 * i;

    Blob<TypeParam> wantedOut(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i)
        wantedOut.mutable_cpu_data()[i] = (i > 2 && i < 8) ? (i - 2) : 2 * i;

    copyBlock(src, dst, std::vector<int>(1, 5), std::vector<int>(1, 1),
              std::vector<int>(1, 3), true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock1DPartialDifferentShapesValidOutRangeZeroPad) {

    Blob<TypeParam> src(std::vector<int>(1, 7));
    for (int i = 0; i < 7; ++i) src.mutable_cpu_data()[i] = i;

    Blob<TypeParam> dst(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i) dst.mutable_cpu_data()[i] = 2 * i;

    TypeParam wantedOutData[] = { 0, 2, 4, 4, 5, 6, 0, 0, 16, 18 };
    Blob<TypeParam> wantedOut(std::vector<int>(1, 10));
    std::memcpy(
        wantedOut.mutable_cpu_data(), wantedOutData, 10 * sizeof(TypeParam));

    copyBlock(src, dst, std::vector<int>(1, 5), std::vector<int>(1, 4),
              std::vector<int>(1, 3), false);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock1DPartialDifferentShapesValidOutRangeMirror) {

    Blob<TypeParam> src(std::vector<int>(1, 7));
    for (int i = 0; i < 7; ++i) src.mutable_cpu_data()[i] = i;

    Blob<TypeParam> dst(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i) dst.mutable_cpu_data()[i] = 2 * i;

    TypeParam wantedOutData[] = { 0, 2, 4, 4, 5, 6, 5, 4, 16, 18 };
    Blob<TypeParam> wantedOut(std::vector<int>(1, 10));
    std::memcpy(
        wantedOut.mutable_cpu_data(), wantedOutData, 10 * sizeof(TypeParam));

    copyBlock(src, dst, std::vector<int>(1, 5), std::vector<int>(1, 4),
              std::vector<int>(1, 3), true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock1DPartialDifferentShapesInvalidLeftOutRangeMirror) {

    Blob<TypeParam> src(std::vector<int>(1, 7));
    for (int i = 0; i < 7; ++i) src.mutable_cpu_data()[i] = i;

    Blob<TypeParam> dst(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i) dst.mutable_cpu_data()[i] = 2 * i;

    TypeParam wantedOutData[] = { 1, 0, 1, 6, 8, 10, 12, 14, 16, 18 };
    Blob<TypeParam> wantedOut(std::vector<int>(1, 10));
    std::memcpy(
        wantedOut.mutable_cpu_data(), wantedOutData, 10 * sizeof(TypeParam));

    copyBlock(src, dst, std::vector<int>(1, 5), std::vector<int>(1, -3),
              std::vector<int>(1, -2), true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock1DPartialDifferentShapesInvalidLeftOutRangeZero) {

    Blob<TypeParam> src(std::vector<int>(1, 7));
    for (int i = 0; i < 7; ++i) src.mutable_cpu_data()[i] = i;

    Blob<TypeParam> dst(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i) dst.mutable_cpu_data()[i] = 2 * i;

    TypeParam wantedOutData[] = { 0, 0, 1, 6, 8, 10, 12, 14, 16, 18 };
    Blob<TypeParam> wantedOut(std::vector<int>(1, 10));
    std::memcpy(
        wantedOut.mutable_cpu_data(), wantedOutData, 10 * sizeof(TypeParam));

    copyBlock(src, dst, std::vector<int>(1, 5), std::vector<int>(1, -3),
              std::vector<int>(1, -2), false);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock1DPartialDifferentShapesInvalidRightOutRangeMirror) {

    Blob<TypeParam> src(std::vector<int>(1, 7));
    for (int i = 0; i < 7; ++i) src.mutable_cpu_data()[i] = i;

    Blob<TypeParam> dst(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i) dst.mutable_cpu_data()[i] = 2 * i;

    TypeParam wantedOutData[] = { 0, 2, 4, 6, 8, 10, 12, 5, 6, 5 };
    Blob<TypeParam> wantedOut(std::vector<int>(1, 10));
    std::memcpy(
        wantedOut.mutable_cpu_data(), wantedOutData, 10 * sizeof(TypeParam));

    copyBlock(src, dst, std::vector<int>(1, 5), std::vector<int>(1, 5),
              std::vector<int>(1, 7), true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock1DPartialDifferentShapesInvalidRightOutRangeZero) {

    Blob<TypeParam> src(std::vector<int>(1, 7));
    for (int i = 0; i < 7; ++i) src.mutable_cpu_data()[i] = i;

    Blob<TypeParam> dst(std::vector<int>(1, 10));
    for (int i = 0; i < 10; ++i) dst.mutable_cpu_data()[i] = 2 * i;

    TypeParam wantedOutData[] = { 0, 2, 4, 6, 8, 10, 12, 5, 6, 0 };
    Blob<TypeParam> wantedOut(std::vector<int>(1, 10));
    std::memcpy(
        wantedOut.mutable_cpu_data(), wantedOutData, 10 * sizeof(TypeParam));

    copyBlock(src, dst, std::vector<int>(1, 5), std::vector<int>(1, 5),
              std::vector<int>(1, 7), false);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestCopyBlock2DFull) {

    std::vector<int> srcShape(2);
    srcShape[0] = 10;
    srcShape[1] = 20;
    Blob<TypeParam> src(srcShape);
    for (int i = 0; i < src.count(); ++i) src.mutable_cpu_data()[i] = i;

    std::vector<int> dstShape(srcShape);
    Blob<TypeParam> dst(dstShape);
    for (int i = 0; i < dst.count(); ++i) dst.mutable_cpu_data()[i] = 2 * i;

    Blob<TypeParam> wantedOut(srcShape);
    std::memcpy(
        wantedOut.mutable_cpu_data(), src.cpu_data(),
        src.count() * sizeof(TypeParam));

    copyBlock(src, dst, src.shape(), std::vector<int>(2, 0),
              std::vector<int>(2, 0), true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestCopyBlock2DPartial) {

    std::vector<int> srcShape(2);
    srcShape[0] = 6;
    srcShape[1] = 15;
    Blob<TypeParam> src(srcShape);
    for (int i = 0; i < src.count(); ++i) src.mutable_cpu_data()[i] = i;

    std::vector<int> dstShape(2);
    dstShape[0] = 10;
    dstShape[1] = 20;
    Blob<TypeParam> dst(dstShape);
    for (int i = 0; i < dst.count(); ++i) dst.mutable_cpu_data()[i] = 2 * i;

    std::vector<int> blockShape(2);
    blockShape[0] = 3;
    blockShape[1] = 5;

    std::vector<int> srcPos(2);
    srcPos[0] = 1;
    srcPos[1] = 8;

    std::vector<int> dstPos(2);
    dstPos[0] = 3;
    dstPos[1] = 9;

    Blob<TypeParam> wantedOut(dstShape);
    for (int y = 0; y < dstShape[0]; ++y)
        for (int x = 0; x < dstShape[1]; ++x)
            wantedOut.mutable_cpu_data()[y * dstShape[1] + x] =
                (y >= dstPos[0] && y < dstPos[0] + blockShape[0] &&
                 x >= dstPos[1] && x < dstPos[1] + blockShape[1]) ?
                src.cpu_data()[(y - dstPos[0] + srcPos[0]) *
                               srcShape[1] + (x -dstPos[1] + srcPos[1])] :
                dst.cpu_data()[y * dstShape[1] + x];

    copyBlock(src, dst, blockShape, srcPos, dstPos, true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock2DPartialInvalidRightMirror) {

    std::vector<int> srcShape(2);
    srcShape[0] = 8;
    srcShape[1] = 9;
    Blob<TypeParam> src(srcShape);
    for (int i = 0; i < src.count(); ++i) src.mutable_cpu_data()[i] = i;

    std::vector<int> dstShape(2);
    dstShape[0] = 5;
    dstShape[1] = 6;
    Blob<TypeParam> dst(dstShape);
    for (int i = 0; i < dst.count(); ++i) dst.mutable_cpu_data()[i] = 2 * i;

    std::vector<int> blockShape(2);
    blockShape[0] = 6;
    blockShape[1] = 7;

    std::vector<int> srcPos(2);
    srcPos[0] = 5;
    srcPos[1] = 3;

    std::vector<int> dstPos(2);
    dstPos[0] = 3;
    dstPos[1] = 4;

    Blob<TypeParam> wantedOut(dstShape);
    for (int y = 0; y < dstShape[0]; ++y) {
      int yr = y - dstPos[0] + srcPos[0];
      if (yr < 0) yr = -yr;
      int factor = yr / (srcShape[0] - 1);
      if (factor % 2 == 0) yr = yr - factor * (srcShape[0] - 1);
      else yr = (factor + 1) * (srcShape[0] - 1) - yr;
      for (int x = 0; x < dstShape[1]; ++x) {
        int xr = x - dstPos[1] + srcPos[1];
        if (xr < 0) xr = -xr;
        factor = xr / (srcShape[1] - 1);
        if (factor % 2 == 0) xr = xr - factor * (srcShape[1] - 1);
        else xr = (factor + 1) * (srcShape[1] - 1) - xr;
        wantedOut.mutable_cpu_data()[y * dstShape[1] + x] =
            (y >= dstPos[0] && y < dstPos[0] + blockShape[0] &&
             x >= dstPos[1] && x < dstPos[1] + blockShape[1]) ?
            src.cpu_data()[yr * srcShape[1] + xr] :
            dst.cpu_data()[y * dstShape[1] + x];
      }
    }

    copyBlock(src, dst, blockShape, srcPos, dstPos, true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock2DPartialInvalidLeftMirror) {

    std::vector<int> srcShape(2);
    srcShape[0] = 8;
    srcShape[1] = 9;
    Blob<TypeParam> src(srcShape);
    for (int i = 0; i < src.count(); ++i) src.mutable_cpu_data()[i] = i;

    std::vector<int> dstShape(2);
    dstShape[0] = 5;
    dstShape[1] = 6;
    Blob<TypeParam> dst(dstShape);
    for (int i = 0; i < dst.count(); ++i) dst.mutable_cpu_data()[i] = 2 * i;

    std::vector<int> blockShape(2);
    blockShape[0] = 6;
    blockShape[1] = 7;

    std::vector<int> srcPos(2);
    srcPos[0] = 5;
    srcPos[1] = 3;

    std::vector<int> dstPos(2);
    dstPos[0] = -2;
    dstPos[1] = 4;

    Blob<TypeParam> wantedOut(dstShape);
    for (int y = 0; y < dstShape[0]; ++y) {
      int yr = y - dstPos[0] + srcPos[0];
      if (yr < 0) yr = -yr;
      int factor = yr / (srcShape[0] - 1);
      if (factor % 2 == 0) yr = yr - factor * (srcShape[0] - 1);
      else yr = (factor + 1) * (srcShape[0] - 1) - yr;
      for (int x = 0; x < dstShape[1]; ++x) {
        int xr = x - dstPos[1] + srcPos[1];
        if (xr < 0) xr = -xr;
        factor = xr / (srcShape[1] - 1);
        if (factor % 2 == 0) xr = xr - factor * (srcShape[1] - 1);
        else xr = (factor + 1) * (srcShape[1] - 1) - xr;
        wantedOut.mutable_cpu_data()[y * dstShape[1] + x] =
            (y >= dstPos[0] && y < dstPos[0] + blockShape[0] &&
             x >= dstPos[1] && x < dstPos[1] + blockShape[1]) ?
            src.cpu_data()[yr * srcShape[1] + xr] :
            dst.cpu_data()[y * dstShape[1] + x];
      }
    }

    copyBlock(src, dst, blockShape, srcPos, dstPos, true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestCopyBlock5DPartial) {

    std::vector<int> srcShape(5);
    srcShape[0] = 2;
    srcShape[1] = 3;
    srcShape[2] = 6;
    srcShape[3] = 15;
    srcShape[4] = 12;
    Blob<TypeParam> src(srcShape);
    for (int i = 0; i < src.count(); ++i) src.mutable_cpu_data()[i] = i;

    std::vector<int> dstShape(5);
    dstShape[0] = 1;
    dstShape[1] = 3;
    dstShape[2] = 10;
    dstShape[3] = 20;
    dstShape[4] = 20;
    Blob<TypeParam> dst(dstShape);
    for (int i = 0; i < dst.count(); ++i) dst.mutable_cpu_data()[i] = 2 * i;

    std::vector<int> blockShape(5);
    blockShape[0] = 1;
    blockShape[1] = 3;
    blockShape[2] = 5;
    blockShape[3] = 6;
    blockShape[4] = 7;

    std::vector<int> srcPos(5);
    srcPos[0] = 1;
    srcPos[1] = 0;
    srcPos[2] = 1;
    srcPos[3] = 5;
    srcPos[4] = 3;

    std::vector<int> dstPos(5);
    dstPos[0] = 0;
    dstPos[1] = 0;
    dstPos[2] = 2;
    dstPos[3] = 3;
    dstPos[4] = 4;

    Blob<TypeParam> wantedOut(dstShape);
    for (int n = 0; n < dstShape[0]; ++n) {
      for (int c = 0; c < dstShape[1]; ++c) {
        for (int z = 0; z < dstShape[2]; ++z) {
          for (int y = 0; y < dstShape[3]; ++y) {
            for (int x = 0; x < dstShape[4]; ++x) {
              wantedOut.mutable_cpu_data()[
                  (((n * dstShape[1] + c) * dstShape[2] + z) *
                   dstShape[3] + y) * dstShape[4] + x] =
                  (n >= dstPos[0] && n < dstPos[0] + blockShape[0] &&
                   c >= dstPos[1] && c < dstPos[1] + blockShape[1] &&
                   z >= dstPos[2] && z < dstPos[2] + blockShape[2] &&
                   y >= dstPos[3] && y < dstPos[3] + blockShape[3] &&
                   x >= dstPos[4] && x < dstPos[4] + blockShape[4]) ?
                  src.cpu_data()[
                      ((((n - dstPos[0] + srcPos[0]) *
                         srcShape[1] + (c - dstPos[1] + srcPos[1])) *
                        srcShape[2] + (z - dstPos[2] + srcPos[2])) *
                       srcShape[3] + (y - dstPos[3] + srcPos[3])) *
                      srcShape[4] + (x - dstPos[4] + srcPos[4])] :
                dst.cpu_data()[(((n * dstShape[1] + c) * dstShape[2] + z) *
                                dstShape[3] + y) * dstShape[4] + x];
            }
          }
        }
      }
    }

    copyBlock(src, dst, blockShape, srcPos, dstPos, true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestCopyBlock5DPartialMirror) {

    std::vector<int> srcShape(5);
    srcShape[0] = 2;
    srcShape[1] = 3;
    srcShape[2] = 6;
    srcShape[3] = 8;
    srcShape[4] = 9;
    Blob<TypeParam> src(srcShape);
    for (int i = 0; i < src.count(); ++i) src.mutable_cpu_data()[i] = i;

    std::vector<int> dstShape(5);
    dstShape[0] = 1;
    dstShape[1] = 3;
    dstShape[2] = 10;
    dstShape[3] = 20;
    dstShape[4] = 20;
    Blob<TypeParam> dst(dstShape);
    for (int i = 0; i < dst.count(); ++i) dst.mutable_cpu_data()[i] = 2 * i;

    std::vector<int> blockShape(5);
    blockShape[0] = 1;
    blockShape[1] = 3;
    blockShape[2] = 5;
    blockShape[3] = 6;
    blockShape[4] = 7;

    std::vector<int> srcPos(5);
    srcPos[0] = 1;
    srcPos[1] = 0;
    srcPos[2] = 1;
    srcPos[3] = 5;
    srcPos[4] = 3;

    std::vector<int> dstPos(5);
    dstPos[0] = 0;
    dstPos[1] = 0;
    dstPos[2] = 2;
    dstPos[3] = 3;
    dstPos[4] = 4;

    Blob<TypeParam> wantedOut(dstShape);
    for (int n = 0; n < dstShape[0]; ++n) {
      int nr = n - dstPos[0] + srcPos[0];
      if (nr < 0) nr = -nr;
      int factor = nr / (srcShape[0] - 1);
      if (factor % 2 == 0) nr = nr - factor * (srcShape[0] - 1);
      else nr = (factor + 1) * (srcShape[0] - 1) - nr;
      for (int c = 0; c < dstShape[1]; ++c) {
        int cr = c - dstPos[1] + srcPos[1];
        if (cr < 0) cr = -cr;
        factor = cr / (srcShape[1] - 1);
        if (factor % 2 == 0) cr = cr - factor * (srcShape[1] - 1);
        else cr = (factor + 1) * (srcShape[1] - 1) - cr;
        for (int z = 0; z < dstShape[2]; ++z) {
          int zr = z - dstPos[2] + srcPos[2];
          if (zr < 0) zr = -zr;
          factor = zr / (srcShape[2] - 1);
          if (factor % 2 == 0) zr = zr - factor * (srcShape[2] - 1);
          else zr = (factor + 1) * (srcShape[2] - 1) - zr;
          for (int y = 0; y < dstShape[3]; ++y) {
            int yr = y - dstPos[3] + srcPos[3];
            if (yr < 0) yr = -yr;
            factor = yr / (srcShape[3] - 1);
            if (factor % 2 == 0) yr = yr - factor * (srcShape[3] - 1);
            else yr = (factor + 1) * (srcShape[3] - 1) - yr;
            for (int x = 0; x < dstShape[4]; ++x) {
              int xr = x - dstPos[4] + srcPos[4];
              if (xr < 0) xr = -xr;
              factor = xr / (srcShape[4] - 1);
              if (factor % 2 == 0) xr = xr - factor * (srcShape[4] - 1);
              else xr = (factor + 1) * (srcShape[4] - 1) - xr;
              wantedOut.mutable_cpu_data()[
                  (((n * dstShape[1] + c) * dstShape[2] + z) *
                   dstShape[3] + y) * dstShape[4] + x] =
                  (n >= dstPos[0] && n < dstPos[0] + blockShape[0] &&
                   c >= dstPos[1] && c < dstPos[1] + blockShape[1] &&
                   z >= dstPos[2] && z < dstPos[2] + blockShape[2] &&
                   y >= dstPos[3] && y < dstPos[3] + blockShape[3] &&
                   x >= dstPos[4] && x < dstPos[4] + blockShape[4]) ?
                  src.cpu_data()[
                      (((nr * srcShape[1] + cr) * srcShape[2] + zr) *
                       srcShape[3] + yr) * srcShape[4] + xr] :
                dst.cpu_data()[(((n * dstShape[1] + c) * dstShape[2] + z) *
                                dstShape[3] + y) * dstShape[4] + x];
            }
          }
        }
      }
    }

    copyBlock(src, dst, blockShape, srcPos, dstPos, true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest, TestCopyBlock5DPartialZero) {

    std::vector<int> srcShape(5);
    srcShape[0] = 2;
    srcShape[1] = 3;
    srcShape[2] = 6;
    srcShape[3] = 8;
    srcShape[4] = 9;
    Blob<TypeParam> src(srcShape);
    for (int i = 0; i < src.count(); ++i) src.mutable_cpu_data()[i] = i;

    std::vector<int> dstShape(5);
    dstShape[0] = 1;
    dstShape[1] = 3;
    dstShape[2] = 10;
    dstShape[3] = 20;
    dstShape[4] = 20;
    Blob<TypeParam> dst(dstShape);
    for (int i = 0; i < dst.count(); ++i) dst.mutable_cpu_data()[i] = 2 * i;

    std::vector<int> blockShape(5);
    blockShape[0] = 1;
    blockShape[1] = 3;
    blockShape[2] = 5;
    blockShape[3] = 6;
    blockShape[4] = 7;

    std::vector<int> srcPos(5);
    srcPos[0] = 1;
    srcPos[1] = 0;
    srcPos[2] = 1;
    srcPos[3] = 5;
    srcPos[4] = 3;

    std::vector<int> dstPos(5);
    dstPos[0] = 0;
    dstPos[1] = 0;
    dstPos[2] = 2;
    dstPos[3] = 3;
    dstPos[4] = 4;

    Blob<TypeParam> wantedOut(dstShape);
    for (int n = 0; n < dstShape[0]; ++n) {
      int nr = n - dstPos[0] + srcPos[0];
      for (int c = 0; c < dstShape[1]; ++c) {
        int cr = c - dstPos[1] + srcPos[1];
        for (int z = 0; z < dstShape[2]; ++z) {
          int zr = z - dstPos[2] + srcPos[2];
          for (int y = 0; y < dstShape[3]; ++y) {
            int yr = y - dstPos[3] + srcPos[3];
            for (int x = 0; x < dstShape[4]; ++x) {
              int xr = x - dstPos[4] + srcPos[4];
              bool skip =
                  nr < 0 || nr >= srcShape[0] ||
                  cr < 0 || cr >= srcShape[1] ||
                  zr < 0 || zr >= srcShape[2] ||
                  yr < 0 || yr >= srcShape[3] ||
                  xr < 0 || xr >= srcShape[4];
              wantedOut.mutable_cpu_data()[
                  (((n * dstShape[1] + c) * dstShape[2] + z) *
                   dstShape[3] + y) * dstShape[4] + x] =
                  (n >= dstPos[0] && n < dstPos[0] + blockShape[0] &&
                   c >= dstPos[1] && c < dstPos[1] + blockShape[1] &&
                   z >= dstPos[2] && z < dstPos[2] + blockShape[2] &&
                   y >= dstPos[3] && y < dstPos[3] + blockShape[3] &&
                   x >= dstPos[4] && x < dstPos[4] + blockShape[4]) ?
                  (skip ? 0.0f : src.cpu_data()[
                      (((nr * srcShape[1] + cr) * srcShape[2] + zr) *
                       srcShape[3] + yr) * srcShape[4] + xr]) :
                dst.cpu_data()[(((n * dstShape[1] + c) * dstShape[2] + z) *
                                dstShape[3] + y) * dstShape[4] + x];
            }
          }
        }
      }
    }

    copyBlock(src, dst, blockShape, srcPos, dstPos, false);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock5DPartialInvalidRightMirror) {

    std::vector<int> srcShape(5);
    srcShape[0] = 2;
    srcShape[1] = 3;
    srcShape[2] = 6;
    srcShape[3] = 8;
    srcShape[4] = 9;
    Blob<TypeParam> src(srcShape);
    for (int i = 0; i < src.count(); ++i) src.mutable_cpu_data()[i] = i;

    std::vector<int> dstShape(5);
    dstShape[0] = 1;
    dstShape[1] = 3;
    dstShape[2] = 10;
    dstShape[3] = 5;
    dstShape[4] = 6;
    Blob<TypeParam> dst(dstShape);
    for (int i = 0; i < dst.count(); ++i) dst.mutable_cpu_data()[i] = 2 * i;

    std::vector<int> blockShape(5);
    blockShape[0] = 1;
    blockShape[1] = 3;
    blockShape[2] = 5;
    blockShape[3] = 6;
    blockShape[4] = 7;

    std::vector<int> srcPos(5);
    srcPos[0] = 1;
    srcPos[1] = 0;
    srcPos[2] = 1;
    srcPos[3] = 5;
    srcPos[4] = 3;

    std::vector<int> dstPos(5);
    dstPos[0] = 0;
    dstPos[1] = 0;
    dstPos[2] = 2;
    dstPos[3] = 3;
    dstPos[4] = 4;

    Blob<TypeParam> wantedOut(dstShape);
    for (int n = 0; n < dstShape[0]; ++n) {
      int nr = n - dstPos[0] + srcPos[0];
      if (nr < 0) nr = -nr;
      int factor = nr / (srcShape[0] - 1);
      if (factor % 2 == 0) nr = nr - factor * (srcShape[0] - 1);
      else nr = (factor + 1) * (srcShape[0] - 1) - nr;
      for (int c = 0; c < dstShape[1]; ++c) {
        int cr = c - dstPos[1] + srcPos[1];
        if (cr < 0) cr = -cr;
        factor = cr / (srcShape[1] - 1);
        if (factor % 2 == 0) cr = cr - factor * (srcShape[1] - 1);
        else cr = (factor + 1) * (srcShape[1] - 1) - cr;
        for (int z = 0; z < dstShape[2]; ++z) {
          int zr = z - dstPos[2] + srcPos[2];
          if (zr < 0) zr = -zr;
          factor = zr / (srcShape[2] - 1);
          if (factor % 2 == 0) zr = zr - factor * (srcShape[2] - 1);
          else zr = (factor + 1) * (srcShape[2] - 1) - zr;
          for (int y = 0; y < dstShape[3]; ++y) {
            int yr = y - dstPos[3] + srcPos[3];
            if (yr < 0) yr = -yr;
            factor = yr / (srcShape[3] - 1);
            if (factor % 2 == 0) yr = yr - factor * (srcShape[3] - 1);
            else yr = (factor + 1) * (srcShape[3] - 1) - yr;
            for (int x = 0; x < dstShape[4]; ++x) {
              int xr = x - dstPos[4] + srcPos[4];
              if (xr < 0) xr = -xr;
              factor = xr / (srcShape[4] - 1);
              if (factor % 2 == 0) xr = xr - factor * (srcShape[4] - 1);
              else xr = (factor + 1) * (srcShape[4] - 1) - xr;
              wantedOut.mutable_cpu_data()[
                  (((n * dstShape[1] + c) * dstShape[2] + z) *
                   dstShape[3] + y) * dstShape[4] + x] =
                  (n >= dstPos[0] && n < dstPos[0] + blockShape[0] &&
                   c >= dstPos[1] && c < dstPos[1] + blockShape[1] &&
                   z >= dstPos[2] && z < dstPos[2] + blockShape[2] &&
                   y >= dstPos[3] && y < dstPos[3] + blockShape[3] &&
                   x >= dstPos[4] && x < dstPos[4] + blockShape[4]) ?
                  src.cpu_data()[
                      (((nr * srcShape[1] + cr) * srcShape[2] + zr) *
                       srcShape[3] + yr) * srcShape[4] + xr] :
                dst.cpu_data()[(((n * dstShape[1] + c) * dstShape[2] + z) *
                                dstShape[3] + y) * dstShape[4] + x];
            }
          }
        }
      }
    }

    copyBlock(src, dst, blockShape, srcPos, dstPos, true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

  TYPED_TEST(TiledPredictHelpersTest,
             TestCopyBlock5DPartialInvalidLeftMirror) {

    std::vector<int> srcShape(5);
    srcShape[0] = 2;
    srcShape[1] = 3;
    srcShape[2] = 6;
    srcShape[3] = 8;
    srcShape[4] = 9;
    Blob<TypeParam> src(srcShape);
    for (int i = 0; i < src.count(); ++i) src.mutable_cpu_data()[i] = i;

    std::vector<int> dstShape(5);
    dstShape[0] = 1;
    dstShape[1] = 3;
    dstShape[2] = 10;
    dstShape[3] = 5;
    dstShape[4] = 6;
    Blob<TypeParam> dst(dstShape);
    for (int i = 0; i < dst.count(); ++i) dst.mutable_cpu_data()[i] = 2 * i;

    std::vector<int> blockShape(5);
    blockShape[0] = 1;
    blockShape[1] = 3;
    blockShape[2] = 5;
    blockShape[3] = 6;
    blockShape[4] = 7;

    std::vector<int> srcPos(5);
    srcPos[0] = 1;
    srcPos[1] = 0;
    srcPos[2] = 1;
    srcPos[3] = 5;
    srcPos[4] = 3;

    std::vector<int> dstPos(5);
    dstPos[0] = 0;
    dstPos[1] = 0;
    dstPos[2] = 2;
    dstPos[3] = -2;
    dstPos[4] = 4;

    Blob<TypeParam> wantedOut(dstShape);
    for (int n = 0; n < dstShape[0]; ++n) {
      int nr = n - dstPos[0] + srcPos[0];
      if (nr < 0) nr = -nr;
      int factor = nr / (srcShape[0] - 1);
      if (factor % 2 == 0) nr = nr - factor * (srcShape[0] - 1);
      else nr = (factor + 1) * (srcShape[0] - 1) - nr;
      for (int c = 0; c < dstShape[1]; ++c) {
        int cr = c - dstPos[1] + srcPos[1];
        if (cr < 0) cr = -cr;
        factor = cr / (srcShape[1] - 1);
        if (factor % 2 == 0) cr = cr - factor * (srcShape[1] - 1);
        else cr = (factor + 1) * (srcShape[1] - 1) - cr;
        for (int z = 0; z < dstShape[2]; ++z) {
          int zr = z - dstPos[2] + srcPos[2];
          if (zr < 0) zr = -zr;
          factor = zr / (srcShape[2] - 1);
          if (factor % 2 == 0) zr = zr - factor * (srcShape[2] - 1);
          else zr = (factor + 1) * (srcShape[2] - 1) - zr;
          for (int y = 0; y < dstShape[3]; ++y) {
            int yr = y - dstPos[3] + srcPos[3];
            if (yr < 0) yr = -yr;
            factor = yr / (srcShape[3] - 1);
            if (factor % 2 == 0) yr = yr - factor * (srcShape[3] - 1);
            else yr = (factor + 1) * (srcShape[3] - 1) - yr;
            for (int x = 0; x < dstShape[4]; ++x) {
              int xr = x - dstPos[4] + srcPos[4];
              if (xr < 0) xr = -xr;
              factor = xr / (srcShape[4] - 1);
              if (factor % 2 == 0) xr = xr - factor * (srcShape[4] - 1);
              else xr = (factor + 1) * (srcShape[4] - 1) - xr;
              wantedOut.mutable_cpu_data()[
                  (((n * dstShape[1] + c) * dstShape[2] + z) *
                   dstShape[3] + y) * dstShape[4] + x] =
                  (n >= dstPos[0] && n < dstPos[0] + blockShape[0] &&
                   c >= dstPos[1] && c < dstPos[1] + blockShape[1] &&
                   z >= dstPos[2] && z < dstPos[2] + blockShape[2] &&
                   y >= dstPos[3] && y < dstPos[3] + blockShape[3] &&
                   x >= dstPos[4] && x < dstPos[4] + blockShape[4]) ?
                  src.cpu_data()[
                      (((nr * srcShape[1] + cr) * srcShape[2] + zr) *
                       srcShape[3] + yr) * srcShape[4] + xr] :
                dst.cpu_data()[(((n * dstShape[1] + c) * dstShape[2] + z) *
                                dstShape[3] + y) * dstShape[4] + x];
            }
          }
        }
      }
    }

    copyBlock(src, dst, blockShape, srcPos, dstPos, true);

    for (int i = 0; i < dst.count(); ++i)
        ASSERT_EQ(wantedOut.cpu_data()[i], dst.cpu_data()[i]);
  }

}
