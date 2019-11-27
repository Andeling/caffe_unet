#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/util/vector_helper.hpp"
#include "caffe/test/test_caffe_main.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

  template<typename Dtype>
  class VectorHelperTest : public ::testing::Test {
  protected:
    VectorHelperTest(){
    }

    virtual void SetUp() {
    }

    virtual ~VectorHelperTest() {
    }

  };

  TYPED_TEST_CASE(VectorHelperTest, TestDtypes);

  TYPED_TEST(VectorHelperTest, TestBasics) {
    vector<TypeParam> A = make_vec<TypeParam>(5,4,3,2,1);
    EXPECT_EQ( "(5,4,3,2,1)", toString(A));
  }

  TYPED_TEST(VectorHelperTest, TestShift3D) {
    vector<TypeParam> A = make_vec<TypeParam>(
        3,3,2,3,
        1,2,5,4,
        2,1,1,2,
        0,0,0,1);
    vector<TypeParam> B = m_shift3D<TypeParam>(7, 4, 12, A);

    vector<TypeParam> R = make_vec<TypeParam>(
        3,3,2,10,
        1,2,5,8,
        2,1,1,14,
        0,0,0,1);
    EXPECT_EQ( toString(R), toString(B));
  }

  TYPED_TEST(VectorHelperTest, TestScale3D) {
    // test scaling of unit matrix
    vector<TypeParam> A = make_vec<TypeParam>(
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1);
    vector<TypeParam> B = m_scale3D<TypeParam>(4, 3, 2, A);
    EXPECT_EQ( "(4,0,0,0,"
               "0,3,0,0,"
               "0,0,2,0,"
               "0,0,0,1)", toString(B));

    // test scaling forward and inverse
    vector<TypeParam> C = make_vec<TypeParam>(
        3,3,2,3,
        1,2,5,4,
        2,1,1,2,
        0,0,0,1);

    vector<TypeParam> D = m_scale3D<TypeParam>(4, 3, 2, C);
    vector<TypeParam> E = m_scale3D<TypeParam>(1.0/4, 1.0/3, 1.0/2, D);

    EXPECT_EQ( toString(E), toString(C));
  }

  TYPED_TEST(VectorHelperTest, TestRotate3D) {
    vector<TypeParam> A = make_vec<TypeParam>(
        4,0,0,0,
        0,3,0,0,
        0,0,2,0,
        0,0,0,1);

    // rotation around first axis
    vector<TypeParam> B = m_rotate3D<TypeParam>(90, 0, 0, A);
    for (int i = 0; i < 16; ++i) {
      B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
    }

    EXPECT_EQ( "(4,0,0,0,"
               "0,0,-2,0,"
               "0,3,0,0,"
               "0,0,0,1)", toString(B));

    // rotation around second axis
    B = m_rotate3D<TypeParam>(0, 90, 0, A);
    for (int i = 0; i < 16; ++i) {
      B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
    }

    EXPECT_EQ( "(0,0,-2,0,"
               "0,3,0,0,"
               "4,0,0,0,"
               "0,0,0,1)", toString(B));

    // rotation around third axis
    B = m_rotate3D<TypeParam>(0, 0, 90, A);
    for (int i = 0; i < 16; ++i) {
      B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
    }

    EXPECT_EQ( "(0,-3,0,0,"
               "4,0,0,0,"
               "0,0,2,0,"
               "0,0,0,1)", toString(B));

    // rotate forward backward
    vector<TypeParam> C = make_vec<TypeParam>(
        3,3,2,3,
        1,2,5,4,
        2,1,1,2,
        0,0,0,1);
    B = m_rotate3D<TypeParam>(27.5, 0, 0, C);
    B = m_rotate3D<TypeParam>(-27.5, 0, 0, B);

    for (int i = 0; i < 16; ++i) {
      B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
    }

    EXPECT_EQ( toString(B), toString(C));

    B = m_rotate3D<TypeParam>(0, 13.9, 0, C);
    B = m_rotate3D<TypeParam>(0, -13.9, 0, B);

    for (int i = 0; i < 16; ++i) {
      B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
    }

    EXPECT_EQ( toString(B), toString(C));

    B = m_rotate3D<TypeParam>(0, 0, 42.14, C);
    B = m_rotate3D<TypeParam>(0, 0, -42.14, B);

    for (int i = 0; i < 16; ++i) {
      B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
    }

    EXPECT_EQ( toString(B), toString(C));

  }

}  // end namespace caffe
