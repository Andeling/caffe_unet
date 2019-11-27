#include "gtest/gtest.h"
#include "caffe/util/hungarian_algorithm.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class HungarianAlgorithmTest : public ::testing::Test {};

TEST_F(HungarianAlgorithmTest, TestQuadratic) {

  double costMatrix[] = {
      5, 2, 0, 7, 3,
      1, 0, 7, 4, 6,
      0, 3, 2, 2, 9,
      5, 8, 1, 0, 3,
      0, 0, 4, 9, 7 };

  int assignmentVector[] = { 4, 1, 2, 3, 0 };

  std::vector< std::vector<double> > cost(5, std::vector<double>(5));
  for (int r = 0; r < 5; ++r)
      for (int c = 0; c < 5; ++c)
          cost[r][c] = costMatrix[r * 5 + c];
  std::vector<int> assignment;
  HungarianAlgorithm ha;
  double finalCost = ha.Solve(cost, assignment);

  EXPECT_EQ(5, finalCost);
  for (size_t i = 0; i < 5; ++i) EXPECT_EQ(assignmentVector[i], assignment[i]);
}

TEST_F(HungarianAlgorithmTest, TestRectangular) {

  double costMatrix[] = {
      5, 2, 0, 7, 3, 1,
      1, 0, 7, 4, 6, 4,
      0, 3, 2, 2, 9, 8,
      5, 8, 1, 0, 3, 0,
      0, 0, 4, 9, 7, 5 };

  int assignmentVector[] = { 2, 1, 3, 5, 0 };

  std::vector< std::vector<double> > cost(5, std::vector<double>(6));
  for (int r = 0; r < 5; ++r)
      for (int c = 0; c < 6; ++c)
          cost[r][c] = costMatrix[r * 6 + c];
  std::vector<int> assignment;
  HungarianAlgorithm ha;
  double finalCost = ha.Solve(cost, assignment);

  EXPECT_EQ(2, finalCost);
  for (size_t i = 0; i < 5; ++i) EXPECT_EQ(assignmentVector[i], assignment[i]);
}

}
