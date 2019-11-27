#include <string>
#include <vector>

#include "hdf5.h"

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class HDF5Test : public ::testing::Test {};

TEST_F(HDF5Test, TestHDF5SaveLoadInt) {

  std::string tmpFileName;
  MakeTempFilename(&tmpFileName);

  hid_t file_hid = H5Fcreate(
      tmpFileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  EXPECT_GE(file_hid, 0);
  const int writtenValue = 42;
  hdf5_save_int(file_hid, "/dataset", writtenValue);
  H5Fclose(file_hid);

  file_hid = H5Fopen(tmpFileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  EXPECT_GE(file_hid, 0);
  int readValue = hdf5_load_int(file_hid, "/dataset");
  H5Fclose(file_hid);

  EXPECT_EQ(readValue, writtenValue);
}



}
