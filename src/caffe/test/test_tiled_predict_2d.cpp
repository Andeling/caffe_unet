#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/tiled_predict.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

  template <typename TypeParam>
  class TiledPredictTest2d : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

  protected:
    TiledPredictTest2d()
            : reference_output_() {
      MakeTempFilename(&output_file_prefix_);
      modeldef_file_name_ = output_file_prefix_ + "_modeldef.h5";
      data_file_name_ = output_file_prefix_ + "_data.h5";
      weights_file_name_ = output_file_prefix_ + "_weights.h5";
    }

    // write sample hdf5 data, model and weights for all the tests
    virtual void SetUp() {

      std::vector<int> dataShape(4);
      dataShape[0] = 2;
      dataShape[1] = 1;
      dataShape[2] = 20;
      dataShape[3] = 18;

      Blob<Dtype> data(dataShape);
      for (int i = 0; i < data.count(); ++i) data.mutable_cpu_data()[i] = i;

      {
        hid_t file_id = H5Fcreate(data_file_name_.c_str(), H5F_ACC_TRUNC,
                                  H5P_DEFAULT, H5P_DEFAULT);
        ASSERT_GE(file_id, 0) << "Failed to create HDF5 file" <<
            data_file_name_;
        hdf5_save_nd_dataset(file_id, "/data", data);
        herr_t status = H5Fclose(file_id);
        EXPECT_GE(status, 0) << "Failed to close HDF5 file " << data_file_name_;
      }

      // Setup sample architecture
      std::stringstream md;
      md << "name: 'sample_net_tiled_predict_2d'" << std::endl;
      md << "force_backward: true" << std::endl;
      md << "layer { bottom: 'data' top: 'd0a' name: 'conv_data-d0a' "
         << "type: 'Convolution' param { lr_mult: 1 decay_mult: 1 } "
         << "param { lr_mult: 2 decay_mult: 0 } "
         << "convolution_param { num_output: 1 pad: 0 kernel_size: 5 "
         << "weight_filler { type: 'msra' }} }" << std::endl;
      md << "layer { bottom: 'd0a' top: 'd1a' name: 'conv_d0a-d1a' "
         << "type: 'Convolution' param { lr_mult: 1 decay_mult: 1 } "
         << "param { lr_mult: 2 decay_mult: 0 } "
         << "convolution_param { num_output: 1 pad: 0 kernel_size: 2 "
         << "stride: 2 } }" << std::endl;
      md << "layer { bottom: 'd1a' top: 'd1b' name: 'conv_d1a-d1b' "
         << "type: 'Convolution' param { lr_mult: 1 decay_mult: 1 } "
         << "param { lr_mult: 2 decay_mult: 0 } "
         << "convolution_param { num_output: 1 pad: 0 kernel_size: 3 "
         << "weight_filler { type: 'msra' }} }" << std::endl;
      md << "layer { bottom: 'd1b' top: 'u0a' name: 'upconv_d1b_u0a' "
         << "type: 'Deconvolution' param { lr_mult: 1 decay_mult: 1 } "
         << "param { lr_mult: 2 decay_mult: 0 } "
         << "convolution_param { num_output: 1 pad: 0 kernel_size: 2 "
         << "stride: 2 weight_filler { type: 'msra' }} }" << std::endl;
      md << "layer { bottom: 'u0a' top: 'score' name: 'conv_u0a-score' "
         << "type: 'Convolution' param { lr_mult: 1 decay_mult: 1 } "
         << "param { lr_mult: 2 decay_mult: 0 } "
         << "convolution_param { num_output: 2 pad: 0 kernel_size: 9 "
         << "weight_filler { type: 'msra' }} }" << std::endl;
      md << "layer { bottom: 'u0a' top: 'score2' name: 'conv_u0a-score2' "
         << "type: 'Convolution' param { lr_mult: 1 decay_mult: 1 } "
         << "param { lr_mult: 2 decay_mult: 0 } "
         << "convolution_param { num_output: 1 pad: 0 kernel_size: 9 "
         << "weight_filler { type: 'msra' }} }" << std::endl;

      // Setup sample solver
      std::stringstream sd;
      sd << "net: 'sample_net_tiled_predict_2d-train.prototxt'"
         << std::endl;
      sd << "base_lr: 0.00001" << std::endl;
      sd << "momentum: 0.9" << std::endl;
      sd << "lr_policy: \"fixed\"" << std::endl;
      sd << "max_iter: 600000" << std::endl;
      sd << "display: 1" << std::endl;
      sd << "snapshot: 10000" << std::endl;
      sd << "snapshot_prefix: 'snapshot'" << std::endl;
      sd << "snapshot_format: HDF5" << std::endl;
      sd << "type: \"SGD\"" << std::endl;
      sd << "solver_mode: GPU" << std::endl;
      sd << "debug_info: 0" << std::endl;

      // Create modeldef.h5 file
      {
        hid_t file_id = H5Fcreate(modeldef_file_name_.c_str(), H5F_ACC_TRUNC,
                                  H5P_DEFAULT, H5P_DEFAULT);
        ASSERT_GE(file_id, 0) << "Failed to create HDF5 file" <<
            modeldef_file_name_;
        hdf5_save_string(file_id, ".unet-ident", "sample_net_tiled_predict_2d");
        H5Gcreate(file_id, "/unet_param", H5P_DEFAULT, H5P_DEFAULT,
                  H5P_DEFAULT);
        hdf5_save_string(
            file_id, "/unet_param/name", "sample_net_tiled_predict_2d");
        hdf5_save_string(
            file_id, "/unet_param/description", "sample_net_tiled_predict_2d");
        hdf5_save_string(file_id, "/unet_param/input_blob_name", "data");
        hdf5_save_string(file_id, "/unet_param/padding", "mirror");
        hdf5_save_int_vec(
            file_id, "/unet_param/element_size_um", std::vector<int>(2, 1));
        hdf5_save_int(file_id, "/unet_param/input_num_channels", 1);
        hdf5_save_int(file_id, "/unet_param/downsampleFactor", 2);
        hdf5_save_int(file_id, "/unet_param/padInput", 4);
        hdf5_save_int(file_id, "/unet_param/padOutput", -12);
        hdf5_save_string(file_id, "/model_prototxt", md.str());
        hdf5_save_string(file_id, "/solver_prototxt", sd.str());

        std::vector<int> mapShape(2, 2);
        mapShape[1] = 84;
        Blob<int> mapInputNumPxGPUMemMB(mapShape);
        int mapData[] = {
            0, 324, 360, 400, 440, 484, 528, 576, 624, 676, 728, 784,
            840, 900, 960, 1024, 1088, 1156, 1224, 1296, 1368, 1444,
            1520, 1600, 1680, 1764, 1848, 1936, 2024, 2116, 2208, 2304,
            2400, 2500, 2600, 2704, 2808, 2916, 3024, 3136, 3248, 3364,
            3480, 3600, 3720, 3844, 3968, 4096, 4224, 4356, 4488, 4624,
            4760, 4900, 5040, 5184, 5328, 5476, 5624, 5776, 5928, 6084,
            6240, 6400, 6560, 6724, 6888, 7056, 7224, 7396, 7568, 7744,
            7920, 8100, 8280, 8464, 8648, 8836, 9024, 9216, 9408, 9604,
            9800, 10000,
            0, 178, 178, 178, 178, 178, 178, 178, 178, 179, 179, 179,
            179, 179, 179, 179, 179, 179, 180, 180, 180, 180, 180, 181,
            181, 181, 181, 181, 181, 181, 182, 182, 182, 184, 183, 183,
            183, 184, 185, 185, 185, 185, 186, 187, 187, 187, 187, 187,
            188, 188, 189, 189, 190, 190, 190, 191, 192, 192, 192, 193,
            192, 192, 192, 192, 193, 194, 194, 195, 195, 195, 196, 197,
            197, 198, 198, 199, 199, 199, 200, 200, 201, 201, 203, 203
        };
        std::memcpy(mapInputNumPxGPUMemMB.mutable_cpu_data(), mapData,
                    mapShape[0] * mapShape[1] * sizeof(int));
        hdf5_save_nd_dataset(
            file_id, "/unet_param/mapInputNumPxGPUMemMB",
            mapInputNumPxGPUMemMB);
        herr_t status = H5Fclose(file_id);
        EXPECT_GE(status, 0)
            << "Failed to close HDF5 file " << modeldef_file_name_;
      }

      // Pad data (mirror)
      std::vector<int> paddedShape(dataShape);
      paddedShape[2] += 16;
      paddedShape[3] += 16;
      Blob<Dtype> dataPadded(paddedShape);
      Dtype *outP = dataPadded.mutable_cpu_data();
      int nSlices = paddedShape[0] * paddedShape[1];
      for (int i = 0; i < nSlices; ++i) {
        Dtype const *sliceIn =
            data.cpu_data() + i * dataShape[2] * dataShape[3];
        for (int y = 0; y < paddedShape[2]; ++y) {
          int yIn = y - 8;
          if (yIn < 0) yIn = -yIn;
          int factor = yIn / (dataShape[2] - 1);
          if (factor % 2 == 0) yIn = yIn - factor * (dataShape[2] - 1);
          else yIn = (factor + 1) * (dataShape[2] - 1) - yIn;
          for (int x = 0; x < paddedShape[3]; ++x, ++outP) {
            int xIn = x - 8;
            if (xIn < 0) xIn = -xIn;
            factor = xIn / (dataShape[3] - 1);
            if (factor % 2 == 0) xIn = xIn - factor * (dataShape[3] - 1);
            else xIn = (factor + 1) * (dataShape[3] - 1) - xIn;
            *outP = sliceIn[yIn * dataShape[3] + xIn];
          }
        }
      }

      // Create test.prototxt file
      std::ofstream model_file(
          (output_file_prefix_ + "_test.prototxt").c_str(), std::ios::trunc);
      model_file << "state: { phase: TEST }" << std::endl;
      model_file << "layer { name: 'data' type: 'Input' top: 'data' "
                 << "input_param { shape: { dim: " << paddedShape[0]
                 << " dim: " << paddedShape[1] << " dim: " << paddedShape[2]
                 << " dim: " << paddedShape[3] << " } } }" << std::endl;
      model_file << md.str();
      model_file.close();

      // instantiate the caffe net
      Net<Dtype> caffe_net(
          (output_file_prefix_ + "_test.prototxt").c_str(), caffe::TEST);

      // Set weights

      // conv_data-d0a <== identity transform
      {
        shared_ptr< Blob<Dtype> > &w =
            caffe_net.layer_by_name("conv_data-d0a")->blobs()[0];
        ASSERT_EQ(25, w->count());
        std::memset(w->mutable_cpu_data(), 0, w->count() * sizeof(Dtype));
        w->mutable_cpu_data()[12] = Dtype(1);
        shared_ptr< Blob<Dtype> > &b =
            caffe_net.layer_by_name("conv_data-d0a")->blobs()[1];
        ASSERT_EQ(1, b->count());
        b->mutable_cpu_data()[0] = Dtype(0);
      }

      // conv_d0a-d1a <== average sub-sampling
      {
        shared_ptr< Blob<Dtype> > &w =
            caffe_net.layer_by_name("conv_d0a-d1a")->blobs()[0];
        ASSERT_EQ(4, w->count());
        w->mutable_cpu_data()[0] = Dtype(0.25);
        w->mutable_cpu_data()[1] = Dtype(0.25);
        w->mutable_cpu_data()[2] = Dtype(0.25);
        w->mutable_cpu_data()[3] = Dtype(0.25);
        shared_ptr< Blob<Dtype> > &b =
            caffe_net.layer_by_name("conv_d0a-d1a")->blobs()[1];
        ASSERT_EQ(1, b->count());
        b->mutable_cpu_data()[0] = Dtype(0);
      }

      // conv_d1a-d1b <== identity transform
      {
        shared_ptr< Blob<Dtype> > &w =
            caffe_net.layer_by_name("conv_d1a-d1b")->blobs()[0];
        ASSERT_EQ(9, w->count());
        std::memset(w->mutable_cpu_data(), 0, w->count() * sizeof(Dtype));
        w->mutable_cpu_data()[4] = Dtype(1);
        shared_ptr< Blob<Dtype> > &b =
            caffe_net.layer_by_name("conv_d1a-d1b")->blobs()[1];
        ASSERT_EQ(1, b->count());
        b->mutable_cpu_data()[0] = Dtype(0);
      }

      // upconv_d1b_u0a <== average up-sampling
      {
        shared_ptr< Blob<Dtype> > &w =
            caffe_net.layer_by_name("upconv_d1b_u0a")->blobs()[0];
        ASSERT_EQ(4, w->count());
        for (int i = 0; i < w->count(); ++i)
            w->mutable_cpu_data()[i] = Dtype(0.25);
        shared_ptr< Blob<Dtype> > &b =
            caffe_net.layer_by_name("upconv_d1b_u0a")->blobs()[1];
        ASSERT_EQ(1, b->count());
        b->mutable_cpu_data()[0] = Dtype(0);
      }

      // conv_d1a-d1b <== ch0: id, ch1: -id
      {
        shared_ptr< Blob<Dtype> > &w =
            caffe_net.layer_by_name("conv_u0a-score")->blobs()[0];
        ASSERT_EQ(162, w->count());
        std::memset(w->mutable_cpu_data(), 0, w->count() * sizeof(Dtype));
        w->mutable_cpu_data()[40] = Dtype(1);
        w->mutable_cpu_data()[121] = Dtype(-1);
        shared_ptr< Blob<Dtype> > &b =
            caffe_net.layer_by_name("conv_u0a-score")->blobs()[1];
        ASSERT_EQ(2, b->count());
        b->mutable_cpu_data()[0] = Dtype(0);
        b->mutable_cpu_data()[1] = Dtype(0);
      }

      // conv_u0a-score2 <== identity transform
      {
        shared_ptr< Blob<Dtype> > &w =
            caffe_net.layer_by_name("conv_u0a-score2")->blobs()[0];
        ASSERT_EQ(81, w->count());
        std::memset(w->mutable_cpu_data(), 0, w->count() * sizeof(Dtype));
        w->mutable_cpu_data()[40] = Dtype(1);
        shared_ptr< Blob<Dtype> > &b =
            caffe_net.layer_by_name("conv_u0a-score2")->blobs()[1];
        ASSERT_EQ(1, b->count());
        b->mutable_cpu_data()[0] = Dtype(0);
      }

      // store weights
      caffe_net.ToHDF5(weights_file_name_);

      // Forward pass to generate reference output scores
      if (reference_output_.size() != 2)
      {
        caffe_net.blob_by_name("data")->Reshape(paddedShape);
        std::memcpy(
            caffe_net.blob_by_name("data")->mutable_cpu_data(),
            dataPadded.cpu_data(), dataPadded.count() * sizeof(Dtype));
        std::vector<Blob<Dtype>*> const &caffeScores = caffe_net.Forward();
        ASSERT_EQ(2, caffeScores.size());
        ASSERT_EQ(dataShape[0], caffeScores[0]->shape(0));
        ASSERT_EQ(2, caffeScores[0]->shape(1));
        ASSERT_EQ(dataShape[2], caffeScores[0]->shape(2));
        ASSERT_EQ(dataShape[3], caffeScores[0]->shape(3));
        ASSERT_EQ(dataShape[0], caffeScores[1]->shape(0));
        ASSERT_EQ(1, caffeScores[1]->shape(1));
        ASSERT_EQ(dataShape[2], caffeScores[1]->shape(2));
        ASSERT_EQ(dataShape[3], caffeScores[1]->shape(3));
        reference_output_.resize(caffeScores.size(), NULL);
        for (size_t i = 0; i < caffeScores.size(); ++i) {
          reference_output_[i] = new Blob<Dtype>(caffeScores[i]->shape());
          std::memcpy(reference_output_[i]->mutable_cpu_data(),
                      caffeScores[i]->cpu_data(),
                      caffeScores[i]->count() * sizeof(Dtype));

          // // Save reference output scores for debugging
          // hid_t file_id = H5Fopen(
          //     data_file_name_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
          // ASSERT_GE(file_id, 0) << "Failed to open HDF5 file" <<
          //     data_file_name_;
          // hdf5_save_nd_dataset(
          //     file_id, caffe_net.blob_names()[
          //         caffe_net.output_blob_indices()[i]],
          //         *reference_output_[i]);
          // herr_t status = H5Fclose(file_id);
          // EXPECT_GE(status, 0)
          //     << "Failed to close HDF5 file " << data_file_name_;
        }
      }
    }

    virtual ~TiledPredictTest2d() {
      for (size_t i = 0; i < reference_output_.size(); ++i)
          delete reference_output_[i];
    }

    void CheckBlobEqual(
        const Blob<Dtype>& reference, const Blob<Dtype>& toTest,
        int iterations);

    void TestTiledPredict(
        std::string const &param_name, std::string const &param_value,
        int iterations, bool average_mirror, bool average_rotate);

    std::string output_file_prefix_;
    std::string modeldef_file_name_;
    std::string data_file_name_;
    std::string weights_file_name_;

    std::vector<Blob<Dtype>*> reference_output_;

  };

  // adapted from HDF5OutputLayerTest<TypeParam>::CheckBlobEqual
  template<typename TypeParam>
  void TiledPredictTest2d<TypeParam>::CheckBlobEqual(
      const Blob<Dtype>& reference, const Blob<Dtype>& toTest, int iterations) {
    if (iterations <= 0 || iterations > reference.shape(0))
        EXPECT_EQ(reference.shape(0), toTest.shape(0));
    else EXPECT_EQ(iterations, toTest.shape(0));
    EXPECT_EQ(reference.shape(1), toTest.shape(1));
    EXPECT_EQ(reference.shape(2), toTest.shape(2));
    EXPECT_EQ(reference.shape(3), toTest.shape(3));
    const Dtype min_precision = 1e-5;
    for (int i = 0; i < toTest.count(); ++i) {
      Dtype expected_value = reference.cpu_data()[i];
      Dtype precision = std::max(
          Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
      EXPECT_NEAR(expected_value, toTest.cpu_data()[i], precision);
    }
  }

  // test tiled predict option defined by "param_name" for the
  // parameter value defined by "param_value", and check if the outputs
  // equal the reference
  template<typename TypeParam>
  void TiledPredictTest2d<TypeParam>::TestTiledPredict(
      std::string const &param_name, std::string const &param_value,
      int iterations, bool average_mirror, bool average_rotate) {

    bool skipTest;
    if (Caffe::mode() == Caffe::CPU &&
        ( param_name.compare("gpu_mem_available_MB") == 0 ||
          param_name.compare("") == 0 )) {
      skipTest = true;
      //VLOG(1) << "skip test" << std::endl;
    } else {
      skipTest = false;
    }

    if (!skipTest) {
      std::string gpu_mem_available_MB("");
      std::string mem_available_px("");
      std::string n_tiles("");
      std::string tile_size("");

      // run tiled prediction for the set of parameters
      if (param_name == "gpu_mem_available_MB")
          gpu_mem_available_MB = param_value;
      else if (param_name == "mem_available_px") mem_available_px = param_value;
      else if (param_name == "n_tiles") n_tiles = param_value;
      else if (param_name == "tile_size") tile_size = param_value;

      // run tiled prediction and write result to hdf5
      caffe::TiledPredict<Dtype>(
          data_file_name_, output_file_prefix_ + ".h5", modeldef_file_name_,
          weights_file_name_, iterations, gpu_mem_available_MB,
          mem_available_px, n_tiles, tile_size, average_mirror,
          average_rotate);

      // read results from hdf5 output file
      std::vector<Blob<Dtype>*> predictedScores(2, NULL);
      predictedScores[0] = new Blob<Dtype>();
      predictedScores[1] = new Blob<Dtype>();
      hid_t file_id = H5Fopen(
          (output_file_prefix_ + ".h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      ASSERT_GE(file_id, 0) << "Failed to open HDF5 file"
                            << output_file_prefix_ << ".h5";
      hdf5_load_nd_dataset(
          file_id, "score", 1, INT_MAX, predictedScores[0], true);
      hdf5_load_nd_dataset(
          file_id, "score2", 1, INT_MAX, predictedScores[1], true);
      herr_t status = H5Fclose(file_id);
      EXPECT_GE(status, 0) << "Failed to close HDF5 file "
                           << output_file_prefix_ << ".h5";

      // compare results
      CheckBlobEqual(*reference_output_[0], *predictedScores[0], iterations);
      CheckBlobEqual(*reference_output_[1], *predictedScores[1], iterations);

      // Clean up
      delete predictedScores[0];
      delete predictedScores[1];
    }
  }

  TYPED_TEST_CASE(TiledPredictTest2d, TestDtypesAndDevices);

  TYPED_TEST(TiledPredictTest2d, TestTiledPredictTileSize) {
    this->TestTiledPredict("tile_size", "20x20", 1, false, false);
    this->TestTiledPredict("tile_size", "20x10", 1, false, false);
    this->TestTiledPredict("tile_size", "7x20", 1, false, false);
    this->TestTiledPredict("tile_size", "10x10", 1, false, false);
  }

  TYPED_TEST(TiledPredictTest2d, TestTiledPredictNTilesYX) {
    this->TestTiledPredict("n_tiles", "1x1", 1, false, false);
    this->TestTiledPredict("n_tiles", "1x2", 1, false, false);
    this->TestTiledPredict("n_tiles", "3x1", 1, false, false);
    this->TestTiledPredict("n_tiles", "2x2", 1, false, false);
  }

  TYPED_TEST(TiledPredictTest2d, TestTiledPredictNTiles) {
    this->TestTiledPredict("n_tiles", "1", 1, false, false);
    this->TestTiledPredict("n_tiles", "2", 1, false, false);
    this->TestTiledPredict("n_tiles", "3", 1, false, false);
    this->TestTiledPredict("n_tiles", "4", 1, false, false);
  }

  TYPED_TEST(TiledPredictTest2d, TestTiledPredictMemPx) {
    this->TestTiledPredict("mem_available_px", "2500", 1, false, false);
    this->TestTiledPredict("mem_available_px", "900", 1, false, false);
    this->TestTiledPredict("mem_available_px", "750", 1, false, false);
  }

  TYPED_TEST(TiledPredictTest2d,
             TestTiledPredictTileSizeIterAllAverageMirror) {
    this->TestTiledPredict("tile_size", "20x20", 0, true, false);
    this->TestTiledPredict("tile_size", "10x10", 0, true, false);
  }

  TYPED_TEST(TiledPredictTest2d,
             TestTiledPredictTileSizeIterAllAverageRotate) {
    this->TestTiledPredict("tile_size", "20x20", 0, false, true);
    this->TestTiledPredict("tile_size", "10x10", 0, false, true);
  }

  TYPED_TEST(TiledPredictTest2d, TestTiledPredictGpuMemMB) {
    this->TestTiledPredict("gpu_mem_available_MB", "1000", 1, false, false);
    this->TestTiledPredict("gpu_mem_available_MB", "200", 1, false, false);
    this->TestTiledPredict("gpu_mem_available_MB", "179", 1, false, false);
    this->TestTiledPredict("gpu_mem_available_MB", "178", 1, false, false);
  }

  TYPED_TEST(TiledPredictTest2d, TestTiledPredictNoOption) {
    this->TestTiledPredict("", "", 1, false, false);
  }

}  // namespace caffe
