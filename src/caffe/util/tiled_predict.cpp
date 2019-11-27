#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include <google/protobuf/text_format.h>
#include "caffe/util/math_functions.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/tiled_predict.hpp"

namespace caffe {

  std::vector<std::string> split(std::string const &str, char delim) {
    std::vector<std::string> res;
    std::stringstream strStream(str);
    std::string segment;
    while(std::getline(strStream, segment, delim)) res.push_back(segment);
    return res;
  }

  template<typename NumericT>
  std::vector<NumericT> toNumericVector(std::vector<std::string> const &in) {
    std::vector<NumericT> out;
    NumericT x = NumericT();
    for (int i = 0; i < in.size(); ++i) {
      std::istringstream(in[i]) >> x;
      out.push_back(x);
    }
    return out;
  }

  enum TilingParameter { None, MemMB, MemPx, NTiles, Tiling, Shape };

  std::vector<int> getTiling(
      TilingParameter tilingOption, std::vector<int> const &tilingParam,
      std::vector<int> const &dataShape, std::vector<int> const &dsFactor,
      std::vector<int> const &padIn, std::vector<int> const &padOut,
      int memAvailPx, std::vector<int> &inTileShape,
      std::vector<int> &outTileShape)
  {
    std::vector<int> tiling;

    int nDims = dataShape.size();
    std::vector<int> tmp_outTileShape;

    switch(tilingOption) {
    case MemPx: {
      CHECK_EQ(nDims, 2)
          << "Tiling option memAvailPx only supported for 2D models";

      // check minimum input size
      std::vector<int> d4a_s(
          vector_cast<int>(
              ceil(vector_cast<float>(1 - padOut) /
                   vector_cast<float>(dsFactor))));
      std::vector<int> min_inTileShape(dsFactor * d4a_s + padIn);

      CHECK(memAvailPx >= product(min_inTileShape))
          << "memory available must be at least the minimum input "
          << "size of the network, which is " << product(min_inTileShape)
          << " = " << toString(min_inTileShape) << "!";

      // find maximum reasonable number of tiles in rows
      std::vector<int> border_px(
          vector_cast<int>(
              round(vector_cast<float>(padIn - padOut) / 2.0f)));

      std::vector<int> curr_ntiles(nDims, 0);
      curr_ntiles[1] = 1;
      std::vector<int> outShapeMax(nDims, -1);
      while (outShapeMax[1] <= 0) {
        curr_ntiles[0] = curr_ntiles[0] + 1;
        d4a_s = vector_cast<int>(
            ceil(ceil(vector_cast<float>(dataShape) /
                      vector_cast<float>(curr_ntiles)) -
                 vector_cast<float>(padOut)) /
            vector_cast<float>(dsFactor));
        std::vector<int> inTileShape(dsFactor * d4a_s + padIn);
        outShapeMax[1] = static_cast<int>(
            std::floor(static_cast<float>(memAvailPx) /
                       static_cast<float>(inTileShape[0])) - 2 * border_px[0]);
        d4a_s = vector_cast<int>(
            floor((static_cast<float>(outShapeMax[1]) -
                   vector_cast<float>(padOut)) /
                  vector_cast<float>(dsFactor)));
        outShapeMax[1] = dsFactor[0] * d4a_s[0] + padOut[0];
      }
      // find maximum reasonable number of tiles in cols
      curr_ntiles[0] = 1;
      curr_ntiles[1] = 0;
      while (outShapeMax[0] <= 0) {
        curr_ntiles[1] = curr_ntiles[1] + 1;
        d4a_s = vector_cast<int>(
            ceil(ceil(vector_cast<float>(dataShape) /
                      vector_cast<float>(curr_ntiles)) -
                 vector_cast<float>(padOut)) /
            vector_cast<float>(dsFactor));
        std::vector<int> inTileShape(dsFactor * d4a_s + padIn);
        outShapeMax[0] = static_cast<int>(
            std::floor(static_cast<float>(memAvailPx) /
                       static_cast<float>(inTileShape[1])) - 2 * border_px[1]);
        d4a_s = vector_cast<int>(
            floor((static_cast<float>(outShapeMax[0]) -
                   vector_cast<float>(padOut)) /
                  vector_cast<float>(dsFactor)));
        outShapeMax[0] = dsFactor[1] * d4a_s[1] + padOut[1];
      }
      // find optimal tiling layout
      // 	solution with minimum total input size
      std::vector<int> max_ntiles(
          vector_cast<int>(ceil(vector_cast<float>(dataShape) /
                                vector_cast<float>(outShapeMax))));
      LOG(INFO) << "max_ntiles: " << toString(max_ntiles);
      d4a_s = std::vector<int>(nDims, 0);
      std::vector<int> inTileShape(nDims);
      std::vector<int> outTileShape(nDims);
      uint32_t M_tile;
      uint32_t Neff_tiles;
      uint64_t min_M_total = std::numeric_limits<uint64_t>::max();
      tiling = std::vector<int>(nDims, 1);
      for (int ny = 1; ny <= max_ntiles[0]; ++ny) {
        for (int nx = 1; nx <= max_ntiles[1]; ++nx) {
          std::vector<int> testedTiling(nDims);
          testedTiling[0] = ny;
          testedTiling[1] = nx;
          d4a_s = vector_cast<int>(
              ceil((ceil(vector_cast<float>(dataShape) /
                         vector_cast<float>(testedTiling)) -
                    vector_cast<float>(padOut)) /
                   vector_cast<float>(dsFactor)));
          std::vector<int> inTileShape(dsFactor * d4a_s + padIn);
          std::vector<int> outTileShape(dsFactor * d4a_s + padOut);
          M_tile = product(inTileShape);
          Neff_tiles = static_cast<int>(
              product(ceil(vector_cast<float>(dataShape) /
                           vector_cast<float>(outTileShape))));
          if (M_tile <= memAvailPx && M_tile * Neff_tiles < min_M_total) {
            min_M_total = M_tile * Neff_tiles;
            tiling = testedTiling;
          }
        }
      }
      break;
    }
    case NTiles: {
      CHECK_EQ(nDims, 2)
          << "Option 'n_tiles' specifying the total number of tiles is only "
          << "supported for 2D models";

      // find optimal tiling layout
      // 	solution with minimum tile input size / minimum total input size
      std::vector<int> max_ntiles(tilingParam);
      std::vector<int> d4a_s;
      std::vector<int> inTileShape;
      std::vector<int> outTileShape;
      uint32_t M_tile = 0;
      uint32_t Neff_tiles = 0;
      uint32_t max_Neff_tiles = 0;
      uint32_t min_M_tile = std::numeric_limits<uint32_t>::max();
      tiling = std::vector<int>(nDims, 1);
      for (int tileIdx = 0; tileIdx < product(max_ntiles); ++tileIdx) {
        std::vector<int> testedTiling(nDims);
        int tmp = tileIdx;
        for (int d = nDims - 1; d >= 0; --d)
        {
          testedTiling[d] = tmp % max_ntiles[d] + 1;
          tmp /= max_ntiles[d];
        }
        if (product(testedTiling) > tilingParam[0]) continue;
        d4a_s = vector_cast<int>(
            ceil((ceil(vector_cast<float>(dataShape) /
                       vector_cast<float>(testedTiling)) -
                  vector_cast<float>(padOut)) /
                 vector_cast<float>(dsFactor)));
        inTileShape = dsFactor * d4a_s + padIn;
        outTileShape = dsFactor * d4a_s + padOut;
        M_tile = product(inTileShape);
        Neff_tiles = static_cast<int>(
            product(ceil(vector_cast<float>(dataShape) /
                         vector_cast<float>(outTileShape))));
        if (Neff_tiles > max_Neff_tiles) {
          max_Neff_tiles = Neff_tiles;
          min_M_tile = M_tile;
          tiling = testedTiling;
        }
        if (Neff_tiles == max_Neff_tiles) {
          if (M_tile < min_M_tile) {
            min_M_tile = M_tile;
            tiling = testedTiling;
          }
        }
      }
      CHECK_GT(max_Neff_tiles, 0);
      break;
    }
    case Tiling: {
      tiling = tilingParam;
      std::vector<int> d4a_s(
          vector_cast<int>(
              ceil((ceil(vector_cast<float>(dataShape) /
                         vector_cast<float>(tiling)) -
                    vector_cast<float>(padOut)) /
                   vector_cast<float>(dsFactor))));
      std::vector<int> outTileShape(dsFactor * d4a_s + padOut);
      tiling = vector_cast<int>(ceil(vector_cast<float>(dataShape) /
                                     vector_cast<float>(outTileShape)));
      break;
    }
    case Shape:
    {
      std::vector<int> outTileShape(tilingParam);
      std::vector<int> d4a_s(
          vector_cast<int>(
              ceil((vector_cast<float>(outTileShape) -
                    vector_cast<float>(padOut)) /
                   vector_cast<float>(dsFactor))));
      tmp_outTileShape = dsFactor * d4a_s + padOut;
      break;
    }
    default:
      CHECK(false) << "Unknown tiling option";
    }

    /*
     * compute input and output sizes (for u-shaped network)
     */
    std::vector<int> d4a_s(nDims);
    if (tilingOption != Shape) {
      d4a_s = vector_cast<int>(
          ceil((ceil(vector_cast<float>(dataShape) /
                     vector_cast<float>(tiling)) -
                vector_cast<float>(padOut)) /
               vector_cast<float>(dsFactor)));
    } else {
      d4a_s = vector_cast<int>(
          (vector_cast<float>(tmp_outTileShape) -
           vector_cast<float>(padOut)) /
          vector_cast<float>(dsFactor));
      tiling = vector_cast<int>(ceil(vector_cast<float>(dataShape) /
                                     vector_cast<float>(tmp_outTileShape)));
    }
    inTileShape = dsFactor * d4a_s + padIn;
    outTileShape = dsFactor * d4a_s + padOut;
    return tiling;
  }

  // Tiled predict: score a model in overlap-tile strategy for passing
  // large images through caffe
  template <typename Dtype>
  void TiledPredict(
      const string& infileH5, const string& outfileH5, const string& model,
      const string& weights, int iterations,
      const string& gpu_mem_available_MB_str,
      const string& mem_available_px_str, const string& n_tiles_str,
      const string& tile_size_str, bool average_mirror, bool average_rotate) {

    /*
     * Check and parse input parameters
     */
    CHECK_GT(infileH5.size(), 0) << "Need an hdf5 input file.";
    CHECK_GT(outfileH5.size(), 0) << "Need an hdf5 output file.";
    CHECK_GT(model.size(), 0) << "Need a model definition to score.";
    if (weights.size() == 0)
        LOG(INFO) << "Warning: No model weights are given. "
                  << "Net will be initialized without model weights!";
    CHECK_GE(iterations, 0) << "Number of iterations must be >= 0.";

    // Get requested tiling option
    // alternative options defining tiling layout
    std::vector<TilingParameter> tilingParameters;
    if (gpu_mem_available_MB_str.size() > 0)
        tilingParameters.push_back(MemMB);
    if (mem_available_px_str.size() > 0)
        tilingParameters.push_back(MemPx);
    if (n_tiles_str.size() > 0) tilingParameters.push_back(NTiles);
    if (tile_size_str.size() > 0) tilingParameters.push_back(Shape);
    if (Caffe::mode() == Caffe::CPU) {
      CHECK_EQ(tilingParameters.size(), 1)
          << "Exactly one of the alternative options 'mem_available_px', "
          << "'n_tiles', or 'tile_size' has to be specified in CPU mode!";
      CHECK_NE(tilingParameters[0], MemMB)
          << "Option 'gpu_mem_available_MB' is not supported in CPU mode.";
    } else
        CHECK_LE(tilingParameters.size(), 1)
            << "At most one of the alternative options 'gpu_mem_available_MB', "
            << "'mem_available_px', 'n_tiles', or 'tile_size' has to be "
            << "specified in GPU mode!";

    TilingParameter tilingParameter = (tilingParameters.size() == 0) ?
        None : tilingParameters[0];

    // parse tiling layout cmdline parameters
    TilingParameter useOption = MemPx;
    int gpu_mem_MB = 0;
    int memAvailPx = 0;
    std::vector<int> ntiles_param;
    std::vector<int> tile_size_param;
    switch(tilingParameter) {
    case None: {
      size_t cudaMem_free = 0;	// /1024/1024 --> MB
      size_t cudaMem_total = 0;
#ifndef CPU_ONLY
      CUDA_CHECK( cudaMemGetInfo(&cudaMem_free, &cudaMem_total));
#else
      NO_GPU;
#endif
      LOG(INFO) << "cudaMemGetInfo";
      LOG(INFO) << "free: " << cudaMem_free;
      LOG(INFO) << "total: " << cudaMem_total;
      gpu_mem_MB = int(std::floor(cudaMem_free / 1024.f / 1024.f));
      CHECK_GE(gpu_mem_MB, 1) << "Specified GPU memory available must be >= 1!";
      LOG(INFO) << "gpu_mem_MB: " << gpu_mem_MB;
      break;
    }
    case MemMB: {
      std::istringstream(gpu_mem_available_MB_str) >> gpu_mem_MB;
      CHECK_GE(gpu_mem_MB, 1) << "Specified GPU memory available must be >= 1!";
      LOG(INFO) << "gpu_mem_MB: " << gpu_mem_MB;
      break;
    }
    case MemPx: {
      std::istringstream(mem_available_px_str) >> memAvailPx;
      CHECK_GE(memAvailPx, 1) << "Specified memory available must be >= 1!";
      LOG(INFO) << "memAvailPx: " << memAvailPx;
      break;
    }
    case NTiles: {
      ntiles_param = toNumericVector<int>(split(n_tiles_str, 'x'));
      CHECK(ntiles_param.size() >= 1 && ntiles_param.size() <= 3);
      for (int d = 0; d < ntiles_param.size(); ++d)
          CHECK_GE(ntiles_param[d], 1)
              << "Number of tiles must be >= 1 for all dimensions.";
      LOG(INFO) << "n_tiles: " << toString(ntiles_param);
      useOption = (ntiles_param.size() == 1) ? NTiles : Tiling;
      break;
    }
    case Shape: {
      tile_size_param = toNumericVector<int>(split(tile_size_str ,'x'));
      CHECK(tile_size_param.size() >= 2 && tile_size_param.size() <= 3);
      for (int d = 0; d < tile_size_param.size(); ++d)
          CHECK_GE(tile_size_param[d], 1)
              << "Tile size must be >= 1 for all dimensions.";
      LOG(INFO) << "tile_size: " << toString(tile_size_param);
      useOption = Shape;
      break;
    }
    default:
      useOption = None;
    }

    CHECK(!(average_mirror && average_rotate))
        << "Options average_mirror and average_rotate are mutually exclusive!";
    LOG(INFO) << "average_mirror: " << average_mirror;
    LOG(INFO) << "average_rotate: " << average_rotate;
    int numAveragePasses = (average_mirror || average_rotate) ? 4 : 1;

    /*---------------------------------------------------------------------
     *  Read model definition
     *---------------------------------------------------------------------*/
    // prototxt
    std::string model_prototxt;
    // unet parameters
    std::string unet_ident;
    std::string input_dataset_name("data");
    std::string input_blob_name;
    int nDims_model;
    std::string padding;
    std::vector<int> dsFactor;
    std::vector<int> padIn;
    std::vector<int> padOut;

    CHECK(model.size() > 3 && model.compare(model.size() - 3, 3, ".h5") == 0)
        << "Model file type is not supported. HDF5 file is expected.";

    LOG(INFO) << "Reading model definition from: " << model.c_str();
    hid_t file_hid = H5Fopen(model.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK_GE(file_hid, 0) << "Failed to open HDF5 file " << model;
    unet_ident = hdf5_load_string(file_hid, "/.unet-ident");

    try {
      input_dataset_name = hdf5_load_string(
          file_hid, "/unet_param/input_dataset_name", true);
    }
    catch (std::exception &) {}

    // read unet parameters
    input_blob_name = hdf5_load_string(file_hid, "/unet_param/input_blob_name");
    nDims_model = hdf5_get_dataset_shape(
        file_hid, "/unet_param/element_size_um")[0];
    if (tilingParameter == NTiles) {
      if (ntiles_param.size() == 1)
          ntiles_param = std::vector<int>(nDims_model, ntiles_param[0]);
      CHECK_EQ(ntiles_param.size(), nDims_model)
          << "n_tiles dimensions (" << ntiles_param.size()
          << ") does not match input blob dimensions (" << nDims_model << ")";
    }
    if (tilingParameter == Shape)
        CHECK_EQ(tile_size_param.size(), nDims_model)
            << "Tile shape dimensions (" << tile_size_param.size()
            << "( do not match input blob dimensions (" << nDims_model << ")";
    padding = hdf5_load_string(file_hid, "/unet_param/padding");
    dsFactor = hdf5_load_int_vec(file_hid, "/unet_param/downsampleFactor");
    if (dsFactor.size() == 1) dsFactor.resize(nDims_model, dsFactor[0]);
    CHECK_EQ(dsFactor.size(), nDims_model)
        << "Number of downsample factors does not match model dimension";
    padIn = hdf5_load_int_vec(file_hid, "/unet_param/padInput");
    if (padIn.size() == 1) padIn.resize(nDims_model, padIn[0]);
    CHECK_EQ(padIn.size(), nDims_model)
        << "padInput does not match model dimension";
    padOut = hdf5_load_int_vec(file_hid, "/unet_param/padOutput");
    if (padOut.size() == 1) padOut.resize(nDims_model, padOut[0]);
    CHECK_EQ(padOut.size(), nDims_model)
        << "padOutput does not match model dimension";

    // read model_prototxt
    model_prototxt = hdf5_load_string(file_hid, "/model_prototxt");

    // read unet parameter 'mapInputNumPxGPUMemMB'
    //		int array size 2xN
    //			row 0: InputNumPx
    //			row 1: GPUMemMB
    // set 'memAvailPx' based on 'gpu_mem_MB'
    if ((tilingParameter == None || tilingParameter == MemMB) &&
        useOption == MemPx) {
      const std::string group_name("/unet_param");
      hid_t group_hid = H5Gopen(file_hid, group_name.c_str(), H5P_DEFAULT);
      CHECK_GE(group_hid, 0) << "Failed to open HDF5 group " << group_name;
      Blob<int>* mapInputNumPxGPUMemMB = new Blob<int>();
      hdf5_load_nd_dataset(
          group_hid, "mapInputNumPxGPUMemMB", 2, 2, mapInputNumPxGPUMemMB,
          true);
      LOG(INFO) << "\tmapInputNumPxGPUMemMB size: "
                << toString(mapInputNumPxGPUMemMB->shape());

      // find largest value GPUMemMB <= gpu_mem_MB
      // set 'memAvailPx' to corresponding value InputNumPx
      const int* map = mapInputNumPxGPUMemMB->cpu_data();
      const int mapNx = mapInputNumPxGPUMemMB->shape(-1);
      const int iy = 1;
      int ix = 0;
      int curr_GPUMemMB = map[iy * mapNx + ix];
      while (curr_GPUMemMB <= gpu_mem_MB && ix < mapNx ) {
        ++ix;
        if (ix < mapNx) curr_GPUMemMB = map[iy * mapNx + ix];
      }
      int ix_found = std::max(0, ix-1);
      memAvailPx = map[ 0 * mapNx + ix_found ];
      curr_GPUMemMB = map[ 1 * mapNx + ix_found ];
      LOG(INFO) << "Set memAvailPx: " << memAvailPx << " (based on map entry ["
                << memAvailPx << ", " << curr_GPUMemMB
                << "] from /unet_param/mapInputNumPxGPUMemMB)";
      CHECK_GE(memAvailPx, 1)
          << "Specified GPU memory available must be >= 1!";

      delete mapInputNumPxGPUMemMB;
    }
    H5Fclose(file_hid);

    CHECK(padding == "zero" || padding == "mirror")
        << "Supported padding options are 'zero' and 'mirror'.";
    bool mirrorPadFlag = (padding == "mirror");
    LOG(INFO) << "unet parameters";
    LOG(INFO) << "\t.unet-ident: " << unet_ident;
    LOG(INFO) << "\tinput_blob_name: " << input_blob_name;
    LOG(INFO) << "\tinput_num_spatial_dims: " << nDims_model;
    LOG(INFO) << "\tpadding: " << padding;
    LOG(INFO) << "\tdownsampleFactor: " << toString(dsFactor);
    LOG(INFO) << "\tpadInput: " << toString(padIn);
    LOG(INFO) << "\tpadOutput: " << toString(padOut);

    /*---------------------------------------------------------------------
     *  Read input data
     *---------------------------------------------------------------------*/
    LOG(INFO) << "Loading HDF5 input file: " << infileH5;
    hid_t file_id = H5Fopen(infileH5.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK_GE(file_id, 0) << "Failed to open HDF5 file " << infileH5;

    // read 'data' dataset to caffe blob
    const std::string dataset_name(input_dataset_name);
    Blob<Dtype>* dataBlob = new Blob<Dtype>();
    LOG(INFO) << "\tloading dataset: " << dataset_name;
    hdf5_load_nd_dataset(file_id, dataset_name.c_str(), 1, INT_MAX, dataBlob,
                         true);
    LOG(INFO) << "\tdata blob size: " << toString(dataBlob->shape());
    herr_t status = H5Fclose(file_id);
    CHECK_GE(status, 0) << "Failed to close HDF5 file " << infileH5;
    int nDims = dataBlob->num_axes() - 2;
    CHECK_EQ(nDims, nDims_model)
        << "Model requires " << nDims_model
        << "-D data, but the given input blob is" << nDims << "-D.";
    int nChannels = dataBlob->shape(1);

    // Crop data blob according to number of iterations
    if (iterations <= 0 || iterations > dataBlob->shape(0))
        iterations = dataBlob->shape(0);
    if (iterations < dataBlob->shape(0))
    {
      Blob<Dtype> *tmp = dataBlob;
      std::vector<int> croppedDataBlobShape(dataBlob->shape());
      croppedDataBlobShape[0] = iterations;
      dataBlob = new Blob<Dtype>(croppedDataBlobShape);
      std::memcpy(dataBlob->mutable_cpu_data(), tmp->cpu_data(),
                  dataBlob->count() * sizeof(Dtype));
      delete tmp;
    }

    std::vector<int> dataShape(nDims);
    for (int d = 0; d < nDims; ++d) dataShape[d] = dataBlob->shape(d + 2);

    std::vector<int> outputBlobIndices;
    std::vector<std::string> scoreBlobNames;
    std::vector<Blob<Dtype>*> scoreBlobs;
    std::vector<int> lastInTileShapeUsed;
    Net<Dtype> *caffe_net = NULL;

    /*---------------------------------------------------------------------
     *  Start tiled prediction with averaging
     *---------------------------------------------------------------------*/
    for (int avg = 0; avg < numAveragePasses; ++avg)
    {

      // Create flipped/rotated input blob
      Blob<Dtype> *transformedBlob = dataBlob;
      std::vector<Blob<Dtype>*> outBlobs;
      if (avg > 0)
      {
        transformedBlob = new Blob<Dtype>();
        if (average_mirror) {
          // Get flip configuration
          std::vector<bool> flipConfig(nDims, false);
          int tmp = avg;
          for (int d = nDims - 1; d >= 0; --d, tmp = tmp >> 1)
              flipConfig[d] = (tmp % 2 == 1);
          flip(*dataBlob, *transformedBlob, flipConfig);
        }
        else rotate90(*dataBlob, *transformedBlob, avg);
      }
      std::vector<int> transformedShape(nDims);
      for (int d = 0; d < nDims; ++d)
          transformedShape[d] = transformedBlob->shape(d + 2);

      std::vector<int> inTileShape, outTileShape;
      std::vector<int> tiling;

      // Get tiling layout and tile shape
      tiling = getTiling(
          useOption,
          (useOption == NTiles || useOption == Tiling) ? ntiles_param :
          ((useOption == Shape) ? tile_size_param : std::vector<int>()),
          transformedShape, dsFactor, padIn, padOut, memAvailPx, inTileShape,
          outTileShape);
      int ntiles = product(tiling);
      std::vector<int> border(
          vector_cast<int>(
              round(vector_cast<float>(inTileShape - outTileShape) / 2.0f)));
      LOG(INFO) << "Tile shape ([z] y x): input = " << toString(inTileShape)
                << ", output = " << toString(outTileShape);
      LOG(INFO) << "Selected tiling ([z] y x): " << toString(tiling);

      // Networks needs to be reinitialized due to tile shape change
      if (lastInTileShapeUsed != inTileShape) {

        delete caffe_net;

        // write temporary model definition stream (for phase: TEST)
        std::stringstream model_stream;
        model_stream << "state: { phase: TEST }" << std::endl;
        model_stream << "layer {" << std::endl;
        model_stream << "name: \"" << input_blob_name << "\"" << std::endl;
        model_stream << "type: \"Input\"" << std::endl;
        model_stream << "top: \"" << input_blob_name << "\"" << std::endl;
        model_stream << "input_param { shape: { dim: 1 dim: " << nChannels;
        for (int d = 0; d < nDims; ++d)
            model_stream << " dim: " << inTileShape[d];
        model_stream << " } }" << std::endl;
        model_stream << "}" << std::endl;
        model_stream << model_prototxt;

        NetParameter param;
        CHECK(google::protobuf::TextFormat::ParseFromString(
                  model_stream.str(), &param))
            << "Could not parse prototxt\n" << model_stream.str();
        param.mutable_state()->set_phase(caffe::TEST);

        // instantiate the caffe net
        caffe_net = new Net<Dtype>(param);

        // Reshape network input blob to padded tile shape
        std::vector<int> inTileBlobShape(make_int_vect(1, nChannels));
        inTileBlobShape.insert(
            inTileBlobShape.end(), inTileShape.begin(), inTileShape.end());
        caffe_net->blob_by_name(input_blob_name)->Reshape(inTileBlobShape);

        if (weights.size() > 0) caffe_net->CopyTrainedLayersFrom(weights);
        else LOG(INFO) << "Warning: No model weights are given. Net will be "
                       << "initialized without model weights!";

        // If this is the first network initialization, initialize
        // the score blobs
        if (avg == 0) {
          outputBlobIndices = caffe_net->output_blob_indices();
          scoreBlobs.resize(outputBlobIndices.size());
          scoreBlobNames.resize(outputBlobIndices.size());
          outBlobs.resize(outputBlobIndices.size());
          for (size_t i = 0; i < outputBlobIndices.size(); ++i) {
            scoreBlobNames[i] = caffe_net->blob_names()[outputBlobIndices[i]];
            std::vector<int> scoresShape(1, iterations);
            scoresShape.push_back(
                caffe_net->blobs()[outputBlobIndices[i]]->shape(1));
            scoresShape.insert(
                scoresShape.end(), dataShape.begin(), dataShape.end());
            scoreBlobs[i] = new Blob<Dtype>(scoresShape);
            outBlobs[i] = scoreBlobs[i];
          }
        }
        lastInTileShapeUsed = inTileShape;
      }

      if (avg > 0) {
        outBlobs.resize(outputBlobIndices.size());
        for (size_t i = 0; i < outputBlobIndices.size(); ++i) {
          std::vector<int> transformedBlobShape(1, iterations);
          transformedBlobShape.push_back(scoreBlobs[i]->shape(1));
          transformedBlobShape.insert(
              transformedBlobShape.end(), transformedShape.begin(),
              transformedShape.end());
          outBlobs[i] = new Blob<Dtype>(transformedBlobShape);
        }
      }

      // Process requested samples
      for (int n = 0; n < iterations; ++n) {

        // Process all tiles
        for (int tileIdx = 0; tileIdx < ntiles; ++tileIdx) {

          std::cout << "Processing batch " << avg * iterations + n + 1
                    << "/" << numAveragePasses * iterations
                    << ", tile " << tileIdx + 1 << "/" << ntiles << std::endl;

          // Get tile in grid and compute tile start position
          std::vector<int> tile(nDims);
          int tmp = tileIdx;
          bool skip = false;
          for (int d = nDims - 1; d >= 0; --d) {
            tile[d] = tmp % tiling[d];
            skip |= (tile[d] * outTileShape[d] >
                     transformedShape[d] + 2 * border[d]);
            tmp /= tiling[d];
          }
          if (skip)
          {
            LOG(INFO) << "----> skip " << toString(tile) << " (out of bounds)";
            continue;
          }
          else LOG(INFO) << "----> tile " << toString(tile) << " / "
                         << toString(tiling);
          std::vector<int> inTilePos(tile * outTileShape - border);
          std::vector<int> outTilePos(tile * outTileShape);
          std::vector<int> inPos(1, n);
          inPos.push_back(0);
          inPos.insert(inPos.end(), inTilePos.begin(), inTilePos.end());
          std::vector<int> outPos(1, n);
          outPos.push_back(0);
          outPos.insert(outPos.end(), outTilePos.begin(), outTilePos.end());

          // Pass cropped tile to network
          shared_ptr< Blob<Dtype> > inputBlob =
              caffe_net->blob_by_name(input_blob_name);
          copyBlock(
              *transformedBlob, *inputBlob, inputBlob->shape(),
              inPos, std::vector<int>(inputBlob->shape().size(), 0),
              mirrorPadFlag);
          vector<Blob<Dtype>*> const &caffeScores = caffe_net->Forward();
          for (size_t i = 0; i < caffeScores.size(); ++i)
          {
            copyBlock(
                *caffeScores[i], *outBlobs[i], caffeScores[i]->shape(),
                std::vector<int>(caffeScores[i]->shape().size(), 0), outPos,
                false);
          }
        }
      }

      // Undo transformation and accumulate scores
      if (avg > 0) {
        for (size_t i = 0; i < outBlobs.size(); ++i) {
          if (average_mirror) {
            // Get flip configuration
            std::vector<bool> flipConfig(nDims, false);
            int tmp = avg;
            for (int d = nDims - 1; d >= 0; --d, tmp = tmp >> 1)
                flipConfig[d] = (tmp % 2 == 1);
            flip(*outBlobs[i], *outBlobs[i], flipConfig);
          }
          else rotate90(*outBlobs[i], *outBlobs[i], -avg);
          Dtype const *inP = outBlobs[i]->cpu_data();
          Dtype *outP = scoreBlobs[i]->mutable_cpu_data();
          for (size_t j = 0; j < scoreBlobs[i]->count(); ++j, ++inP, ++outP)
              *outP += *inP;
          delete outBlobs[i];
        }
        delete transformedBlob;
      }
    }

    delete caffe_net;

    // Divide output scores by number of passes
    if (numAveragePasses > 1)
    {
      for (size_t i = 0; i < scoreBlobs.size(); ++i) {
        Dtype *outP = scoreBlobs[i]->mutable_cpu_data();
        for (size_t j = 0; j < scoreBlobs[i]->count(); ++j, ++outP)
            *outP /= numAveragePasses;
      }
    }

    delete dataBlob;

    // write hdf5 output file
    LOG(INFO) << "Saving HDF5 output file " << outfileH5;
    file_id = H5Fcreate(
        outfileH5.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(file_id, 0) << "Failed to create or reopen HDF5 file"
                         << outfileH5;
    for (size_t outIdx = 0; outIdx < scoreBlobs.size(); ++outIdx)
    {
      hdf5_save_nd_dataset(
          file_id, scoreBlobNames[outIdx], *scoreBlobs[outIdx]);
      delete scoreBlobs[outIdx];
    }
    status = H5Fclose(file_id);
    CHECK_GE(status, 0) << "Failed to close HDF5 file " << outfileH5;
  }

  template void TiledPredict<float>(
      const string& infileH5, const string& outfileH5, const string& model,
      const string& weights, int iterations,
      const string& gpu_mem_available_MB_str,
      const string& mem_available_px_str, const string& n_tiles_str,
      const string& tile_size_str, bool average_mirror, bool average_rotate);

  template void TiledPredict<double>(
      const string& infileH5, const string& outfileH5, const string& model,
      const string& weights, int iterations,
      const string& gpu_mem_available_MB_str,
      const string& mem_available_px_str, const string& n_tiles_str,
      const string& tile_size_str, bool average_mirror, bool average_rotate);

}  // namespace caffe
