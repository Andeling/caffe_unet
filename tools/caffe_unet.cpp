#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/tiled_predict.hpp"
#include "caffe/util/vector_helper.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::NetParameter;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using caffe::hdf5_load_int;
using caffe::hdf5_load_string;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(model, "",
    "The model definition hdf5 file.");
DEFINE_string(weights, "",
    "Optional; The pretrained weights for net initialization."
    "When no model weights are given the net will be initialized "
    "without model weights.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run. Set to zero to run over all images.");

// for command tiled_predict only
DEFINE_string(
    infileH5, "", "The hdf5 input file.");
DEFINE_string(
    outfileH5, "", "The hdf5 output file.");
DEFINE_string(
    gpu_mem_available_MB, "", "Tiling option 1 (GPU mode only, 2d input data "
    "only): In GPU mode you can give the available GPU memory in MB (Specify "
    "as a string, e.g. '6000'). Internally this will be transformed to "
    "mem_available_px via the memory usage table that was recorded during "
    "creation of the modelfile. The modelfile can be created using the matlab "
    "interface in matlab/unet/createModeldef_H5.m. If no such table is "
    "available, the function will fail! (This option is used implicitly when "
    "no tiling option is specified, the parameter is then determined "
    "automatically using Cuda).");

DEFINE_string(
    mem_available_px, "", "Tiling option 2 (2d input data only): You can give "
    "the number of input pixels that can be maximally input to the network "
    "(Specify as a string, e.g. '291600', which corresponds to '540x540' "
    "pixels.). This value must be given including required padding.");
DEFINE_string(
    n_tiles, "", "Tiling option 3 (2d and 3d input data): (2d input data "
    "only) You can give the total number of tiles (Specify as a string, e.g. "
    "'4'). The tile shape will be determined to minimize memory consumption. "
    "(2d and 3d input data) Alternatively you can directly specify the tile "
    "layout [(nz) ny nx] (Specify as string, e.g. '2x2', or '2x2x2').");
DEFINE_string(
    tile_size, "", "Tiling option 4 (2d and 3d input data): The tile shape "
    "[(z) y x] in pixels (Specify as a string, e.g. '540x540', or "
    "'28x44x44').");
DEFINE_bool(
    average_mirror, false, "Use the average over mirrored versions of the "
    "input tiles for the final segmentation. Cannot be used in conjunction "
    "with average_rotate.");
DEFINE_bool(
    average_rotate, false, "Use the average over rotated versions of the "
    "input tiles for the final segmentation. Cannot be used in conjunction "
    "with average_mirror.");

// for command check_model_and_weights_h5 only
DEFINE_int32(n_channels, 1,
    "The number of input channels. Required in check_model_and_weights_h5 "
    "mode to create a test blob of appropriate dimensionality.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe_unet actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// caffe_unet commands to call by
//     caffe_unet <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);


// check_model_and_weights_h5: Check whether the given model and weights are
// valid and compatible
int check_model_and_weights_h5() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to check.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to check.";
  LOG(INFO) << "Checking model " << FLAGS_model << " and weights "
            << FLAGS_weights;

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // Disable automatic printing of hdf5 error stack
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);

  std::string model_prototxt_file(FLAGS_model);

  if (FLAGS_model.size() < 3 ||
      FLAGS_model.compare(FLAGS_model.size() - 3, 3, ".h5") != 0) return 1;

  // Get U-net ID from model
  LOG(INFO) << "Reading unet ID from '" << FLAGS_model.c_str() << "'";
  hid_t file_hid = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Could not open '" << FLAGS_model << "'";
  std::string unet_ident_model = hdf5_load_string(file_hid, "/.unet-ident");
  LOG(INFO) << "  unet ID (model) = '" << unet_ident_model << "'";

  std::string input_blob_name =
      caffe::hdf5_load_string(file_hid, "/unet_param/input_blob_name");
  int nDims =
      caffe::hdf5_get_dataset_shape(file_hid, "/unet_param/element_size_um")[0];
  int nChannels = FLAGS_n_channels;
  std::vector<int> dsFactor(
      caffe::hdf5_load_int_vec(file_hid, "/unet_param/downsampleFactor"));
  if (dsFactor.size() == 1) dsFactor.resize(nDims, dsFactor[0]);
  std::vector<int> padIn(
      caffe::hdf5_load_int_vec(file_hid, "/unet_param/padInput"));
  if (padIn.size() == 1) padIn.resize(nDims, padIn[0]);
  std::vector<int> padOut(
      caffe::hdf5_load_int_vec(file_hid, "/unet_param/padOutput"));
  if (padOut.size() == 1) padOut.resize(nDims, padOut[0]);
  std::string model_prototxt =
      caffe::hdf5_load_string(file_hid, "/model_prototxt");
  H5Fclose(file_hid);

  std::vector<int> minBottomBlobSize(
      caffe::vector_cast<int>(
          caffe::operator+(
              1, caffe::operator-(
                  0, caffe::vector_cast<int>(
                      caffe::operator/(
                          caffe::vector_cast<float>(padOut),
                          caffe::vector_cast<float>(dsFactor)))))));
  std::vector<int> minShape(
      caffe::operator+(caffe::operator*(minBottomBlobSize, dsFactor), padIn));

  if (FLAGS_weights.size() >= 3 &&
      FLAGS_weights.compare(FLAGS_weights.size() - 3, 3, ".h5") == 0) {

    // Get U-net ID from weights
    LOG(INFO) << "Reading unet ID from '" << FLAGS_weights.c_str() << "'";
    file_hid = H5Fopen(
        FLAGS_weights.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK_GE(file_hid, 0) << "Could not open '" << FLAGS_weights << "'";
    size_t size;
    H5T_class_t class_;
    herr_t status = H5LTget_dataset_info(
        file_hid, "/.unet-ident", NULL, &class_, &size);
    if (status >= 0) {
      char *buf = new char[size];
      buf[0] = 0;
      status = H5LTread_dataset_string(file_hid, "/.unet-ident", buf);
      std::string unet_ident_weights(buf);
      delete[] buf;
      H5Fclose(file_hid);

      if (status < 0) {
        LOG(INFO) << " Could not read '/.unet_ident' in '" << FLAGS_weights
                  << "'";
        return 1;
      }

      LOG(INFO) << "  unet ID (weights) = '" << unet_ident_weights << "'";
      CHECK(unet_ident_model == unet_ident_weights)
          << "Unet IDs do not match. Check failed!";

      return 0;
    }
    else
    {
      H5Fclose(file_hid);
      LOG(INFO) << "'" << FLAGS_weights << "' does not contain '/.unet_ident'";
    }
  }

  // write temporary model definition stream (for phase: TEST)
  std::stringstream model_stream;
  model_stream << "state: { phase: TEST }" << std::endl;
  model_stream << "layer {" << std::endl;
  model_stream << "name: \"" << input_blob_name << "\"" << std::endl;
  model_stream << "type: \"Input\"" << std::endl;
  model_stream << "top: \"" << input_blob_name << "\"" << std::endl;
  model_stream << "input_param { shape: { dim: 1 dim: " << nChannels;
  for (int d = 0; d < nDims; ++d) model_stream << " dim: " << minShape[d];
  model_stream << " } }" << std::endl;
  model_stream << "}" << std::endl;
  model_stream << model_prototxt;

  NetParameter param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
            model_stream.str(), &param))
      << "Could not parse prototxt\n" << model_stream.str();
  param.mutable_state()->set_phase(caffe::TEST);

  Net<float> caffe_net(param);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  shared_ptr<Blob<float> > input_blob = caffe_net.blob_by_name(input_blob_name);
  std::vector<int> inputBlobShape;
  inputBlobShape.push_back(1);
  inputBlobShape.push_back(nChannels);
  inputBlobShape.insert(inputBlobShape.end(), minShape.begin(), minShape.end());
  input_blob->Reshape(inputBlobShape);

  float iter_loss;
  caffe_net.Forward(&iter_loss);

  return 0;
}
RegisterBrewFunction(check_model_and_weights_h5);


// Tiled predict (wrapper): score a model in overlap-tile strategy for
// passing large images through caffe
int tiled_predict() {

  // set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // calls Tiled predict and passes cmdline flags
  caffe::TiledPredict<float>(
      FLAGS_infileH5, FLAGS_outfileH5, FLAGS_model, FLAGS_weights,
      FLAGS_iterations, FLAGS_gpu_mem_available_MB, FLAGS_mem_available_px,
      FLAGS_n_tiles, FLAGS_tile_size, FLAGS_average_mirror,
      FLAGS_average_rotate );

  return 0;

}
RegisterBrewFunction(tiled_predict);


int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe_unet <command> <args>\n\n"
      "commands:\n"
      "  tiled_predict                score a model in overlap-tile strategy\n"
      "  check_model_and_weights_h5   check given model (hdf5) and weights\n"
      "  device_query                 show GPU diagnostic information");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe_unet");
  }
}
