#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "caffe/caffe.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/vector_helper.hpp"

#include "caffe_unet.h"

template<typename T>
void copyBlock(
    caffe::Blob<T> const &in, caffe::Blob<T> &out, std::vector<int> shape,
    std::vector<int> inPos, std::vector<int> outPos,
    bool padMirror);

class _UNetModel {
public:
    std::string model_prototxt;  // Network definition prototxt file
    std::string ident;           // Unique model identifier
    std::string name;            // Model name
    std::string description;     // Model description
    std::string input_blob_name; // Name of input blob in TEST phase

    int32_t n_dims;                     // Number of spatial dimensions of the data (2 or 3)
    int32_t n_channels;                 // Number of channels of the input data
    int32_t n_classes;                  // Number of classes of the output data
    std::vector<double> pixel_size_um;  // Input data pixel sizes in micrometer
    int32_t normalization_type;         // Input data normalization type
    bool padMirror;                     // Pad input data by mirroring
    std::vector<int32_t> dsFactor;      // Total downsampling factor
    std::vector<int32_t> padIn;         // Input padding
    std::vector<int32_t> padOut;        // Output padding

    caffe::Net<float> *caffe_net;
    std::vector<int> inTileShape;
    std::vector<int> outTileShape;
};

#ifdef __cplusplus
extern "C" {
#endif

void UNet_SetGPUDevice(uint8_t device_id) {
    caffe::Caffe::SetDevice(device_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
}

UNet_Err UNet_LoadModel(UNetModel* model, char* model_filename, char* weight_filename) {
    // Disable automatic printing of hdf5 error stack
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    // Set min logging level to WARNING.
    FLAGS_minloglevel = google::WARNING;
    google::InitGoogleLogging("caffe_unet");

    class _UNetModel *m = new _UNetModel();
    *model = (UNetModel)m;

    using namespace caffe;

    // -------------------------
    // Read model definition
    // -------------------------
    hid_t modeldef_hid = H5Fopen(model_filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (modeldef_hid < 0) {
        printf("Failed to open weight file\n");
        return UNet_ErrFileNotFound;
    }

    m->model_prototxt = hdf5_load_string(modeldef_hid, "/model_prototxt");
    m->ident = hdf5_load_string(modeldef_hid, "/.unet-ident");
    m->name = hdf5_load_string(modeldef_hid, "/unet_param/name");
    m->description = hdf5_load_string(modeldef_hid, "/unet_param/description");
    m->input_blob_name = hdf5_load_string(modeldef_hid, "/unet_param/input_blob_name");
    
    m->n_dims = hdf5_get_dataset_shape(modeldef_hid, "/unet_param/element_size_um")[0];
    m->pixel_size_um.resize(m->n_dims);
    herr_t status = H5LTread_dataset_double(modeldef_hid, "/unet_param/element_size_um", m->pixel_size_um.data());
    if (status < 0) {
        fprintf(stderr, "failed to load /unet_param/element_size_um\n");
        H5Fclose(modeldef_hid);
        return UNet_ErrInvalidModel;
    }
    
    m->normalization_type = hdf5_load_int(modeldef_hid, "/unet_param/normalization_type");

    std::string padding = hdf5_load_string(modeldef_hid, "/unet_param/padding");
    if (padding == "mirror") {
        m->padMirror = true;
    } else if (padding == "zero") {
        m->padMirror = false;
    } else {
        fprintf(stderr, "padding should be either 'zero' or 'mirror'\n");
        H5Fclose(modeldef_hid);
        return UNet_ErrInvalidModel;
    }

    m->dsFactor = hdf5_load_int_vec(modeldef_hid, "/unet_param/downsampleFactor");
    m->padIn = hdf5_load_int_vec(modeldef_hid, "/unet_param/padInput");
    m->padOut = hdf5_load_int_vec(modeldef_hid, "/unet_param/padOutput");
    
    // If any of these is a scalar, convert it to a vector of n_dims dimensions.
    if (m->dsFactor.size() == 1) {
        m->dsFactor.resize(m->n_dims, m->dsFactor[0]);
    }
    if (m->padIn.size() == 1) {
        m->padIn.resize(m->n_dims, m->padIn[0]);
    }
    if (m->padOut.size() == 1) {
        m->padOut.resize(m->n_dims, m->padOut[0]);
    }

    // Check number of dimensions
    if (m->dsFactor.size() != m->n_dims) {
        fprintf(stderr, "Number of downsample factors does not match model dimension\n");
        H5Fclose(modeldef_hid);
        return UNet_ErrInvalidModel;
    }
    if (m->padIn.size() != m->n_dims) {
        fprintf(stderr, "padInput does not match model dimension\n");
        H5Fclose(modeldef_hid);
        return UNet_ErrInvalidModel;
    }
    if (m->padOut.size() != m->n_dims) {
        fprintf(stderr, "padOut does not match model dimension\n");
        H5Fclose(modeldef_hid);
        return UNet_ErrInvalidModel;
    }

    H5Fclose(modeldef_hid);

    // -------------------------
    // Open weight file to get n_channels
    // -------------------------
    hid_t weight_hid = H5Fopen(weight_filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (weight_hid < 0) {
        printf("Failed to open weight file\n");
        return UNet_ErrFileNotFound;
    }
    m->n_channels = hdf5_get_dataset_shape(weight_hid, "/data/conv_d0a-b/0")[1];
    H5Fclose(weight_hid);

    // -------------------------
    // Set tile shape to minimal
    // -------------------------

    std::vector<int> outTileShapeMin;
    outTileShapeMin.assign(m->n_dims, 1);

    std::vector<int> d4a_size_min = vector_cast<int>(
        ceil(vector_cast<float>(outTileShapeMin - m->padOut) / 
             vector_cast<float>(m->dsFactor)));
    m->inTileShape = m->dsFactor * d4a_size_min + m->padIn;
    m->outTileShape = m->dsFactor * d4a_size_min + m->padOut;

    // --------------------
    // Initialized network
    // --------------------
    
    // write temporary model definition stream (for phase: TEST)
    std::stringstream model_stream;
    model_stream << "layer {" << std::endl;
    model_stream << "name: \"" << m->input_blob_name << "\"" << std::endl;
    model_stream << "type: \"Input\"" << std::endl;
    model_stream << "top: \"" << m->input_blob_name << "\"" << std::endl;
    model_stream << "input_param { shape: { dim: 1 dim: " << m->n_channels;
    for (int d = 0; d < m->n_dims; ++d) {
        model_stream << " dim: " << m->inTileShape[d];
    }
    model_stream << " } }" << std::endl;
    model_stream << "}" << std::endl;
    model_stream << m->model_prototxt;
    
    caffe::NetParameter param;
    google::protobuf::TextFormat::ParseFromString(model_stream.str(), &param);
    param.mutable_state()->set_phase(caffe::TEST);

    // Init caffe net
    m->caffe_net = new caffe::Net<float>(param);

    // Load trained weights from file
    m->caffe_net->CopyTrainedLayersFromHDF5(weight_filename);

    // Find num of output classes
    std::vector<int> outputBlobIndices = m->caffe_net->output_blob_indices();
    if (outputBlobIndices.size() != 1) {
        fprintf(stderr, "exactly 1 output blob is expected, got %ld\n", outputBlobIndices.size());
        return UNet_ErrInvalidModel;
    }
    int outputBlobIndex = outputBlobIndices[0];
    m->n_classes = m->caffe_net->blobs()[outputBlobIndex]->shape(1);

    return UNet_OK;
}

void UNet_Close(UNetModel model) {
    class _UNetModel *m = (class _UNetModel *)model;

    delete m->caffe_net;
    delete m;
}

UNet_Err UNet_SetTileShape(UNetModel model, int32_t* tile_shape) {
    using namespace caffe;
    
    class _UNetModel *m = (class _UNetModel *)model;
    std::vector<int> tileShape(tile_shape, tile_shape + m->n_dims);
    std::vector<int> d4a_size =
        vector_cast<int>(
        ceil((vector_cast<float>(tileShape - m->padOut)) /
              vector_cast<float>(m->dsFactor)));
    m->inTileShape = m->dsFactor * d4a_size + m->padIn;
    m->outTileShape = m->dsFactor * d4a_size + m->padOut;

    return UNet_OK;
}

UNet_Err UNet_GetTileShape(UNetModel model, int32_t* tile_shape) {
    class _UNetModel *m = (class _UNetModel *)model;
    for (int i = 0; i < m->n_dims; i++) {
        tile_shape[i] = m->outTileShape[i];
    }
    return UNet_OK;
}

UNet_Err UNet_TiledPredict(UNetModel model, int32_t* image_shape, float* image, float* score) {
    using namespace caffe;
    class _UNetModel *m = (class _UNetModel *)model;

    // Tiling shape
    std::vector<int32_t> ImageShape(image_shape, image_shape + m->n_dims);
    std::vector<int32_t> tiling = vector_cast<int32_t>(ceil(vector_cast<float>(ImageShape) /
                                    vector_cast<float>(m->outTileShape)));
    std::vector<int32_t> border = vector_cast<int32_t>(round(vector_cast<float>(m->inTileShape - m->outTileShape) / 2.0f));
    LOG(INFO) << "Selected tiling ([z] y x): " << toString(tiling);

    // Create input image blob
    std::vector<int32_t> imageBlobShape;
    imageBlobShape.push_back(1);
    imageBlobShape.push_back(m->n_channels);
    imageBlobShape.insert(imageBlobShape.end(), image_shape, image_shape + m->n_dims);
    Blob<float>* imageBlob = new Blob<float>(imageBlobShape);
    imageBlob->set_cpu_data(image);

    // Create output score blob
    std::vector<int32_t> scoreBlobShape;
    scoreBlobShape.push_back(1);
    scoreBlobShape.push_back(m->n_classes);
    scoreBlobShape.insert(scoreBlobShape.end(), image_shape, image_shape + m->n_dims);
    int32_t scoreBlobSize = product(scoreBlobShape);
    Blob<float>* scoreBlob = new Blob<float>(scoreBlobShape);
    
    // Get input title blob and set the blob shape
    std::vector<int32_t> inputTileBlobShape;
    inputTileBlobShape.push_back(1);
    inputTileBlobShape.push_back(m->n_channels);
    inputTileBlobShape.insert(inputTileBlobShape.end(), m->inTileShape.begin(), m->inTileShape.end());
    shared_ptr<caffe::Blob<float> > inputTileBlob = m->caffe_net->blob_by_name(m->input_blob_name);
    if (inputTileBlob->shape() != inputTileBlobShape) {
        inputTileBlob->Reshape(inputTileBlobShape);
    }

    // Process tiles
    int n_tiles = product(tiling);
    for (int i_tile = 0; i_tile < n_tiles; i_tile++) {
        LOG(INFO) << "Processing tile " << i_tile + 1 << "/" << n_tiles;

        // Get tile in grid and compute tile start position
        std::vector<int> tile(m->n_dims);
        int tmp = i_tile;
        bool skip = false;
        for (int d = m->n_dims - 1; d >= 0; --d) {
            tile[d] = tmp % tiling[d];
            skip |= (tile[d] * m->outTileShape[d] > ImageShape[d] + 2 * border[d]);
            tmp /= tiling[d];
        }
        if (skip) {
            LOG(INFO) << "----> skip " << toString(tile) << " (out of bounds)";
            continue;
        } else {
            LOG(INFO) << "----> tile " << toString(tile) << " / "
                        << toString(tiling);
        }
        std::vector<int> inTilePos(tile * m->outTileShape - border);
        std::vector<int> outTilePos(tile * m->outTileShape);
        std::vector<int> inPos;
        inPos.push_back(0);
        inPos.push_back(0);
        inPos.insert(inPos.end(), inTilePos.begin(), inTilePos.end());
        std::vector<int> outPos;
        outPos.push_back(0);
        outPos.push_back(0);
        outPos.insert(outPos.end(), outTilePos.begin(), outTilePos.end());

        // Pass tile through U-Net
        copyBlock(
            *imageBlob, *inputTileBlob, inputTileBlob->shape(),
            inPos, std::vector<int>(inputTileBlob->shape().size(), 0),
            m->padMirror);
        std::vector<Blob<float>*> const &outputBlobs = m->caffe_net->Forward();
        if (outputBlobs.size() != 1) {
            fprintf(stderr, "expecting 1 output blob from caffe_net->Forward(), got %ld\n", outputBlobs.size());
            return UNet_ErrInvalidModel;
        }
        Blob<float>* outputTileBlob = outputBlobs[0];
        copyBlock(
            *outputTileBlob, *scoreBlob, outputTileBlob->shape(),
            std::vector<int>(outputTileBlob->shape().size(), 0), outPos,
            m->padMirror);
    }
    memcpy(score, scoreBlob->cpu_data(), sizeof(float) * scoreBlobSize);
    return UNet_OK;
}

UNet_Err UNet_ModelID(UNetModel model, char *ident, uint32_t size) {
    class _UNetModel *m = (class _UNetModel *)model;
    uint32_t size_copy = m->ident.length() + 1;
    if (size_copy > size) {
        return UNet_ErrBufferSize;
    }
    memcpy(ident, m->ident.c_str(), size_copy);
    return UNet_OK;
}

UNet_Err UNet_ModelName(UNetModel model, char *name, uint32_t size)  {
    class _UNetModel *m = (class _UNetModel *)model;
    uint32_t size_copy = m->ident.length() + 1;
    if (size_copy > size) {
        return UNet_ErrBufferSize;
    }
    memcpy(name, m->ident.c_str(), size_copy);
    return UNet_OK;
}

UNet_Err UNet_ModelDescription(UNetModel model, char *description, uint32_t size)  {
    class _UNetModel *m = (class _UNetModel *)model;
    uint32_t size_copy = m->ident.length() + 1;
    if (size_copy > size) {
        return UNet_ErrBufferSize;
    }
    memcpy(description, m->ident.c_str(), size_copy);
    return UNet_OK;
}

int32_t UNet_NumDims(UNetModel model) {
    class _UNetModel *m = (class _UNetModel *)model;
    return m->n_dims;
}

int32_t UNet_NumChannels(UNetModel model) {
    class _UNetModel *m = (class _UNetModel *)model;
    return m->n_channels;
}

int32_t UNet_NumClasses(UNetModel model) {
    class _UNetModel *m = (class _UNetModel *)model;
    return m->n_classes;
}

void UNet_PixelSize(UNetModel model, double *pixel_size_um) {
    class _UNetModel *m = (class _UNetModel *)model;
    for (int i = 0; i < m->n_dims; i++) {
        pixel_size_um[i] = m->pixel_size_um[i];
    }
}

int32_t UNet_NormalizationType(UNetModel model) {
    class _UNetModel *m = (class _UNetModel *)model;
    return m->normalization_type;
}

#ifdef __cplusplus
}
#endif