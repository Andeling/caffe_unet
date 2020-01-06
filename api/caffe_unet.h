#ifndef _CAFFE_UNET_H
#define _CAFFE_UNET_H

typedef void *UNetModel;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    UNet_OK = 0,
    UNet_ErrMisc = -1,
    UNet_ErrFileNotFound = -2,
    UNet_ErrInvalidModel = -3,
    UNet_ErrInvalidWeight = -4,
    UNet_ErrBufferSize = -5,
    UNet_ErrInvalidParameter = -6,
} UNet_Err;

void UNet_SetGPUDevice(uint8_t device_id);

UNet_Err UNet_LoadModel(UNetModel* model, char* model_filename, char* weight_filename);
void UNet_Close(UNetModel model);

UNet_Err UNet_SetTileShape(UNetModel model, int32_t* tile_shape);
UNet_Err UNet_GetTileShape(UNetModel model, int32_t* tile_shape);
UNet_Err UNet_TiledPredict(UNetModel model, int32_t* image_shape, float* image, float* score);

UNet_Err UNet_GetID(UNetModel model, char *ident, uint32_t size);
UNet_Err UNet_GetName(UNetModel model, char *name, uint32_t size);
UNet_Err UNet_GetDescription(UNetModel model, char *description, uint32_t size);
int32_t UNet_GetNumDims(UNetModel model);
int32_t UNet_GetNumChannels(UNetModel model);
int32_t UNet_GetNumClasses(UNetModel model);
void UNet_GetPixelSize(UNetModel model, double *pixel_size_um);
int32_t UNet_GetNormalizationType(UNetModel model);

#ifdef __cplusplus
}
#endif
#endif
