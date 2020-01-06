#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "tiffio.h"
#include "api/caffe_unet.h"

uint16_t *tiff_imread_u16(char *filename, uint32_t *width, uint32_t *height) {
    TIFF *tif = TIFFOpen(filename, "r");
    if (tif == NULL) {
        return NULL;
    }

    uint16_t bits_per_sample;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
    if (bits_per_sample != 16) {
        fprintf(stderr, "expecting BitsPerSample=16, got %d\n", bits_per_sample);
        TIFFClose(tif);
        return NULL;
    }

    uint16_t sample_per_pixel;
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &sample_per_pixel);
    if (sample_per_pixel != 1) {
        fprintf(stderr, "expecting SamplesPerPixel = 1, got %d\n", sample_per_pixel);
        TIFFClose(tif);
        return NULL;
    }

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, height);
    
    uint16_t *imbuf = malloc(sizeof(uint16_t) * (*width) * (*height));

    uint32_t row;
    for (row = 0; row < *height; row++) {
        TIFFReadScanline(tif, (void *)(imbuf + row * (*width)), row, 0);
    }

    TIFFClose(tif);

    return imbuf;
}

void tiff_imwrite_u8(char *filename, uint8_t *buf, uint32_t width, uint32_t height) {
    TIFF *tif = TIFFOpen(filename, "w");

    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1); // min-is-black
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);

    uint32_t row;
    for (row = 0; row < height; row++) {
        TIFFWriteScanline(tif, (void *)(buf + row * width), row, 0);
    }

    TIFFClose(tif);
    return;
}

int main() {
    UNet_SetGPUDevice(0);

    struct timespec time_start, time_end;
    float time_elapsed;

    UNetModel model;

    printf("Loading model...\n");
    clock_gettime(CLOCK_REALTIME, &time_start);
    UNet_Err ret = UNet_LoadModel(&model,
        "/tmp/caffe_unet/2d_cell_net_v0_model/2d_cell_net_v0.modeldef.h5",
        "/tmp/caffe_unet/2d_cell_net_v0_model/2d_cell_net_v0.caffemodel.h5");
    if (ret != UNet_OK) {
        fprintf(stderr, "failed to load model\n");
        return -1;
    }
    clock_gettime(CLOCK_REALTIME, &time_end);
    time_elapsed = (time_end.tv_sec - time_start.tv_sec) + 1.0e-9 * (time_end.tv_nsec - time_start.tv_nsec);
    printf("Finished in %.6f s.\n", time_elapsed);

    int32_t tileShape[2] = {528, 592};
    ret = UNet_SetTileShape(model, (int32_t *)&tileShape);

    // Get model info
    int len_buf = 4096;
    char buf[len_buf];
    UNet_ModelID(model, buf, len_buf);
    printf("Model ID: %s\n", buf);
    UNet_ModelName(model, buf, len_buf);
    printf("Model Name: %s\n", buf);

    int32_t n_dims = UNet_NumDims(model);
    int32_t n_channels = UNet_NumChannels(model);
    int32_t n_classes = UNet_NumClasses(model);
    printf("Spatial Dimensions: %d\n", n_dims);
    printf("Input Image Channels: %d\n", n_channels);
    printf("Output Score Classes: %d\n", n_classes);

    // Read input image
    uint32_t width, height;
    uint16_t *im = tiff_imread_u16("/tmp/caffe_unet/sampledata/BF-Microspores/BF-C2DH-MiSp_01.tif", &width, &height);
  
    // Normalize to [0, 1]
    float *imnorm = malloc(sizeof(float) * width * height);
    float im_max = im[0];
    float im_min = im[0];
    int i;
    for (i = 1; i < width * height; i++) {
        if (im[i] > im_max) {
            im_max = im[i];
        }
        if (im[i] < im_min) {
            im_min = im[i];
        }
    }
    float im_range = im_max - im_min;
    for (i = 0; i < width * height; i++) {
        imnorm[i] = ((float)im[i] - im_min) / im_range;
    }
  
    // Perform U-Net tiled predict
    float *score = malloc(sizeof(float) * n_classes * width * height);
    int im_shape[2] = {height, width};

    printf("Performing tiled predict...\n");
    clock_gettime(CLOCK_REALTIME, &time_start);
    ret = UNet_TiledPredict(model, (int *)&im_shape, imnorm, score);
    if (ret != 0) {
        fprintf(stderr, "UNet_TiledPredict returned %d\n", ret);
        return -1;
    }
    clock_gettime(CLOCK_REALTIME, &time_end);
    time_elapsed = (time_end.tv_sec - time_start.tv_sec) + 1.0e-9 * (time_end.tv_nsec - time_start.tv_nsec);
    printf("Finished in %.6f s.\n", time_elapsed);

    // Generate segmentation mask assuming n_classes = 2
    uint8_t *immask = malloc(sizeof(uint8_t) * width * height);
    for (i = 0; i < width * height; i++) {
        if (score[i] < score[i + width * height]) {
            immask[i] = 255;
        } else {
            immask[i] = 0;
        }
    }

    // Write segmentation mask to tiff file
    tiff_imwrite_u8("/tmp/caffe_unet/output.tif", immask, width, height);

    free(im);
    free(imnorm);
    free(immask);

    return 0;
}