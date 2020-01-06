FIND_PACKAGE( TIFF )

add_executable(caffe_unet_example example/unet_api_example.c)
target_link_libraries(caffe_unet_example caffeunet ${TIFF_LIBRARIES})
caffe_default_properties(caffe_unet_example)
include_directories(${PROJECT_SOURCE_DIR}/src/unet_api ${TIFF_INCLUDE_DIR})