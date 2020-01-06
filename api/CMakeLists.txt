add_library(caffeunet caffe_unet.cpp utils.cpp)
target_link_libraries(caffeunet ${Caffe_LINK})
caffe_default_properties(caffeunet)

FIND_PACKAGE( TIFF )

add_executable(caffe_unet_example example/unet_api_example.c)
target_link_libraries(caffe_unet_example caffeunet ${TIFF_LIBRARIES})
caffe_default_properties(caffe_unet_example)
include_directories(${PROJECT_SOURCE_DIR} ${TIFF_INCLUDE_DIR})