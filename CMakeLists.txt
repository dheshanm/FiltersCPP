cmake_minimum_required(VERSION 3.25)
project(vision_cpp)

set(CMAKE_CXX_STANDARD 23)

add_executable(app src/main.cpp src/utils/camera/camera.cpp src/utils/camera/camera.h src/utils/filters.h src/utils/watch_channel.h src/utils/processor/processor.cpp src/utils/processor/processor.h src/constants.h src/tasks/greyscale.h src/tasks/blur.h src/tasks/negative.h src/tasks/sobel.h src/utils/kernels.h src/tasks/magnitude.h src/tasks/task.h src/tasks/quantize.h)

# OpenCV
FIND_PACKAGE( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
TARGET_LINK_LIBRARIES (app ${OpenCV_LIBS})

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(app OpenMP::OpenMP_CXX)
endif()


