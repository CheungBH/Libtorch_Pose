#project(Test)
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(Test)

include_directories(include)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

set(SOURCES main.cpp Darknet.cpp Hungarian.cpp KalmanTracker.cpp Sppe.cpp TrackingBox.cpp utils.cpp Tracker.cpp)

add_executable(Test ${SOURCES})

target_link_libraries(Test ${TORCH_LIBRARIES} ${OpenCV_LIBS})

set_property(TARGET Test PROPERTY CXX_STANDARD 14)



