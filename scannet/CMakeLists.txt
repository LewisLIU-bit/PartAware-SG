cmake_minimum_required(VERSION 3.16)
project(scannet_sg LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

find_package(jsoncpp CONFIG REQUIRED)

find_package(PCL REQUIRED COMPONENTS
    common
    io
    filters
    segmentation
    visualization
)

find_package(Boost REQUIRED COMPONENTS filesystem)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(
    openset_ply_map
    src/openset_ply_map.cpp
)

target_link_libraries(
    openset_ply_map
    PRIVATE
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Boost::filesystem
    dw
    elf
)

add_executable(
    generate_json
    src/generate_json.cpp
)

target_link_libraries(
    generate_json
    PRIVATE
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Boost::filesystem
    dw
    elf
)
