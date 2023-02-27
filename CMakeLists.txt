cmake_minimum_required(VERSION 3.24)
project(opencv-examples)

# 让find_package找到conan安装的包
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/build)  #添加到find_package module模式搜索路径
# list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/build)  #添加到find_package config模式搜索路径

if(WIN32)
	add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
	find_path(OpenCV_DIR OpenCVConfig.cmake PATHS 
	          $ENV{OpenCV_DIR}
		"D:/opencv/opencv-4.7.0" # PC Amaury Mines
			  )
	find_package(OpenCV REQUIRED)
else()
	find_package(OpenCV REQUIRED)
endif()

#
# OpenCV自带cmake不支持target链接方式
# 这里自己创建一个，方便子工程链接
#
MESSAGE(STATUS ${OpenCV_LIBS})
MESSAGE(STATUS ${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS ${OpenCV_SHARED})
add_library(OpenCV INTERFACE)
target_link_libraries(OpenCV INTERFACE ${OpenCV_LIBS})

# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
#全部Target都C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# 项目配置
# 
set(DataDir "${CMAKE_SOURCE_DIR}/data/")

#
# 子工程
#
add_subdirectory(examples)

#
# 其他配置
#

# 代码分组
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${CommonHeadersFiles})
# 开启工程分组
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# 关闭编译器优化
set(CMAKE_CXX_FLAGS "-fno-elide-constructors ${CMAKE_CXX_FLAGS}")