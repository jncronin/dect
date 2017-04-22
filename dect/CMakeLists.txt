include(FindPkgConfig)
include(CheckLibraryExists)
include(CheckIncludeFiles)
pkg_search_module(OCL OpenCL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(DECT_SOURCES "main.cpp")
set(EXTRA_LIBS ${EXTRA_LIBS} dectlib)

find_path(LibTiffIncludeDir tiffio.h NO_DEFAULT_PATH)
if(NOT LibTiffIncludeDir)
	message(FATAL_ERROR "Cannot find libtiff include directory")
endif()

include_directories(../libdect)
include_directories(${LibTiffIncludeDir})

find_library(LibTiffLibrary tiff NO_DEFAULT_PATH)
if(NOT LibTiffLibrary)
	message(FATAL_ERROR "Cannot find libtiff")
endif()

add_executable(dect ${DECT_SOURCES})
target_link_libraries(dect ${EXTRA_LIBS})
target_link_libraries(dect ${LibTiffLibrary})
target_link_libraries(dect ${OCL_LIBRARIES})

install (TARGETS dect RUNTIME DESTINATION bin)

