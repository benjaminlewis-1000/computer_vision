cmake_minimum_required(VERSION 2.8)
project( Tennis )
find_package( OpenCV REQUIRED )
find_package(Threads REQUIRED )

add_executable( tennis tennis_ball.cpp )
target_link_libraries( tennis ${OpenCV_LIBS}  )

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3")        ## Optimize
    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()
