cmake_minimum_required(VERSION 2.8)
project( AppProject )
find_package( OpenCV REQUIRED )

add_executable(ROI findROI.cpp)
target_link_libraries( ROI ${OpenCV_LIBS} )
add_executable(inpaint oculus_base.cpp)
target_link_libraries(inpaint ${OpenCV_LIBS} )
