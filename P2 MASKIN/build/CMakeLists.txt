cmake_minimum_required(VERSION 3.20)
project(neural_network_cpp)
include_directories(../inc)
add_executable(run_neural_network ../src/main.cpp 
                                  ../src/dense_layer.cpp 
                                  ../src/neural_network.cpp
                                  ../src/button.cpp
                                  ../src/led.cpp
                                  ../src/gpiod_utils.c)
target_compile_options(run_neural_network PRIVATE -Wall -Werror)
target_link_libraries(run_neural_network gpiod)
set_target_properties(run_neural_network PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../output)