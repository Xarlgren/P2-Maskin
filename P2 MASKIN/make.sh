#!/bin/bash

################################################################################
# @brief Script for building and running the neural network.
#
# @note Run the script in this directory by using the source command:
#       $ source make.sh
################################################################################
cd build                    # Redirects to the build subdirectory.
make                        # Builds the project with CMake.
cd ..                       # Redirects to the base directory.
./output/run_neural_network # Runs the program.