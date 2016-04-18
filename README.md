# mantis
Deep learning models of machine translation using attentional and structural biases

## Introduction

### Features

### Building

mantis has been developed using external libraries, including cylab's cnn (https://github.com/clab/cnn.git) and eigen (https://bitbucket.org/eigen/eigen).

Let's assume:

+ PATH_TO_CNN=$HOME/cnn/

+ PATH_TO_EIGEN=$HOME/cnn/eigen/

First, we need to build cnn both in CPU and GPU versions.

* To build cnn with CPU-based version:

>> mkdir PATH_TO_CNN/build

>> cd PATH_TO_CNN/build

>> cmake .. -DEIGEN3_INCLUDE_DIR=PATH_TO_EIGEN

>> make -j 4

* To build cnn with GPU-based version:

>> mkdir PATH_TO_CNN/build-cuda

>> cd PATH_TO_CNN/build-cuda

>> cmake .. -DBACKEND=cuda -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-7.5/

>> make -j 4

Next, we build our attentional model as follows:

>> cd $HOME/mantis/src

+ CPU version: 

>> g++ -g -o attentional attentional.cc -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN -std=c++11 -L/usr/lib -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -L/$PATH_TO_CNN/build/cnn -lcnn
 
+ GPU version: 

>> g++ -g -o attentional-gpu attentional.cc -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN -I/usr/local/cuda-7.0/include -std=c++11 -L/usr/lib -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -L/$PATH_TO_CNN/build-cuda/cnn -lcnncuda -DHAVE_CUDA -L/usr/local/cuda-7.0/targets/x86_64-linux/lib -lcudart -lcublas

## Contacts

1) Trevor Cohn (t.cohn@unimelb.edu.au)

2) Hoang Cong Duy Vu (vhoang2@student.unimelb.edu.au or duyvuleo@gmail.com)

3) Reza Haffari (gholamreza.haffari@monash.edu)

---
Updated on April 2016

