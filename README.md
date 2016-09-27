# mantis
Deep learning models of machine translation using attentional and structural biases

## Introduction

This code is an implementation of the following work:

Incorporating Structural Alignment Biases into an Attentional Neural Translation Model. Trevor Cohn, Cong Duy Vu Hoang, Ekaterina Vymolova, Kaisheng Yao, Chris Dyer and Gholamreza Haffari. In Proceedings of NAACL-16, 2016. (long paper)

### Features

(to be updated)

### Building

mantis has been developed using external libraries, including cylab's cnn (https://github.com/clab/cnn.git) and eigen (https://bitbucket.org/eigen/eigen).

First, please clone it via our github link (https://github.com/trevorcohn/mantis.git), then do the followings:

>> git clone https://github.com/trevorcohn/mantis.git

>> cd $HOME/mantis

>> git submodule init 

>> git submodule update

>> cd $HOME/mantis && hg clone https://bitbucket.org/eigen/eigen/

Currently, mantis has been upgraded to be compatible with cnn version 2. Thus, please use the cnn version 2 by git-cloning it via "git clone -b v2 https://github.com/clab/cnn.git" instead.

Let's assume:

+ $PATH_TO_CNN=$HOME/mantis/cnn-v2/

+ $PATH_TO_EIGEN=$HOME/mantis/eigen/

+ $PATH_TO_CUDA=/usr/local/cuda-7.5/

First, we need to build cnn both in CPU and GPU versions.

* To build cnn with CPU-based version:

>> mkdir $PATH_TO_CNN/build

>> cd $PATH_TO_CNN/build

>> cmake .. -DEIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN

>> make -j 4

* To build cnn with GPU-based version:

>> mkdir $PATH_TO_CNN/build-cuda

>> cd $PATH_TO_CNN/build-cuda

>> cmake .. -DBACKEND=cuda -DCUDA_TOOLKIT_ROOT_DIR=$PATH_TO_CUDA

>> make -j 4

Please note that if you encounter the following compilation error (which should not happen in the latest cnn-v2), e.g.:

---

Linking CXX shared library libcnncuda_shared.so

/usr/bin/ld: CMakeFiles/cnncuda_shared.dir/./cnncuda_shared_intermediate_link.o: relocation R_X86_64_32S against `__nv_module_id' can not be used when making a shared object; recompile with -fPIC
CMakeFiles/cnncuda_shared.dir/./cnncuda_shared_intermediate_link.o: error adding symbols: Bad value

---

then, do the following:

>> do the "cmake ..." thing as mentioned earlier

(inside build-cuda directory) >> vim -v ./cnn/CMakeFiles/cnncuda_shared.dir/build.make

>> Add ' --compiler-options "-fPIC"' to the following line:

cd $PATH_TO_CNN/build-cuda/cnn && /usr/local/cuda-7.0/bin/nvcc -m64 -ccbin "/usr/bin/cc" <b>--compiler-options "-fPIC"</b> -dlink $PATH_TO_CNN/build-cuda/cnn/CMakeFiles/cnncuda_shared.dir//./cnncuda_shared_generated_gpu-ops.cu.o -o $PATH_TO_CNN/build-cuda/cnn/CMakeFiles/cnncuda_shared.dir/./cnncuda_shared_intermediate_link.o

>> make clean && make -j 4

(make sure the progress is 100% done!)

Next, we build our attentional model as follows:

>> cd $HOME/mantis/src

+ CPU version: 

>> g++ -g -o attentional attentional.cc -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN -std=c++11 -L/usr/lib -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -L/$PATH_TO_CNN/build/cnn -lcnn

>> g++ -g -o biattentional biattentional.cc -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN -std=c++11 -L/usr/lib -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -L/$PATH_TO_CNN/build/cnn -lcnn
 
+ GPU version: 

>> g++ -g -o attentional-gpu attentional.cc -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN -I/usr/local/cuda-7.0/include -std=c++11 -L/usr/lib -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -L/$PATH_TO_CNN/build-cuda/cnn -lcnn -lcnncuda -DHAVE_CUDA -L/usr/local/cuda-7.0/targets/x86_64-linux/lib -lcudart -lcublas

>> g++ -g -o biattentional-gpu biattentional.cc -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN -I/usr/local/cuda-7.0/include -std=c++11 -L/usr/lib -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -L/$PATH_TO_CNN/build-cuda/cnn -lcnn -lcnncuda -DHAVE_CUDA -L/usr/local/cuda-7.0/targets/x86_64-linux/lib -lcudart -lcublas

## Contacts

1) Trevor Cohn 

2) Hoang Cong Duy Vu

3) Reza Haffari 

---
Updated on April 2016

