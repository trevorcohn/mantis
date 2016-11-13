# mantis

Deep learning models of machine translation using attention and structural bias. This is build on top of the cnn neural network library, using
C++. Please refer to the [cnn github page](http://github.com/clab/cnn) for more details, including some issues with compiling and running with
the library. 

This code is an implementation of the following paper:

    Incorporating Structural Alignment Biases into an Attentional Neural Translation Model. 
    Trevor Cohn, Cong Duy Vu Hoang, Ekaterina Vymolova, Kaisheng Yao, Chris Dyer and Gholamreza Haffari. 
    In Proceedings of NAACL-16, 2016. (long paper)

Please cite the above paper if you use or extend this code.

### Dependencies

Before compiling cnn, you need:
 * [Eigen](https://bitbucket.org/eigen/eigen), using the development version (not release), e.g. 3.3.beta2
 * [cuda](https://developer.nvidia.com/cuda-toolkit) version 7.5 or higher

### Building

First, clone the repository

    git clone https://github.com/trevorcohn/mantis.git

Next pull down the submodules (cnn)

    cd mantis
    git submodule init 
    git submodule update

As mentioned above, you'll need the latest development version of eigen

    hg clone https://bitbucket.org/eigen/eigen/

#### CPU build

Compiling to execute on a CPU is as follows

    mkdir build_cpu
    cd build_cpu
    cmake .. -DEIGEN3_INCLUDE_DIR=eigen
    make -j 2

MKL support. If you have Intel's MKL library installed on your machine, you can speed up the computation on the CPU by:

    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN -DMKL=TRUE -DMKL_ROOT=MKL

substituting in different paths to EIGEN and MKL if you have placed them in different directories. 

This will build the two binaries
    
    build_cpu/src/attentional
    build_cpu/src/biattentional


#### GPU build

Building on the GPU uses the Nvidia CUDA library, currently tested against version 7.5.
The process is as follows

    mkdir build_gpu
    cd build_gpu
    cmake .. -DBACKEND=cuda -DEIGEN3_INCLUDE_DIR=EIGEN -DCUDA_TOOLKIT_ROOT_DIR=CUDA
    make -j 2

substituting in your Eigen and CUDA folders, as appropriate.

This will result in the two binaries

    build_gpu/src/attentional
    build_gpu/src/biattentional

#### Using the model

The model can be run as follows

    ./build_cpu/src/attentional -t sample-data/train.de-en.unk.cap -d sample-data/dev.de-en.unk.cap 

which will train a small model on a tiny training set, i.e.,

    (CPU)
    [cnn] random seed: 978201625
    [cnn] allocating memory: 512MB
    [cnn] memory allocation done.
    Reading training data from sample-data/train.de-en.unk.cap...
    5000 lines, 117998 & 105167 tokens (s & t), 2738 & 2326 types
    Reading dev data from sample-data/dev.de-en.unk.cap...
    100 lines, 1800 & 1840 tokens (s & t), 2738 & 2326 types
    Parameters will be written to: am_1_64_32_RNN_b0_g000_d0-pid48778.params
    %% Using RNN recurrent units
    **SHUFFLE
    [epoch=0 eta=0.1 clips=50 updates=50]  E = 5.77713 ppl=322.832 [completed in 192.254 ms]
    [epoch=0 eta=0.1 clips=50 updates=50]  E = 5.12047 ppl=167.415 [completed in 188.866 ms]
    [epoch=0 eta=0.1 clips=50 updates=50]  E = 5.36808 ppl=214.451 [completed in 153.08 ms]
    ...

    (GPU)
    [cnn] initializing CUDA
    Request for 1 GPU ...
    [cnn] Device Number: 0
    [cnn]   Device name: GeForce GTX TITAN X
    [cnn]   Memory Clock Rate (KHz): 3505000
    [cnn]   Memory Bus Width (bits): 384
    [cnn]   Peak Memory Bandwidth (GB/s): 336.48
    [cnn]   Memory Free (GB): 0.0185508/12.8847
    [cnn]
    [cnn] Device Number: 1
    [cnn]   Device name: GeForce GTX TITAN X
    [cnn]   Memory Clock Rate (KHz): 3505000
    [cnn]   Memory Bus Width (bits): 384
    [cnn]   Peak Memory Bandwidth (GB/s): 336.48
    [cnn]   Memory Free (GB): 6.31144/12.8847
    [cnn]
    [cnn] Device Number: 2
    [cnn]   Device name: GeForce GTX TITAN X
    [cnn]   Memory Clock Rate (KHz): 3505000
    [cnn]   Memory Bus Width (bits): 384
    [cnn]   Peak Memory Bandwidth (GB/s): 336.48
    [cnn]   Memory Free (GB): 0.0185508/12.8847
    [cnn] ...
    [cnn] Device(s) selected: 6
    [cnn] random seed: 2080175584
    [cnn] allocating memory: 512MB
    [cnn] memory allocation done.
    Reading training data from sample-data/train.de-en.unk.cap...
    5000 lines, 117998 & 105167 tokens (s & t), 2738 & 2326 types
    Reading dev data from sample-data/dev.de-en.unk.cap...
    100 lines, 1800 & 1840 tokens (s & t), 2738 & 2326 types
    Parameters will be written to: am_1_64_32_RNN_b0_g000_d0-pid14453.params
    %% Using RNN recurrent units
    **SHUFFLE
    [epoch=0 eta=0.01 clips=0 updates=50]  E = 6.12625 ppl=457.718 [completed in 724.351 ms]
    [epoch=0 eta=0.01 clips=0 updates=50]  E = 5.23731 ppl=188.163 [completed in 714.797 ms]
    [epoch=0 eta=0.01 clips=0 updates=50]  E = 5.37111 ppl=215.102 [completed in 796.774 ms]
    ...

Every so often the development performance is measured, and the best scoring model will be saved to disk.

If you want to build a large network, you will need to indicate the memory usage (*--cnn-mem FORWARD_MEM,BACKWARD_MEM,PARAMETERS_MEM*) for cnn backend, e.g.,

    ./build_cpu/src/attentional --cnn-mem 3000 -t sample-data/train.de-en.unk.cap -d sample-data/dev.de-en.unk.cap
  
    ./build_cpu/src/attentional --cnn-mem 1000,1000,2000 -t sample-data/train.de-en.unk.cap -d sample-data/dev.de-en.unk.cap

The binaries have command line help, and their usage is illustrated in the *scripts/* folder. This includes
decoding.

## Contacts

Trevor Cohn, Hoang Cong Duy Vu and Reza Haffari 

---
Updated October 2016
