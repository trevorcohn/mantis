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
 * [Eigen](https://bitbucket.org/eigen/eigen), using the development version (not release)
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

substiting in a different path to eigen if you have placed in a different directory.

This will build the two binaries
    
    build_cpu/src/attentional
    build_cpu/src/biattentional


#### GPU build

Building on the GPU uses the Nvida CUDA library, currently tested against version 7.5.
The process is as follows

    mkdir build_gpu
    cd build_gpu
    cmake .. -DBACKED=cuda -DEIGEN3_INCLUDE_DIR=eigen -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda 
    make -j 2

substituting in your Eigen and CUDA folder, as appropriate.

This will result in the two binaries

    build_gpu/src/attentional
    build_gpu/src/biattentional

#### Using the model

The model can be run as follows

    ./build_cpu/src/attentional -t sample-data/train.de-en.unk.cap -d sample-data/dev.de-en.unk.cap 

which will train a small model on a tiny training set, i.e.,

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

Every so often the development performance is measured, and the best scoring model will be saved to disk.

The binaries have command line help, and their usage is illustrated in the *scripts/* folder. This includes
decoding.

## Contacts

Trevor Cohn, Hoang Cong Duy Vu and Reza Haffari 

---
Updated October 2016
