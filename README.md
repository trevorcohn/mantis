# mantis
Deep learning model of machine translation using attentional and structural biases

## Introduction

### Features

### Building

cd $HOME/mantis/src

PATH_TO_CNN=$HOME/cnn/
PATH_TO_EIGEN=$HOME/cnn/eigen/

CPU: g++ -g -o attentional attentional.cc -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN -std=c++11 -L/usr/lib -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -L/$PATH_TO_CNN/build/cnn -lcnn
 
GPU: g++ -g -o attentional-gpu attentional.cc -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN -I/usr/local/cuda-7.0/include -std=c++11 -L/usr/lib -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -L/$PATH_TO_CNN/build-cuda/cnn -lcnncuda -DHAVE_CUDA -L/usr/local/cuda-7.0/targets/x86_64-linux/lib -lcudart -lcublas

## Contacts

1) Trevor Cohn (t.cohn@unimelb.edu.au)

2) Hoang Cong Duy Vu (vhoang2@student.unimelb.edu.au or duyvuleo@gmail.com)

3) Reza Haffari (gholamreza.haffari@monash.edu)

---
Updated on April 2016

