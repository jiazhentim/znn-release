#!/bin/bash
#scl enable devtoolset-2 bash
module add boost
module add intel
icc \
    -std=c++11 \
    $1.cpp \
    -I../../.. \
    -I../../../src/include \
    -DNDEBUG -O3 \
    -DZNN_CUBE_POOL_LOCKFREE \
    -L/usr/people/vyf/work/xeon/fftw-3.3.4/.libs \
    -L/usr/local/lib -L/usr/lib -L/usr/lib64 \
    -DZNN_USE_FLOATS \
    -lfftw3f -mkl=sequential -lm -lpthread -lrt -o $1 -static-intel -ggdb
#    -DZNN_USE_MKL_NATIVE_FFT \
#    -DZNN_FFTW_PLANNING_MODE=FFTW_MEASURE \
