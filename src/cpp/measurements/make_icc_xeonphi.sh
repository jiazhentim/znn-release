#!/bin/bash
module add gcc/4.9
module add intel/compilervars/15
icc -mmic -std=c++11 $1.cpp -I../../.. -I/usr/include -I../../../src/include -I../../../boost_1_58_0 -DNDEBUG -O3 -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS -L/usr/people/vyf/work/phi/fftw-3.3.4/.libs -lfftw3f -lm -lpthread -lrt -o $1 -static-intel -DZNN_NO_THREAD_LOCAL -DZNN_XEON_PHI -D_GLIBCXX_USE_SCHED_YIELD -ggdb -DZNN_DONT_CACHE_FFTS
