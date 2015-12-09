//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include <cassert>
#include <iostream>
#include <future>
#include <functional>
#include <thread>
#include "network/parallel/network.hpp"
#include "network/trivial/trivial_fft_network.hpp"
#include "network/trivial/trivial_network.hpp"
#include "network/helpers.hpp"
#include <zi/zargs/zargs.hpp>
#include <fstream>

using namespace znn::v4;

typedef float real;
typedef std::vector<cube_p<real>> cubevec;

template<typename F, typename G>
static std::pair<double, double> bench(cubevec vec, F fwd, G bwd, int rounds) {
    // Warmup
    auto t = fwd(std::move(vec[0]));
    vec[0] = bwd(std::move(t));
    std::vector<std::future<std::vector<double>>> futs;

    for (int i = 0; i < vec.size(); ++i) {
        futs.push_back(std::async(std::launch::async, [&](int ii) {
                    std::vector<double> times(rounds);
                    zi::wall_timer wt;
                    // TODO: try saved-intermediate ffts as well.

                    wt.reset();

                    for (size_t j = 0; j < rounds; ++j) {
                        auto t = fwd(std::move(vec[ii]));
                        vec[ii] = bwd(std::move(t));
                        times[j] = wt.lap<double>();
                    }

                    return times;
                }, i));
    };

    std::vector<double> cat;
    for (auto& fut : futs) {
        auto res = fut.get();
        cat.insert(cat.end(), res.begin(), res.end());
    }

    return measured(cat);
}

struct del_plan {
    void operator()(void* plan) {
        FFT_DESTROY_PLAN((fft_plan) plan);
    }
};

typedef std::unique_ptr<typename std::remove_pointer<fft_plan>::type,
                        del_plan> fftplanptr;

static fftplanptr make2dplan(int y, int z, bool r2c) {
    const int inner_real_dim = y, outer_real_dim = z,
        inner_complex_dim = y, outer_complex_dim = z / 2 + 1;
    auto in = std::unique_ptr<real[]>(new real[inner_real_dim * outer_real_dim]);
    auto out = std::unique_ptr<complex[]>(new complex[inner_complex_dim *
                                                      outer_complex_dim]);
    auto iptr = in.get();
    auto optr = reinterpret_cast<fft_complex*>(out.get());
    auto plan = r2c ?
        fftwf_plan_dft_r2c_2d(inner_real_dim, outer_real_dim,
                              iptr, optr, ZNN_FFTW_PLANNING_MODE)
        : fftwf_plan_dft_c2r_2d(inner_complex_dim, outer_complex_dim,
                                optr, iptr, ZNN_FFTW_PLANNING_MODE);
    return fftplanptr(plan, del_plan());
}

static fftplanptr make1dplan(int x, bool r2c) {
    auto in = std::unique_ptr<complex[]>(new complex[x]);
    auto out = std::unique_ptr<complex[]>(new complex[x]);
    auto iptr = reinterpret_cast<fft_complex*>(in.get());
    auto optr = reinterpret_cast<fft_complex*>(out.get());
    auto plan = fftwf_plan_dft_1d(x, iptr, optr,
                                  r2c ? FFTW_FORWARD : FFTW_BACKWARD,
                                  ZNN_FFTW_PLANNING_MODE);

    return fftplanptr(plan, del_plan());
}

// complex x-z transpose, assumes real dims x y z
static fftplanptr maketransposeplan(int x, int y, int z, float* ptr) {
    z = (z / 2 + 1) * 2;
    // TODO out-of-place transpose to real maybe?

    const int dims = 3;
    fftw_iodim howmany_dims[dims] = {
        { .n = x, .is = y * z, .os = 2 },
        { .n = y, .is = z, .os = x * 2 },
        { .n = z, .is = 1, .os = y * x * 2 }};

    auto plan = fftwf_plan_guru_r2r(/*rank*/0, /*dims*/NULL,
                                    dims, howmany_dims,
                                    ptr, ptr, /*kind*/NULL,
                                    ZNN_FFTW_PLANNING_MODE);
    //delete[] tmp;
    //printf("%d\n", (uintptr_t)(void*)plan);
    //return fftplanptr(plan, del_plan());
    return nullptr;
}

struct fwd_transform {
    fftplanptr yzfft, zfft;
    int x, y, z;
    fwd_transform(int x, int y, int z) {
        x = x; y = y; z = z;
        yzfft = make2dplan(y, z, true);
        zfft = make1dplan(x, true);
    }
    cube_p<complex> operator()(cube_p<real>&& cube_real) {
        auto real_data = cube_real->data();
        auto complex_cube = get_cube<complex>(fft_complex_size(*cube_real));
        auto complex_data = reinterpret_cast<fft_complex*>(complex_cube->data());
        for (int i = 0; i < x; ++i) {
            fftwf_execute_dft_r2c(yzfft.get(), real_data + i * y * z,
                                  complex_data + i * y * z);
        }

        // transpose
        // for z, y, fft(x)
        // transpose

        // TODO: save on the final transpose (dot product should still be valid)
        // then inverse would not have an initial transpose, and do 1d first)
        //....
        // transpose urls:
        //http://stackoverflow.com/questions/6021740/how-do-i-use-fftw-plan-many-dft-on-a-transposed-array-of-data
        //http://agentzlerich.blogspot.com/2010/01/using-fftw-for-in-place-matrix.html
        return nullptr;
    }
};

static std::function<cube_p<real>(cube_p<complex>&&)>
make_bwd(int x, int y, int z) {
}

int main(int argc, char** argv)
{
    if (argc != 6) {
        std::cerr << "Usage: fft_tr x y z rounds nthreads\n";
        return 1;
    }
    int x = atoi(argv[1]);
    int y = atoi(argv[2]);
    int z = atoi(argv[3]);
    size_t rounds = atoi(argv[4]);
    size_t max_threads = atoi(argv[5]);

    uniform_init init(1);
    cubevec vec(max_threads);
    for (int i = 0; i < max_threads; ++i) {
        vec[i] = get_cube<real>({x,y,z});
        init.initialize(vec[i]);
    }

    fftw::transformer fft({x,y,z});
    auto fwd = [&](cube_p<real>&& x) {
        return fft.forward(std::move(x));
    };
    auto bwd = [&](cube_p<complex>&& x) {
        return fft.backward(std::move(x));
    };
    auto m = bench(vec, fwd, bwd, rounds);
    std::cout << "LONE-FFTW Average Forward/Backward: " << m.first * 1000
              << " +/- " << m.second * 1000 << " ms\n";

    //auto fwd2 = make_fwd(x, y, z);
    //auto bwd2 = make_bwd(x, y, z);
    //m = bench(vec, std::move(fwd2), std::move(bwd2), rounds);
    //std::cout << "TRANSPOSE Average Forward/Backward: " << m.first * 1000
    //<< " +/- " << m.second * 1000 << " ms\n";

    // complex dims 2x3x4 correspond to real dims 2x3x6
    fft_complex im234[] = {
        {0, 0}, {0, 1}, {0, 2}, {0, 3},
        {1, 0}, {1, 1}, {1, 2}, {1, 3},
        {2, 0}, {2, 1}, {2, 2}, {2, 3},

        {0, -10}, {0, -1}, {0, -2}, {0, -3},
        {1, -10}, {1, -1}, {1, -2}, {1, -3},
        {2, -10}, {2, -1}, {2, -2}, {2, -3}};

    auto pr = [](float* a, int x, int y, int z) {
        int ctr = 0;

        printf("%dx%dx%d array\n", x, y, z);
        for (int k = 0; k < x; ++k) {
            for (int i = 0; i < y; ++i) {
                for (int j = 0; j < z; ++j) {
                    printf("%d(%d) ", (int) a[ctr], (int) a[ctr+1]);
                    ctr += 2;
                }
                printf("\n");
            }
            printf("\n");
        }
    };

    pr((float*) im234, 2, 3, 4);
    auto trplan = maketransposeplan(2, 3, 6, (float*) im234);
    //fftwf_execute_r2r(trplan.get(), (float*) im234, (float*) im234);
    printf("transposed\n");
    pr((float*) im234, 4, 3, 2);
}
