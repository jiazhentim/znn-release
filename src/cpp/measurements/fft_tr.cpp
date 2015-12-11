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
#include <fftw3.h>

#include <mkl_trans.h>

using namespace znn::v4;

typedef float real;
typedef std::vector<cube_p<real>> cubevec;

cubevec deepcpy(const cubevec& x) {
    auto r = x;
    for (int i = 0; i < r.size(); ++i) r[i] = get_copy(*r[i]);
    return r;
}

template<typename F, typename G>
static std::pair<double, double> bench(cubevec vec, F fwd, G bwd,
                                       int rounds) {
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

// complex x-z transpose, assumes complex dims x y z. Only tranposes
// xz plane at yaxis into yzx.
static void xztranspose(int x, int y, int z, fftwf_complex* a, int yaxis,
                        fftwf_complex* yzx) {
    // No idea why MKL_Complex version doesn't work
    auto src = reinterpret_cast</*MKL_Complex8*/double*>(a + z * yaxis);
    auto dst = reinterpret_cast</*MKL_Complex8**/double*>(yzx + z * x * yaxis);

    // float-based complex transpose
    mkl_domatcopy('R', // row major
                  'T', // transpose
                  x, // rows
                  z, // cols
                  1.0/*{ .real = 1.0, .imag = 1.0 }*/, // alpha
                  src, // xz-plane slice
                  y * z, // row stride
                  dst, // destination zx plane
                  x); // transposed row stride
}

enum {PAD_TO = 16};
enum {COMPLEX_PAD = PAD_TO / sizeof (fftwf_complex)};

template<typename T>
void padit(T& x) {
    x = (!!(x % COMPLEX_PAD) + x / COMPLEX_PAD) * COMPLEX_PAD;
}

struct fwd_transform {
    int x, y, z;
    fftwf_plan yzfft, xfft;
    fwd_transform(vec3i s)
        : x(s[0]), y(s[1]), z(s[2]) {
        auto in = get_cube<real>(s);

        auto tru_size = fft_complex_size(s);
        auto pad_size = tru_size;
        padit(pad_size[2]);
        auto out = get_cube<complex>(pad_size);

        auto iptr = reinterpret_cast<float*>(in->data());
        auto optr = reinterpret_cast<fftwf_complex*>(out->data());

        int rank = 2;
        int dims2[] = {y, z};
        int howmany = x;
        int idist = y * z;
        int odist = pad_size[1] * pad_size[2];
        int istride = 1;
        int ostride = 1;
        yzfft = fftwf_plan_many_dft_r2c(rank, dims2, howmany, iptr, NULL, istride,
                                        idist, optr, NULL, ostride, odist,
                                        ZNN_FFTW_PLANNING_MODE);

        auto tra_size = vec3i(tru_size[1], tru_size[2], tru_size[0]);
        padit(tra_size[2]);
        auto tra = get_cube<complex>(tra_size);

        auto tptr = reinterpret_cast<fftwf_complex*>(tra->data());
        rank = 1;
        int dims1[] = {x};
        howmany = tra_size[0] * tra_size[1];
        idist = 1;
        odist = tra_size[2];
        istride = pad_size[1] * pad_size[2];
        ostride = 1;
        xfft = fftwf_plan_many_dft(rank, dims1, howmany, optr, NULL, istride,
                                   idist, tptr, NULL, ostride, odist,
                                   FFTW_FORWARD, ZNN_FFTW_PLANNING_MODE);
        /*fftwf_print_plan(yzfft);
        printf("\n");
        fftwf_print_plan(xfft);
        printf("\n");*/
    }
    ~fwd_transform() {
        fftwf_destroy_plan(yzfft);
        fftwf_destroy_plan(xfft);
    }
    cube_p<complex> operator()(cube_p<real>&& cube_real) {
        auto real_data = cube_real->data();
        auto asize = fft_complex_size(*cube_real);
        auto csize = asize;
        padit(csize[2]);
        auto complex_cube = get_cube<complex>(csize);
        auto complex_data = reinterpret_cast<fftwf_complex*>(complex_cube->data());
        // TODO don't rely on x y z but array views
        fftwf_execute_dft_r2c(yzfft, real_data, complex_data);
        vec3i tsize = {asize[1], asize[2], asize[0]};
        padit(tsize[2]);
        auto transpose_cube = get_cube<complex>(tsize);
        auto transpose_data =
            reinterpret_cast<fftwf_complex*>(transpose_cube->data());
        fftwf_execute_dft(xfft, complex_data, transpose_data);
        return transpose_cube;
    }
};

// TODO correctness

/*struct bwd_transform {
    fftplanptr yzfft, zfft;
    int x, y, z;
    bwd_transform(int x, int y, int z) {
        x = x; y = y; z = z;
        yzfft = make2dplan(y, z, false);
        zfft = make1dplan(x, false);
    }
    cube_p<real> operator()(cube_p<complex>&& transpose_cube) {
        auto transpose_data = transpose>data();
        auto complex_cube = get_cube<complex>(fft_complex_size(*cube_real));
        auto complex_data = reinterpret_cast<fft_complex*>(complex_cube->data());
        // TODO don't rely on x y z but array views
        // TODO try out-of-place FFT to transpose
        for (int i = 0; i < x; ++i) {
            fftwf_execute_dft_r2c(yzfft.get(), real_data + i * y * z,
                                  complex_data + i * y * z);
        }
        auto csize = size(*complex_cube);
        vec3i tsize = {csize[1], csize[2], csize[0]};
        auto transpose_cube = get_cube<complex>(tsize);
        auto transpose_data = reinterpret_cast<fft_complex*>(transpose_cube->
                                                             data());
        for (int i = 0; i < tsize[0]; ++i) {
            xztranspose(csize[0], csize[1], csize[2], complex_data, i,
                        transpose_data);
            for (int j = 0; j < tsize[1]; ++j) {
                auto ptr = transpose_data
                    + i * tsize[1] * tsize[2] + j * tsize[2];
                fftwf_execute_dft(zfft.get(), ptr, ptr);
            }
        }

        return transpose_cube;
    }
    };*/

int main(int argc, char** argv)
{
    if (argc != 7) {
        std::cerr << "Usage: fft_tr x y z rounds nthreads type\n";
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
    //    auto m = bench(deepcpy(vec), fwd, bwd, rounds);
    //std::cout << "LONE-FFTW Average Forward/Backward: " << m.first * 1000
    //<< " +/- " << m.second * 1000 << " ms\n";

    fwd_transform fwd2(vec3i(x, y, z));
    auto v1 = deepcpy(vec);
    double t1 = 0;
    fwd(std::move(vec[0]));
    zi::wall_timer wt; wt.reset();
    for (int i = 0; i < v1.size(); ++i) {
        if (argv[6][0] == 'f')
            fwd(std::move(v1[i]));
        else if (argv[6][0] == 't')
            fwd2(std::move(v1[i]));
        else {
            std::cerr << "UNKNOWN TYPE " << argv[6] << std::endl;
            return 1;
        }
        t1 += wt.lap<double>();
    }

    const char* name = "X";
    switch(argv[6][0]) {
    case 'f': name = "3DFFTW-ORMKL-only"; break;
    case 't': name = "2DFFTW+1DFFTW-out-of-place"; break;
    }

    printf("%30s %6f\n", name, t1);

    /*
    auto s1 = get_copy(*vec[0]);
    auto s2 = get_copy(*s1);
    auto r1 = fwd2(std::move(s1));
    auto r2 = fwd(std::move(s2));
    for (int i = 0; i < size(*r1)[0]; ++i) {
        for (int j = 0; j < size(*r1)[0]; ++j) {
            for (int k = 0; k < size(*r1)[0]; ++k) {
                auto c1 = (*r1)[i][j][k];
                auto c2 = (*r2)[j][k][i]; // intentional
                if (c1.real() != c2.real()) {
                    printf("Re %f != %f @ %d %d %d\n", c1.real(), c2.real(),
                          i, j, k);
                }
                if (c1.imag() != c2.imag()) {
                    printf("Im %f != %f @ %d %d %d\n", c1.imag(), c2.imag(),
                           i, j, k);
                           }
            }
        }
        }*/

    //auto bwd2 = make_bwd(x, iy, z);
    //m = bench(vec, std::move(fwd2), std::move(bwd2), rounds);
    //std::cout << "TRANSPOSE Average Forward/Backward: " << m.first * 1000
    //<< " +/- " << m.second * 1000 << " ms\n";

    // complex dims 2x3x4 correspond to real dims 2x3x6
    /*    fft_complex im234[] = {
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
                    printf("%f(%f) ", a[ctr], a[ctr+1]);
                    ctr += 2;
                }
                printf("\n");
            }
            printf("\n");
        }
    };

    pr((float*) im234, 2, 3, 4);
    fft_complex im342[sizeof(im234) / sizeof(*im234)] = {0};
    xztranspose(2, 3, 4, im234, 0, im342);
    printf("transposed only 0\n");
    pr((float*) im234, 4, 3, 2);
    printf("transposed\n");
    for (int i = 1; i < 3; ++i)
        xztranspose(2, 3, 4, im234, i, im342);
    pr((float*) im342, 3, 4, 2); */
}
