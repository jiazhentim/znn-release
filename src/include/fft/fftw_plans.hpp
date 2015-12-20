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
#pragma once

#ifdef ZNN_USE_MKL_FFT
#  include <fftw/fftw3.h>
#else
#  include <fftw3.h>
#endif

#include "../types.hpp"
#include "../cube/cube.hpp"

#include <zi/utility/singleton.hpp>
#include <zi/time/time.hpp>

#include <map>
#include <iostream>
#include <unordered_map>
#include <type_traits>
#include <mutex>

#ifndef ZNN_FFTW_PLANNING_MODE
#  define ZNN_FFTW_PLANNING_MODE (FFTW_ESTIMATE)
#endif

namespace znn { namespace v4 {

#ifdef ZNN_USE_FLOATS

#define FFT_DESTROY_PLAN fftwf_destroy_plan
#define FFT_CLEANUP      fftwf_cleanup
#define FFT_PLAN_C2R     fftwf_plan_dft_c2r_3d
#define FFT_PLAN_R2C     fftwf_plan_dft_r2c_3d
typedef fftwf_plan    fft_plan   ;
typedef fftwf_complex fft_complex;

#else

#define FFT_DESTROY_PLAN fftw_destroy_plan
#define FFT_CLEANUP      fftw_cleanup
#define FFT_PLAN_C2R     fftw_plan_dft_c2r_3d
#define FFT_PLAN_R2C     fftw_plan_dft_r2c_3d
typedef fftw_plan    fft_plan   ;
typedef fftw_complex fft_complex;

#endif

inline vec3i fft_complex_size(const vec3i& s)
{
    auto r = s;
    r[2] /= 2;
    r[2] += 1;
    return r;
}

template< typename T >
inline vec3i fft_complex_size(const cube<T>& c)
{
    return fft_complex_size(size(c));
}

inline vec3i transpose_size(const vec3i& t) {
  return vec3i(t[1], t[2], t[0]);
}

template<typename T>
inline vec3i transpose_size(const cube<T>& t) {
  return transpose_size(size(t));
}

inline vec3i transformed_size(const vec3i& t) {
  return transpose_size(fft_complex_size(t));
}

template<typename T>
inline vec3i inverse_transformed_size(const cube<T>& t) {
  return inverse_transformed_size(t);
}

class fft_plan_fwd {
  fftwf_plan yzfft, xfft;

public:
  fft_plan_fwd(vec3i s) {
    auto in = get_cube<real>(s);
    auto yz = get_cube<complex>(fft_complex_size(s));

    auto inptr = in->data();
    auto yzptr = reinterpret_cast<fftwf_complex*>(yz->data());

    auto in_logical_size = s;
    auto in_physical_size = in->shape();
    auto yz_logical_size = size(*yz);
    auto yz_physical_size = yz->shape();

    int rank = 2;
    int dims2[] = {in_logical_size[1], in_logical_size[2]};
    int howmany = in_logical_size[0];
    int idist = in_physical_size[1] * in_physical_size[2];
    int odist = yz_physical_size[1] * yz_physical_size[2];
    int istride = 1;
    int ostride = 1;
    yzfft = fftwf_plan_many_dft_r2c(rank, dims2, howmany, inptr, NULL, istride,
                                    idist, yzptr, NULL, ostride, odist,
                                    ZNN_FFTW_PLANNING_MODE);

    auto tr = get_cube<complex>(transpose_size(yz_logical_size));
    auto tr_logical_size = size(*tr);
    auto tr_physical_size = tr->shape();
    auto trptr = reinterpret_cast<fftwf_complex*>(tr->data());

    rank = 1;
    int dims1[] = {tr_logical_size[0]};
    howmany = tr_logical_size[0] * tr_logical_size[1];
    idist = 1;
    odist = tr_physical_size[2];
    istride = yz_physical_size[1] * yz_physical_size[2];
    ostride = 1;
    xfft = fftwf_plan_many_dft(rank, dims1, howmany, yzptr, NULL, istride,
                               idist, trptr, NULL, ostride, odist,
                               FFTW_FORWARD, ZNN_FFTW_PLANNING_MODE);
  }

  cube_p<complex> operator()(cube_p<real>&& in) {
    auto yz = get_cube<complex>(fft_complex_size(*in));
    auto inptr = in->data();
    auto yzptr = reinterpret_cast<fftwf_complex*>(yz->data());

    fftwf_execute_dft_r2c(yzfft, inptr, yzptr);

    inptr = nullptr;
    auto tr = get_cube<complex>(transpose_size(*yz));
    auto trptr = reinterpret_cast<fftwf_complex*>(tr->data());

    fftwf_execute_dft(xfft, yzptr, trptr);

    return tr;
  }

  ~fft_plan_fwd() {
    fftwf_destroy_plan(yzfft);
    fftwf_destroy_plan(xfft);
  }
};

class fft_plan_bwd {
  fftwf_plan yzfft, xfft;
  vec3i sz;

public:
  fft_plan_bwd(const vec3i& s) {
    sz = s;

    auto yz = get_cube<complex>(fft_complex_size(s));
    auto tr = get_cube<complex>(transpose_size(size(*yz)));

    auto yzptr = reinterpret_cast<fftwf_complex*>(yz->data());
    auto trptr = reinterpret_cast<fftwf_complex*>(tr->data());

    auto yz_logical_size = size(*yz);
    auto yz_physical_size = yz->shape();
    auto tr_logical_size = size(*tr);
    auto tr_physical_size = tr->shape();

    int rank = 1;
    int dims1[] = {tr_logical_size[0]};
    int howmany = tr_logical_size[0] * tr_logical_size[1];
    int idist = tr_physical_size[2];
    int odist = 1;
    int istride = 1;
    int ostride = yz_physical_size[1] * yz_physical_size[2];
    xfft = fftwf_plan_many_dft(rank, dims1, howmany, trptr, NULL, istride,
                               idist, yzptr, NULL, ostride, odist,
                               FFTW_BACKWARD, ZNN_FFTW_PLANNING_MODE);

    auto out = get_cube<real>(s);
    auto outptr = out->data();
    auto out_logical_size = s;
    auto out_physical_size = out->shape();

    rank = 2;
    int dims2[] = {out_logical_size[1], out_logical_size[2]};
    howmany = out_logical_size[0];
    idist = yz_physical_size[1] * yz_physical_size[2];
    odist = out_physical_size[1] * out_physical_size[2];
    istride = 1;
    ostride = 1;
    yzfft = fftwf_plan_many_dft_c2r(rank, dims2, howmany, yzptr, NULL, istride,
                                    idist, outptr, NULL, ostride, odist,
                                    ZNN_FFTW_PLANNING_MODE);

  }

  cube_p<real> operator()(cube_p<complex>&& tr) {
    auto yz = get_cube<complex>(fft_complex_size(sz));
    auto trptr = reinterpret_cast<fftwf_complex*>(tr->data());
    auto yzptr = reinterpret_cast<fftwf_complex*>(yz->data());

    fftwf_execute_dft(xfft, trptr, yzptr);

    trptr = nullptr;
    auto out = get_cube<real>(sz);
    auto outptr = out->data();

    fftwf_execute_dft_c2r(yzfft, yzptr, outptr);

    return out;
  }

  ~fft_plan_bwd() {
    fftwf_destroy_plan(yzfft);
    fftwf_destroy_plan(xfft);
  }
};

class fft_plans_impl
{
private:
    std::mutex                                           m_          ;
    real                                                 time_       ;
    std::unordered_map<vec3i, std::unique_ptr<fft_plan_fwd>, vec_hash<vec3i>>
      fwd_;
    std::unordered_map<vec3i, std::unique_ptr<fft_plan_bwd>, vec_hash<vec3i>>
      bwd_;

public:
    ~fft_plans_impl()
    {
        fwd_.clear();
        bwd_.clear();
        FFT_CLEANUP();
    }

    fft_plans_impl(): m_(), fwd_(), bwd_(), time_(0)
    {
    }

    fft_plan_fwd* get_forward( const vec3i& s )
    {
        guard g(m_);

        auto& ret = fwd_[s];

        if (!ret) {
          zi::wall_timer wt; wt.reset();
          ret.reset(new fft_plan_fwd(s));
          time_ += wt.elapsed<real>();
        }

        return ret.get();
    }

    fft_plan_bwd* get_backward( const vec3i& s )
    {
        guard g(m_);

        auto& ret = bwd_[s];

        if (!ret) {
          zi::wall_timer wt; wt.reset();
          ret.reset(new fft_plan_bwd(s));
          time_ += wt.elapsed<real>();
        }

        return ret.get();
    }

}; // class fft_plans_impl

namespace {
fft_plans_impl& fft_plans =
    zi::singleton<fft_plans_impl>::instance();
} // anonymous namespace


}} // namespace znn::v4

#undef FFT_DESTROY_PLAN
#undef FFT_CLEANUP
#undef FFT_PLAN_R2C
#undef FFT_PLAN_C2R
