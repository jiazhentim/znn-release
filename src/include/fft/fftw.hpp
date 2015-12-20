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

#ifdef ZNN_USE_MKL_NATIVE_FFT
#  include "fftmkl.hpp"
#else

#include "fftw_plans.hpp"

#include <zi/time.hpp>

#ifdef ZNN_MEASURE_FFT_RUNTIME
#  define ZNN_MEASURE_FFT_START() zi::wall_timer wt
#  define ZNN_MEASURE_FFT_END() fftw_stats.add(wt.elapsed<double>())
#else
#  define ZNN_MEASURE_FFT_START() static_cast<void>(0)
#  define ZNN_MEASURE_FFT_END() static_cast<void>(0)
#endif

#ifdef ZNN_USE_FLOATS
#  define FFT_EXECUTE_DFT_R2C
#  define FFT_EXECUTE_DFT_C2R fftwf_execute_dft_c2r
#else
#  define FFT_EXECUTE_DFT_R2C fftw_execute_dft_r2c
#  define FFT_EXECUTE_DFT_C2R fftw_execute_dft_c2r
#endif

namespace znn { namespace v4 {

class fft_stats_impl
{
private:
    double              total_time_;
    std::size_t         total_     ;
    mutable std::mutex  m_         ;

public:
    fft_stats_impl()
        : total_time_(0)
        , total_(0)
        , m_()
    { }

    double get_total_time() const
    {
        guard g(m_);
        return total_time_;
    }

    void reset_total_time()
    {
        guard g(m_);
        total_time_ = 0;
    }

    size_t get_total() const
    {
        guard g(m_);
        return total_;
    }

    void add(double time)
    {
        guard g(m_);
        ++total_;
        total_time_ += time;
    }
};

namespace {
fft_stats_impl& fft_stats = zi::singleton<fft_stats_impl>::instance();
} // anonymous namespace

class fftw
{
public:
    class transformer
    {
    private:
        vec3i    sz           ;
        fft_plan_fwd* forward_plan ;
        fft_plan_bwd* backward_plan;

    public:
        transformer(const vec3i& s)
            : sz(s)
            , forward_plan(fft_plans.get_forward(s))
            , backward_plan(fft_plans.get_backward(s))
        {}

      /*
        void forward( cube<real>& in,
                      cube<complex>& out )
        {
            ZI_ASSERT(size(out)==transformed_size(in));
            ZI_ASSERT(size(in)==sz);

            ZNN_MEASURE_FFT_START();
            FFT_EXECUTE_DFT_R2C(forward_plan,
                                 reinterpret_cast<real*>(in.data()),
                                 reinterpret_cast<fft_complex*>(out.data()));
            ZNN_MEASURE_FFT_END();
        }

        void backward( cube<complex>& in,
                       cube<real>& out )
        {
            ZI_ASSERT(size(in)==transformed_size(out));
            ZI_ASSERT(size(out)==sz);

            ZNN_MEASURE_FFT_START();
            FFT_EXECUTE_DFT_C2R(backward_plan,
                                 reinterpret_cast<fft_complex*>(in.data()),
                                 reinterpret_cast<real*>(out.data()));
            ZNN_MEASURE_FFT_END();
            }*/  // TODO add various size asserts

        cube_p<complex> forward( cube_p<real>&& in )
        {
          ZNN_MEASURE_FFT_START();
          auto ret = (*forward_plan)(std::forward<cube_p<real>>(in));
          ZNN_MEASURE_FFT_END();
          return ret;
        }

        cube_p<complex> forward_pad( const ccube_p<real>& in )
        {
            cube_p<real> pin = pad_zeros(*in, sz);
            ZNN_MEASURE_FFT_START();
            auto ret = (*forward_plan)(std::forward<cube_p<real>>(pin));
            ZNN_MEASURE_FFT_END();
            return ret;
        }

        cube_p<real> backward( cube_p<complex>&& in )
        {
          ZNN_MEASURE_FFT_START();
          auto ret = (*backward_plan)(std::forward<cube_p<complex>>(in));
          ZNN_MEASURE_FFT_END();
          return ret;
        }
    };


public:
  /*
    static void forward( cube<real>& in,
                         cube<complex>& out )
    {
      ZI_ASSERT(size(transformed_size(out)) == size(in));

      auto plan = fft_plans.get_forward(
                                        vec3i(size(in)[0],size(in)[1],size(in)[2]));

        ZNN_MEASURE_FFT_START();
        FFT_EXECUTE_DFT_R2C(plan,
                             reinterpret_cast<real*>(in.data()),
                             reinterpret_cast<fft_complex*>(out.data()));
        ZNN_MEASURE_FFT_END();
    }

    static void backward( cube<complex>& in,
                          cube<real>& out )
    {
      ZI_ASSERT(size(transformed_size(out)) == size(in));

      auto plan = fft_plans.get_backward(
                                               vec3i(size(out)[0],size(out)[1],size(out)[2]));

        ZNN_MEASURE_FFT_START();
        FFT_EXECUTE_DFT_C2R(plan,
                             reinterpret_cast<fft_complex*>(in.data()),
                             reinterpret_cast<real*>(out.data()));
        ZNN_MEASURE_FFT_END();
    }*/

        static cube_p<complex> forward( cube_p<real>&& in )
        {
          transformer t(size(*in));
          return t.forward(std::forward<cube_p<real>>(in));
        }

  static cube_p<complex> forward_pad( const ccube_p<real>& in, const vec3i& s)
        {
          transformer t(s);
          return t.forward_pad(in);
        }

        static cube_p<real> backward( cube_p<complex>&& in, const vec3i& s)
        {
          transformer t(s);
          return t.backward(std::forward<cube_p<complex>>(in));
        }
}; // class fftw

}} // namespace znn::v4

#undef ZNN_MEASURE_FFT_START
#undef ZNN_MEASURE_FFT_END

#undef FFT_EXECUTE_DFT_R2C
#undef FFT_EXECUTE_DFT_C2R


#endif
