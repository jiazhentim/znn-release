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

#include "../../../cube/cube.hpp"
#include "../../../types.hpp"
#include "../../../meta.hpp"

namespace znn { namespace v4 { namespace detail {

template< typename T >
inline void convolve_sparse_add( cube<T> const & a,
                                 cube<T> const & b,
                                 vec3i const & s,
                                 cube<T> & r) noexcept
{
  size_t ax = size(a)[0];
  size_t ay = size(a)[1];
  size_t az = size(a)[2];

  size_t bx = size(b)[0];
  size_t by = size(b)[1];
  size_t bz = size(b)[2];

    size_t rbx = (bx-1) * s[0] + 1;
    size_t rby = (by-1) * s[1] + 1;
    size_t rbz = (bz-1) * s[2] + 1;

    size_t rx = ax - rbx + 1;
    size_t ry = ay - rby + 1;
    size_t rz = az - rbz + 1;

    ZI_ASSERT(size(r)[0]==rx);
    ZI_ASSERT(size(r)[1]==ry);
    ZI_ASSERT(size(r)[2]==rz);

    for ( size_t x = 0; x < rx; ++x )
        for ( size_t y = 0; y < ry; ++y )
            for ( size_t z = 0; z < rz; ++z )
                for ( size_t dx = x, wx = bx-1; dx < rbx + x; dx += s[0], --wx )
                    for ( size_t dy = y, wy = by-1; dy < rby + y; dy += s[1], --wy )
                        for ( size_t dz = z, wz = bz-1; dz < rbz + z; dz += s[2], --wz )
                            r[x][y][z] += a[dx][dy][dz] * b[wx][wy][wz];
}

template< typename T >
inline void convolve_sparse( cube<T> const & a,
                             cube<T> const & b,
                             vec3i const & s,
                             cube<T> & r) noexcept
{
  size_t ax = size(a)[0];
  size_t ay = size(a)[1];
  size_t az = size(a)[2];

  size_t bx = size(b)[0];
  size_t by = size(b)[1];
  size_t bz = size(b)[2];

    size_t rbx = (bx-1) * s[0] + 1;
    size_t rby = (by-1) * s[1] + 1;
    size_t rbz = (bz-1) * s[2] + 1;

    size_t rx = ax - rbx + 1;
    size_t ry = ay - rby + 1;
    size_t rz = az - rbz + 1;

    ZI_ASSERT(size(r)[0]==rx);
    ZI_ASSERT(size(r)[1]==ry);
    ZI_ASSERT(size(r)[2]==rz);

    for ( size_t x = 0; x < rx; ++x )
        for ( size_t y = 0; y < ry; ++y )
            for ( size_t z = 0; z < rz; ++z )
            {
                r[x][y][z] = 0;
                for ( size_t dx = x, wx = bx-1; dx < rbx + x; dx += s[0], --wx )
                    for ( size_t dy = y, wy = by-1; dy < rby + y; dy += s[1], --wy )
                        for ( size_t dz = z, wz = bz-1; dz < rbz + z; dz += s[2], --wz )
                            r[x][y][z] += a[dx][dy][dz] * b[wx][wy][wz];
            }
}


template< typename T >
inline void convolve_sparse_flipped( cube<T> const & a,
                                     cube<T> const & b,
                                     vec3i const & s,
                                     cube<T> & r ) noexcept
{
  size_t ax = size(a)[0];
  size_t ay = size(a)[1];
  size_t az = size(a)[2];

  size_t bx = size(b)[0];
  size_t by = size(b)[1];
  size_t bz = size(b)[2];

    size_t rx = (ax - bx) / s[0] + 1;
    size_t ry = (ay - by) / s[1] + 1;
    size_t rz = (az - bz) / s[2] + 1;

    ZI_ASSERT(size(r)[0]==rx);
    ZI_ASSERT(size(r)[1]==ry);
    ZI_ASSERT(size(r)[2]==rz);

    for ( size_t qx = 0, x = 0; qx < rx; ++qx, x += s[0] )
        for ( size_t qy = 0, y = 0; qy < ry; ++qy, y += s[1] )
            for ( size_t qz = 0, z = 0; qz < rz; ++qz, z += s[2] )
            {
                r[qx][qy][qz] = 0;
                for ( size_t dx = 0; dx < bx; ++dx )
                    for ( size_t dy = 0; dy < by; ++dy )
                        for ( size_t dz = 0; dz < bz; ++dz )
                            r[qx][qy][qz] +=
                                a[ax-1-x-dx][ay-1-y-dy][az-1-z-dz] *
                                b[bx-1-dx][by-1-dy][bz-1-dz];
            }
}

template< typename T >
inline void convolve_sparse_inverse_add( cube<T> const & a,
                                         cube<T> const & b,
                                         vec3i const & s,
                                         cube<T> & r ) noexcept
{
  size_t ax = size(a)[0];
  size_t ay = size(a)[1];
  size_t az = size(a)[2];

  size_t bx = size(b)[0];
  size_t by = size(b)[1];
  size_t bz = size(b)[2];

#   ifndef NDEBUG
    size_t rbx = (bx-1) * s[0] + 1;
    size_t rby = (by-1) * s[1] + 1;
    size_t rbz = (bz-1) * s[2] + 1;

    size_t rx = ax + rbx - 1;
    size_t ry = ay + rby - 1;
    size_t rz = az + rbz - 1;

    ZI_ASSERT(size(r)[0]==rx);
    ZI_ASSERT(size(r)[1]==ry);
    ZI_ASSERT(size(r)[2]==rz);
#   endif

    for ( size_t wx = 0; wx < bx; ++wx )
        for ( size_t wy = 0; wy < by; ++wy )
            for ( size_t wz = 0; wz < bz; ++wz )
            {
                size_t fx = bx - 1 - wx;
                size_t fy = by - 1 - wy;
                size_t fz = bz - 1 - wz;

                size_t ox = fx * s[0];
                size_t oy = fy * s[1];
                size_t oz = fz * s[2];

                for ( size_t x = 0; x < ax; ++x )
                    for ( size_t y = 0; y < ay; ++y )
                        for ( size_t z = 0; z < az; ++z )
                            r[x+ox][y+oy][z+oz] += a[x][y][z] * b[wx][wy][wz];
            }
}


template< typename T >
inline void convolve_sparse_inverse( cube<T> const & a,
                                     cube<T> const & b,
                                     vec3i const & s,
                                     cube<T> & r ) noexcept
{
    fill(r,0);
    ::znn::v4::detail::convolve_sparse_inverse_add(a,b,s,r);
}


}}} // namespace znn::v4::detail
