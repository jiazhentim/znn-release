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

#include <zi/utility/singleton.hpp>
#include <boost/lockfree/stack.hpp>
#include <boost/lockfree/queue.hpp>
#include <array>

#include "../../types.hpp"
#include "../../lockfree_allocator.hpp"

#ifdef ZNN_XEON_PHI
#  include <mkl.h>
#endif

namespace znn { namespace v4 {

#ifdef ZNN_XEON_PHI
#  define __ZNN_ALIGN 0x3F // 64 byte alignment
#else
#  define __ZNN_ALIGN 0xF // 16 byte alignment
#endif


inline void* znn_malloc(size_t s)
{
    void* r = malloc(s);
    if ( !r ) throw std::bad_alloc();
    return r;
}

inline void znn_free(void* ptr)
{
    free(ptr);
}

template <typename T> struct cube: boost::multi_array_ref<T,3>
{
private:
    using base_type =  boost::multi_array_ref<T,3>;

    vec3i s;

public:
    explicit cube(const vec3i& p, const vec3i& s, T* data)
        : boost::multi_array_ref<T,3>(data,extents[p[0]][p[1]][p[2]])
    {
        this->s = s;
    }

    ~cube()
    {
        znn_free(this);
    }

    cube& operator=(const cube& x)
    {
        base_type::operator=(static_cast<base_type>(x));
        s = x.s;
        return *this;
    }

    template< class Array >
    cube& operator=(const Array& x)
    {
        base_type::operator=(x);
        s = vec3i(x.shape()[0], x.shape()[1], x.shape()[2]);
        return *this;
    }

    const vec3i& unpadded_size() const { return s; }
};


template <typename T> struct qube: boost::multi_array_ref<T,4>
{
    explicit qube(const vec4i& s, T* data)
        : boost::multi_array_ref<T,4>(data,extents[s[0]][s[1]][s[2]][s[3]])
    {
    }

    ~qube()
    {
        znn_free(this);
    }
};

template<typename T>
struct __znn_aligned_size
{
    static const size_t value = ((sizeof(T)-1) | __ZNN_ALIGN) + 1;
    static_assert((value&__ZNN_ALIGN)==0, "bad value");
};


template<class T>
inline T* __offset_cast(void* mem, size_t off)
{
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(mem)+off);
}


class memory_bucket
{
public:
    std::size_t                   mem_size_;
    boost::lockfree::queue<void*> stack_   ;

public:
    memory_bucket(size_t ms = 0)
        : mem_size_(ms)
        , stack_(65536*4)
    {}

    void clear()
    {
        void * p;
        while ( stack_.unsynchronized_pop(p) )
        {
            znn_free(p);
        }
    }

public:
    void return_memory( void* c )
    {
        while ( !stack_.push(c) );
    }

public:
    ~memory_bucket()
    {
        clear();
    }

    void* get()
    {
        void* r;
        if ( stack_.pop(r) ) return r;
        return znn_malloc(mem_size_);
    }
};

template< typename T >
class single_type_cube_pool
{
private:
    std::array<memory_bucket,32> buckets_;

public:
    single_type_cube_pool()
    {
        for ( size_t i = 0; i < 32; ++i )
        {
            buckets_[i].mem_size_ = static_cast<size_t>(1) << i;
        }
    }

public:
    std::shared_ptr<cube<T>> get( const vec3i& s )
    {
        auto p = s;
        /* Secretly allocate more space so that the last dimension is padded */
        p[2] = pad(s[2]);
        const auto raw_data_size = p[0] * p[1] * p[2] * sizeof (T);
        size_t bucket = 64 - __builtin_clzl( __znn_aligned_size<cube<T>>::value
                                             + raw_data_size - 1 );

        void*    mem  = buckets_[bucket].get();
        T*       data = __offset_cast<T>(mem, __znn_aligned_size<cube<T>>::value);
        assert(data & __ZNN_ALIGN == 0);
        cube<T>* c    = new (mem) cube<T>(p, s, data);

        return std::shared_ptr<cube<T>>(c,[this,bucket](cube<T>* c) {
                this->buckets_[bucket].return_memory(c);
            }, allocator<cube<T>>());
    }

private:
    static int pad(int x) {
        x *= sizeof (T);
        int rem = x & __ZNN_ALIGN;
        int div = x & ~__ZNN_ALIGN;
        div += rem == 0 ? 0 : __ZNN_ALIGN + 1;
        return div / sizeof (T);
    }
}; // single_type_cube_pool


template< typename T >
struct pool
{
private:
    static single_type_cube_pool<T>& instance;

public:
    static std::shared_ptr<cube<T>> get( const vec3i& s )
    {
        return instance.get(s);
    }

    static std::shared_ptr<cube<T>> get( size_t x, size_t y, size_t z )
    {
        return instance.get( vec3i(x,y,z) );
    }
};

template< typename T >
single_type_cube_pool<T>& pool<T>::instance =
    zi::singleton<single_type_cube_pool<T>>::instance();


template<typename T>
std::shared_ptr<cube<T>> get_cube(const vec3i& s)
{
    return pool<T>::get(s);
}

}} // namespace znn::v4
