#ifndef TN_TRIANGLE_HPP
#define TN_TRIANGLE_HPP

namespace TN {

struct Triangle
{
    int64_t indices[ 3 ];
    Triangle() {}
    Triangle( int64_t i1, int64_t i2, int64_t i3 )
    {
        indices[ 0 ] = i1;
        indices[ 1 ] = i2;
        indices[ 2 ] = i3;
    }
    int64_t & operator[] ( int64_t i )
    {
        return indices[ i ];
    }

    int64_t operator[] ( int64_t i ) const
    {
        return indices[ i ];
    }
};

}

#endif