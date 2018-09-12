#ifndef TN_MESH_UTILS_HPP
#define TN_MESH_UTILS_HPP

#include <iostream>
#include <vector>
#include <algorithm>

namespace TN
{

namespace Mesh
{

template< typename FloatType, typename IndexType >
inline void sortNeighborhoodsCCW( 
    const std::vector< FloatType > & x,
    const std::vector< FloatType > & y,    
          std::vector< IndexType > & neighborhoods,
    const std::vector< IndexType > & neighborhoodSums )
{
    const size_t END = x.size();
	#pragma omp parallel for
	for( size_t i = 0; i < END; ++i )
	{
        const IndexType OFFSET = i > 0 ? neighborhoodSums[ i - 1 ] : 0;
        const IndexType NUM_NEIGHBORS = neighborhoodSums[ i ] - OFFSET;

        std::sort( 
        	neighborhoods.begin() + OFFSET,
            neighborhoods.begin() + OFFSET + NUM_NEIGHBORS,
            [&x, &y, i ]( const IndexType & a, const IndexType & b ) -> bool
			{   
			    return ( y[ a ] >= y[ i ] && y[ b ] >= y[ i ] ) ? x[ a ] > x[ b ] :
			           ( y[ a ]  < y[ i ] && y[ b ]  < y[ i ] ) ? x[ a ] < x[ b ] :
			                                                      y[ a ] > y[ b ];
			}
        );
	}
}

template< typename IndexType >
inline IndexType maxNeighborhoodSize( 
    const std::vector< IndexType > & neighborhoodSums )
{
	IndexType maxN = 0;
    const size_t END = neighborhoodSums.size();
	for( size_t i = 0; i < END; ++i )
	{
        const IndexType OFFSET = i > 0 ? neighborhoodSums[ i - 1 ] : 0;
        const IndexType NUM_NEIGHBORS = neighborhoodSums[ i ] - OFFSET;
        maxN = std::max( maxN, NUM_NEIGHBORS );
	}
	return maxN;
}

}

}

#endif