#ifndef TN_MESH_UTILS_HPP
#define TN_MESH_UTILS_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace TN
{

namespace Mesh
{

enum VertexFlag
{
    boundary = 1
};

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

template< typename IndexType >
inline bool shareEdge( 
    IndexType i,
	IndexType j,
    const std::vector< IndexType > & neighborhoods,
    const std::vector< IndexType > & neighborhoodSums )
{
    const IndexType Oi = i > 0 ? neighborhoodSums[ i - 1 ] : 0;
    const IndexType Ni = neighborhoodSums[ i ] - Oi;

    for( IndexType k = Oi; k < Oi + Ni; ++k )
    {
    	if( j == k )
    	{
    		return true;
    	}
    }    
    return false;
}


template< typename IndexType,
          typename FloatType >
inline IndexType computeVertexFlags( 
	const std::vector< FloatType > & x,
	const std::vector< FloatType > & y,
    const std::vector< IndexType > & neighborhoods,	
    const std::vector< IndexType > & neighborhoodSums,
    std::vector< std::uint8_t > & flags )
{
    const size_t END = x.size();
	#pragma omp parallel for
	for( size_t i = 0; i < END; ++i )
	{
        flags[ i ] = 0;

        const IndexType OFFSET = i > 0 ? neighborhoodSums[ i - 1 ] : 0;
        const IndexType NUM_NEIGHBORS = neighborhoodSums[ i ] - OFFSET;
        
        IndexType previousVertex = neighborhoods[ i ];

        bool closedLoop = true;
        for( IndexType j = previousVertex + 1; j < NUM_NEIGHBORS + 1; ++j )
        {
        	IndexType currentVertext = j % NUM_NEIGHBORS;

            if( ! shareEdge( 
            	previousVertex, 
            	currentVertext,
            	neighborhoods,
            	neighborhoodSums ) )
            {
            	closedLoop = false;
                break;
            }
            previousVertex = currentVertext;
        }

        if( closedLoop == false )
        {
        	flags[ i ] |= VertexFlag::boundary;
        }
        else
        {
        	std::uint8_t tmp = flags[ i ] & ~ VertexFlag::boundary;
        }
    }
}

}

}

#endif