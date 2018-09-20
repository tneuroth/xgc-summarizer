#ifndef TN_TRACKER_HPP
#define TN_TRACKER_HPP

#include "Summary.hpp"
#include "XGCAggregator.hpp"

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <iostream>

namespace TN
{
	
namespace Tracker
{

template < typename ValueType > 
inline void identifyNewSuperParticles( 
    const std::vector< int64_t > & ids,
    const std::vector< ValueType > & phase,
    std::vector< int64_t > & newSuperParticles,
    double threshold )
{
	const size_t N_PTCL = ids.size();
    const int64_t W0_OFFSET = XGC_PHASE_INDEX_MAP.at( "w0" ) * N_PTCL;
    const int64_t W1_OFFSET = XGC_PHASE_INDEX_MAP.at( "w1" ) * N_PTCL;

    int nthreads = 1;
    #pragma omp parallel
    {
    	nthreads = omp_get_num_threads();
    }

    std::vector< std::vector< int64_t > > perThreadNewSupers( nthreads );

    #pragma omp parallel for
	for( size_t i = 0; i < N_PTCL; ++i )
	{
		// if it meets criteria
        if( std::abs( phase[ i + W0_OFFSET ] * phase[ i + W1_OFFSET ] ) >= threshold )
        {
            perThreadNewSupers[ omp_get_thread_num() ].push_back( ids[ i ] );
        }
    }

    int64_t total = 0;
    for( int i = 0; i < nthreads; ++i )
    {
    	total += perThreadNewSupers[ i ].size();
    }

    newSuperParticles.clear();
    newSuperParticles.reserve( total ); 

    for( int i = 0; i < nthreads; ++i )
    {
    	newSuperParticles.insert( 
    		newSuperParticles.end(), 
    		perThreadNewSupers[ i ].begin(), 
    		perThreadNewSupers[ i ].end() );
    }
}

template < typename ValueType > 
inline void trackParticles(  
    const std::vector< int64_t > & ids,
    const std::vector< ValueType > phase,
    std::unordered_set< int64_t > & trackedParticleIds,
    PhasePathStep< ValueType > & pathsStep )
{    
	const size_t N_PTCL = ids.size();
	for( size_t i = 0; i < N_PTCL; ++i )
	{
        if( trackedParticleIds.count( ids[ i ] ) )
        {
        	pathsStep.ids.push_back( ids[ i ] );
        	for( auto var : XGC_PHASE_INDEX_MAP )
        	{
        		pathsStep.values.push_back( phase[ i + var.second * N_PTCL ] );
        	}
        }
    }
}

}

}

#endif