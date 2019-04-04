#ifndef TN_REDUCE_HPP
#define TN_REDUCE_HPP

#include <mpi.h>
#include <vector>
#include <cstdint>

namespace TN
{

namespace MPI
{

template< typename T > 
inline void localDivide( 
    std::vector< T > & result, 
    T denominator,
    bool safe )
{
    const int64_t SZ = result.size();
    
    if( safe )
    {
        #pragma omp parallel for
        for( int64_t i = 0; i < SZ; ++i )
        {
            if( denominator[ i ] != 0 )
            {
                result[ i ] /= denominator;
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for( int64_t i = 0; i < SZ; ++i )
        {
            result[ i ] /= denominator;
        }
    }
}

template< typename T > 
inline void localDivide( 
    std::vector< T > & result, 
    const std::vector< T > & denominator,
    bool safe )
{
    const int64_t SZ = result.size();

    if( safe )
    {
        #pragma omp parallel for
        for( int64_t i = 0; i < SZ; ++i )
        {
            if( denominator[ i ] != 0 )
            {
                result[ i ] /= denominator[ i ];
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for( int64_t i = 0; i < SZ; ++i )
        {
            result[ i ] /= denominator[ i ];
        }
    }
}

template< typename T > 
inline void localSqrtAndDivide( 
    std::vector< T > & result, 
    const std::vector< T > & denominator,
    bool safe )
{
    const int64_t SZ = result.size();
    
    if( safe )
    {
        #pragma omp parallel for
        for( int64_t i = 0; i < SZ; ++i )
        {
            if( denominator[ i ] != 0 )
            {
                result[ i ] = std::sqrt( result[ i ] ) / denominator[ i ];
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for( int64_t i = 0; i < SZ; ++i )
        {
            result[ i ] = std::sqrt( result[ i ] ) / denominator[ i ];
        }
    }
}

template< typename T > 
inline void ReduceOpMPI( 
    int rank, 
    std::vector< T > & myData,
    MPI_Op op,
    MPI_Comm comm )
{
    MPI_Reduce(
        rank == 0 ? MPI_IN_PLACE : myData.data(),
        myData.data(),
        myData.size(),
        sizeof( T ) == 4 ? MPI_FLOAT : MPI_DOUBLE,
        op,
        0,
        comm );
}

template< typename T > 
inline void ReduceMean( 
    int rank, 
    std::vector< T > & sums, 
    const std::vector< T > & reducedCounts,
    MPI_Comm comm )
{
    ReduceOpMPI( rank, sums, MPI_SUM, comm );

    if( rank == 0 )
    {
        localDivide( sums, reducedCounts, true );
    }
}

template< typename T > 
inline void ReduceVariance( 
    int rank, 
    std::vector< T > & mySumOfSquaredDeviations,
    const std::vector< T > & reducedCounts,
    MPI_Comm comm )
{
    ReduceOpMPI( rank, mySumOfSquaredDeviations, MPI_SUM, comm );
 
    if( rank == 0 )
    {
        localSqrtAndDivide( mySumOfSquaredDeviations, reducedCounts, true );
    }
}

template< typename T > 
inline void ReduceRMS( 
    int rank, 
    std::vector< T > & mySumOfSquares, 
    const std::vector< T > & reducedCounts,
    MPI_Comm comm )
{
    ReduceOpMPI( rank, mySumOfSquares, MPI_SUM, comm );

    if( rank == 0 )
    {
        localSqrtAndDivide( mySumOfSquares, reducedCounts, true );
    }
}

template< typename T > 
inline void ReduceSkewness( 
    int rank, 
    std::vector< T > & myData,
    MPI_Comm comm )
{

}

template< typename T > 
inline void ReduceKurtosis( 
    int rank, 
    std::vector< T > & myData,
    MPI_Comm comm )
{

}

template< typename T > 
inline void ReduceInterquartileRange( 
    int rank, 
    std::vector< T > & myData,
    MPI_Comm comm )
{

}

template< typename T > 
inline void ReduceShannonEntropy( 
    int rank, 
    std::vector< T > & myData,
    MPI_Comm comm )
{

}

}

}

#endif