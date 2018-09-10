#ifndef TN_XGC_PARTICLE_READER
#define TN_XGC_PARTICLE_READER

#include <adios2.h>
#include <mpi.h>

#include <map>
#include <vector>
#include <string>
#include <iostream>

namespace TN
{

inline std::vector< std::pair< int64_t, int64_t > > split( int64_t N, int64_t k )
{
    std::vector< std::pair< int64_t, int64_t > > result( k );
    for( int64_t i = 0, offset = 0; i < k; ++i )
    {
        result[ i ].first  = offset;
        result[ i ].second = ( N - offset ) / ( k - i );
        offset += result[ i ].second;
    }
    return result;
}

template< typename PhaseType, typename TargetFloatType >
inline void copySwitchOrder(
    const std::vector< PhaseType > & chunk,
    std::vector< TargetFloatType > & result,
    const uint64_t WIDTH,
    const uint64_t offset )
{
    const uint64_t CHUNK_SIZE =  chunk.size() / WIDTH;
    const uint64_t FULL_SIZE  = result.size() / WIDTH;

    #pragma omp parallel for
    for( uint64_t pIDX = 0; pIDX < CHUNK_SIZE; ++pIDX )
    {
        #pragma omp simd
        for( uint64_t vIDX = 0; vIDX < WIDTH; ++vIDX )
        {
            result[  vIDX * FULL_SIZE + ( offset + pIDX ) ] = chunk[ ( pIDX ) * WIDTH + vIDX ];
        }
    }
}

template<
    typename PhaseType,
    typename TimeStepType,
    typename RealTimeType,
    typename TargetFloatType >
inline int64_t readBPParticleDataStep(
    std::vector< TargetFloatType > & result,
    const std::string & ptype,
    const std::string & path,
    int rank,
    int nRanks,
    adios2::IO     & bpIO,
    adios2::Engine & reader,
    adios2::Variable< PhaseType > & phaseVar,
    adios2::Variable< TimeStepType > & timestepVar,
    adios2::Variable< RealTimeType > & timeVar,
    int64_t & simstep,
    double  & realtime,
    bool splitByBlocks )
{
    auto dims = phaseVar.Shape();
    uint64_t SZ = dims[ 0 ];

    if( splitByBlocks )
    {
        std::cout << "splitting " << SZ << " particles by blocks" << std::endl;

        std::vector<typename adios2::Variable< PhaseType >::Info> blocks =
            reader.BlocksInfo( phaseVar, reader.CurrentStep() );

        std::cout << "found " << blocks.size() << " blocks" << std::endl;        

        auto splitPoints = split( blocks.size(), nRanks )[ rank ];

        uint64_t MY_SIZE = 0;
        for( int i = splitPoints.first; i < splitPoints.first + splitPoints.second; ++i )
        {
            MY_SIZE += blocks[ i ].Count[ 1 ];
        }

        result.resize( MY_SIZE * dims[ 1 ] );
        std::vector< PhaseType > tmp;

        std::cout << "RANK: " << rank
                  << ", from " << splitPoints.first
                  << ", reading " << splitPoints.second
                  << " blocks, with " << MY_SIZE
                  << " particles " << std::endl;

        int64_t copy_offset = 0;
        for( int i = splitPoints.first; i < splitPoints.first + splitPoints.second; ++i )
        {
            std::cout << "rank: " << rank << " reading block " << i << std::endl; 

            phaseVar.SetSelection(
            {
                { blocks[ i ].Start[ 1 ],         0 },
                { blocks[ i ].Count[ 1 ], dims[ 1 ] }
            } );
            
            std::cout << "rank: " << rank << " reading block " << i << "set selection " 
                      << blocks[ i ].Start[ 1 ] << " to " << blocks[ i ].Count[ 1 ] 
                      << " with " << dims[ 1 ] << " variables" << std::endl; 

            tmp.clear();
            reader.Get( phaseVar, tmp, adios2::Mode::Sync );

            std::cout << "rank: " << rank << " called Get" << std::endl; 

            copySwitchOrder(
                tmp,
                result,
                dims[ 1 ],
                copy_offset );

            std::cout << "rank: " << rank << " changed column order" << std::endl; 
            copy_offset += blocks[ i ].Count[ 1 ];
        }
    }
    else
    {
        uint64_t CS = SZ / nRanks;
        uint64_t MY_START = rank * CS;
        uint64_t MY_SIZE = ( rank < nRanks - 1 ? CS : SZ - rank*CS );

        phaseVar.SetSelection(
        {
            { MY_START,        0 },
            { MY_SIZE, dims[ 1 ] }
        } );

        std::vector< PhaseType > tmp;
        reader.Get( phaseVar, tmp, adios2::Mode::Sync );
        result.resize( MY_SIZE * dims[ 1 ] );

        copySwitchOrder(
            tmp,
            result,
            dims[ 1 ],
            0 );
    }

    std::cout << "rank: " << rank << " finished reading particles" << std::endl; 

    TimeStepType tStepRead;
    reader.Get( timestepVar, &tStepRead, adios2::Mode::Sync );
    simstep = static_cast< int64_t >( tStepRead );

    std::cout << "rank: " << rank << " read simstep" << std::endl; 

    RealTimeType timeRead;
    reader.Get( timeVar, &timeRead, adios2::Mode::Sync );
    realtime = static_cast< double >( timeRead );

    std::cout << "rank: " << rank << " read realtime" << std::endl; 

    return SZ;
}

template< typename TargetFloatType >
inline int64_t readBPParticleDataStep(
    std::vector< TargetFloatType > & result,
    const std::string & ptype,
    const std::string & path,
    int rank,
    int nRanks,
    adios2::IO     & bpIO,
    adios2::Engine & reader,
    int64_t & simstep,
    double  & realtime,
    bool splitByBlocks )
{
    adios2::Variable< int > stepV = bpIO.InquireVariable< int >( "timestep" );
    adios2::Variable< double > timeV = bpIO.InquireVariable< double >( "time" );

    if( ! stepV )
    {
        std::cerr << "couldn't find timestep as an int" << std::endl;
        exit( 1 );
    }

    if( ! timeV )
    {
        std::cerr << "couldn't find time as a double" << std::endl;
        exit( 1 );
    }

    std::string phaseName = ptype == "ions" ? "iphase" : "ephase";
    adios2::Variable< double > phaseDouble = bpIO.InquireVariable< double >( phaseName );
    adios2::Variable< float  > phaseFloat  = bpIO.InquireVariable< float   >( phaseName );

    if( phaseDouble )
    {
        readBPParticleDataStep(
            result,
            ptype,
            path,
            rank,
            nRanks,
            bpIO,
            reader,
            phaseDouble,
            stepV,
            timeV,
            simstep,
            realtime,
            splitByBlocks );
    }
    else if( phaseFloat )
    {
        readBPParticleDataStep(
            result,
            ptype,
            path,
            rank,
            nRanks,
            bpIO,
            reader,
            phaseFloat,
            stepV,
            timeV,
            simstep,
            realtime,
            splitByBlocks );
    }
    else
    {
        std::cerr << "couldn't find " << phaseName << " as either float or double" << std::endl;
        exit( 1 );
    }
}
template< typename TargetFloatType >
inline int64_t readBPParticleDataStep(
    std::vector< TargetFloatType > & result,
    const std::string & ptype,
    const std::string & path,
    int rank,
    int nRanks,
    int64_t & simstep,
    double  & realtime )
{
    adios2::ADIOS adios( MPI_COMM_SELF, adios2::DebugOFF );
    adios2::IO bpIO = adios.DeclareIO( "Particle-IO-Self-" + std::to_string( rank ) );
    adios2::Engine bpReader = bpIO.Open( path, adios2::Mode::Read );

    auto totalNumParticles = readBPParticleDataStep(
                                 result,
                                 ptype,
                                 path,
                                 rank,
                                 nRanks,
                                 bpIO,
                                 bpReader,
                                 simstep,
                                 realtime,
                                 true
                             );

    bpReader.Close();
    return totalNumParticles;
}

}

#endif