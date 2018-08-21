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
    uint64_t MY_SIZE = 0;

    if( splitByBlocks )
    {
        std::vector<typename adios2::Variable< PhaseType >::Info> blocks =
           reader.BlocksInfo( phaseVar, reader.CurrentStep() );

        uint64_t BCS = blocks.size() / nRanks;
        uint64_t MY_START_BLOC = rank * BCS;
        uint64_t MY_NUM_BLOCKS  = ( rank < nRanks - 1 ? BCS : blocks.size() - rank*BCS );

        uint64_t MY_START = blocks[ MY_START_BLOC ].Start;
        MY_SIZE  = 0;

        for( int i = MY_START_BLOC; i < MY_START_BLOC + MY_NUM_BLOCKS; ++i )
        {
            MY_SIZE += blocks[ i ].Count;
        }

        phaseVar.SetSelection(
        {
            { MY_START,         0 },
            { MY_SIZE, dims[ 1 ] }
        } );
    }
    else
    {
        uint64_t CS = SZ / nRanks;
        uint64_t MY_START = rank * CS;
        uint64_t MY_SIZE = ( rank < nRanks - 1 ? CS : SZ - rank*CS );

        phaseVar.SetSelection(
        {
            { MY_START,         0 },
            { MY_SIZE, dims[ 1 ] }
        } );
    }

    std::vector< PhaseType > tmp;
    reader.Get( phaseVar, tmp, adios2::Mode::Sync );

    result.resize( MY_SIZE * dims[ 1 ] );

    #pragma omp parallel for
    for( uint64_t pIDX = 0; pIDX < MY_SIZE; ++pIDX )
    {
        #pragma omp simd
        for( uint64_t vIDX = 0; vIDX < dims[ 1 ]; ++vIDX )
        {
            result[ vIDX * MY_SIZE + pIDX ] = tmp[ ( pIDX ) * dims[ 1 ] + vIDX ];
        }
    }

    TimeStepType tStepRead;
    reader.Get( timestepVar, &tStepRead, adios2::Mode::Sync );
    simstep = static_cast< int64_t >( tStepRead );

    RealTimeType timeRead;
    reader.Get( timeVar, &timeRead, adios2::Mode::Sync );
    realtime = static_cast< double >( timeRead );

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
    adios2::ADIOS adios(MPI_COMM_WORLD, adios2::DebugOFF );
    adios2::IO bpIO = adios.DeclareIO( "IO" );
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