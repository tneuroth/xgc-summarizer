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

template< typename FloatType >
inline void readBPParticleDataStep(
    std::vector< FloatType > & result,
    const std::string & ptype,
    const std::string & path,
    int rank,
    int nRanks )
{
    adios2::ADIOS adios(MPI_COMM_WORLD, adios2::DebugOFF );
    adios2::IO bpIO = adios.DeclareIO( "IO" );
    adios2::Engine bpReader = bpIO.Open( path, adios2::Mode::Read );

    std::map< std::string, adios2::Params > variableInfo = bpIO.AvailableVariables();
    adios2::Variable<double> phase = bpIO.InquireVariable< double >(
        ptype == "ions" ? "iphase" : "ephase" );
    
    if( phase )
    {
        auto dims = phase.Shape();

        const uint64_t SZ = dims[ 0 ];
        const uint64_t CS = SZ / nRanks;
        const uint64_t MY_START = rank * CS;
        const uint64_t MY_CHUNK = rank < nRanks - 1 ? CS : SZ - rank*CS;

        phase.SetSelection(
        {
            { MY_START,         0 },
            { MY_CHUNK, dims[ 1 ] }
        } );

        std::vector< double > tmp;
        bpReader.Get( phase, tmp, adios2::Mode::Sync );

        result.resize( MY_CHUNK * dims[ 1 ] );

        #pragma omp parallel for
        for( uint64_t pIDX = 0; pIDX < MY_CHUNK; ++pIDX )
        {
            #pragma omp simd
            for( uint64_t vIDX = 0; vIDX < dims[ 1 ]; ++vIDX )
            {
                result[ vIDX * MY_CHUNK + pIDX ] = tmp[ ( pIDX ) * dims[ 1 ] + vIDX ];
            }
        }
    }
    else
    {
        std::cerr << "couldn't find iphase" << std::endl;
    }

    bpReader.Close();
}

}

#endif