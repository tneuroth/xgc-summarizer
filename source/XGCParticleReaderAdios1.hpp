#ifndef TN_XGC_PARTICLE_READER
#define TN_XGC_PARTICLE_READER

#include <adios_read.h>
#include <mpi.h>

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
    ADIOS_FILE * f = adios_read_open_file ( path.c_str(), ADIOS_READ_METHOD_BP, MPI_COMM_WORLD );

    if (f == NULL)
    {
        std::cerr << adios_errmsg() << std::endl;
        exit( 1 );
    }

    ADIOS_VARINFO * v = adios_inq_var ( f, ptype == "ions" ? "iphase" : "ephase" );

    const uint64_t SZ = v->dims[ 0 ];
    const uint64_t CS = SZ / nRanks;
    const uint64_t MY_START = rank * CS;
    const uint64_t MY_LAST = rank < nRanks-1 ? MY_START + CS - 1 : SZ - 1;
    const uint64_t MY_CHUNKSIZE = MY_LAST - MY_START + 1;

    result.resize( MY_CHUNKSIZE * 9 );

    uint64_t start[ 2 ] = { MY_START,        0 };
    uint64_t count[ 2 ] = { MY_CHUNKSIZE,    9 };

    std::cout << "my start " << MY_START << " my size " << MY_CHUNKSIZE << " total " << SZ << std::endl;

    ADIOS_SELECTION * selection = adios_selection_boundingbox( v->ndim, start, count );

    if( std::is_same< FloatType, float >::value )
    {
        std::vector< double > tmp( SZ * 9 );
        adios_schedule_read ( f, selection, "iphase", 0, 1, tmp.data() );
        adios_perform_reads ( f, 1 );

        #pragma omp parallel for
        for( uint64_t pIDX = 0; pIDX < MY_CHUNKSIZE; ++pIDX )
        {
            #pragma omp simd
            for( uint64_t vIDX = 0; vIDX < 9; ++vIDX )
            {
                result[ vIDX * MY_CHUNKSIZE + pIDX ] = tmp[ ( pIDX ) * 9 + vIDX ];
            }
        }
    }
    else if( std::is_same< FloatType, double >::value )
    {
        adios_schedule_read ( f, selection, "iphase", 0, 1, result.data() );
        adios_perform_reads ( f, 1 );
    }

    adios_selection_delete ( selection );
    adios_free_varinfo ( v );
    adios_read_close ( f );
}

}

#endif