
#include "XGCAggregator.hpp"
#include "Summary.hpp"
#include "SummaryWriterAdios2.hpp"
#include "Reduce/Reduce.hpp"

#include <adios_read.h>
#include <mpi.h>

#include <iostream>
#include <chrono>

using namespace std;
using namespace TN;
using namespace chrono;

typedef float ValueType;

int main( int argc, char** argv )
{
    if( argc < 7 )
    {
        cerr << "expected:\
            <executable> \
                <mesh path> \
                <bfield path> \
                <particle data base path> \
                <units.m path> \
                <bool split particle data> \
                <bool in-situ> \
                <bool single particle file> \
                <optional|reduced mesh> <outpath>\n";
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int wrank = 0, wnproc = 1;
    int rank  = 0, nRanks = 1;

    MPI_Comm mpiReaderComm;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &wrank  );
    MPI_Comm_size( MPI_COMM_WORLD, &wnproc );

    const unsigned int color = 99;
    MPI_Comm_split( MPI_COMM_WORLD, color, wrank, &mpiReaderComm );

    MPI_Comm_rank( mpiReaderComm, &rank   );
    MPI_Comm_size( mpiReaderComm, &nRanks );

    int err  = adios_read_init_method ( ADIOS_READ_METHOD_BP, mpiReaderComm, "verbose=3" );

    if( rank == 0 )
    {
        cout << "nRanks:\t" << nRanks << endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const string adios2conf              = argv[ 1 ];
    const string meshpath                = argv[ 2 ];
    const string bfieldpath              = argv[ 3 ];
    const string particle_data_base_path = argv[ 4 ];
    const string units_path              = argv[ 5 ];
    const string outpath                 = argv[ 6 ];
    const bool splitByBlocks             = std::string( argv[ 7 ] ) == "true";
    const bool inSitu                    = std::string( argv[ 8 ] ) == "true";
    const bool appendedReadMode          = std::string( argv[ 9 ] ) == "true";

    TN::XGCAggregator< ValueType > aggregator(
        adios2conf,
        meshpath,
        bfieldpath,
        particle_data_base_path,
        units_path,
        outpath,
    { "ions" , "electrons" },
    inSitu,
    splitByBlocks,
    rank,
    nRanks,
    mpiReaderComm );

    if( argc == 11 )
    {
        aggregator.reduceMesh( argv[ 10 ] );
    }

    if( rank == 0 )
    {
        aggregator.writeMesh();
    }

    // finalize the adios 1 mesh reader
    adios_read_finalize_method ( ADIOS_READ_METHOD_BP );

    aggregator.run();

    MPI_Finalize();
    return 0;
}
