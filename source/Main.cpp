
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
    if( argc < 11 )
    {
        cerr << "expected 10 arguments" << endl;
        exit( 1 ); 
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int wrank = 0, wnproc = 1;
    int rank  = 0, nRanks = 1;

    MPI_Comm mpiReaderComm;

    MPI_Init( &argc, &argv );

    MPI_Comm_rank( MPI_COMM_WORLD, &wrank  );
    MPI_Comm_size( MPI_COMM_WORLD, &wnproc );

    const unsigned int color = 12899;
    int comm_err = MPI_Comm_split( MPI_COMM_WORLD, color, wrank, &mpiReaderComm );

    if( comm_err != MPI_SUCCESS )
    {
        if( comm_err == MPI_ERR_COMM )
        {
            cerr << "wrank: error in MPI_Comm_Split, " << wrank << "MPI_ERR_COMM" << endl;
        }
        else if( comm_err == MPI_ERR_INTERN )
        {
            cerr << "wrank: error in MPI_Comm_Split, " << wrank << "MPI_ERR_INTERN" << endl;   
        }
        else
        {
            cerr << "wrank: error in MPI_Comm_Split, " << wrank << " " << comm_err << endl;      
        }
    }

    MPI_Comm_rank( mpiReaderComm, &rank   );
    MPI_Comm_size( mpiReaderComm, &nRanks );

    int err  = adios_read_init_method ( ADIOS_READ_METHOD_BP, MPI_COMM_SELF, "verbose=3" );


    omp_set_dynamic( 0 );
    omp_set_num_threads( 16 );

    int nt = 1;
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
    }

    cout << "set omp num threads " << nt << endl;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const string adios2conf              = argv[ 1 ];
    const string meshpath                = argv[ 2 ];
    const string bfieldpath              = argv[ 3 ];
    const string particle_data_base_path = argv[ 4 ];
    const string units_path              = argv[ 5 ];
    const string outpath                 = argv[ 6 ];
    const bool splitByBlocks             = string( argv[ 7 ] )  == "true";
    const bool inSitu                    = string( argv[ 8 ] )  == "true";
    const bool appendedReadMode          = string( argv[ 9 ] )  == "true";
    const bool tryUsingCuda              = string( argv[ 10 ] ) == "true";

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
        mpiReaderComm,
        tryUsingCuda );

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
