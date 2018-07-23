#include <adios2.h>
#include <mpi.h>

#include <iostream>
#include <vector>
#include <map>

using namespace std;

int main( int argc, char** argv )
{
    MPI_Init(NULL, NULL);
    int nRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if( argc < 2 )
    {
        cerr << "pass bp file path as command line argument" << endl;
        exit( 1 );
    }

    const string path = argv[ 1 ];

    try
    {
        adios2::ADIOS adios(MPI_COMM_WORLD, adios2::DebugON );
        adios2::IO bpIO = adios.DeclareIO( "XGCParticleIO" );

        adios2::Engine bpReader = bpIO.Open(
                                      path,
                                      adios2::Mode::Read );

        std::map< std::string, adios2::Params > variableInfo = bpIO.AvailableVariables();
        adios2::Variable<double> iphase = bpIO.InquireVariable< double >("iphase");

        if( iphase )
        {
            auto dims = iphase.Shape();

            const int64_t SZ = dims[ 0 ];
            const int64_t CS = SZ / nRanks;
            const int64_t MY_START  = rank * CS;
            const int64_t MY_END    = rank < nRanks-1 ? MY_START + CS - 1 : SZ - 1;
            const int64_t MY_CHUNKSIZE = MY_END - MY_START + 1;

            iphase.SetSelection(
            {
                { MY_START,     0 },
                { MY_CHUNKSIZE, 1 }
            } );

            vector< double > tmp;
            bpReader.Get( iphase, tmp, adios2::Mode::Sync );
        }
        else
        {
            cerr << "couldn't find iphase" << endl;
        }

        bpReader.Close();
    }
    catch (std::invalid_argument &e)
    {
        std::cout << "Invalid argument exception, STOPPING PROGRAM from rank "
                  << rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::ios_base::failure &e)
    {
        std::cout << "IO System base failure exception, STOPPING PROGRAM "
                  "from rank "
                  << rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::exception &e)
    {
        std::cout << "Exception, STOPPING PROGRAM from rank " << rank << "\n";
        std::cout << e.what() << "\n";
    }

    MPI_Finalize();
    return 0;
}