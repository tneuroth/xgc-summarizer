
#include "XGCParticleReader.hpp"
#include "XGCMeshReader.hpp"
#include "XGCConstantReader.hpp"
#include "XGCAggregator.hpp"
#include "Summary.hpp"
#include "SummaryWriter.hpp"
#include "Types/Vec.hpp"

#include <adios_read.h>
#include <mpi.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace TN;
using namespace chrono;

int main( int argc, char** argv )
{
    if( argc < 6 )
    {
        cerr << "expected: <executable> <mesh path> <bfield path> <particle data base path> <units.m path> <optional|reduced mesh>  <outpath>\n";
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    MPI_Init(NULL, NULL);
    int nRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int err  = adios_read_init_method ( ADIOS_READ_METHOD_BP, MPI_COMM_WORLD, "verbose=3" );

    // cout << "rank:\t" << rank << "\n";
    if( rank == 0 )
    {
        cout << "nRanks:\t" << nRanks << endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const string meshpath                = argv[ 1 ];
    const string bfieldpath              = argv[ 2 ];
    const string particle_data_base_path = argv[ 3 ];
    const string units_path              = argv[ 4 ];
    const string outpath                 = argv[ 5 ];

    TN::XGCAggregator aggregator( 
        meshpath,
        bfieldpath,
        particle_data_base_path,
        units_path,
        outpath,
        { "ions" , "electrons" },
        rank,
        nRanks );

    if( argc == 7 )
    {
        aggregator.reduceMesh( argv[ 6 ] );
    }

    if( rank == 0 )
    {
        aggregator.writeMesh();
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    double summaryStepTime = 0.0;
    int64_t outputStep     = 0;

    std::vector< int64_t > steps = { 200, 400, 4000, 4200 };

    SummaryStep summaryStep;
    for( auto tstep : steps )
    {
        high_resolution_clock::time_point st1 = high_resolution_clock::now();

        aggregator.computeSummaryStep(
            summaryStep,
            "ions",
            tstep );

        MPI_Barrier(MPI_COMM_WORLD);

        high_resolution_clock::time_point st2 = high_resolution_clock::now();
        if( rank == 0 )
        {
            std::cout << "compute summary step took a total of " 
                      <<  duration_cast<milliseconds>( st2 - st1 ).count() 
                      << " milliseconds" << std::endl;
        }

        st1 = high_resolution_clock::now();

        MPI_Reduce(
            rank == 0 ? MPI_IN_PLACE : summaryStep.velocityDistribution.data(),
            summaryStep.velocityDistribution.data(),
            summaryStep.velocityDistribution.size(),
            MPI_FLOAT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD );

        MPI_Reduce(
            rank == 0 ? MPI_IN_PLACE : summaryStep.w0w1_mean.data(),
            summaryStep.w0w1_mean.data(),
            summaryStep.w0w1_mean.size(),
            MPI_FLOAT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD );

        MPI_Reduce(
            rank == 0 ? MPI_IN_PLACE : summaryStep.w0w1_rms.data(),
            summaryStep.w0w1_rms.data(),
            summaryStep.w0w1_rms.size(),
            MPI_FLOAT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD );

        MPI_Reduce(
            rank == 0 ? MPI_IN_PLACE : summaryStep.w0w1_min.data(),
            summaryStep.w0w1_min.data(),
            summaryStep.w0w1_min.size(),
            MPI_FLOAT,
            MPI_MIN,
            0,
            MPI_COMM_WORLD );

        MPI_Reduce(
            rank == 0 ? MPI_IN_PLACE : summaryStep.w0w1_max.data(),
            summaryStep.w0w1_max.data(),
            summaryStep.w0w1_max.size(),
            MPI_FLOAT,
            MPI_MAX,
            0,
            MPI_COMM_WORLD );

        MPI_Reduce(
            rank == 0 ? MPI_IN_PLACE : summaryStep.num_particles.data(),
            summaryStep.num_particles.data(),
            summaryStep.num_particles.size(),
            MPI_FLOAT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD );

        MPI_Reduce(
            rank == 0 ? MPI_IN_PLACE : summaryStep.w0w1_variance.data(),
            summaryStep.w0w1_variance.data(),
            summaryStep.w0w1_variance.size(),
            MPI_FLOAT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD );

        MPI_Barrier(MPI_COMM_WORLD);

        st2 = high_resolution_clock::now();

        /////////////////////////////////////////////////////////////////////////

        if( rank == 0 )
        {
            std::cout << "mpi reduction took " << duration_cast<milliseconds>( st2 - st1 ).count() << " milliseconds" << std::endl;
            std::cout << "normalizing " << std::endl;

            const size_t NUM_CELLS = summaryStep.w0w1_mean.size();
            #pragma omp parallel for simd
            for( size_t i = 0; i < NUM_CELLS; ++i )
            {
                if( summaryStep.num_particles[ i ] > 0 )
                {
                    summaryStep.w0w1_mean[ i ] /= summaryStep.num_particles[ i ];
                    summaryStep.w0w1_rms[  i ]  = sqrt( summaryStep.w0w1_rms[ i ] ) / summaryStep.num_particles[ i ];
                    summaryStep.w0w1_variance[  i ] = sqrt( summaryStep.w0w1_variance[ i ] ) / summaryStep.num_particles[ i ];                    
                }
                else
                {
                    summaryStep.w0w1_mean[ i ] = 0;
                    summaryStep.w0w1_rms[  i ] = 0;
                    summaryStep.w0w1_variance[ i ] = 0;
                    summaryStep.w0w1_min[  i ] = 0;
                    summaryStep.w0w1_max[  i ] = 0;
                }
            }
            double realtime = ( double ) tstep / 2.0;
            writeSummaryStep( summaryStep, "ions", tstep, realtime, outputStep++, outpath, tstep != steps.front() );
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    const double N_COMPUTED = steps.size();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

    if( rank == 0 )
    {
        cout << "Summarization step took " << duration / ( N_COMPUTED ) << " milliseconds per step.\n";
    }

    adios_read_finalize_method ( ADIOS_READ_METHOD_BP );
    MPI_Finalize();
    return 0;
}
