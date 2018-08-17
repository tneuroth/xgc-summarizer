
#include "XGCAggregator.hpp"
#include "Summary.hpp"
#include "SummaryWriterAdios2.hpp"

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
        cerr << "expected: 
            <executable> 
                <mesh path> 
                <bfield path> 
                <particle data base path> 
                <units.m path> 
                <bool split particle data>
                <optional|reduced mesh> <outpath>\n";
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    MPI_Init(NULL, NULL);
    int nRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int err  = adios_read_init_method ( ADIOS_READ_METHOD_BP, MPI_COMM_WORLD, "verbose=3" );

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
    const bool splitParticleData = argv[ 6 ] == "true";

    int particleChunkIndex = splitParticleData ?      0 : rank;
    int numParticleChunks  = splitParticleData ? nRanks :    1;

    TN::XGCAggregator< ValueType > aggregator( 
        meshpath,
        bfieldpath,
        particle_data_base_path,
        units_path,
        outpath,
        { "ions" , "electrons" },
        particleChunkIndex,
        numParticleChunks );

    if( argc == 8 )
    {
        aggregator.reduceMesh( argv[ 7 ] );
    }

    if( rank == 0 )
    {
        aggregator.writeMesh();
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    double summaryStepTime = 0.0;
    int64_t outputStep     = 0;

    std::vector< int64_t > steps = { 200, 400 };
    
    SummaryStep2< ValueType > summaryStep;

    for( auto tstep : steps )
    {
        high_resolution_clock::time_point st1 = high_resolution_clock::now();

        summaryStep.setStep( outputStep, tstep, outputStep );
        summaryStep.objectIdentifier = "ions";
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

        high_resolution_clock::time_point rt1 = high_resolution_clock::now();

        for( auto & hist : summaryStep.histograms )
        {
            TN::MPI::ReduceOpMPI( rank, hist.second.values, MPI_SUM );
        }

        for( auto & var : summaryStep.variableStatistics )
        {
            auto & myCounts = var.second.values.at( 
                ScalarVariableStatistics< ValueType >::Statistic::Count );

            TN::MPI::ReduceOpMPI(
                rank, 
                myCounts, 
                MPI_SUM );

            if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Min ) )
            {
                TN::MPI::ReduceOpMPI(
                    rank, 
                    var.second.values.at( 
                        ScalarVariableStatistics< ValueType >::Statistic::Min ),
                    MPI_MIN );
            }

            if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Max ) )
            {
                TN::MPI::ReduceOpMPI(
                    rank, 
                    var.second.values.at( 
                        ScalarVariableStatistics< ValueType >::Statistic::Max ),
                    MPI_MAX );
            }

            if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Mean ) )
            {
                TN::MPI::ReduceMean( 
                    rank,
                    var.second.values.at( 
                        ScalarVariableStatistics< ValueType >::Statistic::Mean ),
                    myCounts );
            }

            if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Variance ) )
            {
                TN::MPI::ReduceVariance( 
                    rank,
                    var.second.values.at( 
                        ScalarVariableStatistics< ValueType >::Statistic::Variance ),
                    myCounts );  
            }

            if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::RMS ) )
            {
                TN::MPI::ReduceRMS( 
                    rank,
                    var.second.values.at( 
                        ScalarVariableStatistics< ValueType >::Statistic::RMS ),
                    myCounts );    
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if( rank == 0 )
        {   
            high_resolution_clock::time_point rt2 = high_resolution_clock::now();
            std::cout << "reduce step took " << duration_cast<milliseconds>( rt2 - rt1 ).count()
                      << " milliseconds"  << std::endl;

            high_resolution_clock::time_point wt1 = high_resolution_clock::now();
            
            writeSummaryStepBP( summaryStep, outpath );

            high_resolution_clock::time_point wt2 = high_resolution_clock::now();        
            std::cout << "write step took " << duration_cast<milliseconds>( wt2 - wt1 ).count()
                      << " milliseconds\n"  << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    const double N_COMPUTED = steps.size();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

    if( rank == 0 )
    {
        cout << "Summarization took on average " << duration / ( N_COMPUTED ) << " milliseconds per step.\n";
    }

    adios_read_finalize_method ( ADIOS_READ_METHOD_BP );
    MPI_Finalize();
    return 0;
}
