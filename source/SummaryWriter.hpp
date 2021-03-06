#ifndef TN_SUMMARY_WRITER
#define TN_SUMMARY_WRITER

#include "Summary.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstdint>

namespace TN
{

inline void writeTriangularMeshObj(
    const std::vector< float > & r,
    const std::vector< float > & z,
    const std::vector< TN::Triangle > & mesh,
    const std::string & outpath )
{
    std::ofstream outfile( outpath );
    for( std::size_t i = 0, end = r.size(); i < end; ++i )
    {
        outfile << "v " << r[ i ] << " " << z[ i ] << " 0\n";
    }
    outfile << "\n";

    for( std::size_t i = 0, end = mesh.size(); i < end; ++i )
    {
        outfile << "f";
        for( std::size_t j = 0; j < 3; ++j )
        {
            outfile << " " << mesh[ i ][ j ] + 1;
        }
        outfile << "\n";
    }
    outfile.close();
}


inline void writeSummaryGrid(
    const SummaryGrid & summaryGrid,
    const std::string & outpath )
{
    const int64_t N_CELLS = summaryGrid.probes.r.size();
    std::string filepath = outpath + "/" + "summary.grid.dat";
    std::ofstream outFile( filepath, std::ios::out | std::ios::binary );
    outFile.write( (char*) summaryGrid.probes.r.data(), sizeof( float ) * N_CELLS );
    outFile.write( (char*) summaryGrid.probes.z.data(), sizeof( float ) * N_CELLS );
    outFile.write( (char*) summaryGrid.probes.psin.data(), sizeof( float ) * N_CELLS );
    outFile.write( (char*) summaryGrid.probes.poloidalAngle.data(), sizeof( float ) * N_CELLS );
    outFile.write( (char*) summaryGrid.probes.B.data(), sizeof( float ) * N_CELLS );
    outFile.write( (char*) summaryGrid.probes.volume.data(), sizeof( float ) * N_CELLS );
    outFile.close();

    writeTriangularMeshObj(
        summaryGrid.probes.r,
        summaryGrid.probes.z,
        summaryGrid.probeTriangulation,
        outpath + "/summary.grid.delaunay.obj" );

    outFile.open( outpath + "/summary.meta.txt" );
    outFile << "delta_v "    << std::to_string( SummaryStep::DELTA_V ) << "\n";
    outFile << "vpara_bins " << std::to_string( SummaryStep::NC      ) << "\n";
    outFile << "vperp_bins " << std::to_string( SummaryStep::NR      ) << "\n";
    outFile << "num_cells  " << std::to_string( N_CELLS              ) << "\n";
    // partNormFactor ??
    // particle_ratio ??
    outFile.close();

    // neighborhoods

    outFile.open( outpath + "/" + "summary.grid.neighbors.dat", std::ios::out | std::ios::binary );
    outFile.write( (char*) summaryGrid.neighborhoods.data(), sizeof( int64_t ) * summaryGrid.neighborhoods.size() );
    outFile.close();

    outFile.open( outpath + "/" + "summary.grid.neighbor.counts.dat", std::ios::out | std::ios::binary );
    outFile.write( (char*) summaryGrid.neighborhoodSums.data(), sizeof( int64_t ) * summaryGrid.neighborhoodSums.size() );
    outFile.close();
}

inline void writeSummaryStep(
    const SummaryStep & summaryStep,
    const std::string & ptype,
    int64_t tstep,
    double realtime,
    int64_t outputStep,
    const std::string & outpath,
    bool append )
{
    std::string summary_path  = outpath + "/" + ptype + ".summary.dat";
    std::string tsteps_path   = outpath + "/tsteps.dat";
    std::string realtime_path = outpath + "/realtime.dat";

    if( ! append )
    {
        std::ofstream ofs;
        ofs.open( summary_path, std::ofstream::out | std::ofstream::trunc);
        ofs.close();

        ofs.open( tsteps_path, std::ofstream::out | std::ofstream::trunc);
        ofs.close();

        ofs.open( realtime_path, std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }

    // write results to disk
    std::ofstream outFile( summary_path, std::ios::binary | std::ios_base::app );
    outFile.write( (char*) summaryStep.velocityDistribution.data(), sizeof( float ) * summaryStep.velocityDistribution.size() );
    outFile.write( (char*) summaryStep.w0w1_mean.data(), sizeof( float ) * summaryStep.w0w1_mean.size() );
    outFile.write( (char*) summaryStep.w0w1_rms.data(), sizeof( float )  * summaryStep.w0w1_rms.size() );
    outFile.write( (char*) summaryStep.w0w1_variance.data(), sizeof( float ) * summaryStep.w0w1_variance.size() );
    outFile.write( (char*) summaryStep.w0w1_min.data(), sizeof( float )  * summaryStep.w0w1_min.size() );
    outFile.write( (char*) summaryStep.w0w1_max.data(), sizeof( float )  * summaryStep.w0w1_max.size() );
    outFile.write( (char*) summaryStep.num_particles.data(), sizeof( float ) * summaryStep.num_particles.size() );
    outFile.close();

    const int64_t sim_step = tstep;
    std::ofstream tsFile(  tsteps_path, std::ios::binary | std::ios_base::app );
    tsFile.write( (char*) & outputStep, sizeof( outputStep ) ); // or sim_step ...
    tsFile.close();

    std::ofstream realTimeFile( realtime_path, std::ios::binary | std::ios_base::app );
    realTimeFile.write( (char*) & realtime, sizeof( realtime ) );
}

}

#endif