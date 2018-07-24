
#include "kdtree/KdTreeSearch2D.hpp"
#include "XGCBFieldInterpolator.hpp"
#include "XGCPsinInterpolator.hpp"
#include "XGCMeshLoader.hpp"
#include "Summary.hpp"
#include "XGCGridBuilder.hpp"
#include "SummaryUtils.hpp"

#include "FieldInterpolator.hpp"
#include "Types/Vec.hpp"

#include "adios_read.h"
#include <hdf5.h>
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

const map< string, int > attrKeyToPhaseIndex =
{
    { "r",            0 },   // Major radius [m]
    { "z",            1 },   // Azimuthal direction [m]
    { "zeta",         2 },   // Toroidal angle
    { "rho_parallel", 3 },   // Parallel Larmor radius [m]
    { "w1",           4 },   // Grid weight 1
    { "w2",           5 },   // Grid weight 2
    { "mu",           6 },   // Magnetic moment
    { "w0",           7 },   // Grid weight
    { "f0",           8 }
}; // Grid distribution function

map< string, double > constants_map;
XGCBFieldInterpolator bFieldInterpolator;
XGCPsinInterpolator psinInterpolator;
Vec2< float > poloidal_center;

void readBPParticleDataStep(
    vector< float > & result,
    const string & ptype,
    const string & path,
    int rank,
    int nRanks )
{
    ADIOS_FILE * f = adios_read_open_file ( path.c_str(), ADIOS_READ_METHOD_BP, MPI_COMM_WORLD );

    if (f == NULL) 
    {
        cout << adios_errmsg() << endl;
        exit( 1 );
    }
    
    ADIOS_VARINFO * v = adios_inq_var ( f, "iphase" );

    const uint64_t SZ = v->dims[ 0 ];
    const uint64_t CS = SZ / nRanks;
    const uint64_t MY_START = rank * CS;
    const uint64_t MY_LAST = rank < nRanks-1 ? MY_START + CS - 1 : SZ - 1;
    const uint64_t MY_CHUNKSIZE = MY_LAST - MY_START + 1;

    uint64_t start[ 2 ] = { MY_START,        0 }; 
    uint64_t count[ 2 ] = { MY_CHUNKSIZE,    9 };

    cout << "my start " << MY_START << " my size " << MY_CHUNKSIZE << " total " << SZ << endl;

    ADIOS_SELECTION * selection = adios_selection_boundingbox( v->ndim, start, count );

    vector< double > tmp( SZ * 9 );
    
    adios_schedule_read ( f, selection, "iphase", 0, 1, tmp.data() );
    adios_perform_reads ( f, 1 );
    
    result.resize( MY_CHUNKSIZE * 9 );
    for( size_t pIDX = 0; pIDX < MY_CHUNKSIZE; ++pIDX )
    {
        for( size_t vIDX = 0; vIDX < 9; ++vIDX )
        {
            result[ vIDX * MY_CHUNKSIZE + pIDX ] = tmp[ ( pIDX ) * 9 + vIDX ];
        }
    }

    adios_selection_delete ( selection );
    adios_free_varinfo ( v );
    adios_read_close ( f );
}

void loadConstants( const string & units_path )
{
    ifstream inFile;
    inFile.open( units_path );
    string line;

    if( ! inFile.is_open() )
    {
        cout << "couldn't open " << units_path << endl;
    }
    while( inFile )
    {
        if ( ! getline( inFile, line ) ) break;
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        line.pop_back();
        stringstream ss( line );
        string name, valueStr;
        getline( ss, name, '=' );
        getline( ss, valueStr );

        constants_map.insert( { name, std::stod( valueStr ) } );
    }

    inFile.close();

    if( constants_map.count( "eq_axis_r" ) <= 0 || constants_map.count( "eq_axis_z" ) <= 0 )
    {
        cout << "error: missing eq_axis_r, and eq_axis_z, need to compute poloidal angles, these constants should be in units.m";
    }
    else
    {
        poloidal_center.x( constants_map.at( "eq_axis_r" ) );
        poloidal_center.y( constants_map.at( "eq_axis_z" ) );
    }
}

void writeSummaryGrid(
    const SummaryGrid & summaryGrid,
    const string & ptype,
    const string & outpath )
{
    const int64_t N_CELLS = summaryGrid.probes.r.size();

    string filepath = outpath + "/" + ptype + ".summary.grid.dat";
    ofstream outFile( filepath, ios::out | ios::binary );
    outFile.write( (char*) summaryGrid.probes.r.data(), sizeof( float ) * N_CELLS );
    outFile.write( (char*) summaryGrid.probes.z.data(), sizeof( float ) * N_CELLS );
    outFile.write( (char*) summaryGrid.probes.psin.data(), sizeof( float ) * N_CELLS );
    outFile.write( (char*) summaryGrid.probes.poloidalAngle.data(), sizeof( float ) * N_CELLS );
    outFile.write( (char*) summaryGrid.probes.B.data(), sizeof( float ) * N_CELLS );
    outFile.write( (char*) summaryGrid.probes.volume.data(), sizeof( float ) * N_CELLS );
    outFile.close();

    writeTriangularMeshObj( summaryGrid.probes.r, summaryGrid.probes.z, summaryGrid.probeTriangulation, outpath + "/" + ptype + ".summary.grid.delaunay.obj" );

    outFile.open( outpath + "/" + ptype + ".summary.meta.txt" );
    outFile << "delta_v "    << to_string( SummaryStep::DELTA_V ) << "\n";
    outFile << "vpara_bins " << to_string( SummaryStep::NC      ) << "\n";
    outFile << "vperp_bins " << to_string( SummaryStep::NR      ) << "\n";
    outFile << "num_cells  " << to_string( N_CELLS              ) << "\n";
    // partNormFactor ??
    // particle_ratio ??
    outFile.close();

    // neighborhoods

    int64_t num_neighbors = 0;
    std::vector< int64_t > nOffsets( N_CELLS );
    for( int64_t i = 0; i < N_CELLS; ++i )
    {
        num_neighbors += summaryGrid.neighborhoods[ i ].size();
        nOffsets[ i ] = num_neighbors;
    }
    std::vector< int64_t > nh( num_neighbors );
    for( int64_t i = 0, k = 0; i < N_CELLS; ++i )
    {
        for( int j = 0; j < ( i == 0 ? nOffsets[ i ] : nOffsets[ i ] - nOffsets[ i - 1 ] ); ++j )
        {
            nh[ k++ ] = summaryGrid.neighborhoods[ i ][ j ];
        }
    }

    outFile.open( outpath + "/" + ptype + ".summary.grid.neighbors.dat", ios::out | ios::binary );
    outFile.write( (char*) nh.data(), sizeof( int64_t ) * num_neighbors );
    outFile.close();

    outFile.open( outpath + "/" + ptype + ".summary.grid.neighbor.counts.dat", ios::out | ios::binary );
    outFile.write( (char*) nOffsets.data(), sizeof( int64_t ) * N_CELLS );
    outFile.close();
}

void computeSummaryStep(
    SummaryStep & summaryStep,
    const SummaryGrid    & summaryGrid,
    const XGCGridBuilder & gridBuilder,
    int64_t st,
    const string & ptype,
    const string & particle_base_path,
    const string & units_path,
    int rank,
    int nRanks,
    TN::KdTreeSearch2D & kdTree )
{
    // need r, z, phi, B, mu, rho_parallel, w0, w1
    vector< float > B, psin;

    cout << particle_base_path << endl;
    string tstep = to_string( st );

    vector< float > iphase;
    high_resolution_clock::time_point readStartTime = high_resolution_clock::now();
    readBPParticleDataStep(
        iphase,
        ptype,
        particle_base_path + "xgc.restart." + string( 5 - tstep.size(), '0' ) + tstep +  ".bp",
        rank,
        nRanks );
    high_resolution_clock::time_point readStartEnd = high_resolution_clock::now();
    cout << "RANK: " << rank
         << ", adios Read time took "
         << duration_cast<milliseconds>( readStartEnd - readStartTime ).count()
         << " milliseconds " << " for " << iphase.size()/9 << " particles" << endl;

    const size_t SZ = iphase.size() / 9;
    const size_t R_POS = attrKeyToPhaseIndex.at( "r" ) * SZ;
    const size_t Z_POS = attrKeyToPhaseIndex.at( "z" ) * SZ;
    const size_t RHO_POS = attrKeyToPhaseIndex.at( "rho_parallel" ) * SZ;
    const size_t W1_POS = attrKeyToPhaseIndex.at( "w1" ) * SZ;        
    const size_t W0_POS = attrKeyToPhaseIndex.at( "w0" ) * SZ;
    const size_t MU_POS = attrKeyToPhaseIndex.at( "mu" ) * SZ;       

    // get b mapped to particles from field
    B.resize( SZ );
    for( size_t i = 0; i < SZ; ++i )
    {
        Vec3< double > b = bFieldInterpolator( Vec2< double >( iphase[ R_POS + i ], iphase[ Z_POS + i ] ) );
        B[ i ] = sqrt( b.x()*b.x() + b.y()*b.y() + b.z()*b.z() );
    }

    // get psi_n mapped from the field
    psin.resize( SZ );
    for( size_t i = 0; i < SZ; ++i )
    {
        psin[ i ] = psinInterpolator( Vec2< double >( iphase[ R_POS + i ], iphase[ Z_POS + i ] ) );
    }

    // compute velocity and weight
    vector< float > vpara( SZ );
    vector< float > vperp( SZ );
    vector< float >  w0w1( SZ );

    #pragma omp parallel for simd
    for( size_t i = 0; i < SZ; ++i )
    {
        w0w1[  i ] = iphase[ W0_POS + i ] * iphase[ W1_POS + i ];
    }

    const double mass_ratio = 1000.0;
    const double ATOMIC_MASS_UNIT = 1.660539040e-27;
    const double ptl_ion_charge_eu = constants_map.at( "ptl_ion_charge_eu" );
    const double mi_sim = constants_map.at( "ptl_ion_mass_au" ) * ATOMIC_MASS_UNIT;
    const double me_sim = mi_sim / mass_ratio;
    const double e = 1.609e-19;

    if( ptype == "ions")
    {
        #pragma omp parallel for simd
        for( size_t i = 0; i < SZ; ++i )
        {
            vpara[ i ] = B[ i ] * iphase[ RHO_POS + i ] * ( ( ptl_ion_charge_eu * e ) / mi_sim );
            vperp[ i ] = sqrt( ( iphase[   MU_POS + i ] * 2.0 * B[ i ] ) / mi_sim );
        }
    }
    else
    {
        #pragma omp parallel for simd
        for( size_t i = 0; i < SZ; ++i )
        {
            vpara[ i ] =( B[ i ] * iphase[ RHO_POS + i ] * (-e) ) / me_sim;
            vperp[ i ] = sqrt( ( iphase[    MU_POS + i ] * 2.0 * B[ i ]  ) / me_sim  );
        }
    }

    // compute summations over particles in each cell

    high_resolution_clock::time_point st1 = high_resolution_clock::now();

    const int NR = SummaryStep::NR;
    const int NC = SummaryStep::NC;
    const int64_t DIST_STRIDE = NR*NC;

    const size_t N_CELLS = summaryGrid.probes.volume.size();

    summaryStep.w0w1_mean            = std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_rms             = std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_min             = std::vector< float >( N_CELLS,  numeric_limits< float >::max() );
    summaryStep.w0w1_max             = std::vector< float >( N_CELLS, -numeric_limits< float >::max() );
    summaryStep.num_particles        = std::vector< float >( N_CELLS, 0.f );
    summaryStep.num_mapped           = std::vector< float >( N_CELLS, 0.f );
    summaryStep.velocityDistribution = std::vector< float >( N_CELLS*NR*NC, 0.f );

    // for VTKM nearest neighbors //////////////////////////////////////////////////////////

    vector< int64_t > gridMap;
    high_resolution_clock::time_point kdt1 = high_resolution_clock::now();
    
    vector< float > r( SZ );
    vector< float > z( SZ );
    for( size_t i = 0; i < SZ; ++i )
    {
        r[ i ] = iphase[ R_POS + i ];
        z[ i ] = iphase[ Z_POS + i ];
    }

    kdTree.run( gridMap, r, z );

    high_resolution_clock::time_point kdt2 = high_resolution_clock::now();
    cout << "RANK: " << rank
         << ", MPI kdtree mapping CHUNK took "
         << duration_cast<milliseconds>( kdt2 - kdt1 ).count()
         << " milliseconds " << " for " << r.size() << " particles" << endl;

    /////////////////////////////////////////////////////////////////////////////////////////

    #pragma omp simd
    for( size_t i = 0; i < SZ; ++i )
    {
        int64_t index = gridMap[ i ];
        //int64_t index = gridBuilder.nearestNeighborIndex( Vec2< double >( r[ i ], z[ i ] ) );
        
        if( index >= 0 )
        {
            summaryStep.w0w1_mean[ index ] += w0w1[ i ];
            summaryStep.w0w1_rms[  index ] += w0w1[ i ] * w0w1[ i ];
            summaryStep.w0w1_min[  index ] = min( w0w1[ i ], summaryStep.w0w1_min[ index ] );
            summaryStep.w0w1_max[  index ] = max( w0w1[ i ], summaryStep.w0w1_max[ index ] );
            summaryStep.num_particles[ index ] += 1.0;

            // map to velocity distribution bin

            const  float VPARA_MIN = -SummaryStep::DELTA_V;
            const  float VPARA_MAX =  SummaryStep::DELTA_V;

            const  float VPERP_MIN = 0;
            const  float VPERP_MAX = SummaryStep::DELTA_V - VPERP_MIN;

            const float R_WIDTH = VPERP_MAX;
            const float C_WIDTH = VPARA_MAX - VPARA_MIN;

            int r = floor( ( ( vperp[ i ] - VPERP_MIN ) / R_WIDTH ) * NR );
            int c = floor( ( ( vpara[ i ] - VPARA_MIN ) / C_WIDTH ) * NC );

            r = max( min( r, NR - 1 ), 0 );
            c = max( min( c, NC - 1 ), 0 );

            summaryStep.num_mapped[ index ] += 1.0;
            summaryStep.velocityDistribution[ index * DIST_STRIDE + r * NC + c ] += w0w1[ i ];
        }
    }

    high_resolution_clock::time_point st2 = high_resolution_clock::now();
    cout << "RANK: " << rank << ", MPI summarization processing CHUNK took " << duration_cast<milliseconds>( st2 - st1 ).count() << " milliseconds " << " for " << r.size() << " particles" << endl;
}

void writeSummaryStep(
    const SummaryStep & summaryStep,
    const string & ptype,
    int64_t tstep,
    double realtime,
    const string & outpath,
    bool append )
{
    string summary_path  = outpath + "/" + ptype + ".summary.dat";
    string tsteps_path   = outpath + "/tsteps.dat";
    string realtime_path = outpath + "/realtime.dat";

    if( ! append )
    {
        ofstream ofs;
        ofs.open( summary_path, ofstream::out | ofstream::trunc);
        ofs.close();

        ofs.open( tsteps_path, ofstream::out | ofstream::trunc);
        ofs.close();

        ofs.open( realtime_path, ofstream::out | ofstream::trunc);
        ofs.close();
    }

    // write results to disk
    ofstream outFile( summary_path, ios::binary | ios_base::app );
    outFile.write( (char*) summaryStep.velocityDistribution.data(), sizeof( float ) * summaryStep.velocityDistribution.size() );
    outFile.write( (char*) summaryStep.w0w1_mean.data(), sizeof( float ) * summaryStep.w0w1_mean.size() );
    outFile.write( (char*) summaryStep.w0w1_rms.data(), sizeof( float )  * summaryStep.w0w1_rms.size() );
    outFile.write( (char*) summaryStep.w0w1_min.data(), sizeof( float )  * summaryStep.w0w1_min.size() );
    outFile.write( (char*) summaryStep.w0w1_max.data(), sizeof( float )  * summaryStep.w0w1_max.size() );
    outFile.write( (char*) summaryStep.num_particles.data(), sizeof( float ) * summaryStep.num_particles.size() );
    outFile.write( (char*) summaryStep.num_mapped.data(), sizeof( float ) * summaryStep.num_mapped.size() );
    outFile.close();

    const int64_t sim_step = tstep;
    ofstream tsFile(  tsteps_path, ios::binary | ios_base::app );
    tsFile.write( (char*) & sim_step, sizeof( sim_step ) );
    tsFile.close();

    ofstream realTimeFile( realtime_path, ios::binary | ios_base::app );
    realTimeFile.write( (char*) & realtime, sizeof( realtime ) );
}

int main( int argc, char** argv )
{
    if( argc != 6 )
    {
        cerr << "expected: <executable> <mesh path> <bfield path> <particle data base path> <units.m path> <outpath>\n";
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
 
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const string ptype = "ions";

    const string meshpath                = argv[ 1 ];
    const string bfieldpath              = argv[ 2 ];
    const string particle_data_base_path = argv[ 3 ];
    const string units_path              = argv[ 4 ];
    const string outpath                 = argv[ 5 ];

    SummaryGrid    summaryGrid;
    XGCGridBuilder gridBuilder;
    float testVal = 1.0 / nRanks;

    loadConstants( units_path );

    readMeshBP( summaryGrid, poloidal_center, meshpath, bfieldpath );

    cout << "Done reading XGC mesh.\n";

    // get neighborhoods
 
    if( rank == 0 )
    {
        const int64_t N_CELLS = summaryGrid.probes.r.size();
        summaryGrid.neighborhoods.resize(  N_CELLS );

        for( int64_t i = 0; i < N_CELLS; ++i )
        {
            summaryGrid.neighborhoods[ i ].reserve( 6 );
        }

        const int64_t N_TRIANGLES = summaryGrid.probeTriangulation.size();
        for( int i = 0; i < N_TRIANGLES; ++i )
        {
            auto & tri = summaryGrid.probeTriangulation[ i ];
            bool in[ 3 ] = { false, false, false };

            for( int p = 0; p < 3; ++p )
            {
                for( int j = 0; j < 3; ++j )
                {
                    if( p != j )
                    {
                        for( int k = 0; k < summaryGrid.neighborhoods[ tri[ p ] ].size(); ++k )
                        {
                            if( summaryGrid.neighborhoods[ tri[ p ] ][ k ] == tri[ j ] )
                            {
                                in[ j ] = true;
                                break;
                            }
                        }    
                        if( ! in[ j ] )
                        {
                            summaryGrid.neighborhoods[ tri[ p ] ].push_back( tri[ j ] );
                        }
                    }
                }
            }
        }

        cout << "Computed neighborhoods\n";
    }

    psinInterpolator.initializeBP(   meshpath );

    cout << "Psin interpolators initialized.\n";

    bFieldInterpolator.initializeBP( meshpath, bfieldpath );

    cout << "BField interpolators initialized.\n";

    gridBuilder.set( summaryGrid.probes.r, summaryGrid.probes.z );

    if( rank == 0 )
    {
        gridBuilder.save( outpath, ptype );
        writeSummaryGrid( summaryGrid, ptype, outpath  );
        cout << "Delaunay and Voronoi saved to disk.\n";
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    cout << "Building KDTree\n";

    TN::KdTreeSearch2D kdTree;
    kdTree.setGrid( summaryGrid.probes.r, summaryGrid.probes.z );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // cout << "Building summarization grid.\n";
    // gridBuilder.grid( summaryGrid, psinInterpolator, bFieldInterpolator, poloidal_center );
    // cout << "Summary mesh constuction complete. ";

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    const int64_t FRST = 200;
    const int64_t LAST = 400;
    const int64_t STRD = 200;

    double summaryStepTime = 0.0;

    SummaryStep summaryStep;
    for( int64_t tstep = FRST; tstep <= LAST; tstep += STRD )
    {
        // cout << "Summarizing time step " << tstep << ". ";
        high_resolution_clock::time_point st1 = high_resolution_clock::now();

        computeSummaryStep(
            summaryStep,
            summaryGrid,
            gridBuilder,
            tstep,     // simulation tstep that corresponds to file
            ptype,
            particle_data_base_path,
            units_path,
            rank,
            nRanks,
            kdTree );

        high_resolution_clock::time_point st2 = high_resolution_clock::now();

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Reduce(
            rank == 0 ? MPI_IN_PLACE : &testVal,
            &testVal,
            1,
            MPI_FLOAT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD );

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
            rank == 0 ? MPI_IN_PLACE : summaryStep.num_mapped.data(),
            summaryStep.num_mapped.data(),
            summaryStep.num_mapped.size(),
            MPI_FLOAT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD );

        MPI_Barrier(MPI_COMM_WORLD);

        /////////////////////////////////////////////////////////////////////////

        if( rank == 0 )
        {
            cout << "normalizing " << endl;

            const size_t NUM_CELLS = summaryGrid.probes.r.size();
            #pragma omp parallel for simd
            for( size_t i = 0; i < NUM_CELLS; ++i )
            {
                if( summaryStep.num_particles[ i ] > 0 )
                {
                    summaryStep.w0w1_mean[ i ] /= summaryStep.num_particles[ i ];
                    summaryStep.w0w1_rms[  i ]  = sqrt( summaryStep.w0w1_rms[ i ] ) / summaryStep.num_particles[ i ];
                }
                else
                {
                    summaryStep.w0w1_mean[ i ] = 0;
                    summaryStep.w0w1_rms[  i ] = 0;
                    summaryStep.w0w1_min[  i ] = 0;
                    summaryStep.w0w1_max[  i ] = 0;
                }
            }
            double realtime = ( double ) tstep / 2.0;
            writeSummaryStep( summaryStep, ptype, tstep, realtime, outpath, tstep != FRST );
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    const double N_COMPUTED = ( LAST - FRST ) / STRD + 1.0;
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

    if( rank == 0 )
    {
        cout << "Summarization step took " << duration / ( N_COMPUTED ) << " milliseconds (on average) including reduction and file reading.\n";
        cout << testVal << endl;
    }

    adios_read_finalize_method ( ADIOS_READ_METHOD_BP );
    MPI_Finalize();
    return 0;
}