#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <set>

#include <hdf5.h>

#include "XGCBFieldInterpolator.hpp"
#include "XGCPsinInterpolator.hpp"
#include "XGCMeshLoader.hpp"
#include "Summary.hpp"
#include "XGCGridBuilder.hpp"
#include "SummaryUtils.hpp"

#include "../FieldInterpolator.hpp"
#include "../Types/Vec.hpp"

using namespace std;
using namespace TN;

map< string, int > constants_map;

XGCBFieldInterpolator bFieldInterpolator;
XGCPsinInterpolator psinInterpolator;
Vec2< float > poloidal_center;

void readParticleDataStep( vector< float > & result, const string & ptype, const string & attr, const string & path )
{
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


    hid_t file_id = H5Fopen( path.c_str() , H5F_ACC_RDONLY, H5P_DEFAULT );
    hid_t group = H5Gopen2( file_id, "/", H5P_DEFAULT );
    hid_t dataset_id = H5Dopen2( group, ptype == "ions" ? "iphase" : "ephase", H5P_DEFAULT );
    hid_t dataspace_id = H5Dget_space ( dataset_id );
    int ndims = H5Sget_simple_extent_ndims( dataspace_id );
    hsize_t dims[ ndims ];
    H5Sget_simple_extent_dims( dataspace_id, dims, NULL );

    hsize_t offset[ 2 ];
    hsize_t  count[ 2 ];
    hsize_t stride[ 2 ];
    hsize_t  block[ 2 ];

    offset[ 0 ] = 0;
    offset[ 1 ] = attrKeyToPhaseIndex.at( attr );

    count[ 0 ]  = dims[ 0 ];
    count[ 1 ]  = 1;

    stride[ 0 ] = 1;
    stride[ 1 ] = 1;

    block[ 0 ] = 1;
    block[ 1 ] = 1;

    herr_t status = H5Sselect_hyperslab( dataspace_id, H5S_SELECT_SET, offset, stride, count, block );

    hsize_t dimsm[ 1 ];
    dimsm[ 0 ] = dims[ 0 ];
    hid_t memspace_id = H5Screate_simple( 1, dimsm, NULL );

    result.resize( dims[ 0 ] );

    status = H5Dread(  dataset_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, result.data() );
    status = H5Dclose( dataset_id );

    status = H5Gclose(   group );
    status = H5Fclose( file_id );
}

void loadConstants( const string & units_path )
{
    ifstream inFile;
    inFile.open( units_path );
    string line;
    while( inFile )
    {
        if ( ! getline( inFile, line ) ) break;
        istringstream ss( line );
        float value;
        string name;
        string valueStr;
        ss >> name >> valueStr;

        if( valueStr == "=" )
        {
            ss >> value;
        }
        else if( valueStr[ 0 ] == '=' )
        {
            valueStr.erase( valueStr.begin() );
            value = stod( valueStr );
        }
        else
        {
            value = stod( valueStr );
        }

        if( name.back() == '=' )
        {
            name.pop_back();
        }

        constants_map.insert( { name, value } );
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

    cout << summaryGrid.probes.r[ 0 ] << "\n";
    cout << summaryGrid.probes.z[ 0 ] << "\n";
    cout << summaryGrid.probes.psin[ 0 ] << "\n";
    cout << summaryGrid.probes.poloidalAngle[ 0 ] << "\n";
    cout << summaryGrid.probes.B[ 0 ] << "\n";
    cout << summaryGrid.probes.volume[ 0 ] << "\n";


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

    // writePolygonalMeshObj(
    //     summaryGrid.probeBoundaries.r,
    //     summaryGrid.probeBoundaries.z,
    //     outpath + "/" + ptype + ".summary.grid.voronoi.obj" );

    outFile.open( outpath + "/" + ptype + ".summary.meta.txt" );
    outFile << "delta_v "    << to_string( SummaryStep::DELTA_V ) << "\n";
    outFile << "vpara_bins " << to_string( SummaryStep::NC      ) << "\n";
    outFile << "vperp_bins " << to_string( SummaryStep::NR      ) << "\n";
    outFile << "num_cells  " << to_string( N_CELLS              ) << "\n";
    // partNormFactor ??
    // particle_ratio ??
    outFile.close();
}

void computeSummaryStep(
    SummaryStep & summaryStep,
    const SummaryGrid    & summaryGrid,
    const XGCGridBuilder & gridBuilder,
    int64_t st,
    double realtime,
    const string & ptype,
    const string & particle_base_path,
    const string & units_path )
{
    // need r, z, phi, B, mu, rho_parallel, w0, w1
    vector< float >                       r,   z,   mu,   rho_parallel,   w0,   w1, B, psin;
    vector< vector< float > * > ptrs = { &r,  &z,  &mu,  &rho_parallel,  &w0,  &w1  };
    vector< string > keys =            { "r", "z", "mu", "rho_parallel", "w0", "w1" };

    cout << particle_base_path << std::endl;

    string tstep = std::to_string( st );

    for( size_t i = 0; i < ptrs.size(); ++i )
    {
        readParticleDataStep( *( ptrs[ i ] ), ptype, keys[ i ], particle_base_path + "xgc.particle." + string( 5 - tstep.size(), '0' ) + tstep +  ".h5" );
    }

    const size_t SZ = r.size();

    // get b mapped to particles from field
    B.resize( r.size() );
    for( size_t i = 0; i < SZ; ++i )
    {
        Vec3< double > b = bFieldInterpolator( Vec2< double >( r[ i ], z[ i ] ) );
        B[ i ] = sqrt( b.x()*b.x() + b.y()*b.y() + b.z()*b.z() );
    }

    // get psi_n mapped from the field
    psin.resize( r.size() );
    for( size_t i = 0; i < SZ; ++i )
    {
        psin[ i ] = psinInterpolator( Vec2< double >( r[ i ], z[ i ] ) );
    }

    // compute velocity and weight
    vector< float > vpara( r.size() );
    vector< float > vperp( r.size() );
    vector< float >  w0w1( r.size() );

    #pragma omp parallel for simd
    for( size_t i = 0; i < SZ; ++i )
    {
        w0w1[  i ] = w0[ i ] * w1[ i ];
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
            vpara[ i ] = B[ i ] * rho_parallel[ i ] * ( ( ptl_ion_charge_eu * e ) / mi_sim );
            vperp[ i ] = sqrt( ( mu[ i ] * 2.0 * B[ i ] ) / mi_sim );
        }
    }
    else
    {
        #pragma omp parallel for simd
        for( size_t i = 0; i < SZ; ++i )
        {
            vpara[ i ] =( B[ i ] * rho_parallel[ i ] * (-e) ) / me_sim;
            vperp[ i ] = sqrt( ( mu[ i ] * 2.0 * B[ i ]  ) / me_sim  );
        }
    }

    // compute summations over particles in each cell

    const int NR = SummaryStep::NR;
    const int NC = SummaryStep::NC;

    cout << "\n \% complete: " << flush;

    const size_t N_CELLS = summaryGrid.probes.volume.size();
    summaryStep.w0w1_mean.resize( N_CELLS, 0.f );
    summaryStep.w0w1_rms.resize( N_CELLS, 0.f );
    summaryStep.w0w1_min.resize( N_CELLS,  numeric_limits< float >::max() );
    summaryStep.w0w1_max.resize( N_CELLS, -numeric_limits< float >::max() );
    summaryStep.num_particles.resize( N_CELLS, 0.f );
    summaryStep.num_mapped.resize( N_CELLS, 0.f );
    summaryStep.velocityDistribution = std::vector< float[ NR*NC ] >( N_CELLS );
    for( size_t i = 0; i < N_CELLS; ++i )
    {
        for( int b = 0; b < NR*NC; ++b )
        {
            summaryStep.velocityDistribution[ i ][ b ] = 0.f;
        }
    }

    #pragma omp simd
    for( size_t i = 0; i < SZ; ++i )
    {
        int64_t index = gridBuilder.nearestNeighborIndex( Vec2< double >( r[ i ], z[ i ] ) );

        // cout << index << ",";

        if( index >= 0 )
        {
            summaryStep.w0w1_mean[ index ] += w0w1[ i ];
            summaryStep.w0w1_rms[  index ] += w0w1[ i ]*w0w1[ i ];
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
            summaryStep.velocityDistribution[ index ][ r*NC + c ] += w0w1[ i ];
        }

        if( i % ( SZ / 10 ) == 0 )
        {
            cout << ( i / (double) ( SZ - 1 ) )  * 100 << ",";
        }
    }

    // apply normalization

    const size_t NUM_CELLS = summaryGrid.probes.r.size();
    #pragma omp parallel for simd
    for( size_t i = 0; i < NUM_CELLS; ++i )
    {
        const double volume = summaryGrid.probes.volume[ i ];
        if( summaryStep.num_particles[ i ] > 0 )
        {
            summaryStep.w0w1_mean[ i ] /= summaryStep.num_particles[ i ];
            summaryStep.w0w1_rms[ i ]   = sqrt( summaryStep.w0w1_rms[ i ] ) / summaryStep.num_particles[ i ];
        }
    }
}

void writeSummaryStep(
    const SummaryStep & summaryStep,
    const std::string & ptype,
    int64_t tstep,
    double realtime,
    const std::string & outpath,
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
    outFile.write(
        (char*) summaryStep.velocityDistribution.data(),
        sizeof( float ) * ( SummaryStep::NR * SummaryStep::NC ) * summaryStep.velocityDistribution.size() );

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

    const string ptype = "ions";

    const string meshpath   = argv[ 1 ];
    const string bfieldpath = argv[ 2 ];
    const string particle_data_base_path = argv[ 3 ];
    const string units_path = argv[ 4 ];
    const string outpath = argv[ 5 ];

    SummaryGrid    summaryGrid;
    XGCGridBuilder gridBuilder;

    loadConstants( units_path );

    readMesh( summaryGrid, poloidal_center, meshpath, bfieldpath );

    cout << "Done reading XGC mesh.\n";

    bFieldInterpolator.initialize( meshpath, bfieldpath );
    psinInterpolator.initialize( meshpath );

    cout << "Mesh interpolators initialized.\n";

    gridBuilder.set( summaryGrid.probes.r, summaryGrid.probes.z );
    gridBuilder.save( outpath, ptype );

    cout << "Delaunay and Voronoi saved to disk.\n";

    // cout << "Building summarization grid.\n";

    // gridBuilder.grid( summaryGrid, psinInterpolator, bFieldInterpolator, poloidal_center );

    // cout << "Summary mesh constuction complete. ";

    writeSummaryGrid( summaryGrid, ptype, outpath  );

    cout << "Summary mesh written to disk.\n";

    int64_t tstep = 2;
    double realtime = 0.0;
    SummaryStep summaryStep;

    cout << "Summarizing time step " << tstep << ". ";
    computeSummaryStep(
        summaryStep,
        summaryGrid,
        gridBuilder,
        tstep,     // simulation tstep that corresponds to file
        realtime,   // real time
        ptype,
        particle_data_base_path,
        units_path );

    writeSummaryStep( summaryStep, ptype, tstep, realtime, outpath, false );
    writeSummaryStep( summaryStep, ptype, 4, 1.0, outpath, true );


    cout << "Done. Summarization writing to disk. " << ".\n";
}