
#include "kdtree/KdTreeSearch2D.hpp"
#include "XGCBFieldInterpolator.hpp"
#include "XGCPsinInterpolator.hpp"
#include "XGCMeshLoader.hpp"
#include "Summary.hpp"
#include "XGCGridBuilder.hpp"
#include "SummaryUtils.hpp"

#include "FieldInterpolator.hpp"
#include "Types/Vec.hpp"

#include <adios2.h>
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

map< string, int > constants_map;
XGCBFieldInterpolator bFieldInterpolator;
XGCPsinInterpolator psinInterpolator;
Vec2< float > poloidal_center;

void readParticleDataStep( vector< float > & result, const string & ptype, const string & attr, const string & path, int rank, int nRanks )
{
    hid_t file_id = H5Fopen( path.c_str() , H5F_ACC_RDONLY, H5P_DEFAULT );
    hid_t group = H5Gopen2( file_id, "/", H5P_DEFAULT );
    hid_t dataset_id = H5Dopen2( group, ptype == "ions" ? "iphase" : "ephase", H5P_DEFAULT );
    hid_t dataspace_id = H5Dget_space ( dataset_id );
    int ndims = H5Sget_simple_extent_ndims( dataspace_id );
    hsize_t dims[ ndims ];
    H5Sget_simple_extent_dims( dataspace_id, dims, NULL );

    const int64_t SZ = dims[ 0 ];
    const int64_t CS = SZ / nRanks;
    const int64_t MY_START  = rank * CS;
    const int64_t MY_END    = rank < nRanks-1 ? MY_START + CS - 1 : SZ - 1;
    const int64_t MY_CHUNKSIZE = MY_END - MY_START + 1;

    hsize_t offset[ 2 ] = { MY_START, attrKeyToPhaseIndex.at( attr ) };
    hsize_t  count[ 2 ] = { MY_CHUNKSIZE, 1 };
    hsize_t stride[ 2 ] = { 1, 1 };
    hsize_t  block[ 2 ] = { 1, 1 };

    herr_t status = H5Sselect_hyperslab( dataspace_id, H5S_SELECT_SET, offset, stride, count, block );

    hsize_t dimsm = MY_CHUNKSIZE;
    hid_t memspace_id = H5Screate_simple( 1, &dimsm, NULL );

    result.resize( MY_CHUNKSIZE );

    status = H5Dread(  dataset_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, result.data() );
    status = H5Dclose( dataset_id );

    status = H5Gclose(   group );
    status = H5Fclose( file_id );
}

void readBPParticleDataStep(
    vector< float > & result,
    const string & ptype,
    const string & path,
    int rank,
    int nRanks,
    adios2::ADIOS & adios )
{
    adios2::IO bpIO = adios.DeclareIO( path + "." + std::to_string( rank ) );
    adios2::Engine bpReader = bpIO.Open( path, adios2::Mode::Read );
    adios2::Variable<double> iphase = bpIO.InquireVariable< double >("iphase");

    if( iphase )
    {
        auto dims = iphase.Shape();

        const int64_t SZ = dims[ 0 ];
        const int64_t CS = SZ / nRanks;
        const int64_t MY_START = rank * CS;
        const int64_t MY_CHUNKSIZE = MY_LAST - MY_START + 1;

        // iphase.SetSelection(
        // {
        //     { MY_START,     0 },
        //     { MY_CHUNKSIZE, 8 }
        // } );

        vector< double > tmp;
        bpReader.Get( iphase, tmp, adios2::Mode::Sync );

        result.resize( MY_CHUNKSIZE * 8 );
        for( size_t pIDX = 0; pIDX < MY_CHUNKSIZE; ++pIDX )
        {
            for( size_t vIDX = 0; vIDX < 8; ++vIDX )
            {
                result[ vIDX * MY_CHUNKSIZE + pIDX ] = tmp[ ( pIDX + MY_START ) * 9 + vIDX ];
            }
        }
    }
    else
    {
        cerr << "iphase doesn't exist\n";
        exit( 1 );
    }

    bpReader.Close();
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

    // cout << summaryGrid.probes.r[ 0 ] << "\n";
    // cout << summaryGrid.probes.z[ 0 ] << "\n";
    // cout << summaryGrid.probes.psin[ 0 ] << "\n";
    // cout << summaryGrid.probes.poloidalAngle[ 0 ] << "\n";
    // cout << summaryGrid.probes.B[ 0 ] << "\n";
    // cout << summaryGrid.probes.volume[ 0 ] << "\n";

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
    TN::KdTreeSearch2D & kdTree,
    adios2::ADIOS & adios )
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
        nRanks,
        adios );
    high_resolution_clock::time_point readStartEnd = high_resolution_clock::now();
    cout << "RANK: " << rank
         << ", adios Read time took "
         << duration_cast<milliseconds>( readStartEnd - readStartTime ).count()
         << " milliseconds " << " for " << iphase.size()/8 << " particles" << endl;

    const size_t SZ = iphase.size() / 8;
    const size_t R_POS = 0, Z_POS = 2*SZ, RHO_POS = 3*SZ, W1_POS = 4*SZ, W0_POS = 7*SZ, MU_POS = 6*SZ;

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
    summaryStep.w0w1_mean.resize( N_CELLS, 0.f );
    summaryStep.w0w1_rms.resize( N_CELLS, 0.f );
    summaryStep.w0w1_min.resize( N_CELLS,  numeric_limits< float >::max() );
    summaryStep.w0w1_max.resize( N_CELLS, -numeric_limits< float >::max() );
    summaryStep.num_particles.resize( N_CELLS, 0.f );
    summaryStep.num_mapped.resize( N_CELLS, 0.f );
    summaryStep.velocityDistribution.resize( N_CELLS*NR*NC, 0.f );

    // for VTKM nearest neighbors //////////////////////////////////////////////////////////

    // std::vector< int64_t > gridMap;
    // high_resolution_clock::time_point kdt1 = high_resolution_clock::now();
    
    vector< float > r( iphase.begin() + R_POS, iphase.begin() + R_POS + SZ );
    vector< float > z( iphase.begin() + Z_POS, iphase.begin() + Z_POS + SZ );

    // kdTree.run( gridMap, r, z );
    // high_resolution_clock::time_point kdt2 = high_resolution_clock::now();
    // cout << "RANK: " << rank
    //      << ", MPI kdtree mapping CHUNK took "
    //      << duration_cast<milliseconds>( kdt2 - kdt1 ).count()
    //      << " milliseconds " << " for " << r.size() << " particles" << endl;

    /////////////////////////////////////////////////////////////////////////////////////////

    #pragma omp simd
    for( size_t i = 0; i < SZ; ++i )
    {
        //int64_t index = gridMap[ i ];//gridBuilder.nearestNeighborIndex( Vec2< double >( r[ i ], z[ i ] ) );

        int64_t index = gridBuilder.nearestNeighborIndex( Vec2< double >( r[ i ], z[ i ] ) );
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

    adios2::ADIOS adios(MPI_COMM_WORLD, adios2::DebugON);

    // cout << "rank:\t" << rank << "\n";
    if( rank == 0 )
    {
        cout << "nRanks:\t" << nRanks << endl;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    const int64_t FRST = 400;
    const int64_t LAST = 400;
    const int64_t STRD = 200;

    double summaryStepTime = 0.0;

    SummaryStep summaryStep;
    for( int64_t tstep = FRST; tstep <= LAST; tstep += STRD )
    {
        // cout << "Summarizing time step " << tstep << ". ";
        MPI_Barrier(MPI_COMM_WORLD);

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
            kdTree,
            adios );

        high_resolution_clock::time_point st2 = high_resolution_clock::now();

        MPI_Barrier(MPI_COMM_WORLD);
        // cout << "RANK: " << rank << ", MPI summarization step CHUNK took " << duration_cast<milliseconds>( st2 - st1 ).count() << " milliseconds.\n";

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
            const size_t NUM_CELLS = summaryGrid.probes.r.size();
            #pragma omp parallel for simd
            for( size_t i = 0; i < NUM_CELLS; ++i )
            {
                if( summaryStep.num_particles[ i ] > 0 )
                {
                    summaryStep.w0w1_mean[ i ] /= summaryStep.num_particles[ i ];
                    summaryStep.w0w1_rms[ i ]   = sqrt( summaryStep.w0w1_rms[ i ] ) / summaryStep.num_particles[ i ];
                }
            }
            double realtime = ( double ) tstep / 2.0;
            writeSummaryStep( summaryStep, ptype, tstep, realtime, outpath, tstep != FRST );
        }
    }

    const double N_COMPUTED = ( LAST - FRST ) / STRD + 1.0;
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

    if( rank == 0 )
    {
        cout << "Summarization step took " << duration / ( N_COMPUTED ) << " milliseconds (on average) including reduction and file reading.\n";
    }

    MPI_Finalize();
    return 0;
}