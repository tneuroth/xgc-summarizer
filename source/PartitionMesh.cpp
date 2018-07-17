//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*************************************************************************************************************************

Ideas for mesh generation:

1) computation to visualize the local statistical coherencey

2) temporally varying aggregations based on local statistical coherency

3) temporally static aggregation methods based on local statistical coherency

4) fast vs optimal algorithms

5) aim for uniformity, assume existing mesh represents interest, despite local statistical coherency (uniform decimation)

6) combine methods, to get clost to uniform decimation while also using local statistical coherency

7) methods to customize mesh to user interest

8) hierarchical meshes (MLD)

Heuristics:

    - invariance to volume, shape, shift, time

    - approximating parititioning heuristics (such as generating partition)

Evaluation:

    - cross heuristic aggreement

    - maximizing information (in cumulative phase space distribution...)

    - customized or task-driven

    - measures of structural complexity

    - minimize structural complexity, maximize information...
____________________________________________________________________________
____________________________________________________________________________

Ideas for monitoring:

    1) predefine regions of interest, and show zoomed in versions of them, with minimap pointing them out in full overview

    2) How can we best utilize the temporal-histogram isosurfaces?
____________________________________________________________________________
____________________________________________________________________________

Ideas for interaction:

OVERVIEW

1) existing panning window over arbitrary 2D projection

2) static windows warped to static B-field

INSPECTION MODE

1) draw lines in space

2) lines in space-time
    a) based on feature tracking (blobs)
    b) follow particle
    c) based on "energy minimization" (poloidal seam-carving)
        - each grid point goes to nearest grid point at t+1, defining one trajectory per grid point

3)

*************************************************************************************************************************/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <set>

#include <hdf5.h>

#include "MeshLoader.hpp"
#include "../FieldInterpolator.hpp"
#include "../Types/Vec.hpp"

using namespace std;
using namespace TN;

map< string, int > attrKeyToPhaseIndex =
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
FieldInterpolator23 bFieldInterpolator;
FieldInterpolator21 psiFieldInterpolator;
Vec2< float > poloidal_center;
const double VELOCITY_MAX = 3378743.0;

struct Triangle
{
    int64_t indices[ 3 ];
    Triangle() {}
    Triangle( int64_t i1, int64_t i2, int64_t i3 )
    {
        indices[ 0 ] = i1;
        indices[ 1 ] = i2;
        indices[ 2 ] = i3;
    }
    int64_t & operator[] ( int64_t i )
    {
        return indices[ i ];
    }
};


float sign ( Vec2< float > p1, Vec2< float > p2, Vec2< float > p3 )
{
    return ( p1.x() - p3.x() ) * ( p2.y() - p3.y() ) - ( p2.x() - p3.x() ) * ( p1.y() - p3.y() );
}

bool PointInTriangle ( Vec2< float > pt, Vec2< float > v1, Vec2< float > v2, Vec2< float > v3 )
{
    bool b1, b2, b3;

    b1 = sign(pt, v1, v2) < 0.0f;
    b2 = sign(pt, v2, v3) < 0.0f;
    b3 = sign(pt, v3, v1) < 0.0f;

    return ((b1 == b2) && (b2 == b3));
}

struct SearchGrid
{
    Vec2< double > ll;
    Vec2< double > ur;

    int rows;
    int cols;

    vector< vector< int64_t > >  cellLookupTable;

    SearchGrid( const Vec2< double > & _ll, const Vec2< double > & _ur, int _rows, int _cols ) :
        ll( _ll ), ur( _ur ), rows( _rows ), cols( _cols )
    {
        cellLookupTable.resize( rows*cols );
    }

    void mapCells( vector< Triangle > & cells, vector< Vec2< float > > & points )
    {
        const  float X_MIN = ll.x();
        const  float X_MAX = ur.x();

        const  float Y_MIN = ll.y();
        const  float Y_MAX = ur.y();

        const float X_WIDTH = X_MAX - X_MIN;
        const float Y_WIDTH = Y_MAX - Y_MIN;

        const int ROWS = rows;
        const int COLS = cols;

        for( size_t i = 0; i < cells.size(); ++i )
        {
            for( int t = 0; t < 3; ++t )
            {
                int r = std::floor( ( ( points[ cells[ i ][ t ] ].y() - Y_MIN ) / Y_WIDTH ) * ROWS );
                int c = std::floor( ( ( points[ cells[ i ][ t ] ].x() - X_MIN ) / X_WIDTH ) * COLS );
                r = std::max( std::min( r, ROWS - 1 ), 0 );
                c = std::max( std::min( c, COLS - 1 ), 0 );
                cellLookupTable[ r*COLS + c ].push_back( i );
            }
        }
    }

    int64_t probeIndex( const Vec2< float > & point, vector< Triangle > & cells, vector< Vec2< float > > & points )
    {
        const  float X_MIN = ll.x();
        const  float X_MAX = ur.x();

        const  float Y_MIN = ll.y();
        const  float Y_MAX = ur.y();

        const float X_WIDTH = X_MAX - X_MIN;
        const float Y_WIDTH = Y_MAX - Y_MIN;

        const int ROWS = rows;
        const int COLS = cols;

        int r = std::floor( ( ( point.y() - Y_MIN ) / Y_WIDTH ) * ROWS );
        int c = std::floor( ( ( point.x() - X_MIN ) / X_WIDTH ) * COLS );

        r = std::max( std::min( r, ROWS - 1 ), 0 );
        c = std::max( std::min( c, COLS - 1 ), 0 );

        auto & cellIds = cellLookupTable[ r*COLS + c ];

        /// check cells that are in this spatial segment
        for( int64_t i = 0, end = (int64_t) cellIds.size(); i < end; ++i )
        {

            // cerr << "( " << point.x() << ", " << point.y() << "), "
            //      << "( " << points[ cells[ i ][ 0 ] ].x() << ", " << points[ cells[ i ][ 0 ] ].y() << "),  "
            //      << "( " << points[ cells[ i ][ 1 ] ].x() << ", " << points[ cells[ i ][ 1 ] ].y() << "),   "
            //      << "( " << points[ cells[ i ][ 1 ] ].x() << ", " << points[ cells[ i ][ 1 ] ].y() << ")\n";

            if( PointInTriangle( point, points[ cells[ cellIds[ i ] ] [ 0 ] ], points[ cells[ cellIds[ i ] ][ 1 ] ], points[ cells[ cellIds[ i ] ][ 2 ] ] ) )
            {
                return i;
            }
        }

        return -1;
    }
};

struct SummaryCellMeta
{
    float volume;
    float center_r;
    float center_z;
    float center_psin;
    float center_poloidal_angle;
    float center_Bfield_strength;
};

struct LocalSummary
{
    static const int NC = 33;
    static const int NR = 17;

    float velocity_distribution[ NR*NC ];

    float w0w1_mean;
    float w0w1_rms;
    float w0w1_min;
    float w0w1_max;
    float num_particles;
    float num_mapped;

    // float w0w1_coherency;
    // float velocity_coherency;

    //weight distribution ... could use GMM for more compact representation

    LocalSummary() :
        w0w1_mean( 0.0 ),
        w0w1_rms(  0.0 ),
        w0w1_min(  numeric_limits< float >::max() ),
        w0w1_max( -numeric_limits< float >::max() ),
        num_particles( 0.0 ),
        num_mapped( 0.0 )
    {
        for( auto & v : velocity_distribution )
        {
            v = 0.0;
        }
    }
};

void writeMeshObj( vector< Vec2< float > > & probes, vector< Triangle > & mesh, const string & outpath )
{
    ofstream outfile( outpath + "mesh.obj" );
    for( size_t i = 0, end = probes.size(); i < end; ++i )
    {
        outfile << "v " << probes[ i ].x() << " " << probes[ i ].y() << " 0\n";
    }
    outfile << "\n";
    for( size_t i = 0, end = mesh.size(); i < end; ++i )
    {
        outfile << "f";
        for( size_t j = 0; j < 3; ++j )
        {
            outfile << " " << mesh[ i ][ j ] + 1;
        }
        outfile << "\n";
    }
    outfile.close();
}

void readParticleDataStep( vector< float > & result, const string & ptype, const string & attr, const string & path )
{
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
    std::string line;
    while( inFile )
    {
        if ( ! std::getline( inFile, line ) ) break;
        std::istringstream ss( line );
        float value;
        std::string name;
        std::string valueStr;
        ss >> name >> valueStr;

        if( valueStr == "=" )
        {
            ss >> value;
        }
        else if( valueStr[ 0 ] == '=' )
        {
            valueStr.erase( valueStr.begin() );
            value = std::stod( valueStr );
        }
        else
        {
            value = std::stod( valueStr );
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
        std::cout << "error: missing eq_axis_r, and eq_axis_z, need to compute poloidal angles, these constants should be in units.m";
    }
    else
    {
        poloidal_center.x( constants_map.at( "eq_axis_r" ) );
        poloidal_center.y( constants_map.at( "eq_axis_z" ) );
    }
}

void writeSummaryGridAndMeta(
    vector< Vec2< float > > & probes,
    vector< float > & psin_node,
    vector< float > & ptheta_node,
    vector< float > & B_node,
    vector< Triangle > & mesh,
    vector< SummaryCellMeta > & meta,
    const string & ptype,
    const string & outpath )
{

    string filepath = outpath + ptype + ".summary.grid.nodes.dat";
    ofstream outFile( filepath, ios::out | ios::binary );
    outFile.write( (char*) probes.data(), sizeof( Vec2< float > ) * probes.size() );
    outFile.write( (char*) psin_node.data(),   sizeof( float )   *   psin_node.size() );
    outFile.write( (char*) ptheta_node.data(),   sizeof( float ) * ptheta_node.size() );
    outFile.write( (char*) B_node.data(),   sizeof( float ) * B_node.size() );
    outFile.close();

    // write mesh
    outFile.open( outpath + ptype + ".summary.grid.delaunay.dat", ios::out | ios::binary );

    const int64_t N_CELLS = mesh.size();
    const int64_t N_CELL_NODE_INDICES = N_CELLS * 3;
    vector< int64_t > cell_node_indices( N_CELL_NODE_INDICES );

    int64_t nextIDX = 0;
    for( int64_t i = 0; i < N_CELLS; ++i )
    {
        for( size_t j = 0; j < 3; ++j )
        {
            cell_node_indices[ nextIDX++ ] = mesh[ i ][ j ];
        }
    }
    outFile.write( (char*) cell_node_indices.data(), sizeof( int64_t ) * cell_node_indices.size() );
    outFile.close();

    outFile.open( outpath + "summary.meta.txt" );
    outFile << "delta_v "    << to_string( VELOCITY_MAX )     << "\n";
    outFile << "vpara_bins " << to_string( LocalSummary::NC ) << "\n";
    outFile << "vperp_bins " << to_string( LocalSummary::NR ) << "\n";
    outFile << "num_cells  " << to_string( N_CELLS          ) << "\n";
    outFile << "cell_size  " << to_string( sizeof( LocalSummary ) ) << "\n";

    // partNormFactor ??
    // particle_ratio ??

    outFile.close();
}

void computeAndWriteSummarizationStep(
    vector< Vec2< float > > & probes,
    vector< Triangle > & mesh,
    vector< SummaryCellMeta > & meta,
    vector< LocalSummary > & summary,
    SearchGrid & searchGrid,
    const string & tstep,
    const double realtime,
    const string & ptype,
    const string & particle_base_path,
    const string & units_path,
    const string & outpath )
{
    // need r, z, phi, B, mu, rho_parallel, w0, w1
    vector< float >                       r,   z,   mu,   rho_parallel,   w0,   w1, B, psin;
    vector< vector< float > * > ptrs = { &r,  &z,  &mu,  &rho_parallel,  &w0,  &w1  };
    vector< string > keys =            { "r", "z", "mu", "rho_parallel", "w0", "w1" };

    for( size_t i = 0; i < ptrs.size(); ++i )
    {
        readParticleDataStep( *( ptrs[ i ] ), ptype, keys[ i ], particle_base_path + "xgc.particle." + std::string( 5 - tstep.size(), '0' ) + tstep +  ".h5" );
    }

    const size_t SZ = r.size();

    // get b mapped to particles from field
    B.resize( r.size() );
    for( size_t i = 0; i < SZ; ++i )
    {
        Vec3< double > b = bFieldInterpolator.interpLin( Vec2< double >( r[ i ], z[ i ] ) );
        B[ i ] = std::sqrt( b.x()*b.x() + b.y()*b.y() + b.z()*b.z() );
    }

    // get psi_n mapped from the field
    psin.resize( r.size() );
    for( size_t i = 0; i < SZ; ++i )
    {
        psin[ i ] = psiFieldInterpolator.interpLin( Vec2< double >( r[ i ], z[ i ] ) );
    }

    // compute velocity and weight
    std::vector< float > vpara( r.size() );
    std::vector< float > vperp( r.size() );
    std::vector< float >  w0w1( r.size() );

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

    const int NR = LocalSummary::NR;
    const int NC = LocalSummary::NC;

    #pragma omp simd
    for( size_t i = 0; i < SZ; ++i )
    {
        // int64_t index = probeIndex( Vec2< float >( r[ i ], z[ i ] ), mesh, probes, true );

        int64_t index = searchGrid.probeIndex( Vec2< float >( r[ i ], z[ i ] ), probes );
        if( index >= 0 )
        {
            auto & cell_summary = summary[ index ];
            cell_summary.w0w1_mean += w0w1[ i ];
            cell_summary.w0w1_rms  += w0w1[ i ]*w0w1[ i ];
            cell_summary.w0w1_min = std::min( w0w1[ i ], cell_summary.w0w1_min );
            cell_summary.w0w1_max = std::max( w0w1[ i ], cell_summary.w0w1_max );
            cell_summary.num_particles += 1.0;

            // map to velocity distribution bin

            const  float VPARA_MIN = -VELOCITY_MAX;
            const  float VPARA_MAX =  VELOCITY_MAX;

            const  float VPERP_MIN = 0;
            const  float VPERP_MAX = VELOCITY_MAX - VPERP_MIN;

            const float R_WIDTH = VPERP_MAX;
            const float C_WIDTH = VPARA_MAX - VPARA_MIN;

            int r = std::floor( ( ( vperp[ i ] - VPERP_MIN ) / R_WIDTH ) * NR );
            int c = std::floor( ( ( vpara[ i ] - VPARA_MIN ) / C_WIDTH ) * NC );

            r = std::max( std::min( r, NR - 1 ), 0 );
            c = std::max( std::min( c, NC - 1 ), 0 );
            cell_summary.num_mapped += 1.0;
            cell_summary.velocity_distribution[ r*NC + c ] += w0w1[ i ];
        }

        if( i % 1000 == 0 )
        {
            cout << "step: " << i << " complete\n";
        }
    }

    // apply normalization

    const size_t NUM_CELLS = mesh.size();
    #pragma omp parallel for simd
    for( size_t i = 0; i < NUM_CELLS; ++i )
    {
        auto & cell_summary = summary[ i ];
        const double volume = meta[ i ].volume;
        if( cell_summary.num_particles > 0 )
        {
            cell_summary.w0w1_mean /= cell_summary.num_particles;
            cell_summary.w0w1_rms = std::sqrt( cell_summary.w0w1_rms ) / cell_summary.num_particles;
        }
        for( auto & bin : cell_summary.velocity_distribution )
        {
            bin /= volume;
        }
    }

    static bool initialized = false;

    string summary_path  = outpath + ptype + ".summary.dat";
    string tsteps_path   = outpath + ptype + ".tsteps.dat";
    string realtime_path = outpath + ptype + ".realtime.dat";

    if( ! initialized )
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
    ofstream outFile( summary_path, std::ios::binary | std::ios_base::app );
    outFile.write( (char*) summary.data(), sizeof( LocalSummary ) * summary.size() );
    outFile.close();

    const int64_t sim_step = stoll( tstep );
    ofstream tsFile(  tsteps_path, std::ios::binary | std::ios_base::app );
    tsFile.write( (char*) & sim_step, sizeof( sim_step ) );
    tsFile.close();

    ofstream realTimeFile( realtime_path, std::ios::binary | std::ios_base::app );
    realTimeFile.write( (char*) & realtime, sizeof( realtime ) );
    realTimeFile.close();
}

void initializeInterpolators( const string & meshpath, const string & bpath  )
{
    ///////////////////////////////////////////////////////////////////////////////////////////
    //
    //    BField

    std::vector< Vec3< double > > values;
    std::vector< Vec2< double > > rz;

    // Read rz
    hid_t file_id = H5Fopen( meshpath.c_str() , H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset_id = H5Dopen2( file_id, "rz", H5P_DEFAULT );
    hid_t dspace = H5Dget_space( dataset_id );
    int ndims = H5Sget_simple_extent_ndims( dspace );
    hsize_t dims[ ndims ];
    H5Sget_simple_extent_dims( dspace, dims, NULL );
    rz.resize( dims[ 0 ] );
    herr_t status = H5Dread( dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ( double * ) rz.data() );
    status = H5Dclose( dataset_id );
    status = H5Fclose(file_id);

    // Read BField
    file_id = H5Fopen( bpath.c_str() , H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t group = H5Gopen2( file_id, "node_data[0]", H5P_DEFAULT );
    dataset_id = H5Dopen2( group, "values", H5P_DEFAULT );
    dspace = H5Dget_space( dataset_id );
    ndims = H5Sget_simple_extent_ndims( dspace );
    hsize_t v_dims[ ndims ];
    H5Sget_simple_extent_dims( dspace, v_dims, NULL );
    values.resize( v_dims[ 0 ] );
    status = H5Dread( dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data() );
    status = H5Dclose( dataset_id );

    status = H5Gclose( group );
    status = H5Fclose(file_id);

    bFieldInterpolator.set( rz, values );

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Read Psi

    std::vector< double > psi_values;
    file_id = H5Fopen( meshpath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2( file_id, "psi", H5P_DEFAULT );
    H5Sget_simple_extent_dims( dspace, v_dims, NULL );
    psi_values.resize( v_dims[ 0 ] );
    status = H5Dread( dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, psi_values.data() );
    status = H5Dclose( dataset_id );

    status = H5Fclose(file_id);

    const double NM = psi_values.back();
    const int SZ = psi_values.size();
    #pragma omp parallel for simd
    for( int i = 0; i < SZ; ++i )
    {
        psi_values[ i ] /= NM;
    }

    psiFieldInterpolator.set( rz, psi_values );
}

void readMesh( vector< Vec2< float > > & probes, vector< float > & psin, vector< float > & ptheta, vector< float > & B, vector< Triangle > & mesh, vector< SummaryCellMeta > & cell_meta, const string & path, const string & bpath )
{
    probes.clear();
    mesh.clear();
    cell_meta.clear();
    psin.clear();
    ptheta.clear();
    B.clear();

    vector< Vec3< double > > Btmp;
    vector< Vec2< double > > tmp2;
    vector< double > tmp1;

    ////////////////////////   Read rz /////////////////////////////////////////////////////////////////////////////////////////

    hid_t file_id = H5Fopen( path.c_str() , H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset_id = H5Dopen2( file_id, "rz", H5P_DEFAULT );
    hid_t dspace = H5Dget_space( dataset_id );
    int ndims = H5Sget_simple_extent_ndims( dspace );
    hsize_t dims[ ndims ];
    H5Sget_simple_extent_dims( dspace, dims, NULL );
    const int NS = dims[ 0 ] ;
    tmp2.resize( NS );
    herr_t status = H5Dread( dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ( double * ) tmp2.data() );
    status = H5Dclose( dataset_id );

    probes.resize( NS );
    for( size_t i = 0; i < NS; ++i )
    {
        probes[ i ] = Vec2< float >( tmp2[ i ].x(), tmp2[ i ].y() );
    }

    /////////////////////////// Read psi ///////////////////////////////////////////////////////////////////////////////////////

    dataset_id = H5Dopen2( file_id, "psi", H5P_DEFAULT );
    tmp1.resize( NS );
    status = H5Dread( dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp1.data() );
    status = H5Dclose( dataset_id );

    psin.resize( NS );
    const double NM = tmp1.back();
    #pragma omp parallel for simd
    for( int i = 0; i < NS; ++i )
    {
        psin[ i ] = tmp1[ i ] / NM;
    }


    /////////////////////////// Compute Poloidal Angle /////////////////////////////////////////////////////////////////////////

    ptheta.resize( NS );
    #pragma omp parallel for simd
    for( int s = 0; s < NS; ++s )
    {
        ptheta[ s ] = ( Vec2< float >( probes[ s ].x(), probes[ s ].y() ) - poloidal_center ).angle( Vec2< float >( 1.0, 0.0 ) );
    }

    ////////////////////////// read connectivity ///////////////////////////////////////////////////////////////////////////////

    dataset_id = H5Dopen2( file_id, "nd_connect_list", H5P_DEFAULT );
    dspace = H5Dget_space( dataset_id );
    ndims = H5Sget_simple_extent_ndims( dspace );
    hsize_t dimsI[ ndims ];
    H5Sget_simple_extent_dims( dspace, dimsI, NULL );

    cerr << "dims=" << dimsI[ 0 ] << ", " << dimsI[ 1 ] <<  "\n";
    vector< int32_t > indices ( dimsI[ 0 ] * dimsI[ 1 ] );

    status = H5Dread( dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ( int32_t * ) indices.data() );
    status = H5Dclose( dataset_id );

    // // close file
    status = H5Fclose(file_id);


    /////////////////////////// Read bfield ////////////////////////////////////////////////////////////////////////////////////

    file_id = H5Fopen( bpath.c_str() , H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t group = H5Gopen2( file_id, "node_data[0]", H5P_DEFAULT );
    dataset_id = H5Dopen2( group, "values", H5P_DEFAULT );
    dspace = H5Dget_space( dataset_id );
    ndims = H5Sget_simple_extent_ndims( dspace );
    hsize_t b_dims[ ndims ];
    H5Sget_simple_extent_dims( dspace, b_dims, NULL );
    Btmp.resize( b_dims[ 0 ] );
    status = H5Dread( dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Btmp.data() );
    status = H5Dclose( dataset_id );
    status = H5Gclose( group );
    status = H5Fclose(file_id);

    B.resize( Btmp.size() );
    #pragma omp parallel for simd
    for( int s = 0; s < NS; ++s )
    {
        B[ s ] = std::sqrt( Btmp[ s ].x() * Btmp[ s ].x() +  Btmp[ s ].y() * Btmp[ s ].y() + Btmp[ s ].z() * Btmp[ s ].z() );
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // copy to mesh structure
    mesh.resize( dimsI[ 0 ] );
    cell_meta.resize( mesh.size() );

    #pragma omp parallel for simd
    for( size_t i = 0; i < dimsI[ 0 ]; ++i )
    {
        mesh[ i ][ 0 ] = indices[ i*3 + 0 ];
        mesh[ i ][ 1 ] = indices[ i*3 + 1 ];
        mesh[ i ][ 2 ] = indices[ i*3 + 2 ];

        double a = mesh[ i ][ 1 ] - mesh[ i ][ 0 ];
        double b = mesh[ i ][ 2 ] - mesh[ i ][ 1 ];
        double c = mesh[ i ][ 0 ] - mesh[ i ][ 2 ];

        double s = ( a + b + c ) / 2.0;
        double A = std::sqrt( s * ( s - a ) * ( s - b ) * ( s - c ) );
        double radius = ( probes[ mesh[ i ][ 0 ] ].x() + probes[ mesh[ i ][ 1 ] ].x() + probes[ mesh[ i ][ 2 ] ].x()  ) / 3.0;

        cell_meta[ i ].volume = 2 * M_PI * A * radius;
        cell_meta[ i ].center_r = radius;
        cell_meta[ i ].center_z = ( probes[ mesh[ i ][ 0 ] ].y() + probes[ mesh[ i ][ 1 ] ].y() + probes[ mesh[ i ][ 2 ] ].y()  ) / 3.0;
        cell_meta[ i ].center_psin = ( psin[ mesh[ i ][ 0 ] ] + psin[ mesh[ i ][ 1 ] ] + psin[ mesh[ i ][ 2 ] ]  ) / 3.0;
        cell_meta[ i ].center_poloidal_angle = ( ptheta[ mesh[ i ][ 0 ] ] + ptheta[ mesh[ i ][ 1 ] ] + ptheta[ mesh[ i ][ 2 ] ]  ) / 3.0;
        cell_meta[ i ].center_Bfield_strength = ( B[ mesh[ i ][ 0 ] ] + B[ mesh[ i ][ 1 ] ] + B[ mesh[ i ][ 2 ] ]  ) / 3.0;
    }
}

int main( int argc, char** argv )
{
    if( argc != 6 )
    {
        cerr << "expected: <executable> <mesh path> <bfield path> <particle data base path> <units.m path> <outpath>\n";
    }

    const string meshpath   = argv[ 1 ];
    const string bfieldpath = argv[ 2 ];
    const string particle_data_base_path = argv[ 3 ];
    const string units_path = argv[ 4 ];
    const string outpath = argv[ 5 ];

    vector< Vec2< float > > probes;
    vector< float > probes_psin;
    vector< float > probes_ptheta;
    vector< float > probes_B;
    vector< Triangle > mesh;
    vector< SummaryCellMeta > cell_meta;
    vector< LocalSummary > summary;


    readMesh( probes, probes_psin, probes_ptheta, probes_B, mesh, cell_meta, meshpath, bfieldpath );
    summary.resize( mesh.size() );


    double rmin =  std::numeric_limits< double >::max();
    double rmax = -std::numeric_limits< double >::max();
    double zmin =  std::numeric_limits< double >::max();
    double zmax = -std::numeric_limits< double >::max();

    for( size_t i = 0, end = probes.size(); i < end; ++i )
    {
        rmin = min( rmin, (double ) probes[ i ].x() );
        rmax = max( rmax, (double ) probes[ i ].x() );
        zmin = min( zmin, (double ) probes[ i ].y() );
        zmax = max( zmax, (double ) probes[ i ].y() );
    }

    cout << "bounding box: r(" << rmin << ", " << rmax << "), z(" << zmin << ", " << zmax << ")\n";
    SearchGrid searchGrid( { rmin, zmin }, { rmax, zmax }, 200, 100 );
    searchGrid.mapCells( mesh, probes );

    initializeInterpolators( meshpath, bfieldpath );
    loadConstants( units_path );

    // ... reduce mesh ...

    // to inspect in blender after reduction would be applied
    writeMeshObj( probes, mesh, outpath );
    writeSummaryGridAndMeta( probes, probes_psin, probes_ptheta, probes_B, mesh, cell_meta, "ions", outpath );
    computeAndWriteSummarizationStep( probes, mesh, cell_meta, summary, searchGrid, "12", 0.5, "ions", particle_data_base_path, units_path, outpath );
}