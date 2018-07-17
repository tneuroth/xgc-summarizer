#ifndef TN_MESH_LOADER_HPP
#define TN_MESH_LOADER_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <hdf5.h>

#include "Summary.hpp"
#include "SummaryUtils.hpp"
#include "Types/Vec.hpp"

inline void readMesh(
    TN::SummaryGrid & summaryGrid,
    const TN::Vec2< float > & poloidal_center,
    const std::string & path,
    const std::string & bpath )
{
    std::vector< TN::Vec3< double > > Btmp;
    std::vector< TN::Vec2< double > > tmp2;
    std::vector< double > tmp1;

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

    summaryGrid.probes.r.resize( NS );
    summaryGrid.probes.z.resize( NS );

    for( size_t i = 0; i < NS; ++i )
    {
        summaryGrid.probes.r[ i ] = tmp2[ i ].x();
        summaryGrid.probes.z[ i ] = tmp2[ i ].y();
    }

    /////////////////////////// Read psi ///////////////////////////////////////////////////////////////////////////////////////

    dataset_id = H5Dopen2( file_id, "psi", H5P_DEFAULT );
    tmp1.resize( NS );
    status = H5Dread( dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp1.data() );
    status = H5Dclose( dataset_id );

    summaryGrid.probes.psin.resize( NS );
    const double NM = tmp1.back();
    #pragma omp parallel for simd
    for( int i = 0; i < NS; ++i )
    {
        summaryGrid.probes.psin[ i ] = tmp1[ i ] / NM;
    }


    /////////////////////////// Read volume ///////////////////////////////////////////////////////////////////////////////////////

    dataset_id = H5Dopen2( file_id, "node_vol", H5P_DEFAULT );
    tmp1.resize( NS );
    status = H5Dread( dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp1.data() );
    status = H5Dclose( dataset_id );

    summaryGrid.probes.volume.resize( NS );
    #pragma omp parallel for simd
    for( int i = 0; i < NS; ++i )
    {
        summaryGrid.probes.volume[ i ] = tmp1[ i ];
    }

    /////////////////////////// Compute Poloidal Angle /////////////////////////////////////////////////////////////////////////

    summaryGrid.probes.poloidalAngle.resize( NS );
    #pragma omp parallel for simd
    for( int s = 0; s < NS; ++s )
    {
        summaryGrid.probes.poloidalAngle[ s ] =
            ( TN::Vec2< float >( summaryGrid.probes.r[ s ], summaryGrid.probes.z[ s ] )
              - poloidal_center ).angle( TN::Vec2< float >( 1.0, 0.0 ) );
    }

    ////////////////////////// read connectivity ///////////////////////////////////////////////////////////////////////////////

    dataset_id = H5Dopen2( file_id, "nd_connect_list", H5P_DEFAULT );
    dspace = H5Dget_space( dataset_id );
    ndims = H5Sget_simple_extent_ndims( dspace );
    hsize_t dimsI[ ndims ];
    H5Sget_simple_extent_dims( dspace, dimsI, NULL );
    std::vector< int32_t > indices ( dimsI[ 0 ] * dimsI[ 1 ] );

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

    summaryGrid.probes.B.resize( Btmp.size() );
    #pragma omp parallel for simd
    for( int s = 0; s < NS; ++s )
    {
        summaryGrid.probes.B[ s ] = std::sqrt( Btmp[ s ].x() * Btmp[ s ].x() +  Btmp[ s ].y() * Btmp[ s ].y() + Btmp[ s ].z() * Btmp[ s ].z() );
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // copy to mesh structure

    summaryGrid.probeTriangulation.resize( dimsI[ 0 ] );
    #pragma omp parallel for simd
    for( size_t i = 0; i < dimsI[ 0 ]; ++i )
    {
        summaryGrid.probeTriangulation[ i ][ 0 ] = indices[ i*3 + 0 ];
        summaryGrid.probeTriangulation[ i ][ 1 ] = indices[ i*3 + 1 ];
        summaryGrid.probeTriangulation[ i ][ 2 ] = indices[ i*3 + 2 ];
    }
}

#endif
