#ifndef TN_MESH_LOADER_HPP
#define TN_MESH_LOADER_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <set>

#include "adios_read.h"

#include "Summary.hpp"
#include "Types/Vec.hpp"

namespace TN
{

template < typename ValueType >
inline void getNeighborhoods( TN::SummaryGrid< ValueType > & summaryGrid )
{
    const int64_t N_CELLS = summaryGrid.variables.at( "r" ).size();
    std::vector< std::set< int64_t > > neighborSets( N_CELLS );
    const int64_t N_TRIANGLES = summaryGrid.triangulation.size();

    for( int i = 0; i < N_TRIANGLES; ++i )
    {
        auto & tri = summaryGrid.triangulation[ i ];
        for( int p = 0; p < 3; ++p )
        {
            for( int j = 0; j < 3; ++j )
            {
                if( p != j )
                {
                    neighborSets[ tri[ p ] ].insert( tri[ j ] );
                }
            }
        }
    }

    int64_t num_neighbors = 0;
    summaryGrid.neighborhoodSums.resize( N_CELLS );

    for( int64_t i = 0; i < N_CELLS; ++i )
    {
        num_neighbors += neighborSets[ i ].size();
        summaryGrid.neighborhoodSums[ i ] = num_neighbors;
    }

    summaryGrid.neighborhoods.resize( num_neighbors );

    int64_t k = 0;
    for( int64_t i = 0; i < N_CELLS; ++i )
    {
        for( auto id : neighborSets[ i ] )
        {
            summaryGrid.neighborhoods[ k++ ] = id;
        }
    }
}

template < typename ValueType >
inline void readMeshBP(
    TN::SummaryGrid< ValueType > & summaryGrid,
    const TN::Vec2< ValueType > & poloidal_center,
    const std::string & path,
    const std::string & bpath )
{
    std::vector< TN::Vec3< double > > Btmp;
    std::vector< TN::Vec2< double > > tmp2;
    std::vector< double > tmp1;

    ////////////////////////   Read rz //////////////////////////////////////////////////////////

    ADIOS_FILE * f = adios_read_open_file ( path.c_str(), ADIOS_READ_METHOD_BP, MPI_COMM_SELF );

    if (f == NULL)
    {
        std::cout << adios_errmsg() << std::endl;
        exit( 1 );
    }

    ADIOS_VARINFO * v = adios_inq_var ( f, "rz" );
    uint64_t SZ = v->dims[ 0 ];

    uint64_t start[2] = { 0,               0 };
    uint64_t count[2] = { v->dims[ 0 ],    2 };

    ADIOS_SELECTION * selection = adios_selection_boundingbox( v->ndim, start, count );

    tmp2.resize( SZ );

    adios_schedule_read ( f, selection, "rz", 0, 1, tmp2.data() );
    adios_perform_reads ( f, 1 );

    summaryGrid.variables.insert( { "r", std::vector< ValueType >( SZ ) } );
    summaryGrid.variables.insert( { "z", std::vector< ValueType >( SZ ) } );

    auto & r = summaryGrid.variables.at( "r" );
    auto & z = summaryGrid.variables.at( "z" );

    for( size_t i = 0; i < SZ; ++i )
    {
        r[ i ] = tmp2[ i ].x();
        z[ i ] = tmp2[ i ].y();
    }

    adios_selection_delete ( selection );
    adios_free_varinfo ( v );

    ////////////////////////////////////////////////////////////////////////////////////////////

    //  Read psi

    v = adios_inq_var ( f, "psi" );

    SZ = v->dims[ 0 ];
    uint64_t startPsi = 0;
    uint64_t countPsi = SZ;
    selection = adios_selection_boundingbox( v->ndim, &startPsi, &countPsi );
    tmp1.resize( SZ );
    adios_schedule_read ( f, selection, "psi", 0, 1, tmp1.data() );
    adios_perform_reads ( f, 1 );

    summaryGrid.variables.insert( { "psin", std::vector< ValueType >( SZ ) } );
    auto & psin = summaryGrid.variables.at( "psin" );

    const double NM = tmp1.back();

    #pragma omp parallel for
    for( int i = 0; i < SZ; ++i )
    {
        psin[ i ] = tmp1[ i ] / NM;
    }

    adios_free_varinfo ( v );

    ////////////////////////////////////////////////////////////////////////////////////////////

    // Node Vol

    adios_schedule_read ( f, selection, "node_vol", 0, 1, tmp1.data() );
    adios_perform_reads ( f, 1 );

    summaryGrid.variables.insert( { "volume", std::vector< ValueType >( SZ ) } );
    auto & volume = summaryGrid.variables.at( "volume" );

    #pragma omp parallel for
    for( int i = 0; i < SZ; ++i )
    {
        volume[ i ] = tmp1[ i ];
    }

    adios_selection_delete ( selection );

    /////////////////////////// Compute Poloidal Angle /////////////////////////////////////////

    summaryGrid.variables.insert( { "poloidal_angle", std::vector< ValueType >( SZ ) } );
    auto & poloidal_angle = summaryGrid.variables.at( "poloidal_angle" );

    #pragma omp parallel for
    for( int s = 0; s < SZ; ++s )
    {
        poloidal_angle[ s ] =
            ( TN::Vec2< ValueType >( r[ s ], z[ s ] )
              - poloidal_center ).angle( TN::Vec2< ValueType >( 1.0, 0.0 ) );
    }

    ////////////////////////////////////////////////////////////////////////////////////////////

    // connectivity

    v = adios_inq_var ( f, "nd_connect_list" );
    SZ = v->dims[ 0 ];
    uint64_t startConn[ 2 ] = { 0,  0 };
    uint64_t countConn[ 2 ] = { SZ, 3 };
    selection = adios_selection_boundingbox( v->ndim, startConn, countConn );
    std::vector< int32_t > indices( SZ * 3  );
    adios_schedule_read ( f, selection, "nd_connect_list", 0, 1, ( int32_t * ) indices.data() );
    adios_perform_reads ( f, 1 );

    adios_selection_delete ( selection );
    adios_free_varinfo ( v );
    adios_read_close ( f );

    ////////////////////////////////////////////////////////////////////////////////////////////

    // copy to mesh structure

    summaryGrid.triangulation.resize( SZ );
    #pragma omp parallel for
    for( size_t i = 0; i < SZ; ++i )
    {
        summaryGrid.triangulation[ i ][ 0 ] = indices[ i*3 + 0 ];
        summaryGrid.triangulation[ i ][ 1 ] = indices[ i*3 + 1 ];
        summaryGrid.triangulation[ i ][ 2 ] = indices[ i*3 + 2 ];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////

    // BField

    f = adios_read_open_file ( bpath.c_str(), ADIOS_READ_METHOD_BP, MPI_COMM_SELF );

    if (f == NULL)
    {
        std::cout << adios_errmsg() << std::endl;
        exit( 1 );
    }

    v = adios_inq_var ( f, "/node_data[0]/values" );
    SZ = v->dims[ 0 ];

    uint64_t startB[2] = { 0,  0 };
    uint64_t countB[2] = { SZ, 3 };

    selection = adios_selection_boundingbox( v->ndim, startB, countB );

    std::vector< double > tmp3( SZ * 3 );

    adios_schedule_read ( f, selection, "/node_data[0]/values", 0, 1, tmp3.data() );
    adios_perform_reads ( f, 1 );

    summaryGrid.variables.insert( { "B", std::vector< ValueType >( SZ ) } );
    auto & B = summaryGrid.variables.at( "B" );

    #pragma omp parallel for
    for( int s = 0; s < SZ; ++s )
    {
        B[ s ] = std::sqrt(
                     tmp3[ s*3   ]*tmp3[ s*3   ] +
                     tmp3[ s*3+1 ]*tmp3[ s*3+1 ] +
                     tmp3[ s*3+2 ]*tmp3[ s*3+2 ] );
    }

    adios_selection_delete ( selection );
    adios_free_varinfo ( v );
    adios_read_close ( f );

    getNeighborhoods( summaryGrid );
}

}

#endif
