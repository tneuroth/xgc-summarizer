#ifndef TN_XGC_PSI_INTERPOLATOR
#define TN_XGC_PSI_INTERPOLATOR

#include <fstream>
#include <iostream>
#include <vector>
#include <hdf5.h>

#include "FieldInterpolator.hpp"
#include "Types/Vec.hpp"

struct XGCPsinInterpolator
{
    FieldInterpolator21 interpolator;

    void initializeBP( const std::string & meshpath  )
    {
        std::vector< double > psi_values;
        std::vector< TN::Vec2< double > > rz;

        ADIOS_FILE * f = adios_read_open_file ( meshpath.c_str(), ADIOS_READ_METHOD_BP, MPI_COMM_WORLD );

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
        
        rz.resize( SZ );
        adios_schedule_read ( f, selection, "rz", 0, 1, rz.data() );
        adios_perform_reads ( f, 1 );
        
        adios_selection_delete ( selection );
        adios_free_varinfo ( v );

        // psi

        psi_values.resize( SZ );
        uint64_t startPsi = 0;
        uint64_t countPsi = SZ;
        selection = adios_selection_boundingbox( 1, &startPsi, &countPsi );
        adios_schedule_read ( f, selection, "psi", 0, 1, psi_values.data() );
        adios_perform_reads ( f, 1 );
        
        const double NM = psi_values.back();
        #pragma omp parallel for simd
        for( int i = 0; i < SZ; ++i )
        {
            psi_values[ i ] /= NM;
        }

        adios_selection_delete ( selection );
        adios_read_close ( f );

        // set

        interpolator.set( rz, psi_values );
    }

    void initialize( const std::string & meshpath  )
    {
        std::vector< TN::Vec2< double > > rz;

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

        ////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Read Psi

        std::vector< double > psi_values;
        file_id = H5Fopen( meshpath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        dataset_id = H5Dopen2( file_id, "psi", H5P_DEFAULT );
        dspace = H5Dget_space( dataset_id );
        ndims = H5Sget_simple_extent_ndims( dspace );
        hsize_t v_dims[ ndims ];
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

        interpolator.set( rz, psi_values );
    }

    double operator() ( const TN::Vec2< double > & p )
    {
        return interpolator.interpLin( p );
    }
};

#endif