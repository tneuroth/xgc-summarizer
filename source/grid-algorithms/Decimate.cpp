
#include "MeshDecimator.hpp"
#include "../Types/Vec.hpp"

#include <vector>
#include <adios_read.h>

#include <iostream>

int main( int argc, char** argv  )
{
	if( argc != 2 )
    {
        std::cerr << "expected: mesh path \n";
    }

    MPI_Comm dummy;

    int err        = adios_read_init_method ( ADIOS_READ_METHOD_BP, dummy, "verbose=3" );
    ADIOS_FILE * f = adios_read_open_file ( argv[ 1 ], ADIOS_READ_METHOD_BP, dummy );

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

    std::vector< TN::Vec2< double > > rz( v->dims[ 0 ] );

    adios_schedule_read ( f, selection, "rz", 0, 1, rz.data() );
    adios_perform_reads ( f, 1 );
    adios_selection_delete ( selection );
    adios_free_varinfo ( v );

    TN::MeshDecimator decimator;
    decimator.set( rz );
    decimator.writeObj( "triangulation.obj");
}