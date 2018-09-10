template< typename ValueType >
void XGCAggregator< ValueType >::reduceMesh(
    const std::string & reducedMeshFilePath )
{
    std::ifstream meshFile( reducedMeshFilePath );
    std::string line;

    if( ! meshFile.is_open() )
    {
        std::cerr << "couldn't open file: " << reducedMeshFilePath << std::endl;
        exit( 1 );
    }

    SummaryGrid2< ValueType > newGrid;

    while( std::getline( meshFile, line ) )
    {
        if( line.size() > 3 )
        {
            std::stringstream sstr( line );
            std::string v;
            sstr >> v;

            if( v == "v" )
            {
                float x, y, z;
                sstr >> x >> y >> z;
                newGrid.variables.at( "r" ).push_back( x );
                newGrid.variables.at( "z" ).push_back( y );

                //just for now until voronoi cell face area can be calculated.
                newGrid.variables.at( "volume" ).push_back( 1.f );
            }
            else if( v == "f" )
            {
                unsigned int i1, i2, i3;
                sstr >> i1 >> i2 >> i3;
                newGrid.triangulation.push_back(
                {
                    i1 - 1,
                    i2 - 1,
                    i3 - 1
                } );
            }
        }
    }

    MPI_Barrier( MPI_COMM_WORLD );
    std::cout << "getting neighborhoods " << std::endl;
    TN::getNeighborhoods( newGrid );

    std::cout << "Copying over values from old mesh "
              << newGrid.variables.at( "r" ).size()
              << "/" << m_summaryGrid.variables.at( "r" ).size()
              << std::endl;

    newGrid.variables.at( "B" ).resize( newGrid.variables.at( "r" ).size() );
    newGrid.variables.at( "psin" ).resize( newGrid.variables.at( "r" ).size() );
    newGrid.variables.at( "poloidal_angle" ).resize( newGrid.variables.at( "r" ).size() );

    for( int64_t i = 0; i < newGrid.variables.at( "r" ).size(); ++i )
    {
        newGrid.variables.at( "B" )[ i ] = m_summaryGrid.variables.at( "B" )[ i*2 ];
        newGrid.variables.at( "psin" )[ i ] = m_summaryGrid.variables.at( "psin" )[ i*2 ];
        newGrid.variables.at( "poloidal_angle" )[ i ] = m_summaryGrid.variables.at( "poloidal_angle" )[ i*2 ];
    }

    // const int64_t SZ = newGrid.variables.at( "r.size();
    // std::vector< vtkm::Vec< vtkm::Float32, 2 > > pos( SZ );
    // #pragma omp parallel for simd
    // for( int64_t i = 0; i < SZ; ++i )
    // {
    //     pos[ i ] = vtkm::Vec< vtkm::Float32, 2 >( newGrid.variables.at( "r[ i ], newGrid.variables.at( "z[ i ] );
    // }

    // auto posHandle = vtkm::cont::make_ArrayHandle( pos );
    // vtkm::cont::ArrayHandle<vtkm::Id> idHandle;
    // vtkm::cont::ArrayHandle<vtkm::Float32> distHandle;

    // MPI_Barrier( MPI_COMM_WORLD );
    // std::cout << "getting neighbors"
    //           << m_gridHandle.GetNumberOfValues() << " "
    //           << posHandle.GetNumberOfValues()    << " " << std::endl;

    // m_kdTree.Run( m_gridHandle, posHandle, idHandle, distHandle, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
    // std::cout << "finished getting neighbors" << std::endl;

    // if( idHandle.GetNumberOfValues() < SZ )
    // {
    //     std::cout << "wrong number of ids " << std::endl;
    //     exit( 1 );
    // }

    // std::vector< int64_t > ids( SZ );
    // #pragma omp parallel for simd
    // for( int64_t i = 0; i < SZ; ++i )
    // {
    //     ids[ i ] = idHandle.GetPortalConstControl().Get( i );
    // }

    // vtkm::cont::ArrayHandle<vtkm::Float32> fieldResultHandle;

    // // Psi

    // MPI_Barrier( MPI_COMM_WORLD );
    // std::cout << "interpolating psi " << std::endl;

    // auto psinHandle = vtkm::cont::make_ArrayHandle( m_summaryGrid.variables.at( "psin );
    // m_interpolator.run(
    //     posHandle,
    //     idHandle,
    //     m_gridHandle,
    //     psinHandle,
    //     m_gridNeighborhoodsHandle,
    //     m_gridNeighborhoodSumsHandle,
    //     fieldResultHandle,
    //     VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    // if( fieldResultHandle.GetNumberOfValues() < SZ )
    // {
    //     std::cout << "wrong number of psi values " << std::endl;
    //     exit( 1 );
    // }

    // newGrid.variables.at( "psin.resize( SZ );
    // #pragma omp parallel for simd
    // for( int64_t i = 0; i < SZ; ++i )
    // {
    //     newGrid.variables.at( "psin[ i ] = fieldResultHandle.GetPortalConstControl().Get( i );
    // }

    // // B

    // MPI_Barrier( MPI_COMM_WORLD );
    // std::cout << "interpolating B " << std::endl;

    // auto bHandle = vtkm::cont::make_ArrayHandle( m_summaryGrid.variables.at( "B );
    // m_interpolator.run(
    //     posHandle,
    //     idHandle,
    //     m_gridHandle,
    //     bHandle,
    //     m_gridNeighborhoodsHandle,
    //     m_gridNeighborhoodSumsHandle,
    //     fieldResultHandle,
    //     VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    // if( fieldResultHandle.GetNumberOfValues() < SZ )
    // {
    //     std::cout << "wrong number of B values " << std::endl;
    //     exit( 1 );
    // }

    // newGrid.variables.at( "B.resize( SZ );
    // #pragma omp parallel for simd
    // for( int64_t i = 0; i < SZ; ++i )
    // {
    //     newGrid.variables.at( "B[ i ] = fieldResultHandle.GetPortalConstControl().Get( i );
    // }

    // // Poloidal Angle

    // MPI_Barrier( MPI_COMM_WORLD );
    // std::cout << "calculating angles " << std::endl;

    // newGrid.variables.at( "poloidalAngle.resize( SZ );
    // const TN::Vec2< float > poloidal_center = { m_constants.at( "eq_axis_r" ), m_constants.at( "eq_axis_z" ) };
    // #pragma omp parallel for simd
    // for( int64_t i = 0; i < SZ; ++i )
    // {
    //     newGrid.variables.at( "poloidalAngle[ i ] =
    //         ( TN::Vec2< float >( newGrid.variables.at( "r[ i ], newGrid.variables.at( "z[ i ] )
    //           - poloidal_center ).angle( TN::Vec2< float >( 1.0, 0.0 ) );
    // }

    m_summaryGrid = newGrid;

    MPI_Barrier( MPI_COMM_WORLD );
    std::cout << "setting static handles " << std::endl;

    setGrid(
        m_summaryGrid.variables.at( "r" ),
        m_summaryGrid.variables.at( "z" ),
        m_summaryGrid.variables.at( "B" ),
        m_summaryGrid.neighborhoods,
        m_summaryGrid.neighborhoodSums );

    MPI_Barrier( MPI_COMM_WORLD );
    std::cout << "done " << std::endl;
}