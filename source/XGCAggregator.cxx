
#include "XGCAggregator.hpp"
#include "Summary.hpp"
#include "SummaryWriterAdios2.hpp"
#include "XGCMeshReaderAdios1.hpp"
#include "XGCParticleReaderAdios2.hpp"
#include "XGCConstantReader.hpp"
#include "Reduce/Reduce.hpp"
#include "XGCSynchronizer.hpp"

#include <adios2.h>
#include <mpi.h>

#include <vtkm/cont/DeviceAdapterAlgorithm.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vector>
#include <set>
#include <chrono>
#include <exception>

template< typename DeviceAdapter >
void checkDevice(DeviceAdapter)
{
    using DeviceAdapterTraits = vtkm::cont::DeviceAdapterTraits<DeviceAdapter>;
    std::cout << "vtkm is using " << DeviceAdapterTraits::GetName() << std::endl;
}

namespace TN
{

template< typename ValueType >
XGCAggregator< ValueType >::XGCAggregator(
    const std::string & meshFilePath,
    const std::string & bfieldFilePath,
    const std::string & restartPath,
    const std::string & unitsFilePath,
    const std::string & outputDirectory,
    const std::set< std::string > & particleTypes,
    bool inSitu,
    bool splitByBlocks,
    int m_rank,
    int nm_ranks ) : 
        m_meshFilePath( meshFilePath ),
        m_bFieldFilePath( bfieldFilePath ),
        m_restartPath( restartPath ),
        m_unitsMFilePath( unitsFilePath ),
        m_outputDirectory( outputDirectory ),
        m_inSitu( inSitu ),
        m_splitByBlocks( splitByBlocks ),
        m_rank( m_rank ),
        m_nranks( nm_ranks )

{
    TN::loadConstants( m_unitsMFilePath, m_constants );

    std::cout << "reading mesh" << std::endl;

    TN::readMeshBP(
        m_summaryGrid,
        { m_constants.at( "eq_axis_r" ), m_constants.at( "eq_axis_z" ) },
        meshFilePath,
        m_bFieldFilePath ); 
    
    std::cout << "setting grid" << std::endl;

    setGrid(
        m_summaryGrid.variables.at( "r" ),
        m_summaryGrid.variables.at( "z" ),
        m_summaryGrid.variables.at( "B" ),
        m_summaryGrid.neighborhoods,
        m_summaryGrid.neighborhoodSums );

    if( m_rank == 0 )
    {
        checkDevice( VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
    }
}

template< typename ValueType >
void XGCAggregator< ValueType >::writeMesh()
{
    if( m_rank == 0 )
    {
        writeGrid( m_outputDirectory );
    }
}

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
                newGrid.triangulation.push_back( { 
                    i1 - 1,
                    i2 - 1,
                    i3 - 1 } );
            }
        }
    }
 
    MPI_Barrier(MPI_COMM_WORLD);
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

    // MPI_Barrier(MPI_COMM_WORLD);
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
    //     ids[ i ] = idHandle.GetPortalControl().Get( i );
    // }

    // vtkm::cont::ArrayHandle<vtkm::Float32> fieldResultHandle;

    // // Psi

    // MPI_Barrier(MPI_COMM_WORLD);
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
    //     newGrid.variables.at( "psin[ i ] = fieldResultHandle.GetPortalControl().Get( i );
    // }

    // // B

    // MPI_Barrier(MPI_COMM_WORLD);
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
    //     newGrid.variables.at( "B[ i ] = fieldResultHandle.GetPortalControl().Get( i );
    // }

    // // Poloidal Angle

    // MPI_Barrier(MPI_COMM_WORLD);
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

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "setting static handles " << std::endl;

    setGrid(
        m_summaryGrid.variables.at( "r" ),
        m_summaryGrid.variables.at( "z" ),
        m_summaryGrid.variables.at( "B" ),
        m_summaryGrid.neighborhoods,
        m_summaryGrid.neighborhoodSums );

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "done " << std::endl;   
}

template< typename ValueType >
void XGCAggregator< ValueType >::runInSitu()
{    
    const float TIMEOUT = 300.f;
    adios2::ADIOS adios(MPI_COMM_WORLD, adios2::DebugOFF );
    adios2::IO bpIO = adios.DeclareIO( "IO" );

    std::cout << "Waiting for file, RANK: " << m_rank << std::endl;

    TN::Synchro::waitForFileExistence( m_restartPath, TIMEOUT );
    adios2::Engine bpReader = bpIO.Open( m_restartPath, adios2::Mode::Read );

    std::cout << "Found file. Opened. RANK: " << m_rank << std::endl;

    SummaryStep2< ValueType > summaryStep;
    int64_t outputStep = 0;
    
    while( 1 )
    {
        adios2::StepStatus status =
            bpReader.BeginStep(adios2::StepMode::NextAvailable, TIMEOUT );

        if (status == adios2::StepStatus::NotReady)
        {
            std::this_thread::sleep_for(
                    std::chrono::milliseconds( 1000 ) );
            std::cout << "Waiting for step: " << outputStep << ", RANK: " << m_rank << std::endl;
            continue;
        }
        else if (status != adios2::StepStatus::OK)
        {
            std::cout << "step status not OK: " << outputStep << ", RANK: " << m_rank << std::endl;
            break;
        }
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////////

        std::chrono::high_resolution_clock::time_point readStartTime = std::chrono::high_resolution_clock::now();
        
        int64_t simstep;
        double  realtime;

        int64_t totalNumParticles = readBPParticleDataStep(
            m_phase,
            "ions",
            m_restartPath,
            m_rank,
            m_nranks,
            bpIO,
            bpReader,
            simstep,
            realtime,
            true );

        summaryStep.numParticles = totalNumParticles;
        summaryStep.setStep( outputStep, simstep, realtime );
        summaryStep.objectIdentifier = "ions";

        std::chrono::high_resolution_clock::time_point readStartEnd = std::chrono::high_resolution_clock::now();

        std::cout << "RANK: " << m_rank
             << ", adios Read time took "
             << std::chrono::duration_cast<std::chrono::milliseconds>( readStartEnd - readStartTime ).count()
             << " std::chrono::milliseconds " << " for " << m_phase.size()/9 << " particles" << std::endl;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////

        computeSummaryStep(
            m_phase,
            summaryStep,
            "ions" );

        ++outputStep;
    }
}

template< typename ValueType >
void XGCAggregator< ValueType >::runInPost()
{
    double summaryStepTime = 0.0;
    int64_t outputStep     = 0;
    std::vector< int64_t > steps = { 200, 400 };
    SummaryStep2< ValueType > summaryStep;

    for( auto tstep : steps )
    {
        std::string tstepStr = std::to_string( tstep );
        std::chrono::high_resolution_clock::time_point readStartTime = std::chrono::high_resolution_clock::now();
        
        int64_t simstep;
        double  realtime;

        int64_t totalNumParticles = readBPParticleDataStep(
            m_phase,
            "ions",
            m_restartPath + "xgc.restart." + std::string( 5 - tstepStr.size(), '0' ) + tstepStr +  ".bp",
            m_rank,
            m_nranks,
            simstep,
            realtime );

        summaryStep.numParticles = totalNumParticles;
        summaryStep.setStep( outputStep, simstep, realtime );
        summaryStep.objectIdentifier = "ions";

        std::chrono::high_resolution_clock::time_point readStartEnd = std::chrono::high_resolution_clock::now();

        std::cout << "RANK: " << m_rank
             << ", adios Read time took "
             << std::chrono::duration_cast<std::chrono::milliseconds>( readStartEnd - readStartTime ).count()
             << " std::chrono::milliseconds " << " for " << m_phase.size()/9 << " particles" << std::endl;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////

        computeSummaryStep(
            m_phase,
            summaryStep,
            "ions" );

        ++outputStep;
    }
}

template< typename ValueType >
void XGCAggregator< ValueType >::run()
{
    if( m_inSitu )
    {
        runInSitu();
    }
    else
    {
        runInPost();
    }
}

template< typename ValueType >
void XGCAggregator< ValueType >::setGrid(
    const std::vector< ValueType > & r,
    const std::vector< ValueType > & z,
    const std::vector< ValueType > & scalar,
    const std::vector< int64_t > & gridNeighborhoods,
    const std::vector< int64_t > & gridNeighborhoodSums )
{
    const int64_t N_CELLS = r.size();
    m_gridPoints.resize( N_CELLS );

    #pragma omp parallel for simd
    for( int64_t i = 0; i < N_CELLS; ++i )
    {
        m_gridPoints[ i ] = vtkm::Vec< ValueType, 2 >( r[ i ], z[ i ] );
    }

    m_gridHandle = vtkm::cont::make_ArrayHandle( m_gridPoints );

    m_gridNeighborhoods = std::vector< vtkm::Int64 >( gridNeighborhoods.begin(), gridNeighborhoods.end() );
    m_gridNeighborhoodsHandle = vtkm::cont::make_ArrayHandle( m_gridNeighborhoods );

    m_gridNeighborhoodSums = std::vector< vtkm::Int64 >( gridNeighborhoodSums.begin(), gridNeighborhoodSums.end() );
    m_gridNeighborhoodSumsHandle = vtkm::cont::make_ArrayHandle( m_gridNeighborhoodSums );

    m_gridScalars = std::vector< ValueType >( scalar.begin(), scalar.end() );
    m_gridScalarHandle = vtkm::cont::make_ArrayHandle( m_gridScalars );

    m_kdTree.Build( m_gridHandle, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
}

template< typename ValueType >
void XGCAggregator< ValueType >::compute(
    std::vector< int64_t >     & result,
    std::vector< ValueType >       & field,
    const std::vector< ValueType > & r,
    const std::vector< ValueType > & z )
{
    const int64_t SZ = r.size();
    std::vector< vtkm::Vec< ValueType, 2 > > ptclPos( SZ );

    #pragma omp parallel for simd
    for( int64_t i = 0; i < SZ; ++i )
    {
        ptclPos[ i ] = vtkm::Vec< ValueType, 2 >( r[ i ], z[ i ] );
    }

    auto ptclHandle = vtkm::cont::make_ArrayHandle( ptclPos );
    vtkm::cont::ArrayHandle< vtkm::Id  > idHandle;
    vtkm::cont::ArrayHandle< ValueType > distHandle;

    m_kdTree.Run( m_gridHandle, ptclHandle, idHandle, distHandle, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    result.resize( SZ );
    #pragma omp parallel for simd
    for( int64_t i = 0; i < SZ; ++i )
    {
        result[ i ] = idHandle.GetPortalControl().Get( i );
    }

    vtkm::cont::ArrayHandle<vtkm::Float32> fieldResultHandle;

    m_interpolator.run(
        ptclHandle,
        idHandle,
        m_gridHandle,
        m_gridScalarHandle,
        m_gridNeighborhoodsHandle,
        m_gridNeighborhoodSumsHandle,
        fieldResultHandle,
        VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    field.resize( SZ );
    #pragma omp parallel for simd
    for( int64_t i = 0; i < SZ; ++i )
    {
        field[ i ] = fieldResultHandle.GetPortalControl().Get( i );
    }
}

template< typename ValueType >
void XGCAggregator< ValueType >::aggregateOMP(
    const SummaryGrid2< ValueType > & summaryGrid,
    SummaryStep2< ValueType >       & summaryStep,
    const std::vector< ValueType >  & vX,
    const std::vector< ValueType >  & vY,
    const std::vector< ValueType >  & w0,
    const std::vector< ValueType >  & w1,
    const std::vector< int64_t >    & gIDs,
    const int64_t N_CELLS )
{
    /**********************
        Prepare and Size
    **********************/

    // velocity histograms

    HistogramDefinition< ValueType > def1;
    def1.identifier = "vpara-vperp-w1";
    def1.axis = { "vpara", "vperp" };
    def1.dims = { 33, 17 };
    def1.weight = "w1";
    ValueType PartDV = 3378743.0 / 2.0;
    def1.edges = { { -PartDV, PartDV }, { 0, PartDV } };

    HistogramDefinition< ValueType > def2;
    def2.identifier = "vpara-vperp-w0w1";
    def2.axis = { "vpara", "vperp" };
    def2.dims = { 33, 17 };
    def2.weight = "w0w1";
    def2.edges = { { -PartDV, PartDV }, { 0, PartDV } };

    summaryStep.histograms.insert( { def1.identifier, CellHistograms< ValueType >( def1 ) } );
    summaryStep.histograms.insert( { def2.identifier, CellHistograms< ValueType >( def2 ) } );

    // summary statistics
    

    // For some reason this wont compile if template ValueType is used.
    std::set< ScalarVariableStatistics< float >::Statistic > stats =
    {
        ScalarVariableStatistics< float >::Statistic::Count,
        ScalarVariableStatistics< float >::Statistic::Mean,
        ScalarVariableStatistics< float >::Statistic::Variance,  
        ScalarVariableStatistics< float >::Statistic::RMS,  
        ScalarVariableStatistics< float >::Statistic::Min,    
        ScalarVariableStatistics< float >::Statistic::Max                                
    };

    std::vector< std::string > vars    = { "w0", "w1", "w0w1" };
    std::vector< std::vector< const ValueType * > > varVals = { 
        std::vector< const ValueType * >( { w0.data() } ),
        std::vector< const ValueType * >( { w1.data() } ),
        std::vector< const ValueType * >( { w0.data(), w1.data() } )
    }; 

    const int N_VARS = varVals.size();

    for( auto var : vars )
    {
        summaryStep.variableStatistics.insert( { "w0",   ScalarVariableStatistics< ValueType >(   "w0", stats ) } );
        summaryStep.variableStatistics.insert( { "w1",   ScalarVariableStatistics< ValueType >(   "w1", stats ) } );
        summaryStep.variableStatistics.insert( { "w0w1", ScalarVariableStatistics< ValueType >( "w0w1", stats ) } );
    }

    summaryStep.resize( N_CELLS );

    /**********************
             Compute
    **********************/
  
    const int ROWS = def1.dims[ 1 ];
    const int COLS = def1.dims[ 0 ];

    const size_t N_BINS = ROWS * COLS;
    const size_t SZ = vX.size();
    
    const Vec2< ValueType > xRange = { def1.edges[ 0 ][ 0 ], def1.edges[ 0 ][  1 ] };
    const Vec2< ValueType > yRange = { def1.edges[ 1 ][ 0 ], def1.edges[ 1 ][  1 ] };

    const ValueType X_WIDTH = xRange.b() - xRange.a();
    const ValueType Y_WIDTH = yRange.b() - yRange.a();

    auto & hist1 = summaryStep.histograms.at( def1.identifier ).values;
    auto & hist2 = summaryStep.histograms.at( def1.identifier ).values;

    auto & vs = summaryStep.variableStatistics;

    #pragma omp simd
    for( size_t i = 0; i < SZ; ++i )
    {
        const int64_t index = gIDs[ i ];

        int row = std::round( ( ( vY[ i ] - yRange.a() ) / Y_WIDTH ) * ROWS );
        int col = std::round( ( ( vX[ i ] - xRange.a() ) / X_WIDTH ) * COLS );

        if( row < ROWS && col < COLS )
        {
            hist1[ index * N_BINS + row * COLS + col ] += w1[ i ];
            hist2[ index * N_BINS + row * COLS + col ] += w0[ i ] * w1[ i ];            
        }

        for( int v = 0; v < N_VARS; ++v )
        {
            auto & var = vars[ v ];

            auto & count = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Count );
            auto & mean  = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Mean  );
            auto & rms   = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::RMS   );
            auto & mn    = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Min   );
            auto & mx    = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Max   );

            ValueType val = varVals[ v ].size() > 1 ? varVals[ v ][ 0 ][ i ] * varVals[ v ][ 1 ][ i ]  : varVals[ v ][ 0 ][ i ];

            mn[    index ] = std::min( mn[ index ], val );
            mx[    index ] = std::max( mx[ index ], val );            
            mean[  index ] += val;
            rms[   index ] += val*val;
            count[ index ] += 1;            
        }
    }

    #pragma omp simd
    for( size_t i = 0; i < SZ; ++i )
    {
        const int64_t index = gIDs[ i ];

        for( int v = 0; v < N_VARS; ++v )
        {
            auto & var = vars[ v ];
            auto & variance = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Variance );
            auto & mean = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Mean );

            ValueType val = varVals[ v ].size() > 1 ? varVals[ v ][ 0 ][ i ] * varVals[ v ][ 1 ][ i ] : varVals[ v ][ 0 ][ i ];
            val = ( val - mean[ index ] );
            variance[ index ] += val*val;        
        }
    }
}

template< typename ValueType >
void XGCAggregator< ValueType >::aggregateVTKM(
    const SummaryGrid2< ValueType > & summaryGrid,
    SummaryStep2< ValueType >       & summaryStep,
    const std::vector< ValueType >  & vX,
    const std::vector< ValueType >  & vY,
    const std::vector< ValueType >  & w,
    const std::vector< int64_t >    & gIDs,
    const int64_t N_CELLS )
{
    // const int64_t BINS_PER_CELL = SummaryStep::NR*SummaryStep::NC;

    // auto vxHdl = vtkm::cont::make_ArrayHandle( vX );
    // auto vyHdl = vtkm::cont::make_ArrayHandle( vY );
    // auto wHdl  = vtkm::cont::make_ArrayHandle(  w );
    // auto gHdl  = vtkm::cont::make_ArrayHandle(  gIDs );
    // vtkm::worklet::Keys < int64_t > keys( gHdl, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    // vtkm::cont::ArrayHandle< ValueType > meanHdl;
    // vtkm::cont::ArrayHandle< ValueType > rmsHdl;
    // vtkm::cont::ArrayHandle< ValueType > varHdl;
    // vtkm::cont::ArrayHandle< ValueType > minHdl;
    // vtkm::cont::ArrayHandle< ValueType > maxHdl;
    // vtkm::cont::ArrayHandle< ValueType > cntHdl;
    // vtkm::cont::ArrayHandle< vtkm::Vec< ValueType, BINS_PER_CELL > > histHndl;

    // const vtkm::Vec< vtkm::Int32,    2 >  histDims = { SummaryStep::NR,   SummaryStep::NC      };
    // const vtkm::Vec< ValueType,  2 >  xRange   = { -SummaryStep::DELTA_V, SummaryStep::DELTA_V };
    // const vtkm::Vec< ValueType,  2 >  yRange   = { 0,                     SummaryStep::DELTA_V };

    // std::cout << "running aggregate" << std::endl;

    //Run(
    //     N_CELLS,
    //     histDims,
    //     xRange,
    //     yRange,
    //     vxHdl,
    //     vyHdl,
    //     wHdl,
    //     keys,
    //     meanHdl,
    //     rmsHdl,
    //     varHdl,
    //     minHdl,
    //     maxHdl,
    //     cntHdl,
    //     histHndl,
    //     VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    // std::cout << "done aggregating" << std::endl;

    // summaryStep.w0w1_mean            = std::vector< ValueType >( N_CELLS, 0.f );
    // summaryStep.w0w1_rms             = std::vector< ValueType >( N_CELLS, 0.f );
    // summaryStep.w0w1_variance        = std::vector< ValueType >( N_CELLS, 0.f );
    // summaryStep.w0w1_min             = std::vector< ValueType >( N_CELLS,  std::numeric_limits< ValueType >::max() );
    // summaryStep.w0w1_max             = std::vector< ValueType >( N_CELLS, -std::numeric_limits< ValueType >::max() );
    // summaryStep.num_particles        = std::vector< ValueType >( N_CELLS, 0.f );
    // summaryStep.velocityDistribution = std::vector< ValueType >( N_CELLS*SummaryStep::NR*SummaryStep::NC, 0.f );

    // auto uniqueKeys = keys.GetUniqueKeys();
    // const int64_t N_UK = uniqueKeys.GetNumberOfValues();

    // std::cout << histHndl.GetNumberOfValues() << " values ";
    // std::cout << sizeof( histHndl.GetPortalControl().Get( 0 ) ) << " is size of each" << std::endl;
    // std::cout << summaryStep.velocityDistribution.size() << " is summary size of each" << std::endl;

    // std::cout << "copying aggregation results" << std::endl;

    // // #pragma omp parallel for simd
    // for( int64_t i = 0; i < N_UK; ++i )
    // {
    //     int64_t key = static_cast< int64_t >( uniqueKeys.GetPortalControl().Get( i ) );

    //     summaryStep.w0w1_mean[     key ] = meanHdl.GetPortalControl().Get( i );
    //     summaryStep.w0w1_rms[      key ] = rmsHdl.GetPortalControl().Get( i );
    //     summaryStep.w0w1_variance[ key ] = varHdl.GetPortalControl().Get( i );
    //     summaryStep.w0w1_min[      key ] = minHdl.GetPortalControl().Get( i );
    //     summaryStep.w0w1_max[      key ] = maxHdl.GetPortalControl().Get( i );
    //     summaryStep.num_particles[ key ] = cntHdl.GetPortalControl().Get( i );

    //     for( int64_t j = 0; j < BINS_PER_CELL; ++j )
    //     {
    //         if( key * BINS_PER_CELL + j >= N_CELLS * BINS_PER_CELL )
    //         {
    //             std::cout << "error " << key * BINS_PER_CELL + j << " / " << N_CELLS * BINS_PER_CELL << std::endl;
    //             exit( 1 );
    //         }
    //         if( key >= N_CELLS  )
    //         {
    //             std::cout << "error " << key << " / " << N_CELLS << std::endl;
    //             exit( 1 );
    //         }

    //         summaryStep.velocityDistribution[ key * BINS_PER_CELL + j ]
    //             = histHndl.GetPortalControl().Get( i )[ j ];
    //     }
    // }
}

template< typename ValueType >
void XGCAggregator< ValueType >::writeGrid( const std::string & path )
{
    TN::writeSummaryGridBP( m_summaryGrid, path );
}

template< typename ValueType >
void XGCAggregator< ValueType >::computeSummaryStep(
    std::vector< ValueType > & phase,
    TN::SummaryStep2< ValueType > & summaryStep, 
    const std::string & ptype )
{
    const size_t SZ      = phase.size() / 9;
    const size_t R_POS   = XGC_PHASE_INDEX_MAP.at( "r" ) * SZ;
    const size_t Z_POS   = XGC_PHASE_INDEX_MAP.at( "z" ) * SZ;
    const size_t RHO_POS = XGC_PHASE_INDEX_MAP.at( "rho_parallel" ) * SZ;
    const size_t W1_POS  = XGC_PHASE_INDEX_MAP.at( "w1" ) * SZ;
    const size_t W0_POS  = XGC_PHASE_INDEX_MAP.at( "w0" ) * SZ;
    const size_t MU_POS  = XGC_PHASE_INDEX_MAP.at( "mu" ) * SZ;

    // for VTKM nearest neighbors and B field Interpolation //////////////////////
    
    std::chrono::high_resolution_clock::time_point kdt1 = std::chrono::high_resolution_clock::now();
    
    std::vector< int64_t > gridMap;
    std::vector< ValueType > r( SZ );
    std::vector< ValueType > z( SZ );

    for( size_t i = 0; i < SZ; ++i )
    {
        r[ i ] = phase[ R_POS + i ];
        z[ i ] = phase[ Z_POS + i ];
    }

    m_B.resize( SZ );
    gridMap.resize( SZ );

    compute( gridMap, m_B, r, z );

    std::chrono::high_resolution_clock::time_point kdt2 = std::chrono::high_resolution_clock::now();
    std::cout << "RANK: " << m_rank
         << ", kdtree mapping CHUNK took "
         << std::chrono::duration_cast<std::chrono::milliseconds>( kdt2 - kdt1 ).count()
         << " std::chrono::milliseconds " << " for " << r.size() << " particles" << std::endl;

    // compute velocity and weight
    std::vector< ValueType > vpara( SZ );
    std::vector< ValueType > vperp( SZ );
    std::vector< ValueType >    w0( SZ );
    std::vector< ValueType >    w1( SZ );

    #pragma omp parallel for simd
    for( size_t i = 0; i < SZ; ++i )
    {
        w0[  i ] = phase[ W0_POS + i ];
        w1[  i ] = phase[ W1_POS + i ];        
    }

    const double mass_ratio = 1000.0;
    const double ATOMIC_MASS_UNIT = 1.660539040e-27;
    const double ptl_ion_charge_eu = m_constants.at( "ptl_ion_charge_eu" );
    const double mi_sim = m_constants.at( "ptl_ion_mass_au" ) * ATOMIC_MASS_UNIT;
    const double me_sim = mi_sim / mass_ratio;
    const double e = 1.609e-19;

    if( ptype == "ions")
    {
        #pragma omp parallel for simd
        for( size_t i = 0; i < SZ; ++i )
        {
            vpara[ i ] = m_B[ i ] * phase[ RHO_POS + i ] * ( ( ptl_ion_charge_eu * e ) / mi_sim );
            vperp[ i ] = sqrt( ( phase[   MU_POS + i ] * 2.0 * m_B[ i ] ) / mi_sim );
        }
    }
    else
    {
        #pragma omp parallel for simd
        for( size_t i = 0; i < SZ; ++i )
        {
            vpara[ i ] =( m_B[ i ] * phase[ RHO_POS + i ] * (-e) ) / me_sim;
            vperp[ i ] = sqrt( ( phase[    MU_POS + i ] * 2.0 * m_B[ i ]  ) / me_sim  );
        }
    }

    phase.clear();

    // compute summations over particles in each cell

    std::chrono::high_resolution_clock::time_point at1 = std::chrono::high_resolution_clock::now();

    // aggregateVTKM(
    //     m_summaryGrid,
    //     summaryStep,
    //     vpara,
    //     vperp,
    //     w0w1,
    //     gridMap,
    //     m_summaryGrid.variables.at( "volume.size() );

    // Without VTKM

        aggregateOMP(
            m_summaryGrid,
            summaryStep,
            vpara,
            vperp,
            w0,
            w1,
            gridMap,
            m_summaryGrid.variables.at( "volume" ).size() );

    std::chrono::high_resolution_clock::time_point at2 = std::chrono::high_resolution_clock::now();    

    std::cout << "RANK: " << m_rank
         << ", Aggregation step took "
         << std::chrono::duration_cast<std::chrono::milliseconds>( at2 - at1 ).count()
         << " std::chrono::milliseconds " << " for " << r.size() << " particles" << std::endl;    

    std::chrono::high_resolution_clock::time_point rt1 = std::chrono::high_resolution_clock::now();

    for( auto & hist : summaryStep.histograms )
    {
        TN::MPI::ReduceOpMPI( m_rank, hist.second.values, MPI_SUM );
    }

    for( auto & var : summaryStep.variableStatistics )
    {
        auto & myCounts = var.second.values.at( 
            ScalarVariableStatistics< ValueType >::Statistic::Count );

        TN::MPI::ReduceOpMPI(
            m_rank, 
            myCounts, 
            MPI_SUM );

        if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Min ) )
        {
            TN::MPI::ReduceOpMPI(
                m_rank, 
                var.second.values.at( 
                    ScalarVariableStatistics< ValueType >::Statistic::Min ),
                MPI_MIN );
        }

        if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Max ) )
        {
            TN::MPI::ReduceOpMPI(
                m_rank, 
                var.second.values.at( 
                    ScalarVariableStatistics< ValueType >::Statistic::Max ),
                MPI_MAX );
        }

        if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Mean ) )
        {
            TN::MPI::ReduceMean( 
                m_rank,
                var.second.values.at( 
                    ScalarVariableStatistics< ValueType >::Statistic::Mean ),
                myCounts );
        }

        if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Variance ) )
        {
            TN::MPI::ReduceVariance( 
                m_rank,
                var.second.values.at( 
                    ScalarVariableStatistics< ValueType >::Statistic::Variance ),
                myCounts );  
        }

        if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::RMS ) )
        {
            TN::MPI::ReduceRMS( 
                m_rank,
                var.second.values.at( 
                    ScalarVariableStatistics< ValueType >::Statistic::RMS ),
                myCounts );    
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if( m_rank == 0 )
    {   
        std::chrono::high_resolution_clock::time_point rt2 = std::chrono::high_resolution_clock::now();
        std::cout << "reduce step took " << std::chrono::duration_cast<std::chrono::milliseconds>( rt2 - rt1 ).count()
                  << " std::chrono::milliseconds"  << std::endl;

        std::chrono::high_resolution_clock::time_point wt1 = std::chrono::high_resolution_clock::now();
        
        writeSummaryStepBP( summaryStep, m_outputDirectory );

        std::chrono::high_resolution_clock::time_point wt2 = std::chrono::high_resolution_clock::now();        
        std::cout << "write step took " << std::chrono::duration_cast<std::chrono::milliseconds>( wt2 - wt1 ).count()
                  << " std::chrono::milliseconds\n"  << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

template class XGCAggregator<float>;
//template class XGCAggregator<double>;

}