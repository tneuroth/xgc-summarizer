
#include "SimpleAdios2XMLParser.hpp"
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
    const std::string & adiosConfigFilePath,
    const std::string & meshFilePath,
    const std::string & bfieldFilePath,
    const std::string & restartPath,
    const std::string & unitsFilePath,
    const std::string & outputDirectory,
    const std::set< std::string > & particleTypes,
    bool inSitu,
    bool splitByBlocks,
    int m_rank,
    int nm_ranks,
    MPI_Comm communicator ) :
    m_meshFilePath( meshFilePath ),
    m_bFieldFilePath( bfieldFilePath ),
    m_restartPath( restartPath ),
    m_unitsMFilePath( unitsFilePath ),
    m_outputDirectory( outputDirectory ),
    m_inSitu( inSitu ),
    m_splitByBlocks( splitByBlocks ),
    m_rank( m_rank ),
    m_nranks( nm_ranks ),
    m_summaryWriterAppendMode( true ),
    m_mpiCommunicator( communicator )
{
    const std::map< std::string, std::string > ioEngines 
        = TN::XML::extractIoEngines( adiosConfigFilePath );

    if( ioEngines.count( "particles" ) )
    {
        m_particleReaderEngine = ioEngines.at( "particles" );
    }
    else
    {
        m_particleReaderEngine = "BPFile";
    }

    TN::Synchro::waitForFileExistence( m_unitsMFilePath, 100000 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 1000 ) );
    TN::loadConstants( m_unitsMFilePath, m_constants );
    
    TN::Synchro::waitForFileExistence(     meshFilePath, 100000 );
    TN::Synchro::waitForFileExistence( m_bFieldFilePath, 100000 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 1000 ) );

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
    writeGrid( m_outputDirectory );
}

template< typename ValueType >
void XGCAggregator< ValueType >::runInSitu()
{
    const float TIMEOUT = 30000.f;
    std::string particlePath = m_restartPath;
    
    /*************************************************************************/
    // Summary Writer (results are reduced to and written from mpi root)

    std::unique_ptr< adios2::ADIOS > adiosOutPut;    
    std::unique_ptr< adios2::IO > summaryIO;
    std::unique_ptr< adios2::Engine > summaryWriter;

    if( m_summaryWriterAppendMode && m_rank == 0 )
    {
        adiosOutPut = std::unique_ptr< adios2::ADIOS >( 
            new adios2::ADIOS( MPI_COMM_SELF, adios2::DebugOFF ) );
        summaryIO = std::unique_ptr< adios2::IO >( 
            new adios2::IO( adiosOutPut.DeclareIO( "Summary-IO-Root" ) ) );
        summaryWriter = std::unique_ptr< adios2::Engine >( 
            new adios2::Engine( summaryIO->Open( 
                m_outputDirectory + "/summary.bp", adios2::Mode::Write ) ) );
    }

    /*************************************************************************/
    // Reader

    adios2::ADIOS adios( m_mpiCommunicator, adios2::DebugOFF );
    adios2::IO particleIO = adios.DeclareIO( "Particle-IO-Collective" );

    if( m_particleReaderEngine == "SST" )
    {
        particleIO.SetEngine( "Sst" );
        MPI_Barrier( m_mpiCommunicator );
        particlePath = "xgc.particle.bp";
        std::cout << "set engine type to sst";
    }
    else if( m_particleReaderEngine == "InSituMPI" )
    {
        particleIO.SetEngine( "InSituMPI" );
        MPI_Barrier( m_mpiCommunicator );
        particlePath = "xgc.particle.bp";
        std::cout << "set engine type to InSituMPI";
    }

    std::cout << "Trying to open reader: " << particlePath << std::endl;
    adios2::Engine particleReader = particleIO.Open( particlePath, adios2::Mode::Read );

    /************************************************************************************/

    while( 1 )
    {
        /*******************************************************************************/
        // Read Particle Data Step

        std::cout << "Before begin step" << std::endl;

        adios2::StepStatus status =
            particleReader.BeginStep( 
                adios2::StepMode::NextAvailable, TIMEOUT );

        std::cout << "After begin step, status is: "
                  << ( status == adios2::StepStatus::OK          ? "OK"          :
                       status == adios2::StepStatus::OtherError  ? "OtherError"  :
                       status == adios2::StepStatus::EndOfStream ? "EndOfStream" :
                       "OtherError" ) << std::endl;
        
        if ( status == adios2::StepStatus::NotReady )
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds( 1000 ) );
            continue;
        }
        else if ( status != adios2::StepStatus::OK )
        {
            std::cout << "step status not OK: " << outputStep << ", RANK: " << m_rank << std::endl;
            break;
        }

        std::chrono::high_resolution_clock::time_point readStartTime = std::chrono::high_resolution_clock::now();

        int64_t simstep;
        double  realtime;

        std::cout << "Reading Particles" << std::endl;

        int64_t totalNumParticles 
            = readBPParticleDataStep(
                m_phase,
                "ions",
                m_restartPath,
                m_rank,
                m_nranks,
                particleIO,
                particleReader,
                simstep,
                realtime,
                true );

        std::cout << "Before EndStep" << std::endl;
        particleReader.EndStep();
        std::cout << "After EndStep" << std::endl;

        /*******************************************************************************/

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
            "ions",
            summaryIO,
            summaryWriter );

        ++outputStep;
    }

    particleReader.Close();

    if( m_summaryWriterAppendMode && m_rank == 0 )
    {
        summaryWriter->Close();
    }
}

template< typename ValueType >
void XGCAggregator< ValueType >::run()
{
    runInSitu();
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

    result.resize( SZ );
    const auto idControl = idHandle.GetPortalConstControl();

    #pragma omp parallel for simd
    for( int64_t i = 0; i < SZ; ++i )
    {
        result[ i ] =  idControl.Get( i );
    }

    field.resize( SZ );
    const auto fieldControl = fieldResultHandle.GetPortalConstControl();

    #pragma omp parallel for simd
    for( int64_t i = 0; i < SZ; ++i )
    {
        field[ i ] = fieldControl.Get( i );
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
    std::vector< std::vector< const ValueType * > > varVals =
    {
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
    auto & hist2 = summaryStep.histograms.at( def2.identifier ).values;

    auto & vs = summaryStep.variableStatistics;

    #pragma omp simd
    for( size_t i = 0; i < SZ; ++i )
    {
        const int64_t index = gIDs[ i ];

        int row = std::round( ( ( vY[ i ] - yRange.a() ) / Y_WIDTH ) * ROWS );
        int col = std::round( ( ( vX[ i ] - xRange.a() ) / X_WIDTH ) * COLS );

        if( row < ROWS && col < COLS && row >= 0 && col >= 0 )
        {
            hist1[ index * N_BINS + row * COLS + col ] += w1[ i ];
            hist2[ index * N_BINS + row * COLS + col ] += w0[ i ] * w1[ i ];
        }
    }
    
    for( int v = 0; v < N_VARS; ++v )
    {
        const auto & var = vars[ v ];
        auto & count = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Count );
        auto & mean  = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Mean  );
        auto & rms   = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::RMS   );
        auto & mn    = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Min   );
        auto & mx    = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Max   );

        if( var == "w0w1" )
        {
            const auto & w0_values = varVals[ v ][ 0 ];
            const auto & w1_values = varVals[ v ][ 1 ];
            
            #pragma omp simd
            for( size_t i = 0; i < SZ; ++i )
            {
                const int64_t index = gIDs[ i ];
                const auto val = w0_values[ i ] * w1_values[ i ];
                mn[    index ] = std::min( mn[ index ], val );
                mx[    index ] = std::max( mx[ index ], val );
                mean[  index ] += val;
                rms[   index ] += val*val;
                count[ index ] += 1;
            }
        } 
        else
        {
            const auto & values = varVals[ v ][ 0 ];
            
            #pragma omp simd
            for( size_t i = 0; i < SZ; ++i )
            {
                const int64_t index = gIDs[ i ];               
                const auto val = values[ i ];
                mn[    index ] = std::min( mn[ index ], val );
                mx[    index ] = std::max( mx[ index ], val );
                mean[  index ] += val;
                rms[   index ] += val*val;
                count[ index ] += 1;
            }
        }
    }

    for( int v = 0; v < N_VARS; ++v )
    {
        const auto & var = vars[ v ];
        auto & variance = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Variance );
        const auto & mean = vs.at( var ).values.at( ScalarVariableStatistics< ValueType >::Statistic::Mean );

        if( var =="w0w1" )
        {
            const auto & w0_values = varVals[ v ][ 0 ];
            const auto & w1_values = varVals[ v ][ 1 ];
            
            #pragma omp simd
            for( size_t i = 0; i < SZ; ++i )
            {
                const int64_t index = gIDs[ i ];
                ValueType val = w0_values[ i ] * w1_values[ i ];
                val = ( val - mean[ index ] );
                variance[ index ] += val*val;
            }
        }
        else
        {
            for( int v = 0; v < N_VARS; ++v )
            {
                const auto & values = varVals[ v ][ 0 ];

                #pragma omp simd
                for( size_t i = 0; i < SZ; ++i )
                {
                    const int64_t index = gIDs[ i ];
                    ValueType val = values[ i ];
                    val = ( val - mean[ index ] );
                    variance[ index ] += val*val;
                }
            }
        }
    }
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
    const std::string & ptype,
    std::unique_ptr< adios2::IO > & summaryIO,
    std::unique_ptr< adios2::Engine > & summaryWriter )
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

    static std::vector< int64_t > gridMap;

    static std::vector< ValueType > r;
    static std::vector< ValueType > z;

    r.resize( SZ );
    z.resize( SZ );

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

    // compute summations over particles in each cell

    std::chrono::high_resolution_clock::time_point at1 = std::chrono::high_resolution_clock::now();

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

    std::cout << "reducing" << std::endl;

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

    MPI_Barrier( m_mpiCommunicator );

    if( m_rank == 0 )
    {
        std::chrono::high_resolution_clock::time_point rt2 = std::chrono::high_resolution_clock::now();
        std::cout << "reduce step took " << std::chrono::duration_cast<std::chrono::milliseconds>( rt2 - rt1 ).count()
                  << " std::chrono::milliseconds"  << std::endl;

        std::chrono::high_resolution_clock::time_point wt1 = std::chrono::high_resolution_clock::now();

        if( m_summaryWriterAppendMode )
        {
            summaryWriter->BeginStep();
            writeSummaryStepBP( summaryStep, m_outputDirectory, *summaryIO, *summaryWriter );
            summaryWriter->EndStep();
        }
        else
        {
            writeSummaryStepBP( summaryStep, m_outputDirectory );   
        }

        std::chrono::high_resolution_clock::time_point wt2 = std::chrono::high_resolution_clock::now();
        std::cout << "write step took " << std::chrono::duration_cast<std::chrono::milliseconds>( wt2 - wt1 ).count()
                  << " std::chrono::milliseconds\n"  << std::endl;
    }

    MPI_Barrier( m_mpiCommunicator );
}

template class XGCAggregator<float>;
//template class XGCAggregator<double>;

}
