
#include "SimpleAdios2XMLParser.hpp"
#include "XGCAggregator.hpp"
#include "Summary.hpp"
#include "SummaryWriterAdios2.hpp"
#include "XGCMeshReaderAdios1.hpp"
#include "XGCParticleReaderAdios2.hpp"
#include "XGCConstantReader.hpp"
#include "Reduce/Reduce.hpp"
#include "XGCSynchronizer.hpp"
#include "grid-algorithms/MeshUtils.hpp"
#include "Tracker.hpp"
#include <mpi.h>

#include <vector>
#include <set>
#include <chrono>
#include <exception>

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
    MPI_Comm communicator,
    bool tryUsingCuda ) :
        m_meshFilePath( meshFilePath ),
        m_bFieldFilePath( bfieldFilePath ),
        m_particleFile( restartPath ),
        m_unitsMFilePath( unitsFilePath ),
        m_outputDirectory( outputDirectory ),
        m_inSitu( inSitu ),
        m_splitByBlocks( splitByBlocks ),
        m_rank( m_rank ),
        m_nranks( nm_ranks ),
        m_summaryWriterAppendMode( true ),
        m_mpiCommunicator( communicator ),
        m_superParticleThreshold( std::numeric_limits< float >::max() ),
        m_aggregatorTools( tryUsingCuda )
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

    TN::Synchro::waitForFileExistence( m_unitsMFilePath,  100000 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 1000 ) );
    TN::loadConstants( m_unitsMFilePath, m_constants );
    
    TN::Synchro::waitForFileExistence(     meshFilePath,  100000 );
    TN::Synchro::waitForFileExistence( m_bFieldFilePath,  100000 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 1000 ) );

    //std::cout << "reading mesh" << std::endl;

    TN::readMeshBP(
        m_summaryGrid,
    { m_constants.at( "eq_axis_r" ), m_constants.at( "eq_axis_z" ) },
    meshFilePath,
    m_bFieldFilePath );

    //std::cout << "sorting neigborhoods" << std::endl;
    
    TN::Mesh::sortNeighborhoodsCCW( 
        m_summaryGrid.variables.at( "r" ),
        m_summaryGrid.variables.at( "z" ),
        m_summaryGrid.neighborhoods,    
        m_summaryGrid.neighborhoodSums );

    m_summaryGrid.vertexFlags.resize( 
        m_summaryGrid.neighborhoodSums.size() );
    
    TN::Mesh::computeVertexFlags( 
        m_summaryGrid.variables.at( "r" ),
        m_summaryGrid.variables.at( "z" ),
        m_summaryGrid.neighborhoods,    
        m_summaryGrid.neighborhoodSums,
        m_summaryGrid.vertexFlags );

    m_summaryGrid.maxNeighbors = TN::Mesh::maxNeighborhoodSize(
        m_summaryGrid.neighborhoodSums );

    //std::cout << "max neighborhood=" << m_summaryGrid.maxNeighbors;

    //std::cout << "setting grid" << std::endl;

    m_aggregatorTools.setGrid(
        m_summaryGrid.variables.at( "r" ),
        m_summaryGrid.variables.at( "z" ),
        m_summaryGrid.variables.at( "B" ),
        m_summaryGrid.neighborhoods,
        m_summaryGrid.neighborhoodSums,
        m_summaryGrid.vertexFlags );
}

template< typename ValueType >
void XGCAggregator< ValueType >::writeMesh()
{
    writeGrid( m_outputDirectory );
}

template< typename ValueType >
void XGCAggregator< ValueType >::runInSitu()
{
    const float TIMEOUT = 300000.f;
    
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
            new adios2::IO( adiosOutPut->DeclareIO( "Summary-IO-Root" ) ) );

        summaryIO->DefineAttribute<std::string>( "particles", { "ions, electrons" } );

        summaryWriter = std::unique_ptr< adios2::Engine >( 
            new adios2::Engine( summaryIO->Open( 
                m_outputDirectory + "/summary.bp", adios2::Mode::Write ) ) );
    }

    /*************************************************************************/

    // Particle Path Writer, written in parallel

    std::unique_ptr< adios2::ADIOS > adiosParticlePath( new adios2::ADIOS( MPI_COMM_SELF, adios2::DebugOFF )  );    
    std::unique_ptr< adios2::IO > particlePathIO( new adios2::IO( adiosParticlePath->DeclareIO( "ParticlePath-IO" ) ) );
    std::vector< std::string > pathPhase( XGC_PHASE_INDEX_MAP.size() );

    for( auto & var : XGC_PHASE_INDEX_MAP )
    {
        pathPhase[ var.second ] = var.first;
    }

    // particlePathIO->DefineAttribute<std::string>( "variables", pathPhase );

    std::unique_ptr< adios2::Engine > particlePathWriter( 
        new adios2::Engine( particlePathIO->Open( 
                m_outputDirectory + "/particlePaths.bp", adios2::Mode::Write ) ) );

    /*************************************************************************/
    // Reader

    adios2::ADIOS adios( m_mpiCommunicator, adios2::DebugOFF );
    adios2::IO particleIO = adios.DeclareIO( "Particle-IO-Collective" );

    if( m_particleReaderEngine == "SST" )
    {
        particleIO.SetEngine( "Sst" );
        //std::cout << "set engine type to sst";
    }
    else if( m_particleReaderEngine == "InSituMPI" )
    {
        particleIO.SetEngine( "InSituMPI" );
        //std::cout << "set engine type to InSituMPI";
    }

    //std::cout << "Trying to open reader: " << m_particleFile << std::endl;
    adios2::Engine particleReader = particleIO.Open( m_particleFile, adios2::Mode::Read );
    //std::cout << "Summarizer opened reader";

    /************************************************************************************/

    int64_t outputStep = 0;
    TN::SummaryStep< ValueType > summaryStep;   
    std::unordered_set< std::int64_t > trackedParticleIds;

    while( 1 )
    {
        /*******************************************************************************/
        // Read Particle Data Step

        //std::cout << "Before begin step" << std::endl;

        adios2::StepStatus status =
            particleReader.BeginStep( 
                adios2::StepMode::NextAvailable, TIMEOUT );

        //std::cout << "After begin step, status is: "
        //          << ( status == adios2::StepStatus::OK          ? "OK"          :
        //               status == adios2::StepStatus::OtherError  ? "OtherError"  :
        //               status == adios2::StepStatus::EndOfStream ? "EndOfStream" :
        //               "OtherError" ) << std::endl;
        
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

        int64_t simstep;
        double  realtime;

        //std::cout << "Reading Particles" << std::endl;

        int64_t totalNumParticles 
            = readBPParticleDataStep(
                m_phase,
                m_particleIds,
                "ions",
                m_particleFile,
                m_rank,
                m_nranks,
                particleIO,
                particleReader,
                simstep,
                realtime,
                m_splitByBlocks );

        //std::cout << "Before EndStep" << std::endl;
        particleReader.EndStep();
        //std::cout << "After EndStep" << std::endl;

        /*******************************************************************************/

        summaryStep.numParticles = totalNumParticles;
        summaryStep.setStep( outputStep, simstep, realtime );
        summaryStep.objectIdentifier = "ions";

        /////////////////////////////////////////////////////////////////////////////////////////////////////////

        computeSummaryStep(
            m_phase,
            m_particleIds,
            summaryStep,
            trackedParticleIds,            
            "ions",
            summaryIO,
            summaryWriter,
            particlePathIO,
            particlePathWriter );

        ++outputStep;
    }

    particleReader.Close();

    if( m_summaryWriterAppendMode && m_rank == 0 )
    {
        summaryWriter->Close();
    }
}

template< typename ValueType >
void XGCAggregator< ValueType >::runInPost()
{
    double summaryStepTime = 0.0;
    int64_t outputStep     = 0;

    std::vector< int64_t > steps = { 
        200, 
        400, 
        600, 
        800, 
        1000, 
        1200, 
        1400, 
        1600, 
        1800, 
        2000,
        2200, 
        2400, 
        2600, 
        2800, 
        3000,
        3200, 
        3400, 
        3600, 
        3800,
        4000, 
        4200  };

    SummaryStep< ValueType > summaryStep;
    std::unordered_set< std::int64_t > trackedParticleIds;

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
            new adios2::IO( adiosOutPut->DeclareIO( "Summary-IO-Root" ) ) );

        summaryIO->DefineAttribute<std::string>( "particles", { "ions, electrons" } );

        summaryWriter = std::unique_ptr< adios2::Engine >( 
            new adios2::Engine( summaryIO->Open( 
                m_outputDirectory + "/summary.bp", adios2::Mode::Write ) ) );
    }

    /*************************************************************************/

    // Particle Path Writer, written in parallel

    std::unique_ptr< adios2::ADIOS > adiosParticlePath( new adios2::ADIOS( MPI_COMM_SELF, adios2::DebugOFF )  );    
    std::unique_ptr< adios2::IO > particlePathIO( new adios2::IO( adiosParticlePath->DeclareIO( "ParticlePath-IO" ) ) );
    std::vector< std::string > pathPhase( XGC_PHASE_INDEX_MAP.size() );

    for( auto & var : XGC_PHASE_INDEX_MAP )
    {
        pathPhase[ var.second ] = var.first;
    }

    // particlePathIO->DefineAttribute<std::string>( "variables", pathPhase );

    std::unique_ptr< adios2::Engine > particlePathWriter( 
        new adios2::Engine( particlePathIO->Open( 
                m_outputDirectory + "/particlePaths.bp", adios2::Mode::Write ) ) );

    /******************************************************************************/

    for( auto tstep : steps )
    {
        std::string tstepStr = std::to_string( tstep );
        std::chrono::high_resolution_clock::time_point readStartTime = std::chrono::high_resolution_clock::now();

        int64_t simstep;
        double  realtime;

        int64_t totalNumParticles = readBPParticleDataStep(
             m_phase,
             m_particleIds,             
             "ions",
             m_particleFile + "/xgc.restart." + std::string( 5 - tstepStr.size(), '0' ) + tstepStr +  ".bp",
             m_rank,
             m_nranks,
             simstep,
             realtime,
             m_splitByBlocks );

        summaryStep.numParticles = totalNumParticles;
        summaryStep.setStep( outputStep, simstep, realtime );
        summaryStep.objectIdentifier = "ions";

        std::chrono::high_resolution_clock::time_point readStartEnd = std::chrono::high_resolution_clock::now();

        std::cout << "RANK: " << m_rank
                  << ", adios Read time took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( readStartEnd - readStartTime ).count()
                  << " std::chrono::milliseconds " << " for " << m_phase.size() / 9 << " particles" << std::endl;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////

        std::unique_ptr<     adios2::IO > io;
        std::unique_ptr< adios2::Engine > en;

        computeSummaryStep(
            m_phase,
            m_particleIds,
            summaryStep,
            trackedParticleIds,           
            "ions",
            summaryIO,
            summaryWriter,
            particlePathIO,
            particlePathWriter );

        ++outputStep;
    }

    if( m_summaryWriterAppendMode && m_rank == 0 )
    {
        summaryWriter->Close();
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
void XGCAggregator< ValueType >::aggregateOMP(
    const SummaryGrid< ValueType > & summaryGrid,
    SummaryStep< ValueType >       & summaryStep,
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
    const std::vector< ValueType > & phase,
    const std::vector< int64_t > & m_ids,
    TN::SummaryStep< ValueType > & summaryStep,
    std::unordered_set< int64_t > & trackedParticleIds,
    const std::string & ptype,
    std::unique_ptr< adios2::IO > & summaryIO,
    std::unique_ptr< adios2::Engine > & summaryWriter,
    std::unique_ptr< adios2::IO > & particlePathIO,
    std::unique_ptr< adios2::Engine > & particlePathWriter )
{
    bool TRACKING_ENABLED = false;


    const size_t SZ      = phase.size() / 9;
    const size_t R_POS   = XGC_PHASE_INDEX_MAP.at( "r" ) * SZ;
    const size_t Z_POS   = XGC_PHASE_INDEX_MAP.at( "z" ) * SZ;
    const size_t RHO_POS = XGC_PHASE_INDEX_MAP.at( "rho_parallel" ) * SZ;
    const size_t W1_POS  = XGC_PHASE_INDEX_MAP.at( "w1" ) * SZ;
    const size_t W0_POS  = XGC_PHASE_INDEX_MAP.at( "w0" ) * SZ;
    const size_t MU_POS  = XGC_PHASE_INDEX_MAP.at( "mu" ) * SZ;

    // for VTKM nearest neighbors and B field Interpolation //////////////////////

    static std::vector< int64_t > gridMap;

    static std::vector< ValueType > r;
    static std::vector< ValueType > z;

    r.resize( SZ );
    z.resize( SZ );

    #pragma omp parallel for
    for( size_t i = 0; i < SZ; ++i )
    {
        r[ i ] = phase[ R_POS + i ];
        z[ i ] = phase[ Z_POS + i ];
    }

    m_B.resize( SZ );
    gridMap.resize( SZ );

    std::chrono::high_resolution_clock::time_point kdt1 = std::chrono::high_resolution_clock::now();

    m_aggregatorTools.compute( gridMap, m_B, r, z, m_summaryGrid.maxNeighbors );

    std::chrono::high_resolution_clock::time_point kdt2 = std::chrono::high_resolution_clock::now();
    std::cout << "RANK: " << m_rank
              << ", kdtree mapping and interpolation took "
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

    /*********************************************************************************************/
    // Particle Tracking

    if( TRACKING_ENABLED )
    {

    // identify super particles
        std::vector< int64_t > myNewSuperParticles;
        TN::Tracker::identifyNewSuperParticles( 
            m_particleIds,
            m_phase,
            myNewSuperParticles,
            m_superParticleThreshold );

        int64_t myNumNewParticles = myNewSuperParticles.size();
        
        // communicate total number of new super particles to all nodes

        int64_t totalNumNewParticles;

        MPI_Allreduce(
            & myNumNewParticles,
            & totalNumNewParticles,
            1,
            MPI_LONG_LONG_INT,
            MPI_SUM,
            m_mpiCommunicator );

        // gather new super particles to all nodes
        
        std::vector< int64_t > allNewSuperParticles( totalNumNewParticles );
        std::vector< int > newParticleGatherOffsets( m_nranks );
        std::vector< int > toGatherPerNode( m_nranks );

        MPI_Allgather(
            & myNumNewParticles, 
            1, 
            MPI_INT,
            toGatherPerNode.data(), 
            m_nranks, 
            MPI_INT, 
            m_mpiCommunicator );

        int currGatherOffset = 0;
        for( int i = 0; i < m_nranks; ++i )
        {
            newParticleGatherOffsets[ i ] = currGatherOffset;
            currGatherOffset += toGatherPerNode[ i ];
        }

        MPI_Allgatherv(
            myNewSuperParticles.data(), 
            static_cast< int >( myNewSuperParticles.size() ), 
            MPI_LONG_LONG_INT,
            allNewSuperParticles.data(), 
            toGatherPerNode.data(), 
            newParticleGatherOffsets.data(),
            MPI_LONG_LONG_INT, 
            m_mpiCommunicator );

        // add new super particles to set of tracked particles

        trackedParticleIds.insert( 
            allNewSuperParticles.begin(), 
            allNewSuperParticles.end() );

        // record values for super particles on this node

        PhasePathStep < ValueType > phasePaths;

        TN::Tracker::trackParticles(
            m_particleIds,
            m_phase,
            trackedParticleIds,
            phasePaths );

        // get the number of tracked particles found for each node

        int64_t myNumFoundTrackedParticles = phasePaths.ids.size();
        std::vector< int64_t > eachNodesNumFoundTrackedParticles( m_nranks );

        MPI_Allgather(
            & myNumFoundTrackedParticles, 
            1, 
            MPI_LONG_LONG_INT,
            eachNodesNumFoundTrackedParticles.data(), 
            m_nranks, 
            MPI_LONG_LONG_INT, 
            m_mpiCommunicator );

        int64_t myGlobalOffset = 0;
        for( int i = 0; i < m_rank; ++i )
        {
            myGlobalOffset += eachNodesNumFoundTrackedParticles[ i ];
        }

        int64_t totalNumFoundTrackedParticles = 0;
        for( int i = 0; i < m_rank; ++i )
        {
            totalNumFoundTrackedParticles += eachNodesNumFoundTrackedParticles[ i ];
        }

        if( m_summaryWriterAppendMode )
        {
            particlePathWriter->BeginStep();

            writeParticlePathsStep(
                phasePaths,
                myGlobalOffset,
                totalNumFoundTrackedParticles,    
                XGC_PHASE_INDEX_MAP.size(),
                "super", 
                *particlePathIO,
                *particlePathWriter );
            
            particlePathWriter->EndStep();
        }
        else
        {
            writeParticlePathsStep(
                phasePaths,
                myGlobalOffset,
                totalNumFoundTrackedParticles, 
                m_outputDirectory,   
                summaryStep.simStep,
                XGC_PHASE_INDEX_MAP.size(),
                "super",
                m_mpiCommunicator );
        }
    }

    /*********************************************************************************************/

    for( auto & hist : summaryStep.histograms )
    {
        TN::MPI::ReduceOpMPI( m_rank, hist.second.values, MPI_SUM, m_mpiCommunicator );
    }

    for( auto & var : summaryStep.variableStatistics )
    {
        auto & myCounts = var.second.values.at(
                              ScalarVariableStatistics< ValueType >::Statistic::Count );

        TN::MPI::ReduceOpMPI(
            m_rank,
            myCounts,
            MPI_SUM,
            m_mpiCommunicator );

        if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Min ) )
        {
            TN::MPI::ReduceOpMPI(
                m_rank,
                var.second.values.at(
                    ScalarVariableStatistics< ValueType >::Statistic::Min ),
                MPI_MIN,
                m_mpiCommunicator );
        }

        if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Max ) )
        {
            TN::MPI::ReduceOpMPI(
                m_rank,
                var.second.values.at(
                    ScalarVariableStatistics< ValueType >::Statistic::Max ),
                MPI_MAX,
                m_mpiCommunicator );
        }

        if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Mean ) )
        {
            TN::MPI::ReduceMean(
                m_rank,
                var.second.values.at(
                    ScalarVariableStatistics< ValueType >::Statistic::Mean ),
                myCounts,
                m_mpiCommunicator );
        }

        if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::Variance ) )
        {
            TN::MPI::ReduceVariance(
                m_rank,
                var.second.values.at(
                    ScalarVariableStatistics< ValueType >::Statistic::Variance ),
                myCounts,
                m_mpiCommunicator );
        }

        if( var.second.values.count( ScalarVariableStatistics< ValueType >::Statistic::RMS ) )
        {
            TN::MPI::ReduceRMS(
                m_rank,
                var.second.values.at(
                    ScalarVariableStatistics< ValueType >::Statistic::RMS ),
                myCounts,
                m_mpiCommunicator );
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
            writeSummaryStepBP( summaryStep, *summaryIO, *summaryWriter );
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
