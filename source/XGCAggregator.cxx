
#include "XGCAggregator.hpp"
#include "Summary.hpp"
#include "SummaryWriter.hpp"
#include "XGCMeshReader.hpp"
#include "XGCParticleReader.hpp"
#include "XGCConstantReader.hpp"

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

XGCAggregator::XGCAggregator(
    const std::string & meshFilePath,
    const std::string & bfieldFilePath,
    const std::string & restartDirectory,
    const std::string & unitsFilePath,
    const std::string & outputDirectory,
    const std::set< std::string > & particleTypes,
    int rank,
    int nranks ) : 
        m_meshFilePath( meshFilePath ),
        m_bFieldFilePath( bfieldFilePath ),
        m_restartDirectory( restartDirectory ),
        m_unitsMFilePath( unitsFilePath ),
        m_outputDirectory( outputDirectory ),
        m_rank( rank ),
        m_nranks( nranks )

{
    TN::loadConstants( m_unitsMFilePath, m_constants );

    TN::readMeshBP(
        m_summaryGrid,
        { m_constants.at( "eq_axis_r" ), m_constants.at( "eq_axis_z" ) },
        meshFilePath,
        m_bFieldFilePath ); 
    
    setGrid(
        m_summaryGrid.probes.r,
        m_summaryGrid.probes.z,
        m_summaryGrid.probes.B,
        m_summaryGrid.neighborhoods,
        m_summaryGrid.neighborhoodSums );

    if( m_rank == 0 )
    {
        checkDevice( VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
    }
}

void XGCAggregator::writeMesh()
{
    if( m_rank == 0 )
    {
        writeGrid( m_outputDirectory );
    }
}

void XGCAggregator::reduceMesh( 
    const std::string & reducedMeshFilePath )
{
    std::ifstream meshFile( reducedMeshFilePath );
    std::string line;

    if( ! meshFile.is_open() )
    {
        std::cerr << "couldn't open file: " << reducedMeshFilePath << std::endl;
        exit( 1 );
    }

    SummaryGrid newGrid;

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
                newGrid.probes.r.push_back( x );
                newGrid.probes.z.push_back( y );

                //just for now until voronoi cell face area can be calculated.
                newGrid.probes.volume.push_back( 1.f );
            }
            else if( v == "f" )
            {
                unsigned int i1, i2, i3;
                sstr >> i1 >> i2 >> i3;
                newGrid.probeTriangulation.push_back( { 
                    i1 - 1,
                    i2 - 1,
                    i3 - 1 } );
            }
        }
    }
 
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "getting neighborhoods " << std::endl;
    TN::getNeighborhoods( newGrid );

    std::cout << "Copying over values from old mesh " << newGrid.probes.r.size() << "/" << m_summaryGrid.probes.r.size() << std::endl; 
    
    newGrid.probes.B.resize( newGrid.probes.r.size() );
    newGrid.probes.psin.resize( newGrid.probes.r.size() );
    newGrid.probes.poloidalAngle.resize( newGrid.probes.r.size() );

    for( int64_t i = 0; i < newGrid.probes.r.size(); ++i )
    {
        newGrid.probes.B[ i ] = m_summaryGrid.probes.B[ i*2 ];
        newGrid.probes.psin[ i ] = m_summaryGrid.probes.psin[ i*2 ];       
        newGrid.probes.poloidalAngle[ i ] = m_summaryGrid.probes.poloidalAngle[ i*2 ];                
    }

    // const int64_t SZ = newGrid.probes.r.size();
    // std::vector< vtkm::Vec< vtkm::Float32, 2 > > pos( SZ );
    // #pragma omp parallel for simd
    // for( int64_t i = 0; i < SZ; ++i )
    // {
    //     pos[ i ] = vtkm::Vec< vtkm::Float32, 2 >( newGrid.probes.r[ i ], newGrid.probes.z[ i ] );
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

    // auto psinHandle = vtkm::cont::make_ArrayHandle( m_summaryGrid.probes.psin );
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

    // newGrid.probes.psin.resize( SZ );
    // #pragma omp parallel for simd
    // for( int64_t i = 0; i < SZ; ++i )
    // {
    //     newGrid.probes.psin[ i ] = fieldResultHandle.GetPortalControl().Get( i );
    // }

    // // B

    // MPI_Barrier(MPI_COMM_WORLD);
    // std::cout << "interpolating B " << std::endl;   

    // auto bHandle = vtkm::cont::make_ArrayHandle( m_summaryGrid.probes.B );
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

    // newGrid.probes.B.resize( SZ );
    // #pragma omp parallel for simd
    // for( int64_t i = 0; i < SZ; ++i )
    // {
    //     newGrid.probes.B[ i ] = fieldResultHandle.GetPortalControl().Get( i );
    // }

    // // Poloidal Angle

    // MPI_Barrier(MPI_COMM_WORLD);
    // std::cout << "calculating angles " << std::endl;

    // newGrid.probes.poloidalAngle.resize( SZ );
    // const TN::Vec2< float > poloidal_center = { m_constants.at( "eq_axis_r" ), m_constants.at( "eq_axis_z" ) };
    // #pragma omp parallel for simd
    // for( int64_t i = 0; i < SZ; ++i )
    // {
    //     newGrid.probes.poloidalAngle[ i ] =
    //         ( TN::Vec2< float >( newGrid.probes.r[ i ], newGrid.probes.z[ i ] )
    //           - poloidal_center ).angle( TN::Vec2< float >( 1.0, 0.0 ) );
    // }

    m_summaryGrid = newGrid;

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "setting static handles " << std::endl;

    setGrid(
        m_summaryGrid.probes.r,
        m_summaryGrid.probes.z,
        m_summaryGrid.probes.B,
        m_summaryGrid.neighborhoods,
        m_summaryGrid.neighborhoodSums );

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "done " << std::endl;   
}

void XGCAggregator::setGrid(
    const std::vector< float > & r,
    const std::vector< float > & z,
    const std::vector< float > & scalar,
    const std::vector< int64_t > & gridNeighborhoods,
    const std::vector< int64_t > & gridNeighborhoodSums )
{
    const int64_t N_CELLS = r.size();

    m_gridPoints.resize( N_CELLS );
    #pragma omp parallel for simd
    for( int64_t i = 0; i < N_CELLS; ++i )
    {
        m_gridPoints[ i ] = vtkm::Vec< vtkm::Float32, 2 >( r[ i ], z[ i ] );
    }
    m_gridHandle = vtkm::cont::make_ArrayHandle( m_gridPoints );

    m_gridNeighborhoods = std::vector< vtkm::Int64 >( gridNeighborhoods.begin(), gridNeighborhoods.end() );
    m_gridNeighborhoodsHandle = vtkm::cont::make_ArrayHandle( m_gridNeighborhoods );

    m_gridNeighborhoodSums = std::vector< vtkm::Int64 >( gridNeighborhoodSums.begin(), gridNeighborhoodSums.end() );
    m_gridNeighborhoodSumsHandle = vtkm::cont::make_ArrayHandle( m_gridNeighborhoodSums );

    m_gridScalars = std::vector< vtkm::Float32 >( scalar.begin(), scalar.end() );
    m_gridScalarHandle = vtkm::cont::make_ArrayHandle( m_gridScalars );

    m_kdTree.Build( m_gridHandle, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
}

void XGCAggregator::compute(
    std::vector< int64_t >     & result,
    std::vector< float >       & field,
    const std::vector< float > & r,
    const std::vector< float > & z )
{
    if( r.size() != z.size() )
    {
        std::cout << "r and z are different sizes in aggregator compute" << std::endl;
    }

    const int64_t SZ = r.size();
    std::vector< vtkm::Vec< vtkm::Float32, 2 > > ptclPos( SZ );

    #pragma omp parallel for simd
    for( int64_t i = 0; i < SZ; ++i )
    {
        ptclPos[ i ] = vtkm::Vec< vtkm::Float32, 2 >( r[ i ], z[ i ] );
    }

    auto ptclHandle = vtkm::cont::make_ArrayHandle( ptclPos );
    vtkm::cont::ArrayHandle<vtkm::Id> idHandle;
    vtkm::cont::ArrayHandle<vtkm::Float32> distHandle;

    std::cout << "Running kdtree neighbor mapper" << std::endl;

    m_kdTree.Run( m_gridHandle, ptclHandle, idHandle, distHandle, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    if( idHandle.GetPortalControl().GetNumberOfValues() != SZ )
    {
        std::cout << "kdtree returned wrong number of values" << std::endl;
    }

    result.resize( SZ );
    #pragma omp parallel for simd
    for( int64_t i = 0; i < SZ; ++i )
    {
        result[ i ] = idHandle.GetPortalControl().Get( i );
    }

    vtkm::cont::ArrayHandle<vtkm::Float32> fieldResultHandle;

    if( m_gridNeighborhoodSumsHandle.GetNumberOfValues() != m_gridHandle.GetNumberOfValues() )
    {
        std::cout << "grid neighborhoods has wrong number of values" << std::endl;
    }

    if( m_gridNeighborhoodSumsHandle.GetPortalControl().Get( m_gridNeighborhoodSumsHandle.GetNumberOfValues() - 1 ) 
        != m_gridNeighborhoodsHandle.GetNumberOfValues() )
    {
        std::cout << "wrong number of neighbors" << std::endl;
    }

    std::cout << "running interpolator" << std::endl;

    m_interpolator.run(
        ptclHandle,
        idHandle,
        m_gridHandle,
        m_gridScalarHandle,
        m_gridNeighborhoodsHandle,
        m_gridNeighborhoodSumsHandle,
        fieldResultHandle,
        VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    std::cout << "DONE" << std::endl;

    if( fieldResultHandle.GetNumberOfValues() != SZ )
    {
        std::cout << "field result has wrong number of values" << std::endl;
    }

    field.resize( SZ );
    #pragma omp parallel for simd
    for( int64_t i = 0; i < SZ; ++i )
    {
        field[ i ] = fieldResultHandle.GetPortalControl().Get( i );
    }

    std::cout << "copied " << std::endl;
}

void XGCAggregator::aggregate(
    const SummaryGrid & summaryGrid,
    SummaryStep       & summaryStep,
    const std::vector< float > & vX,
    const std::vector< float > & vY,
    const std::vector< float > & w,
    const std::vector< int64_t > & gIDs,
    const int64_t N_CELLS )
{
    const int64_t BINS_PER_CELL = SummaryStep::NR*SummaryStep::NC;

    auto vxHdl = vtkm::cont::make_ArrayHandle( vX );
    auto vyHdl = vtkm::cont::make_ArrayHandle( vY );
    auto wHdl  = vtkm::cont::make_ArrayHandle(  w );
    auto gHdl  = vtkm::cont::make_ArrayHandle(  gIDs );
    vtkm::worklet::Keys < int64_t > keys( gHdl, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    vtkm::cont::ArrayHandle<vtkm::Float32> meanHdl;
    vtkm::cont::ArrayHandle<vtkm::Float32> rmsHdl;
    vtkm::cont::ArrayHandle<vtkm::Float32> varHdl;
    vtkm::cont::ArrayHandle<vtkm::Float32> minHdl;
    vtkm::cont::ArrayHandle<vtkm::Float32> maxHdl;
    vtkm::cont::ArrayHandle<vtkm::Float32> cntHdl;
    vtkm::cont::ArrayHandle< vtkm::Vec< vtkm::Float32, BINS_PER_CELL > > histHndl;

    const vtkm::Vec< vtkm::Int32,    2 >  histDims = { SummaryStep::NR,       SummaryStep::NC      };
    const vtkm::Vec< vtkm::Float32,  2 >  xRange   = { -SummaryStep::DELTA_V, SummaryStep::DELTA_V };
    const vtkm::Vec< vtkm::Float32,  2 >  yRange   = { 0,                     SummaryStep::DELTA_V };

    std::cout << "running aggregate" << std::endl;

    m_aggregator.Run(
        N_CELLS,
        histDims,
        xRange,
        yRange,
        vxHdl,
        vyHdl,
        wHdl,
        keys,
        meanHdl,
        rmsHdl,
        varHdl,
        minHdl,
        maxHdl,
        cntHdl,
        histHndl,
        VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    std::cout << "done aggregating" << std::endl;

    summaryStep.w0w1_mean            = std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_rms             = std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_variance        = std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_min             = std::vector< float >( N_CELLS,  std::numeric_limits< float >::max() );
    summaryStep.w0w1_max             = std::vector< float >( N_CELLS, -std::numeric_limits< float >::max() );
    summaryStep.num_particles        = std::vector< float >( N_CELLS, 0.f );
    summaryStep.velocityDistribution = std::vector< float >( N_CELLS*SummaryStep::NR*SummaryStep::NC, 0.f );

    auto uniqueKeys = keys.GetUniqueKeys();
    const int64_t N_UK = uniqueKeys.GetNumberOfValues();

    std::cout << histHndl.GetNumberOfValues() << " values ";
    std::cout << sizeof( histHndl.GetPortalControl().Get( 0 ) ) << " is size of each" << std::endl;
    std::cout << summaryStep.velocityDistribution.size() << " is summary size of each" << std::endl;

    std::cout << "copying aggregation results" << std::endl;

    // #pragma omp parallel for simd
    for( int64_t i = 0; i < N_UK; ++i )
    {
        int64_t key = static_cast< int64_t >( uniqueKeys.GetPortalControl().Get( i ) );

        summaryStep.w0w1_mean[     key ] = meanHdl.GetPortalControl().Get( i );
        summaryStep.w0w1_rms[      key ] = rmsHdl.GetPortalControl().Get( i );
        summaryStep.w0w1_variance[ key ] = varHdl.GetPortalControl().Get( i );
        summaryStep.w0w1_min[      key ] = minHdl.GetPortalControl().Get( i );
        summaryStep.w0w1_max[      key ] = maxHdl.GetPortalControl().Get( i );
        summaryStep.num_particles[ key ] = cntHdl.GetPortalControl().Get( i );

        for( int64_t j = 0; j < BINS_PER_CELL; ++j )
        {
            if( key * BINS_PER_CELL + j >= N_CELLS * BINS_PER_CELL )
            {
                std::cout << "error " << key * BINS_PER_CELL + j << " / " << N_CELLS * BINS_PER_CELL << std::endl;
                exit( 1 );
            }
            if( key >= N_CELLS  )
            {
                std::cout << "error " << key << " / " << N_CELLS << std::endl;
                exit( 1 );
            }

            summaryStep.velocityDistribution[ key * BINS_PER_CELL + j ]
                = histHndl.GetPortalControl().Get( i )[ j ];
        }
    }
}

void XGCAggregator::writeGrid( const std::string & path )
{
    TN::writeSummaryGrid( m_summaryGrid, path );
}

void XGCAggregator::computeSummaryStep(
    TN::SummaryStep & summaryStep, 
    const std::string & ptype,
    int64_t st )
{
    // need r, z, phi, B, mu, rho_parallel, w0, w1

    std::cout << m_restartDirectory << std::endl;
    std::string tstep = std::to_string( st );

    std::cout << "reading particle step " << std::endl;
    std::chrono::high_resolution_clock::time_point readStartTime = std::chrono::high_resolution_clock::now();
    readBPParticleDataStep(
        m_phase,
        ptype,
        m_restartDirectory + "xgc.restart." + std::string( 5 - tstep.size(), '0' ) + tstep +  ".bp",
        m_rank,
        m_nranks );
    std::chrono::high_resolution_clock::time_point readStartEnd = std::chrono::high_resolution_clock::now();
    std::cout << "RANK: " << m_rank
         << ", adios Read time took "
         << std::chrono::duration_cast<std::chrono::milliseconds>( readStartEnd - readStartTime ).count()
         << " milliseconds " << " for " << m_phase.size()/9 << " particles" << std::endl;

    const size_t SZ      = m_phase.size() / 9;
    const size_t R_POS   = XGC_PHASE_INDEX_MAP.at( "r" ) * SZ;
    const size_t Z_POS   = XGC_PHASE_INDEX_MAP.at( "z" ) * SZ;
    const size_t RHO_POS = XGC_PHASE_INDEX_MAP.at( "rho_parallel" ) * SZ;
    const size_t W1_POS  = XGC_PHASE_INDEX_MAP.at( "w1" ) * SZ;
    const size_t W0_POS  = XGC_PHASE_INDEX_MAP.at( "w0" ) * SZ;
    const size_t MU_POS  = XGC_PHASE_INDEX_MAP.at( "mu" ) * SZ;

    // for VTKM nearest neighbors and B field Interpolation //////////////////////

    std::vector< int64_t > gridMap;
    std::chrono::high_resolution_clock::time_point kdt1 = std::chrono::high_resolution_clock::now();

    std::vector< float > r( SZ );
    std::vector< float > z( SZ );

    std::cout << "end is " << R_POS + SZ << std::endl;

    for( size_t i = 0; i < SZ; ++i )
    {
        r[ i ] = m_phase[ R_POS + i ];
        z[ i ] = m_phase[ Z_POS + i ];
    }

    m_B.resize( SZ );
    gridMap.resize( SZ );

    compute( gridMap, m_B, r, z );

    std::chrono::high_resolution_clock::time_point kdt2 = std::chrono::high_resolution_clock::now();
    std::cout << "RANK: " << m_rank
         << ", MPI kdtree mapping CHUNK took "
         << std::chrono::duration_cast<std::chrono::milliseconds>( kdt2 - kdt1 ).count()
         << " milliseconds " << " for " << r.size() << " particles" << std::endl;

    {
        for( auto gID : gridMap )
        {
            if( gID < 0 || gID >= m_summaryGrid.probes.volume.size() )
            {
                std::cout << "gID invalid " << gID << std::endl;
            }
        }
    }

    // compute velocity and weight
    std::vector< float > vpara( SZ );
    std::vector< float > vperp( SZ );
    std::vector< float >  w0w1( SZ );

    #pragma omp parallel for simd
    for( size_t i = 0; i < SZ; ++i )
    {
        w0w1[  i ] = m_phase[ W0_POS + i ] * m_phase[ W1_POS + i ];
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
            vpara[ i ] = m_B[ i ] * m_phase[ RHO_POS + i ] * ( ( ptl_ion_charge_eu * e ) / mi_sim );
            vperp[ i ] = sqrt( ( m_phase[   MU_POS + i ] * 2.0 * m_B[ i ] ) / mi_sim );
        }
    }
    else
    {
        #pragma omp parallel for simd
        for( size_t i = 0; i < SZ; ++i )
        {
            vpara[ i ] =( m_B[ i ] * m_phase[ RHO_POS + i ] * (-e) ) / me_sim;
            vperp[ i ] = sqrt( ( m_phase[    MU_POS + i ] * 2.0 * m_B[ i ]  ) / me_sim  );
        }
    }

    // compute summations over particles in each cell

    std::chrono::high_resolution_clock::time_point st1 = std::chrono::high_resolution_clock::now();

    // With VTKM

    // aggregate(
    //     m_summaryGrid,
    //     summaryStep,
    //     vpara,
    //     vperp,
    //     w0w1,
    //     gridMap,
    //     m_summaryGrid.probes.volume.size() );

    // Without VTKM

    const int NR = SummaryStep::NR;
    const int NC = SummaryStep::NC;
    const int64_t DIST_STRIDE = NR*NC;

    const size_t N_CELLS = m_summaryGrid.probes.volume.size();

    summaryStep.w0w1_mean            = std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_rms             = std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_min             = std::vector< float >( N_CELLS,  std::numeric_limits< float >::max() );
    summaryStep.w0w1_max             = std::vector< float >( N_CELLS, -std::numeric_limits< float >::max() );
    summaryStep.num_particles        = std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_variance        = std::vector< float >( N_CELLS, 0.f );
    summaryStep.velocityDistribution = std::vector< float >( N_CELLS*NR*NC, 0.f );

    std::cout << "computing histograms " << std::endl;

    #pragma omp simd
    for( size_t i = 0; i < SZ; ++i )
    {
        int64_t index = gridMap[ i ];

        summaryStep.w0w1_mean[ index ] += w0w1[ i ];
        summaryStep.w0w1_rms[  index ] += w0w1[ i ] * w0w1[ i ];
        summaryStep.w0w1_min[  index ] = std::min( w0w1[ i ], summaryStep.w0w1_min[ index ] );
        summaryStep.w0w1_max[  index ] = std::max( w0w1[ i ], summaryStep.w0w1_max[ index ] );
        summaryStep.num_particles[ index ] += 1.0;

        // map to velocity distribution bin

        const  float VPARA_MIN = -SummaryStep::DELTA_V;
        const  float VPARA_MAX =  SummaryStep::DELTA_V;

        const  float VPERP_MIN = 0;
        const  float VPERP_MAX = SummaryStep::DELTA_V - VPERP_MIN;

        const float R_WIDTH = VPERP_MAX;
        const float C_WIDTH = VPARA_MAX - VPARA_MIN;

        int row = std::floor( ( ( vperp[ i ] - VPERP_MIN ) / R_WIDTH ) * NR );
        int col = std::floor( ( ( vpara[ i ] - VPARA_MIN ) / C_WIDTH ) * NC );

        row = std::max( std::min( row, NR - 1 ), 0 );
        col = std::max( std::min( col, NC - 1 ), 0 );

        // summaryStep.num_mapped[ index ] += 1.0;
        summaryStep.velocityDistribution[ index * DIST_STRIDE + row * NC + col ] += w0w1[ i ];
    }

    std::cout << "done computing histograms " << std::endl;

    #pragma omp simd
    for( size_t i = 0; i < SZ; ++i )
    {
        int64_t index = gridMap[ i ];
        const double v = w0w1[ i ] - summaryStep.w0w1_mean[ index ] / summaryStep.num_particles[ index ];
        summaryStep.w0w1_variance[ index ] += v*v;
    }

    std::chrono::high_resolution_clock::time_point st2 = std::chrono::high_resolution_clock::now();
    std::cout << "RANK: "  << m_rank 
              << ", MPI summarization processing CHUNK took " 
              << std::chrono::duration_cast<std::chrono::milliseconds>( st2 - st1 ).count() << " milliseconds " 
              << " for " << r.size() << " particles" << std::endl;
}


}