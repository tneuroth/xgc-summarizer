
#include "XGCAggregator.hpp"
#include "Summary.hpp"
#include "SummarWriter.hpp"
#include "XGCMeshReader.hpp"
#include "XGCParticleReader.hpp"

#include <vtkm/cont/DeviceAdapterAlgorithm.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <std::vector>
#include <chrono>

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
    int nranks )
{
    TN::loadConstants( unitsFilePath, m_constants );

    TN::readMeshBP(
        m_summarGrid,
        { m_constants.at( "eq_axis_r" ), m_constants.at( "eq_axis_z" ) },
        meshPath,
        bFieldPath ); 
    
    setGrid(
        m_summaryGrid.probes.r,
        m_summaryGrid.probes.z,
        m_summaryGrid.probes.B,
        m_summaryGrid.neighborhoods,
        m_summaryGrid.neighborhoodSums );

    if( rank == 0 )
    {
        checkDevice( VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
        writeGrid( outputDirectory );
    }
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

void XGCAggregator::aggregate(
    const SummaryGrid & summaryGrid,
    SummaryStep       & summaryStep,
    const std::std::vector< float > & vX,
    const std::std::vector< float > & vY,
    const std::std::vector< float > & w,
    const std::std::vector< int64_t > & gIDs,
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

    summaryStep.w0w1_mean            = std::std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_rms             = std::std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_variance        = std::std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_min             = std::std::vector< float >( N_CELLS,  numeric_limits< float >::max() );
    summaryStep.w0w1_max             = std::std::vector< float >( N_CELLS, -numeric_limits< float >::max() );
    summaryStep.num_particles        = std::std::vector< float >( N_CELLS, 0.f );
    summaryStep.velocityDistribution = std::std::vector< float >( N_CELLS*SummaryStep::NR*SummaryStep::NC, 0.f );

    auto uniqueKeys = keys.GetUniqueKeys();
    const int64_t N_UK = uniqueKeys.GetNumberOfValues();
    std::cout << histHndl.GetNumberOfValues() << " values ";
    std::cout << sizeof( histHndl.GetPortalControl().Get( 0 ) ) << " is size of each" << std::endl;
    std::cout << summaryStep.velocityDistribution.size() << " is summary size of each" << std::endl;

    // #pragma omp parallel for simd
    for( int64_t i = 0; i < N_UK; ++i )
    {
        int64_t key = static_cast< int64_t >( uniqueKeys.GetPortalControl().Get( i ) );

        summaryStep.w0w1_mean[ key ] = meanHdl.GetPortalControl().Get( i );
        summaryStep.w0w1_rms[  key ] = rmsHdl.GetPortalControl().Get( i );
        // // summaryStep.w0w1_variance[ i ] = varHdl.GetPortalControl().Get( i );
        summaryStep.w0w1_min[ key ] = minHdl.GetPortalControl().Get( i );
        summaryStep.w0w1_max[ key ] = maxHdl.GetPortalControl().Get( i );
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
    int64_t st )
{
    // need r, z, phi, B, mu, rho_parallel, w0, w1
    std::vector< float > B;

    std::cout << particle_base_path << std::endl;
    std::string tstep = to_string( st );

    std::vector< float > iphase;
    high_resolution_clock::time_point readStartTime = high_resolution_clock::now();
    readBPParticleDataStep(
        iphase,
        ptype,
        particle_base_path + "xgc.restart." + string( 5 - tstep.size(), '0' ) + tstep +  ".bp",
        rank,
        nRanks );
    high_resolution_clock::time_point readStartEnd = high_resolution_clock::now();
    std::cout << "RANK: " << rank
         << ", adios Read time took "
         << duration_cast<milliseconds>( readStartEnd - readStartTime ).count()
         << " milliseconds " << " for " << iphase.size()/9 << " particles" << std::endl;

    const size_t SZ      = iphase.size() / 9;
    const size_t R_POS   = XGC_PHASE_INDEX_MAP.at( "r" ) * SZ;
    const size_t Z_POS   = XGC_PHASE_INDEX_MAP.at( "z" ) * SZ;
    const size_t RHO_POS = XGC_PHASE_INDEX_MAP.at( "rho_parallel" ) * SZ;
    const size_t W1_POS  = XGC_PHASE_INDEX_MAP.at( "w1" ) * SZ;
    const size_t W0_POS  = XGC_PHASE_INDEX_MAP.at( "w0" ) * SZ;
    const size_t MU_POS  = XGC_PHASE_INDEX_MAP.at( "mu" ) * SZ;

    // for VTKM nearest neighbors and B field Interpolation //////////////////////

    std::vector< int64_t > gridMap;
    high_resolution_clock::time_point kdt1 = high_resolution_clock::now();

    std::vector< float > r( SZ );
    std::vector< float > z( SZ );

    for( size_t i = 0; i < SZ; ++i )
    {
        r[ i ] = iphase[ R_POS + i ];
        z[ i ] = iphase[ Z_POS + i ];
    }

    B.resize( SZ );
    gridMap.resize( SZ );
    interpolator.compute( gridMap, B, r, z );

    high_resolution_clock::time_point kdt2 = high_resolution_clock::now();
    std::cout << "RANK: " << rank
         << ", MPI kdtree mapping CHUNK took "
         << duration_cast<milliseconds>( kdt2 - kdt1 ).count()
         << " milliseconds " << " for " << r.size() << " particles" << std::endl;

    // compute velocity and weight
    std::vector< float > vpara( SZ );
    std::vector< float > vperp( SZ );
    std::vector< float >  w0w1( SZ );

    #pragma omp parallel for simd
    for( size_t i = 0; i < SZ; ++i )
    {
        w0w1[  i ] = iphase[ W0_POS + i ] * iphase[ W1_POS + i ];
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
            vpara[ i ] = B[ i ] * iphase[ RHO_POS + i ] * ( ( ptl_ion_charge_eu * e ) / mi_sim );
            vperp[ i ] = sqrt( ( iphase[   MU_POS + i ] * 2.0 * B[ i ] ) / mi_sim );
        }
    }
    else
    {
        #pragma omp parallel for simd
        for( size_t i = 0; i < SZ; ++i )
        {
            vpara[ i ] =( B[ i ] * iphase[ RHO_POS + i ] * (-e) ) / me_sim;
            vperp[ i ] = sqrt( ( iphase[    MU_POS + i ] * 2.0 * B[ i ]  ) / me_sim  );
        }
    }

    // compute summations over particles in each cell

    high_resolution_clock::time_point st1 = high_resolution_clock::now();

    // With VTKM

    aggregate(
        m_summaryGrid,
        summaryStep,
        vpara,
        vperp,
        w0w1,
        gridMap,
        m_summaryGrid.probes.volume.size() );

    // Without VTKM

    // const int NR = SummaryStep::NR;
    // const int NC = SummaryStep::NC;
    // const int64_t DIST_STRIDE = NR*NC;

    // const size_t N_CELLS = summaryGrid.probes.volume.size();

    // summaryStep.w0w1_mean            = std::vector< float >( N_CELLS, 0.f );
    // summaryStep.w0w1_rms             = std::vector< float >( N_CELLS, 0.f );
    // summaryStep.w0w1_min             = std::vector< float >( N_CELLS,  numeric_limits< float >::max() );
    // summaryStep.w0w1_max             = std::vector< float >( N_CELLS, -numeric_limits< float >::max() );
    // summaryStep.num_particles        = std::vector< float >( N_CELLS, 0.f );
    // summaryStep.w0w1_variance        = std::vector< float >( N_CELLS, 0.f );
    // summaryStep.velocityDistribution = std::vector< float >( N_CELLS*NR*NC, 0.f );

    // #pragma omp simd
    // for( size_t i = 0; i < SZ; ++i )
    // {
    //     int64_t index = gridMap[ i ];

    //     if( index >= 0 )
    //     {
    //         summaryStep.w0w1_mean[ index ] += w0w1[ i ];
    //         summaryStep.w0w1_rms[  index ] += w0w1[ i ] * w0w1[ i ];
    //         summaryStep.w0w1_min[  index ] = min( w0w1[ i ], summaryStep.w0w1_min[ index ] );
    //         summaryStep.w0w1_max[  index ] = max( w0w1[ i ], summaryStep.w0w1_max[ index ] );
    //         summaryStep.num_particles[ index ] += 1.0;

    //         // map to velocity distribution bin

    //         const  float VPARA_MIN = -SummaryStep::DELTA_V;
    //         const  float VPARA_MAX =  SummaryStep::DELTA_V;

    //         const  float VPERP_MIN = 0;
    //         const  float VPERP_MAX = SummaryStep::DELTA_V - VPERP_MIN;

    //         const float R_WIDTH = VPERP_MAX;
    //         const float C_WIDTH = VPARA_MAX - VPARA_MIN;

    //         int row = floor( ( ( vperp[ i ] - VPERP_MIN ) / R_WIDTH ) * NR );
    //         int col = floor( ( ( vpara[ i ] - VPARA_MIN ) / C_WIDTH ) * NC );

    //         row = max( min( row, NR - 1 ), 0 );
    //         col = max( min( col, NC - 1 ), 0 );

    //         // summaryStep.num_mapped[ index ] += 1.0;
    //         summaryStep.velocityDistribution[ index * DIST_STRIDE + row * NC + col ] += w0w1[ i ];
    //     }
    // }

    high_resolution_clock::time_point st2 = high_resolution_clock::now();
    std::cout << "RANK: "  << rank 
              << ", MPI summarization processing CHUNK took " 
              << duration_cast<milliseconds>( st2 - st1 ).count() << " milliseconds " 
              << " for " << r.size() << " particles" << std::endl;
}


}