
#include "ParticleMeshInterpolator2D.hpp"
#include "../Summary.hpp"

// #include <vtkm/worklet/KdTree3D.h>
#include <vtkm/cont/DeviceAdapterAlgorithm.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vector>
#include <chrono>

using namespace std;

template< typename DeviceAdapter >
void checkDevice(DeviceAdapter)
{
    using DeviceAdapterTraits = vtkm::cont::DeviceAdapterTraits<DeviceAdapter>;
    cout << "vtkm is using " << DeviceAdapterTraits::GetName() << endl;
}

namespace TN
{

ParticleMeshInterpolator2D::ParticleMeshInterpolator2D()
{
    checkDevice( VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
}

void ParticleMeshInterpolator2D::setGrid( 
    const vector< float > & r,
    const vector< float > & z,
    const vector< float > & scalar,
    const vector< int64_t > & gridNeighborhoods,
    const vector< int64_t > & gridNeighborhoodSums )
{ 
    const int64_t N_CELLS = r.size();

    m_gridPoints.resize( N_CELLS );
    #pragma omp parallel for simd
    for( int64_t i = 0; i < N_CELLS; ++i )
    {
        m_gridPoints[ i ] = vtkm::Vec< vtkm::Float32, 2 >( r[ i ], z[ i ] );
    }
    m_gridHandle = vtkm::cont::make_ArrayHandle( m_gridPoints );

    m_gridNeighborhoods = vector< vtkm::Int64 >( gridNeighborhoods.begin(), gridNeighborhoods.end() );
    m_gridNeighborhoodsHandle = vtkm::cont::make_ArrayHandle( m_gridNeighborhoods );

    m_gridNeighborhoodSums = vector< vtkm::Int64 >( gridNeighborhoodSums.begin(), gridNeighborhoodSums.end() );
    m_gridNeighborhoodSumsHandle = vtkm::cont::make_ArrayHandle( m_gridNeighborhoodSums );  

    m_gridScalars = vector< vtkm::Float32 >( scalar.begin(), scalar.end() );  
    m_gridScalarHandle = vtkm::cont::make_ArrayHandle( m_gridScalars );    

    m_kdTree.Build( m_gridHandle, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
}

void ParticleMeshInterpolator2D::compute(
    vector< int64_t > & result,
    vector< float >   & field,
    const vector< float > & r,
    const vector< float > & z )
{
    const int64_t SZ = r.size();
    vector< vtkm::Vec< vtkm::Float32, 2 > > ptclPos( SZ );

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

void ParticleMeshInterpolator2D::aggregate( 
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

    const vtkm::Vec< vtkm::Int32,    2 >  histDims = { SummaryStep::NR,           SummaryStep::NC };  
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

    summaryStep.w0w1_mean            = std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_rms             = std::vector< float >( N_CELLS, 0.f );
    summaryStep.w0w1_variance        = std::vector< float >( N_CELLS, 0.f );    
    summaryStep.w0w1_min             = std::vector< float >( N_CELLS,  numeric_limits< float >::max() );
    summaryStep.w0w1_max             = std::vector< float >( N_CELLS, -numeric_limits< float >::max() );
    summaryStep.num_particles        = std::vector< float >( N_CELLS, 0.f );
    summaryStep.velocityDistribution = std::vector< float >( N_CELLS*SummaryStep::NR*SummaryStep::NC, 0.f );

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

}