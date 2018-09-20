#include "VTKmAggregatorTools.hpp"

namespace TN
{

template< typename ValueType >
VTKmAggregatorTools< ValueType >::VTKmAggregatorTools< ValueType >( bool truUseCuda )
{
    if( tryUseCuda && 
        vtkm::cont::GetGlobalRuntimeDeviceTracker().CanRunOn( 
            vtkm::cont::DeviceAdapterTagCuda() ) )
    {
        vtkm::cont::GetGlobalRuntimeDeviceTracker().ForceDevice( 
            vtkm::cont::DeviceAdapterTagCuda() );
    }
    else
    {
        vtkm::cont::GetGlobalRuntimeDeviceTracker().DisableDevice( 
            vtkm::cont::DeviceAdapterTagCuda() );
    }
}

template< typename ValueType >
void VTKmAggregatorTools< ValueType >::setGrid(
    const std::vector< ValueType > & r,
    const std::vector< ValueType > & z,
    const std::vector< ValueType > & scalar,
    const std::vector< int64_t > & gridNeighborhoods,
    const std::vector< int64_t > & gridNeighborhoodSums,
    const std::vector< uint8_t > & vertexFlags )
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

    m_vertexFlagsHandle = vtkm::cont::make_ArrayHandle( vertexFlags );

    m_kdTree.Build( m_gridHandle, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );

    vtkm::cont:: DeviceAdapterId currentDevice = m_gridHandle.GetDeviceAdapterId();
    std::cout << "VTKM device is " << currentDevice.GetName() << std::endl;
}

template< typename ValueType >
void VTKmAggregatorTools< ValueType >::compute(
    std::vector< int64_t >         & result,
    std::vector< ValueType >       & field,
    const std::vector< ValueType > & r,
    const std::vector< ValueType > & z,
    int64_t maxNeighbors )
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
        m_vertexFlagsHandle,
        m_gridScalarHandle,
        m_gridNeighborhoodsHandle,
        m_gridNeighborhoodSumsHandle,
        maxNeighbors,
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


template class VTKmAggregatorTools<float>;

}