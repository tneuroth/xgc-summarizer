
#include "ParticleMeshInterpolator2D.hpp"

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

}