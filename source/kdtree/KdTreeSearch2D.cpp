
#include "KdTreeSearch2D.hpp"

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

KdTreeSearch2D::KdTreeSearch2D()
{
    checkDevice( VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
}

void KdTreeSearch2D::setGrid( 
		const vector< float > & r,
	    const vector< float > & z )
{
    cout << "Building KDTree\n";
 
    const int64_t N_CELLS = r.size();    
    m_gridPoints.resize( N_CELLS );

    #pragma omp parallel for simd
    for( int64_t i = 0; i < N_CELLS; ++i )
    {
        m_gridPoints[ i ] = vtkm::Vec< vtkm::Float32, 2 >( r[ i ], z[ i ] );
    }    

    vtkm::worklet::KdTree<2> kdtree;
    auto gt1 = chrono::high_resolution_clock::now();
    
    m_gridHandle = vtkm::cont::make_ArrayHandle( m_gridPoints );
    m_kdTree.Build( m_gridHandle, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
    
    auto gt2 = chrono::high_resolution_clock::now();

    cout << "building KdTree took "
              << chrono::duration_cast<chrono::milliseconds>(gt2-gt1).count()
              << " milliseconds\n";
}

void KdTreeSearch2D::run(
    vector< int64_t > & result,
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
}

}