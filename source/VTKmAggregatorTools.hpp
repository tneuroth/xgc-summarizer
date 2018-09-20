#ifndef VTKM_SUMMARIZER_TOOLS
#define VTKM_SUMMARIZER_TOOLS

// #include <vtkm/cont/DeviceAdapterAlgorithm.h>
// #include <vtkm/worklet/DispatcherMapField.h>

#include "VTKmAggregator.hpp"
#include "VTKmInterpolator.hpp"

#include <vector>
#include <cstdint>
#include <KDTree/KdTree.h>
#include "VTKmInterpolator.hpp"
#include "VTKmAggregator.hpp"

namespace TN
{

template< typename ValueType >
struct VTKmAggregatorTools
{
    vtkm::worklet::KdTree< 2 > m_kdTree;
    TN::VTKmInterpolator2D m_interpolator;

    vtkm::cont::ArrayHandle< vtkm::Vec< ValueType, 2 > > m_gridHandle;
    std::vector< vtkm::Vec< ValueType, 2 > > m_gridPoints;

    vtkm::cont::ArrayHandle< ValueType > m_gridScalarHandle;
    std::vector< ValueType > m_gridScalars;

    vtkm::cont::ArrayHandle< vtkm::Int64 > m_gridNeighborhoodsHandle;
    std::vector< vtkm::Int64 > m_gridNeighborhoods;

    vtkm::cont::ArrayHandle< vtkm::Int64 > m_gridNeighborhoodSumsHandle;
    std::vector< vtkm::Int64 > m_gridNeighborhoodSums;

    vtkm::cont::ArrayHandle< vtkm::UInt8 > m_vertexFlagsHandle;

    VTKmAggregatorTools( bool tryUsingCuda );

    void setGrid(
        const std::vector< ValueType > & r,
        const std::vector< ValueType > & z,
        const std::vector< ValueType > & scalar,
        const std::vector< int64_t > & gridNeighborhoods,
        const std::vector< int64_t > & gridNeighborhoodSums,
        const std::vector< uint8_t > & vertexFlags );

    void compute(
        std::vector< int64_t >         & result,
        std::vector< ValueType >       & field,
        const std::vector< ValueType > & r,
        const std::vector< ValueType > & z,
        int64_t maxNeighbors );
};


}

#endif
