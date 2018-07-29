#ifndef TN_PTCL_MESH_INTERPOLATOR_HPP
#define TN_PTCL_MESH_INTERPOLATOR_HPP

#include "Summary.hpp"
#include <KDTree/KdTree.h>
#include "VTKmInterpolator.hpp"
#include "VTKmAggregator.hpp"

#include <vector>

namespace TN
{

class ParticleMeshInterpolator2D
{

private:

    vtkm::worklet::KdTree< 2 > m_kdTree;
    TN::VTKmInterpolator2D m_interpolator;
    TN::VTKmAggregator m_aggregator;

    vtkm::cont::ArrayHandle< vtkm::Vec< vtkm::Float32, 2 > > m_gridHandle;
    std::vector< vtkm::Vec< vtkm::Float32, 2 > > m_gridPoints;

    vtkm::cont::ArrayHandle< vtkm::Float32 > m_gridScalarHandle;
    std::vector< vtkm::Float32 > m_gridScalars;

    vtkm::cont::ArrayHandle< vtkm::Int64 > m_gridNeighborhoodsHandle;
    std::vector< vtkm::Int64 > m_gridNeighborhoods;

    vtkm::cont::ArrayHandle< vtkm::Int64 > m_gridNeighborhoodSumsHandle;
    std::vector< vtkm::Int64 > m_gridNeighborhoodSums;

public:

    void setGrid(
        const std::vector< float >   & r,
        const std::vector< float >   & z,
        const std::vector< float > & scalar,
        const std::vector< int64_t > & gridNeighborhoods,
        const std::vector< int64_t > & gridNeighborhoodSums
    );

    void compute(
        std::vector< int64_t > & neighbors,
        std::vector< float   > & field,
        const std::vector< float > & r,
        const std::vector< float > & z );

    void aggregate(
        const SummaryGrid & summaryGrid,
        SummaryStep & summary,
        const std::vector< float > & vX,
        const std::vector< float > & vY,
        const std::vector< float > & w,
        const std::vector< int64_t > & gIDs,
        const int64_t N_CELLS );

    ParticleMeshInterpolator2D();
};

}

#endif