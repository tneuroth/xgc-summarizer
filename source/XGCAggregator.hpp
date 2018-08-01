#ifndef TN_PTCL_MESH_INTERPOLATOR_HPP
#define TN_PTCL_MESH_INTERPOLATOR_HPP

#include "Summary.hpp"
#include <KDTree/KdTree.h>
#include "VTKmInterpolator.hpp"
#include "VTKmAggregator.hpp"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace TN
{

const std::map< std::string, int > XGC_PHASE_INDEX_MAP =
{
    { "r",            0 },   // Major radius [m]
    { "z",            1 },   // Azimuthal direction [m]
    { "zeta",         2 },   // Toroidal angle
    { "rho_parallel", 3 },   // Parallel Larmor radius [m]
    { "w1",           4 },   // Grid weight 1
    { "w2",           5 },   // Grid weight 2
    { "mu",           6 },   // Magnetic moment
    { "w0",           7 },   // Grid weight
    { "f0",           8 }    // Grid distribution function (?)
};

class XGCAggregator
{

    std::string m_meshFilePath;
    std::string m_bFieldFilePath;
    std::string m_restartDirectory;
    std::string m_unitsMFilePath;
    std::string m_outputDirectory;

    int m_rank; 
    int m_nranks;

    std::vector< float > m_phase;
    std::vector< float > m_B;

public:

	void computeSummaryStep(
	    TN::SummaryStep & summaryStep,
        const std::string & ptype,
	    int64_t st );

	XGCAggregator(
		const std::string & meshFilePath,
		const std::string & bfieldFilePath,
        const std::string & restartDirectory,
        const std::string & unitsFilePath,
        const std::string & outputDirectory,
        const std::set< std::string > & particleTypes,
        int rank,
        int nranks );

    void reduceMesh( const std::string & reducedMeshFilePath );

    void writeMesh();

private:

    TN::SummaryGrid m_summaryGrid;

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

    std::map< std::string, double > m_constants;

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

    void writeGrid( const std::string & path );
};

}

#endif
