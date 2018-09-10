#ifndef TN_PTCL_MESH_INTERPOLATOR_HPP
#define TN_PTCL_MESH_INTERPOLATOR_HPP

#include "Summary.hpp"
#include <adios2.h>
#include <KDTree/KdTree.h>
#include "VTKmInterpolator.hpp"
#include "VTKmAggregator.hpp"
#include <map>
#include <set>
#include <string>
#include <vector>
#include "mpi.h"

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

template < typename ValueType >
class XGCAggregator
{
    std::string m_meshFilePath;
    std::string m_bFieldFilePath;
    std::string m_restartPath;
    std::string m_unitsMFilePath;
    std::string m_outputDirectory;
    std::string m_particleReaderEngine;

    bool m_inSitu;
    bool m_splitByBlocks;
    bool m_summaryWriterAppendMode;

    int m_rank;
    int m_nranks;
    MPI_Comm m_mpiCommunicator;

    std::vector< ValueType > m_phase;
    std::vector< ValueType > m_B;

    void computeSummaryStep(
        std::vector< ValueType > & phase,
        TN::SummaryStep2< ValueType > & summaryStep,
        const std::string & ptype,
        std::unique_ptr< adios2::IO > & summaryIO,
        std::unique_ptr< adios2::Engine > & summaryWriter );

public:


    XGCAggregator(
        const std::string & adiosConfigFilePath,
        const std::string & meshFilePath,
        const std::string & bfieldFilePath,
        const std::string & restartDirectory,
        const std::string & unitsFilePath,
        const std::string & outputDirectory,
        const std::set< std::string > & particleTypes,
        bool inSitu,
        bool m_splitByBlocks,
        int rank,
        int nranks,
        MPI_Comm communicator );

    void runInSitu();
    void runInPost();

    void run();
    void writeMesh();

private:

    TN::SummaryGrid2< ValueType > m_summaryGrid;

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

    std::map< std::string, double > m_constants;

    void setGrid(
        const std::vector< ValueType > & r,
        const std::vector< ValueType > & z,
        const std::vector< ValueType > & scalar,
        const std::vector< int64_t >   & gridNeighborhoods,
        const std::vector< int64_t >   & gridNeighborhoodSums
    );

    void compute(
        std::vector< int64_t > & neighbors,
        std::vector< ValueType > & field,
        const std::vector< ValueType > & r,
        const std::vector< ValueType > & z );

    void aggregateOMP(
        const SummaryGrid2< ValueType > & summaryGrid,
        SummaryStep2< ValueType >       & summaryStep,
        const std::vector< ValueType >  & vX,
        const std::vector< ValueType >  & vY,
        const std::vector< ValueType >  & w0,
        const std::vector< ValueType >  & w1,
        const std::vector< int64_t >    & gIDs,
        const int64_t N_CELLS );

    void writeGrid( const std::string & path );
};

}

#endif
