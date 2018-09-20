#ifndef TN_PTCL_MESH_INTERPOLATOR_HPP
#define TN_PTCL_MESH_INTERPOLATOR_HPP

#include "Summary.hpp"
#include "VTKmAggregatorTools.hpp"

#include <adios2.h>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <vector>
#include <memory>

#include "mpi.h"

namespace TN
{

const std::unordered_map< std::string, int > XGC_PHASE_INDEX_MAP =
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
        MPI_Comm communicator,
        bool tryUsingCuda );

    void run();

    void writeMesh();

private:

    std::string m_meshFilePath;
    std::string m_bFieldFilePath;
    std::string m_particleFile;
    std::string m_unitsMFilePath;
    std::string m_outputDirectory;
    std::string m_particleReaderEngine;

    double m_superParticleThreshold;

    bool m_inSitu;
    bool m_splitByBlocks;
    bool m_summaryWriterAppendMode;

    int m_rank;
    int m_nranks;
    MPI_Comm m_mpiCommunicator;

    std::vector< std::int64_t > m_particleIds;
    std::vector< ValueType > m_phase;
    std::vector< ValueType > m_B;

    void computeSummaryStep(
        const std::vector< ValueType > & phase,
        const std::vector< int64_t > & ids,
        TN::SummaryStep< ValueType > & summaryStep,
        std::unordered_set< int64_t > & trackedParticleIds,
        const std::string & ptype,
        std::unique_ptr< adios2::IO > & summaryIO,
        std::unique_ptr< adios2::Engine > & summaryWriter,
        std::unique_ptr< adios2::IO > & particlePathI,
        std::unique_ptr< adios2::Engine > & particlePathWriter );

    void runInSitu();
    void runInPost();

    TN::SummaryGrid< ValueType > m_summaryGrid;
    TN::VTKmAggregatorTools< ValueType > m_aggregatorTools;

    std::map< std::string, double > m_constants;

    void aggregateOMP(
        const SummaryGrid< ValueType > & summaryGrid,
        SummaryStep< ValueType >       & summaryStep,
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
