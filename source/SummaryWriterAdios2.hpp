#ifndef TN_SUMMARY_WRITER
#define TN_SUMMARY_WRITER

#include "Summary.hpp"

#include <adios2.h>

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstdint>

namespace TN
{

template< typename ValueType >
inline void writeTriangularMeshObj(
    const std::vector< ValueType > & r,
    const std::vector< ValueType > & z,
    const std::vector< TN::Triangle > & mesh,
    const std::string & outpath )
{
    std::ofstream outfile( outpath );

    if( ! outfile.is_open() )
    {
        std::cerr << "failed to open " << outpath << std::endl;
    }

    for( std::size_t i = 0, end = r.size(); i < end; ++i )
    {
        outfile << "v " << r[ i ] << " " << z[ i ] << " 0\n";
    }
    outfile << "\n";

    for( std::size_t i = 0, end = mesh.size(); i < end; ++i )
    {
        outfile << "f";
        for( std::size_t j = 0; j < 3; ++j )
        {
            outfile << " " << mesh[ i ][ j ] + 1;
        }
        outfile << "\n";
    }
    outfile.close();
}

template< typename ValueType >
inline void writeSummaryGridBP(
    const SummaryGrid< ValueType > & summaryGrid,
    const std::string & outpath )
{
    adios2::ADIOS adios( MPI_COMM_SELF, adios2::DebugOFF );
    adios2::IO bpIO = adios.DeclareIO( "XGC-SUMMARY-GRID-IO" );
    adios2::Engine bpWriter = bpIO.Open( outpath + "/summary.mesh.bp", adios2::Mode::Write );

    for( auto & var : summaryGrid.variables )
    {
        bpWriter.Put< ValueType >(
            bpIO.DefineVariable< ValueType >(
            "fields/" + var.first,
        { var.second.size() },
        { 0 },
        { var.second.size() },
        adios2::ConstantDims ),
        var.second.data() );
    }

    bpWriter.Put< int64_t >(
    bpIO.DefineVariable< int64_t >(
        "connectivity/neighborhoods",
        { summaryGrid.neighborhoods.size() },
        { 0 },
        { summaryGrid.neighborhoods.size() },
        adios2::ConstantDims ),
        summaryGrid.neighborhoods.data() );

    bpWriter.Put< int64_t >(
    bpIO.DefineVariable< int64_t >(
        "connectivity/neighborhoodSums",
        { summaryGrid.neighborhoodSums.size() },
        { 0 },
        { summaryGrid.neighborhoodSums.size() },
        adios2::ConstantDims ),
        summaryGrid.neighborhoodSums.data() );

    bpWriter.Put< int64_t >(
    bpIO.DefineVariable< int64_t >(
        "connectivity/triangulation",
        { summaryGrid.triangulation.size() * 3 },
        { 0 },
        { summaryGrid.triangulation.size() * 3 },
        adios2::ConstantDims ),
        reinterpret_cast< const int64_t * >( summaryGrid.triangulation.data() ) );

    bpWriter.Close();

    writeTriangularMeshObj(
        summaryGrid.variables.at( "r" ),
        summaryGrid.variables.at( "z"),
        summaryGrid.triangulation,
        outpath + "/mesh.obj" );
}

template< typename ValueType >
inline void writeParticlePathsStep(
    const PhasePathStep< ValueType > & pathStep,
    const int64_t & offset,
    const int64_t & globalSize,    
    const int phaseDims,
    const std::string & desc, 
    adios2::IO & bpIO,
    adios2::Engine & bpWriter )
{
    auto availableVariables = bpIO.AvailableVariables();
    auto vphasename = "particle_paths/" + desc + "/phase";
    auto vidname = "particle_paths/" + desc + "/id";

    bpWriter.Put< ValueType >(
        availableVariables.count( vphasename ) ? bpIO.InquireVariable< ValueType >( vphasename )
        : bpIO.DefineVariable< ValueType >(
            vphasename,
            { globalSize, phaseDims },
            { offset, 0 },
            { pathStep.values.size(), phaseDims },
        adios2::ConstantDims ),
        pathStep.values.data() );

    bpWriter.Put< int64_t >(
        availableVariables.count( vidname ) ? bpIO.InquireVariable< int64_t >( vidname )
        : bpIO.DefineVariable< int64_t >(
            vphasename,
            { globalSize },
            { offset },
            { pathStep.ids.size() },
        adios2::ConstantDims ),
        pathStep.ids.data() );
}

template< typename ValueType >
inline void writeParticlePathsStep(
    const PhasePathStep< ValueType > & pathStep,
    const int64_t & offset,
    const int64_t & globalSize, 
    const std::string & directory,
    int stepi,   
    const int phaseDims,
    const std::string & desc,
    MPI_Comm communicator )
{
    adios2::ADIOS adios( communicator, adios2::DebugOFF );
    adios2::IO bpIO = adios.DeclareIO( "XGC-ParticlePath-IO" );
    std::string step = std::to_string( stepi );

    adios2::Engine bpWriter = bpIO.Open(
        directory
        + "/particlePaths.bp"
        + std::string( 7 - step.size(), '0' ) + step + ".bp",
        adios2::Mode::Write );

    writeParticlePathsStep(
        pathStep,
        offset,
        globalSize,    
        phaseDims,
        desc, 
        bpIO,
        bpWriter );

    bpWriter.Close();
}

template< typename ValueType >
inline void writeSummaryStepBP(
    const SummaryStep< ValueType > & summaryStep,
    adios2::IO & bpIO,
    adios2::Engine & bpWriter )
{
    auto availableVariables = bpIO.AvailableVariables();

    bpWriter.Put< ValueType >(
        availableVariables.count( "real_time" ) ? bpIO.InquireVariable< ValueType >( "real_time" )
        : bpIO.DefineVariable< ValueType >(
            "real_time",
    { 1 },
    { 0 },
    { 1 },
    adios2::ConstantDims ),
    & summaryStep.realTime );

    bpWriter.Put< int64_t >(
        availableVariables.count( "sim_step" ) ? bpIO.InquireVariable< int64_t >( "sim_step" )  
        : bpIO.DefineVariable< int64_t >(
            "sim_step",
    { 1 },
    { 0 },
    { 1 },
    adios2::ConstantDims ),
    & summaryStep.simStep );

    auto np_name = summaryStep.objectIdentifier + "/" + "num_particles"; 
    bpWriter.Put< int64_t >(
        availableVariables.count( np_name ) ? bpIO.InquireVariable< int64_t >(  np_name )
        : bpIO.DefineVariable< int64_t >(
            np_name,
    { 1 },
    { 0 },
    { 1 },
    adios2::ConstantDims ),
    & summaryStep.numParticles );

    for( auto & var : summaryStep.variableStatistics )
    {
        for( auto & stat : var.second.values )
        {
            auto name = summaryStep.objectIdentifier + "/" + "statistics/"
                    + var.first + "."
                    + ScalarVariableStatistics< ValueType >::StatisticString( stat.first );

            bpWriter.Put< ValueType >(
                availableVariables.count( name ) ? bpIO.InquireVariable< ValueType >( name )
                : bpIO.DefineVariable< ValueType >(
                    name,
                    { stat.second.size() },
                    { 0 },
                    { stat.second.size() },
                    adios2::ConstantDims ),
            stat.second.data() );
        }
    }

    for( auto & hist : summaryStep.histograms )
    {
        auto valuesName = summaryStep.objectIdentifier + "/" + "histograms/" + hist.first + ".values";
        bpWriter.Put< ValueType >(
            availableVariables.count( valuesName ) ? bpIO.InquireVariable< ValueType >( valuesName )
            : bpIO.DefineVariable< ValueType >(
                valuesName,
        { hist.second.values.size() },
        { 0 },
        { hist.second.values.size() },
        adios2::ConstantDims ),
        hist.second.values.data() );

        auto axisName = summaryStep.objectIdentifier + "/" + "histograms/" + hist.first + ".params.axis";
        bpWriter.Put< std::string >(
            availableVariables.count( axisName ) ? bpIO.InquireVariable< std::string >( axisName )
            : bpIO.DefineVariable< std::string >(
                axisName,
        { hist.second.definition.axis.size() },
        { 0 },
        { hist.second.definition.axis.size() },
        adios2::ConstantDims ),
        hist.second.definition.axis.data() );

        auto identifierName = summaryStep.objectIdentifier + "/" + "histograms/" + hist.first + ".params.identifier";
        bpWriter.Put< std::string >(
            availableVariables.count( identifierName ) ? bpIO.InquireVariable< std::string >( identifierName )
            : bpIO.DefineVariable< std::string >(
                identifierName,
        { 1 },
        { 0 },
        { 1 },
        adios2::ConstantDims ),
        & hist.second.definition.identifier );

        auto weightName = summaryStep.objectIdentifier + "/" + "histograms/" + hist.first + ".params.weight";
        bpWriter.Put< std::string >(
            availableVariables.count( weightName ) ? bpIO.InquireVariable< std::string >( weightName )
            : bpIO.DefineVariable< std::string >(
                weightName,
        { 1 },
        { 0 },
        { 1 },
        adios2::ConstantDims ),
        & hist.second.definition.weight );

        auto dimsName = summaryStep.objectIdentifier + "/" + "histograms/" + hist.first + ".params.dims";
        bpWriter.Put< int >(
            availableVariables.count( dimsName ) ? bpIO.InquireVariable< int >( dimsName )
            : bpIO.DefineVariable< int >(
                dimsName,
        { hist.second.definition.dims.size() },
        { 0 },
        { hist.second.definition.dims.size() },
        adios2::ConstantDims ),
        hist.second.definition.dims.data() );

        for( size_t i = 0; i < hist.second.definition.edges.size(); ++i )
        {
            auto name = 
                summaryStep.objectIdentifier + "/" + "histograms/" 
                + hist.first + ".params.edges(" + std::to_string( i ) + ")";

            bpWriter.Put< ValueType >(
                availableVariables.count( name ) ? bpIO.InquireVariable< ValueType >( name )
                : bpIO.DefineVariable< ValueType >(
                name,
            { hist.second.definition.edges[ i ].size() },
            { 0 },
            { hist.second.definition.edges[ i ].size() },
            adios2::ConstantDims ),
            hist.second.definition.edges[ i ].data() );
        }
    }
}

template< typename ValueType >
inline void writeSummaryStepBP(
    const SummaryStep< ValueType > & summaryStep,
    const std::string & directory )
{
    adios2::ADIOS adios( MPI_COMM_SELF, adios2::DebugOFF );
    adios2::IO bpIO = adios.DeclareIO( "XGC-SUMMARY-STEP-IO" );
    
    std::string step = std::to_string( summaryStep.simStep );
    adios2::Engine bpWriter = bpIO.Open(
                                  directory
                                  + "/summary."
                                  + std::string( 7 - step.size(), '0' ) + step + ".bp",
                                  adios2::Mode::Write );

    writeSummaryStepBP( summaryStep, bpIO, bpWriter );
    bpWriter.Close();
}

}

#endif