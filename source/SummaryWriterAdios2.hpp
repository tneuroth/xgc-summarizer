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
    const SummaryGrid2< ValueType > & summaryGrid,
    const std::string & outpath )
{
        adios2::ADIOS adios( adios2::DebugOFF );

        adios2::IO bpIO = adios.DeclareIO( "XGC-SUMMARY-GRID-IO" );
        adios2::Engine bpWriter = bpIO.Open( outpath + "/summary.mesh.bp", adios2::Mode::Write );

        for( auto & var : summaryGrid.variables )
        {
            std::cout << "writing var size is : " << var.second.size() << std::endl;
            bpWriter.Put<ValueType>(
                bpIO.DefineVariable<ValueType>( 
                    var.first, 
                    { var.second.size() }, 
                    { 0 }, 
                    { var.second.size() },
                    adios2::ConstantDims ), 
                var.second.data() );            
        }

        bpWriter.Close();
}

template< typename ValueType >
inline void writeSummaryStepBP(
    const SummaryStep2< ValueType > & summaryStep,
    const std::string & directory )
{
    adios2::ADIOS adios( adios2::DebugOFF );
    adios2::IO bpIO = adios.DeclareIO( "XGC-SUMMARY-STEP-IO" );

    std::string step = std::to_string( summaryStep.simStep );

    adios2::Engine bpWriter = bpIO.Open( 
        directory 
            + "summary." 
            + summaryStep.objectIdentifier + "."
            + std::string( 7 - step.size(), '0' ) + step + ".bp", 
        adios2::Mode::Write );

    for( auto & var : summaryStep.variableStatistics )
    {
        for( auto & stat : var.second.values )
        {
            bpWriter.Put<float>(
                bpIO.DefineVariable<float>( 
                    "statistics/" 
                        + var.first + "." 
                        + ScalarVariableStatistics< ValueType >::StatisticString( stat.first ), 
                    { stat.second.size() },
                    { 0 },
                    { stat.second.size() },
                    adios2::ConstantDims ),
                stat.second.data() );
        }        
    }

    for( auto & hist : summaryStep.histograms )
    {
        bpWriter.Put< ValueType >(
            bpIO.DefineVariable< ValueType >( 
                "histograms/" + hist.first + ".values",
                { hist.second.values.size() }, 
                { 0 }, 
                { hist.second.values.size() }, 
                adios2::ConstantDims ), 
            hist.second.values.data() );    

        bpWriter.Put< std::string >(
            bpIO.DefineVariable< std::string >(
                "histograms/" + hist.first + ".params.axis",
                { hist.second.definition.axis.size() }, 
                { 0 }, 
                { hist.second.definition.axis.size() }, 
                adios2::ConstantDims ), 
            hist.second.definition.axis.data() );   

        bpWriter.Put< std::string >(
            bpIO.DefineVariable< std::string >(
                "histograms/" + hist.first + ".params.identifier",
                { 1 }, 
                { 0 }, 
                { 1 }, 
                adios2::ConstantDims ), 
            & hist.second.definition.identifier );    

        bpWriter.Put< std::string >(
            bpIO.DefineVariable< std::string >(
                "histograms/" + hist.first + ".params.weight",
                { 1 }, 
                { 0 }, 
                { 1 }, 
                adios2::ConstantDims ), 
            & hist.second.definition.weight );    

        bpWriter.Put< int >(
            bpIO.DefineVariable< int >(
                "histograms/" + hist.first + ".params.dims",
                { hist.second.definition.dims.size() }, 
                { 0 }, 
                { hist.second.definition.dims.size() }, 
                adios2::ConstantDims ), 
            hist.second.definition.dims.data() );            

        for( size_t i = 0; i < hist.second.definition.edges.size(); ++i )
        {
            bpWriter.Put< ValueType >(
                bpIO.DefineVariable< ValueType >(
                    "histograms/" + hist.first + ".params.edges(" + std::to_string( i ) + ")",
                    { hist.second.definition.edges[ i ].size() }, 
                    { 0 },
                    { hist.second.definition.edges[ i ].size() }, 
                    adios2::ConstantDims ), 
                hist.second.definition.edges[ i ].data() );   
        }
    }

    bpWriter.Close();
}

}

#endif