#ifndef TN_SUMMARY_HPP
#define TN_SUMMARY_HPP

#include "Types/Vec.hpp"

#include <vector>
#include <set>
#include <cstdint>
#include <map>

namespace TN
{

/* Temporally Static */
template< typename ValueType >
struct SummaryGrid2
{
    /* Temporally Static, Spatially Varying Per Voronoi Cell Fields */
    std::map< std::string, std::vector< ValueType > > variables;

    // delaunay triangulation dual to the voronoi decomposition
    std::vector< Triangle > triangulation;

    // neighborhood
    std::vector< int64_t > neighborhoods;
    std::vector< int64_t > neighborhoodSums;
};

template < typename ValueType >
struct ScalarVariableStatistics
{
    enum Statistic
    {
        Count,
        Mean,
        Variance,
        Skewness,
        kurtosis,
        RMS,
        InterquartileRange,
        Min,
        Max,
        ShannonEntropy
    };

    static const char* StatisticString( Statistic stat )
    {
        switch ( stat )
        {
        case Count                       :
            return "Count";
        case Mean                        :
            return "Mean";
        case Variance                    :
            return "Variance";
        case Skewness                    :
            return "Skewness";
        case kurtosis                    :
            return "kurtosis";
        case RMS                         :
            return "RMS";
        case InterquartileRange          :
            return "InterquartileRange";
        case Min                         :
            return "Min";
        case Max                         :
            return "Max";
        case ShannonEntropy              :
            return "ShannonEntropy";
        }
    }

    std::string variableIdentifier;

    /* Enabled Per Scalar Variable Statistics */
    std::map< Statistic, std::vector< ValueType > > values;

    ScalarVariableStatistics(
        const std::string & id,
        const std::set< Statistic > & enabled,
        size_t size = 0 ) : variableIdentifier( id )
    {
        for( auto & stat : enabled )
        {
            values.insert(
            {
                stat,
                std::vector< ValueType >( size )
            } );
        }
    }

    void resize( size_t size )
    {
        for( auto & stat : values )
        {
            stat.second.resize( size );
        }
    }

    void clear()
    {
        for( auto & stat : values )
        {
            stat.second.clear();
        }
    }
};

template < typename ValueType >
struct HistogramDefinition
{

private:

    void assign( const HistogramDefinition< ValueType > & other  )
    {
        identifier = other.identifier;
        axis = other.axis;
        dims = other.dims;
        weight = other.weight;
        edges = other.edges;
    }

public:

    std::string identifier;
    std::vector< std::string > axis;
    std::vector< int > dims;
    std::string weight;
    std::vector< std::vector< ValueType > > edges;

    HistogramDefinition() {}

    HistogramDefinition( const HistogramDefinition< ValueType > & other )
    {
        assign( other );
    }

    HistogramDefinition & operator=( const HistogramDefinition< ValueType > & other )
    {
        if( this !=  &other )
        {
            assign( other );
        }

        return *this;
    }

    size_t size() const
    {
        size_t sz = 1;
        for( auto d : dims )
        {
            sz *= d;
        }

        return sz;
    }
};

template < typename ValueType >
struct CellHistograms
{
    HistogramDefinition< ValueType > definition;
    std::vector< ValueType > values;

    CellHistograms( const HistogramDefinition< ValueType > & def )
        : definition( def ) {}

    void resize( size_t size )
    {
        values.resize( size * definition.size() );
    }
    void clear()
    {
        values.clear();
    }
};

template < typename ValueType >
struct SummaryStep2
{
    // Scalar Statistics
    std::map< std::string, ScalarVariableStatistics< ValueType > > variableStatistics;
    std::map< std::string, CellHistograms< ValueType > > histograms;

    std::string objectIdentifier;

    int64_t outStep;
    int64_t simStep;
    ValueType realTime;
    int64_t numParticles;


    void setStep( int64_t _outStep, int64_t _simStep, double _realTime )
    {
        outStep  = _outStep;
        simStep  = _simStep;
        realTime = _realTime;
    }

    void resize( std::size_t size )
    {
        for( auto & s : variableStatistics )
        {
            s.second.resize( size );
        }

        for( auto & h : histograms )
        {
            h.second.resize( size );
        }
    }

    void clear()
    {
        for( auto & s : variableStatistics )
        {
            s.second.clear();
        }

        for( auto & h : histograms )
        {
            h.second.clear();
        }
    }
};

}

#endif

