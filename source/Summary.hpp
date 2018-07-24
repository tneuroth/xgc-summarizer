#ifndef TN_SUMMARY_HPP
#define TN_SUMMARY_HPP

#include "SummaryUtils.hpp"

#include <vector>
#include <list>

namespace TN
{

/* Temporally Static */
struct SummaryGrid
{
    /* Per Probe/Voronoi Cell Summary Fields */
    struct
    {
        std::vector< float > volume;
        std::vector< float > r;
        std::vector< float > z;
        std::vector< float > psin;
        std::vector< float > poloidalAngle;
        std::vector< float > B;
    } probes;

    // delaunay triangulation duel to the voronoi decomposition
    std::vector< Triangle > probeTriangulation;

    // neighborhood
    std::vector< std::vector< int64_t > > neighborhoods;
};

/* time varying */
struct SummaryStep
{
    static const int NC = 33;
    static const int NR = 17;
    static constexpr double DELTA_V = 3378743.0;

    /* Per Probe Voronoi Cell Summary Fields */

    std::vector< float > velocityDistribution;
    std::vector< float > w0w1_mean;
    std::vector< float > w0w1_rms;
    std::vector< float > w0w1_min;
    std::vector< float > w0w1_max;
    std::vector< float > num_particles;
    std::vector< float > num_mapped;

    void resize( size_t sz )
    {
        velocityDistribution.resize( sz*NR*NC );
        w0w1_mean.resize( sz );
        w0w1_rms.resize( sz );
        w0w1_min.resize( sz );
        w0w1_max.resize( sz );
        num_particles.resize( sz );
        num_mapped.resize( sz );
    }

    void clear()
    {
        velocityDistribution.clear();
        w0w1_mean.clear();
        w0w1_rms.clear();
        w0w1_min.clear();
        w0w1_max.clear();
        num_particles.clear();
        num_mapped.clear();
    }

    void merge( const SummaryStep & other )
    {
        const std::int64_t SZ = velocityDistribution.size();
        #pragma omp parallel for simd
        for( int64_t i = 0; i < SZ; ++i )
        {
            #pragma omp simd
            for( int64_t b = 0; b < NR*NC; ++b )
            {
                velocityDistribution[ i*NC*NC + b ] = other.velocityDistribution[ i*NC*NC + b ];
            }

            // Pooled mean = (N1*M1+N2*M2+N3*M3)/(N1+N2+N3)
            // Pooled SD ={ (N1-1)*S1+(N2-1)*S2+(N3-1)S3}/(N1+N2+N3-3)
            // Pooled RMS = ?

            w0w1_min[ i ] = std::min( w0w1_min[ i ], other.w0w1_min[ i ] );
            w0w1_max[ i ] = std::min( w0w1_max[ i ], other.w0w1_max[ i ] );
            num_particles[ i ] += other.num_particles[ i ];
        }
    }
};

}

#endif

