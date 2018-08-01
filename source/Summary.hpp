#ifndef TN_SUMMARY_HPP
#define TN_SUMMARY_HPP

#include "Types/Vec.hpp"

#include <vector>
#include <list>
#include <cstdint>

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

    // delaunay triangulation dual to the voronoi decomposition
    std::vector< Triangle > probeTriangulation;

    // neighborhood
    std::vector< int64_t > neighborhoods;
    std::vector< int64_t > neighborhoodSums;
};

/* time varying */
struct SummaryStep
{
    static const int NC = 33;
    static const int NR = 17;
    static constexpr double DELTA_V = 3378743.0 / 2.0;

    /* Per Probe Voronoi Cell Summary Fields */

    std::vector< float > velocityDistribution;
    std::vector< float > w0w1_mean;
    std::vector< float > w0w1_rms;
    std::vector< float > w0w1_variance;
    std::vector< float > w0w1_min;
    std::vector< float > w0w1_max;
    std::vector< float > num_particles;

    void resize( std::size_t sz )
    {
        velocityDistribution.resize( sz*NR*NC );
        w0w1_mean.resize( sz );
        w0w1_rms.resize( sz );
        w0w1_min.resize( sz );
        w0w1_max.resize( sz );
        num_particles.resize( sz );
        w0w1_variance.resize( sz );
    }

    void clear()
    {
        velocityDistribution.clear();
        w0w1_mean.clear();
        w0w1_rms.clear();
        w0w1_min.clear();
        w0w1_max.clear();
        num_particles.clear();
        w0w1_variance.clear();
    }
};

}

#endif

