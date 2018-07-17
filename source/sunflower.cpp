#include <cmath>
#include <vector>

double getRadius( double k, int64_t n, double b)
{
    if( k > ( n - b ) )
    {
        return 1.0; // put on the boundary
    }
    else
    {
        return std::sqrt( k - 0.5 ) / std::sqrt( n - ( b + 1.0 ) / 2.0 ); // apply square root
    }
}

void sunflower( int64_t n, double alpha, Vec2< float > center, double scale, vector< Vec2< float > > & result )
{
    result.clear();
    double b = std::round( alpha * std::sqrt( n ) ); // number of boundary points
    double phi = ( std::sqrt( 5.0 ) + 1.0 ) / 2.0;   // golden ratio
    for( int64_t k = 1; k < n; ++k )
    {
        double r = getRadius( k, n, b ) * scale;
        double theta = ( 2.0 * M_PI * k ) / ( phi*phi );
        result.push_back( Vec2< float >(  r * std::cos( theta ), r * std::sin( theta ) ) + center );
    }
}