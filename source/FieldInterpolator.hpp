#ifndef FIELD_INTERP_23
#define FIELD_INTERP_23

#include "Types/Vec.hpp"

#include <CGAL/Point_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/Triangulation_hierarchy_2.h>
#include <CGAL/Iso_rectangle_2.h>

#include <CGAL/Interpolation_gradient_fitting_traits_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/sibson_gradient_fitting.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <chrono>

// #include <QDebug>

namespace TN
{

inline double distBasedInterpolation(
    const Vec2< double > & p,
    const Vec2< double > & v1,
    const Vec2< double > & v2,
    const Vec2< double > & v3,
    const double f1,
    const double f2,
    const double f3
)
{
    double a1 = p.distance( v1 );
    double a2 = p.distance( v2 );
    double a3 = p.distance( v3 );

    double a = a1 + a2 + a3;

    a1 = ( 1.0 - a1/a ) / 2.0;
    a2 = ( 1.0 - a2/a ) / 2.0;
    a3 = ( 1.0 - a3/a ) / 2.0;

    return f1*a1 + f2*a2 + f3*a3;
}

inline TN::Vec3< double > distBasedInterpolation(
    const Vec2< double > & p,
    const Vec2< double > & v1,
    const Vec2< double > & v2,
    const Vec2< double > & v3,
    const Vec3< double > & f1,
    const Vec3< double > & f2,
    const Vec3< double > & f3
)
{
    double a1 = p.distance( v1 );
    double a2 = p.distance( v2 );
    double a3 = p.distance( v3 );

    double a = a1 + a2 + a3;

    a1 = ( 1.0 - a1/a ) / 2.0;
    a2 = ( 1.0 - a2/a ) / 2.0;
    a3 = ( 1.0 - a3/a ) / 2.0;

    return f1*a1 + f2*a2 + f3*a3;
}

inline double barycentricInterpolation(
    const TN::Vec2< double >  & p,
    const TN::Vec2< double >  & v1,
    const TN::Vec2< double >  & v2,
    const TN::Vec2< double >  & v3,
    const double & f1,
    const double & f2,
    const double & f3
)
{

    double a1 = std::abs( p.x() * (v2.y() - v3.y() )
                          + v2.x() * (v3.y() -  p.y() )
                          + v3.x() * ( p.y() - v2.y() ) ) / 2.0;

    double a2 = std::abs( v1.x() * ( p.y() - v3.y() )
                          +   p.x() * (v3.y() - v1.y() )
                          +  v3.x() * (v1.y() -  p.y() ) ) / 2.0;

    double a3 = std::abs( v1.x() * (v2.y() -  p.y() )
                          + v2.x() * ( p.y() - v1.y() )
                          +  p.x() * (v1.y() - v2.y() ) ) / 2.0;

    double a  = std::abs( v1.x() * (v2.y() - v3.y() )
                          + v2.x() * (v3.y() - v1.y() )
                          + v3.x() * (v1.y() - v2.y() ) ) / 2.0;

    return ( a1*f1 + a2*f2 + a3*f3 ) / a;
}

}

namespace
{

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K>             Vbb;
typedef CGAL::Triangulation_hierarchy_vertex_base_2<Vbb> Vb;
typedef CGAL::Triangulation_face_base_2<K>               Fb;
typedef CGAL::Triangulation_data_structure_2<Vb,Fb>      Tds;
typedef CGAL::Delaunay_triangulation_2<K,Tds>            Dt;
typedef CGAL::Triangulation_hierarchy_2<Dt>              Triangulation;
typedef Triangulation::Face                              Face;
typedef Triangulation::Vertex_handle                     VertexHandle;
typedef Triangulation::Face_handle                       FaceHandle;

typedef K::FT FieldType;
typedef K::Point_2 MeshType;
typedef K::Vector_2 GradientType;

struct KeyHasher
{
    std::size_t operator()( const MeshType& m ) const
    {
        return std::hash<double>()( m[0] ) ^ std::hash<double>()( m[1] );
    }
};

typedef CGAL::Data_access< std::unordered_map< MeshType, FieldType, KeyHasher > > Value_Access;
typedef CGAL::Data_access< std::unordered_map< MeshType, GradientType, KeyHasher > > Gradient_Access;
typedef CGAL::Interpolation_gradient_fitting_traits_2<K> Traits;

}

class FieldInterpolator23
{

    Triangulation tri;
    std::unordered_map< MeshType, FieldType, KeyHasher > mX;
    std::unordered_map< MeshType, FieldType, KeyHasher > mY;
    std::unordered_map< MeshType, FieldType, KeyHasher > mZ;

public:

    TN::Vec2< double > xRange;
    TN::Vec2< double > yRange;

    bool initialized;

    TN::Vec3< double >  interpLin( const TN::Vec2< double > & p ) const
    {

        Triangulation::Face * face = &(*tri.locate( MeshType( p.x(), p.y() ) ) );

        TN::Vec2< double > v1( face->vertex(0)->point().x(), face->vertex(0)->point().y() );
        TN::Vec2< double > v2( face->vertex(1)->point().x(), face->vertex(1)->point().y() );
        TN::Vec2< double > v3( face->vertex(2)->point().x(), face->vertex(2)->point().y() );

        bool c = true;
        if( mX.find( face->vertex(0)->point() ) == mX.end()
                || mX.find( face->vertex(1)->point() ) == mX.end()
                || mX.find( face->vertex(2)->point() ) == mX.end() )
        {
            c = false;
        }

        if( c )
        {
            TN::Vec3< double > f1( mX.find( face->vertex(0)->point() )->second, mY.find( face->vertex(0)->point() )->second, mZ.find( face->vertex(0)->point() )->second );
            TN::Vec3< double > f2( mX.find( face->vertex(1)->point() )->second, mY.find( face->vertex(1)->point() )->second, mZ.find( face->vertex(1)->point() )->second );
            TN::Vec3< double > f3( mX.find( face->vertex(2)->point() )->second, mY.find( face->vertex(2)->point() )->second, mZ.find( face->vertex(2)->point() )->second );

            return distBasedInterpolation( p, v1, v2, v3, f1, f2, f3 );
        }

        return TN::Vec3< double >( NAN, NAN, NAN );
    }

    FieldInterpolator23() : initialized( false ) {}

    void set( const std::vector<TN::Vec2< double >> & mesh, const std::vector<TN::Vec3< double >> & field  )
    {

        //qDebug() << "Triangulating Mesh";

        tri = Triangulation();
        tri.clear();

        mX.clear();
        mY.clear();
        mZ.clear();

        xRange = TN::Vec2< float >( std::numeric_limits< float >::max(), -std::numeric_limits< float >::max() );
        yRange = xRange;

        for( int i = 0; i < mesh.size(); ++i )
        {
            MeshType p( mesh[i].x(), mesh[i].y() );
            tri.insert( p );
            mX.insert( std::make_pair( p, field[i].x() ) );
            mY.insert( std::make_pair( p, field[i].y() ) );
            mZ.insert( std::make_pair( p, field[i].z() ) );

            xRange.a( std::min( xRange.a(), mesh[ i ].x() ) );
            xRange.b( std::max( xRange.b(), mesh[ i ].x() ) );

            yRange.a( std::min( yRange.a(), mesh[ i ].y() ) );
            yRange.b( std::max( yRange.b(), mesh[ i ].y() ) );
        }

        //qDebug() << "done with triangulation " << tri.number_of_faces();
    }
};


#include <CGAL/Iterator_project.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/number_utils_classes.h>
#include <CGAL/utility.h>

#include <list>
#include <utility>


class FieldInterpolator21
{

    Triangulation tri;
    std::unordered_map< MeshType, FieldType, KeyHasher > mX;
    std::unordered_map< MeshType, GradientType, KeyHasher > gradients;

public:

    bool initialized;
    TN::Vec2< double > xRange;
    TN::Vec2< double > yRange;

    double interpLin( const TN::Vec2< double > & _p ) const
    {
        CGAL::Exact_predicates_inexact_constructions_kernel::Point_2 p( _p.x(), _p.y() );
        std::vector< std::pair< MeshType, FieldType > > coords;

        static FaceHandle initialFace = tri.locate( MeshType( _p.x(), _p.y() ) );

        typedef typename Dt::Locate_type Locate_type;

        FaceHandle face = tri.locate( MeshType( _p.x(), _p.y() ), initialFace );
        initialFace = face;

        TN::Vec2< double > v1( face->vertex(0)->point().x(), face->vertex(0)->point().y() );
        TN::Vec2< double > v2( face->vertex(1)->point().x(), face->vertex(1)->point().y() );
        TN::Vec2< double > v3( face->vertex(2)->point().x(), face->vertex(2)->point().y() );

        bool c = true;
        if( mX.find( face->vertex(0)->point() ) == mX.end()
                || mX.find( face->vertex(1)->point() ) == mX.end()
                || mX.find( face->vertex(2)->point() ) == mX.end() )
        {
            c = false;
        }

        if( c )
        {
            double f1( mX.find( face->vertex(0)->point() )->second );
            double f2( mX.find( face->vertex(1)->point() )->second );
            double f3( mX.find( face->vertex(2)->point() )->second );

            return distBasedInterpolation( _p, v1, v2, v3, f1, f2, f3 );

//            FieldType norm =
//              CGAL::natural_neighbor_coordinates_2_Tyson( tri, p, std::back_inserter( coords ), face, lt, li ).second;
//            return CGAL::linear_interpolation( coords.begin(), coords.end(), norm, Value_Access( mX ) );
        }

        //VertexHandle v = tri.nearest_vertex( MeshType( p.x(), p.y() ) );
        //return mX.find( v->point() )->second;

        return NAN;
    }

    double interpQuad( const TN::Vec2< double > & _p ) const
    {
        MeshType p( _p.x(), _p.y() );
        std::vector< std::pair< MeshType, FieldType > > coords;

        FieldType norm =
            CGAL::natural_neighbor_coordinates_2( tri, p, std::back_inserter( coords ) ).second;

        std::pair< FieldType, bool > res2 =
            CGAL::sibson_c1_interpolation_square(
                coords.begin(),
                coords.end(),
                norm,
                p,
                Value_Access( mX ),
                Gradient_Access( gradients ),
                Traits( ) );

        if( res2.second )
        {
            return res2.first;
        }
        else
        {
            return CGAL::linear_interpolation( coords.begin(), coords.end(), norm, Value_Access( mX ) );
        }
    }

    double interp( const TN::Vec2< double > & p ) const
    {

        Triangulation::Face * face = &(*tri.locate( MeshType( p.x(), p.y() ) ) );

        TN::Vec2< double > v1( face->vertex(0)->point().x(), face->vertex(0)->point().y() );
        TN::Vec2< double > v2( face->vertex(1)->point().x(), face->vertex(1)->point().y() );
        TN::Vec2< double > v3( face->vertex(2)->point().x(), face->vertex(2)->point().y() );

        bool c = true;
        if( mX.find( face->vertex(0)->point() ) == mX.end()
                || mX.find( face->vertex(1)->point() ) == mX.end()
                || mX.find( face->vertex(2)->point() ) == mX.end() )
        {
            c = false;
        }

        if( c )
        {
            double f1( mX.find( face->vertex(0)->point() )->second );
            double f2( mX.find( face->vertex(1)->point() )->second );
            double f3( mX.find( face->vertex(2)->point() )->second );

            return distBasedInterpolation( p, v1, v2, v3, f1, f2, f3 );
        }

        VertexHandle v = tri.nearest_vertex( MeshType( p.x(), p.y() ) );
        return mX.find( v->point() )->second;
    }

    FieldInterpolator21() : initialized( false ) {}

    void set( const std::vector< TN::Vec2< double > > & mesh, const std::vector< double > & field  )
    {

        //qDebug() << "Triangulating Mesh";

        tri = Triangulation();
        tri.clear();
        mX.clear();
        gradients.clear();

        xRange = TN::Vec2< float >( std::numeric_limits< float >::max(), -std::numeric_limits< float >::max() );
        yRange = xRange;

        for( int i = 0; i < mesh.size(); ++i )
        {
            MeshType p( mesh[i].x(), mesh[i].y() );
            tri.insert( p );

            mX.insert( std::make_pair( p, field[i] ) );

            xRange.a( std::min( xRange.a(), mesh[ i ].x() ) );
            xRange.b( std::max( xRange.b(), mesh[ i ].x() ) );

            yRange.a( std::min( yRange.a(), mesh[ i ].y() ) );
            yRange.b( std::max( yRange.b(), mesh[ i ].y() ) );
        }

        //qDebug() << "done with triangulation " << tri.number_of_faces();

//        //qDebug() << "fitting gradients";

//        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

//        sibson_gradient_fitting_nn_2(
//            tri,
//            std::inserter(
//                gradients,
//                gradients.begin( ) ),
//            Value_Access( mX ),
//            Traits( ) );

//        //qDebug() << "done fitting";

//        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

//        std::cout << "Fitting took "
//                  << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
//                  << "second\n";

        initialized = true;
    }

    void writeGradients( const std::string & path )
    {
        std::ofstream outFile( path );
        for( auto & g : gradients )
        {
            outFile << g.first.x() << " " << g.first.y() << " " << g.second.x() << g.second.y() << "\n";
        }
        outFile.close();
    }

    void test()
    {
//        set(
//            { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }, { 2, 0 }, {0, 2 }, { 2, 1 }, { 1, 2 }, { 2, 2 }  },
//            { 1, 2, 3, 4, 5, 6, 7, 8, 9 }
//        );

//        //coordiante computation
//        TN::Vec2< double > p( 1.5, 1.0 );

//        //qDebug() << interpLin( p );
//        //qDebug() << interpQuad( p );
    }
};

template< typename T >
struct LookupTexture
{
    std::vector< T > data;
    std::vector< int > dims;
    TN::Vec2< double > xRange;
    TN::Vec2< double > yRange;
};

template< typename T >
inline void generateGridToTexture3(
    const FieldInterpolator23 & interpolator,
    LookupTexture< T > & result1,
    LookupTexture< T > & result2,
    LookupTexture< T > & result3,
    int resolution )
{
    double width = interpolator.xRange.b() - interpolator.xRange.a();
    double height = interpolator.yRange.b() - interpolator.yRange.a();

    if( width < height )
    {
        result1.dims = { ( height / width ) * resolution , resolution };
    }
    else
    {
        result1.dims = { resolution, resolution * ( width / height ) };
    }

    result2.dims = result1.dims;
    result3.dims = result1.dims;

    result1.data.resize( result1.dims[ 0 ] * result1.dims[ 1 ] );
    result2.data.resize( result1.data.size() );
    result3.data.resize( result1.data.size() );

    for( int r = 0; r < result1.dims[ 0 ]; ++r )
    {
        #pragma omp parallel for
        for( int c = 0; c < result1.dims[ 1 ]; ++c )
        {
            double pc = ( c / ( double ) ( result1.dims[ 1 ] - 1 ) );
            double pr = ( r / ( double ) ( result1.dims[ 0 ] - 1 ) );

            TN::Vec2< double > p(
                pc*interpolator.xRange.b() + ( 1 - pc )*interpolator.xRange.a(),
                pr*interpolator.yRange.b() + ( 1 - pr )*interpolator.yRange.a() );

            TN::Vec3< double > res = interpolator.interpLin( p );

            result1.data[ r*result1.dims[ 1 ] + c ] = res.x();
            result2.data[ r*result1.dims[ 1 ] + c ] = res.y();
            result3.data[ r*result1.dims[ 1 ] + c ] = res.z();
        }
    }

    result1.xRange = interpolator.xRange;
    result1.yRange = interpolator.yRange;

    result2.xRange = result1.xRange;
    result2.yRange = result1.yRange;

    result3.xRange = result1.xRange;
    result3.yRange = result1.yRange;
}

template< typename T >
inline void generateGridToTexture(
    const FieldInterpolator21 & interpolator,
    LookupTexture< T > & result,
    int resolution )
{
    double width = interpolator.xRange.b() - interpolator.xRange.a();
    double height = interpolator.yRange.b() - interpolator.yRange.a();

    if( width < height )
    {
        result.dims = { ( height / width ) * resolution , resolution };
    }
    else
    {
        result.dims = { resolution, resolution * ( width / height ) };
    }
    result.data.resize( result.dims[ 0 ] * result.dims[ 1 ] );

    for( int r = 0; r < result.dims[ 0 ]; ++r )
    {
        #pragma omp parallel for
        for( int c = 0; c < result.dims[ 1 ]; ++c )
        {
            double pc = ( c / ( double ) ( result.dims[ 1 ] - 1 ) );
            double pr = ( r / ( double ) ( result.dims[ 0 ] - 1 ) );

            TN::Vec2< double > p(
                pc*interpolator.xRange.b() + ( 1 - pc )*interpolator.xRange.a(),
                pr*interpolator.yRange.b() + ( 1 - pr )*interpolator.yRange.a() );

            result.data[ r*result.dims[ 1 ] + c ] = interpolator.interpLin( p );
        }
    }

    result.xRange = interpolator.xRange;
    result.yRange = interpolator.yRange;
}

#endif // FIELD_INTERP_23
