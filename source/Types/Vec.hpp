//============================================================================
//  Copyright (c) 2016 University of California Davis
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information
//============================================================================

#ifndef VEC_HPP_INCLUDED
#define VEC_HPP_INCLUDED

#include <cmath>
// #include <QVector3D>

namespace TN
{

template< typename T >
struct Range4
{
    T x1Min;
    T x2Min;
    T x1Max;
    T x2Max;
    Range4( T x1, T x2, T x3, T x4 ) : x1Min(x1), x1Max(x2), x2Min(x3), x2Max(x4) {}
    Range4() {}
};

template< typename T >
struct Vec2
{

private:

    T x1;
    T x2;

public:

    double length( ) const
    {
        double l = std::sqrt( (double)x1*(double)x1 + (double)x2*(double)x2 );
        return l;
    }

    Vec2()
    {
        x1 = x2 = 0;
    }

    Vec2(T v1, T v2)
    {
        x1 = v1;
        x2 = v2;
    }

    Vec2( const Vec2 & other )
    {
        x1 = other.x1;
        x2 = other.x2;
    }

    template <class T2>
    Vec2<T>& operator=( const Vec2<T2> & other )
    {
        x1 = static_cast<T>( other.x() );
        x2 = static_cast<T>( other.y() );
        return *this;
    }

    template <class T2>
    bool operator==( const Vec2<T2> & other ) const
    {
        return x1 == other.x1
               && x2 == other.x2;
    }

    double distance( const Vec2 & other ) const
    {
        return std::sqrt( ( (double)x1 - (double)other.x1 ) * ( (double)x1 - (double)other.x1 )
                          + ( (double)x2 - (double)other.x2 ) * ( (double)x2 - (double)other.x2 ) );
    }

    double dot( Vec2 v2 )
    {
        return (double)x1*(double)v2.x1 + (double)x2*(double)v2.x2;
    }

    double dot( Vec2 v1, Vec2 v2 )
    {
        return (double)v1.x1*(double)v2.x1 + (double)v1.x2*(double)v2.x2;
    }

    double angle( Vec2 other )
    {
        double det = x1*other.x2 - x2*other.x1;
        double angle = std::atan2(det, dot( *this, other ) );//  # atan2(y, x) or atan2(sin, cos);
        while( angle < 0 )
        {
            angle += 2*3.14159265359;
        }
        while( angle > 2*3.14159265359 )
        {
            angle -= 2*3.14159265359;
        }
        return angle;
    }

    void normalize()
    {

        double x = x1, y = x2;
        double norm = std::sqrt( x*x + y*y );

        if ( norm == 0 )
        {
            return;
        }

        x /= norm;
        y /= norm;

        x1 = x;
        x2 = y;
    }

    bool operator < ( const Vec2 & other ) const
    {

        return x2 < other.x2;
    }

    const Vec2 & operator = ( const Vec2 & other )
    {
        if ( this == &( other ) )
        {
            return *this;
        }
        x1 = other.x1;
        x2 = other.x2;

        return *this;
    }
    void operator+=( const Vec2 & other )
    {
        x1 += other.x1;
        x2 += other.x2;
    }

    Vec2 operator+( const Vec2 & v2 ) const
    {
        return Vec2( x1 + v2.x(), x2 + v2.y() );
    }

    Vec2 operator-( const Vec2 & v2 ) const
    {
        return Vec2( x1 - v2.x(), x2 - v2.y() );
    }

    Vec2 operator/( T v )
    {
        return Vec2( x1/v, x2/v );
    }

    Vec2 operator*( T v ) const
    {
        return Vec2( x1*v, x2*v );
    }

    double distToLine( Vec2 r, Vec2 d )
    {
        Vec2 w = *this - r;

        double c1 = dot(w,d);
        double c2 = dot(d,d);
        double b = c1 / c2;

        Vec2 pb = r + d*b;

        return this->distance( pb );
    }

    void operator-=( const Vec2 & other )
    {
        x1 -= other.x1;
        x2 -= other.x2;
    }

    Vec2 toPolar()
    {
        Vec2 p;
        p.r( this->distance(  Vec2< T >( 0.0, 0.0 ) ) );
        p.theta( this->angle( Vec2< T >( 1.0, 0.0 ) ) );
        return p;
    }

    Vec2 toCartesian()
    {
        Vec2 p;
        p.x( r() * cos( theta() ) );
        p.y( r() * sin( theta() ) );
        return p;
    }

    void r( T v )
    {
        x1 = v;
    }
    void theta( T v )
    {
        x2 = v;
    }

    void x( T v )
    {
        x1 = v;
    }
    void y( T v )
    {
        x2 = v;
    }

    void a( T v )
    {
        x1 = v;
    }
    void b( T v )
    {
        x2 = v;
    }

    T x() const
    {
        return x1;
    }
    T y() const
    {
        return x2;
    }

    T r()     const
    {
        return x1;
    }
    T theta() const
    {
        return x2;
    }

    T a()     const
    {
        return x1;
    }
    T b() const
    {
        return x2;
    }
};

template < typename T >
struct Vec3
{

private:

    T x1;
    T x2;
    T x3;

public:

    Vec3()
    {
        x1 = x2 = x3 = 0;
    }

    Vec3( float v1, float v2, float v3 )
    {
        x1 = v1;
        x2 = v2;
        x3 = v3;
    }

    Vec3( const Vec2<T> & v )
    {
        x1 = v.x();
        x2 = v.y();
        x3 = 0.f;
    }

    Vec3( const Vec3 & other )
    {
        x1 = other.x1;
        x2 = other.x2;
        x3 = other.x3;
    }

    // Vec3( const QVector3D & v )
    // {
    //     x1 = v.x();
    //     x2 = v.y();
    //     x3 = v.z();
    // }

    const Vec3 & operator = ( const Vec3 & other )
    {
        if ( this == &( other ) )
        {
            return *this;
        }
        x1 = other.x1;
        x2 = other.x2;
        x3 = other.x3;

        return *this;
    }

    Vec3 rounded( int decimalPlaces )
    {
        return Vec3(
                   std::round( x1 * std::pow( 10.0, decimalPlaces ) ) / std::pow( 10.0, decimalPlaces ),
                   std::round( x2 * std::pow( 10.0, decimalPlaces ) ) / std::pow( 10.0, decimalPlaces ),
                   std::round( x3 * std::pow( 10.0, decimalPlaces ) ) / std::pow( 10.0, decimalPlaces )
               );
    }

    static Vec3 surfaceNormal( Vec3 & p1, Vec3 & p2, Vec3 & p3 )
    {

        Vec3 v = p2 - p1;
        Vec3 w = p3 - p1;

        return Vec3(
                   v.y() * w.z() - v.z() * w.y(),
                   v.z() * w.x() - v.x() * w.z(),
                   v.x() * w.y() - v.y() * w.x()
               );
    }

    void operator+=( const Vec3 & other )
    {
        x1 += other.x1;
        x2 += other.x2;
        x3 += other.x3;
    }

    void operator-=( const Vec3 & other )
    {
        x1 -= other.x1;
        x2 -= other.x2;
        x3 -= other.x3;
    }

    void operator/=( float v )
    {
        x1 /= v;
        x2 /= v;
        x3 /= v;
    }

    void normalize()
    {

        double x = x1, y = x2, z = x3;
        double norm = std::sqrt( x*x + y*y + z*z );

        if ( norm == 0 )
        {
            return;
        }

        x /= norm;
        y /= norm;
        z /= norm;

        x1 = x;
        x2 = y;
        x3 = z;
    }

    float distance2( const Vec3 & other )
    {
        return std::sqrt( ( x1 - other.x1 ) * ( x1 - other.x1 )
                          + ( x2 - other.x2 ) * ( x2 - other.x2 ) );
    }

    float distance3( const Vec3 & other ) const
    {
        return std::sqrt( ( x1 - other.x1 ) * ( x1 - other.x1 )
                          + ( x2 - other.x2 ) * ( x2 - other.x2 )
                          + ( x3 - other.x3 ) * ( x3 - other.x3 ) );
    }

    bool operator==( const Vec3 & other ) const
    {
        return distance3( other ) < .00000001;
    }

    float angle( const Vec3 & other )
    {
        float angle = std::atan2( other.y() - x2, other.x() - x1 );
        return angle;
    }

    Vec3 operator / ( const Vec3 & other )
    {
        return Vec3( x1 / other.x(), x2 / other.y(), x3 / other.z() );
    }

    Vec3 operator - ( const Vec3 & other )
    {
        return Vec3( x1 - other.x(), x2 - other.y(), x3 - other.z() );
    }

    Vec3 operator - ( float num )
    {
        return Vec3( x1 - num, x2 - num, x3 - num );
    }

    Vec3 operator + ( const Vec3 & other )
    {
        return Vec3( x1 + other.x(), x2 + other.y(), x3 + other.z() );
    }

    const Vec3 & operator /= ( const Vec3 & other )
    {
        x1 /= other.x1;
        x2 /= other.x2;
        x3 /= other.x3;
        return *this;
    }

    Vec3 operator * ( T v ) const
    {
        return Vec3( x1 * v, x2 * v, x3 * v );
    }

    Vec3 operator / ( T v ) const
    {
        return Vec3( x1 / v, x2 / v, x3 / v );
    }

    bool operator < ( const Vec3 & other ) const
    {

        return x3 < other.x3;
    }

    double length( ) const
    {
        double l = std::sqrt( (double)x1*(double)x1 + (double)x2*(double)x2 + (double)x3*(double)x3  );
        //if ( l != l ) {
        //    qDebug( "not a number %f %f", x1, x2 );
        // }
        return l;
    }

    double xyLength( ) const
    {
        double l = std::sqrt( (double)x1*(double)x1 + (double)x2*(double)x2 );
        //if ( l != l ) {
        //    qDebug( "not a number %f %f", x1, x2 );
        // }
        return l;
    }

    static Vec3< T > RGBtoHSV( Vec3< T > rgb )
    {

        Vec3< T > hsv;

        double  mn, mx, delta;

        mn = rgb.r() < rgb.g() ? rgb.r() : rgb.g();
        mn = mn  < rgb.b() ? mn  : rgb.b();

        mx = rgb.r() > rgb.g() ? rgb.r() : rgb.g();
        mx = mx  > rgb.b() ? mx  : rgb.b();

        hsv.v( mx );
        delta = mx - mn;
        if (delta < 0.00001)
        {
            hsv.s( 0 );
            hsv.h( 0 );
            return hsv;
        }
        if( mx > 0.0 )
        {
            hsv.s( delta / mx );
        }
        else
        {
            hsv.s( 0.0 );
            hsv.h( 0.0 );
            return hsv;
        }
        if( rgb.r() >= mx )
            hsv.h( ( rgb.g() - rgb.b() ) / delta );
        else if( rgb.g() >= mx )
            hsv.h( 2.0 + ( rgb.b() - rgb.r() ) / delta );
        else
            hsv.h( 4.0 + ( rgb.r() - rgb.g() ) / delta );

        hsv.h( hsv.h() * 60.0 );

        if( hsv.h() < 0.0 )
            hsv.h( hsv.h() + 360.0 );

        return hsv;
    }

    static Vec3< T > HSVtoRGB( Vec3< T > hsv )
    {

        double hh, p, q, t, ff;
        long i;

        Vec3< T > rgb;
        if(hsv.s() <= 0.0)
        {
            rgb.r( hsv.v() );
            rgb.g( hsv.v() );
            rgb.b( hsv.v() );
            return rgb;
        }

        hh = hsv.h();

        if(hh >= 360.0) hh = 0.0;

        hh /= 60.0;

        i = (long)hh;

        ff = hh - i;
        p = hsv.v() * (1.0 - hsv.s() );
        q = hsv.v() * (1.0 - (hsv.s() * ff));
        t = hsv.v() * (1.0 - (hsv.s() * (1.0 - ff)));

        switch(i)
        {
        case 0:
            rgb.r( hsv.v() );
            rgb.g( t );
            rgb.b( p );
            break;
        case 1:
            rgb.r( q );
            rgb.g( hsv.v() );
            rgb.b( p );
            break;
        case 2:
            rgb.r( p );
            rgb.g( hsv.v() );
            rgb.b( t );
            break;

        case 3:
            rgb.r( p );
            rgb.g( q );
            rgb.b( hsv.v() );
            break;
        case 4:
            rgb.r( t );
            rgb.g( p );
            rgb.b( hsv.v() );
            break;
        case 5:
        default:
            rgb.r( hsv.v() );
            rgb.g( p );
            rgb.b( q );
            break;
        }
        return rgb;
    }

    void h( T _v )
    {
        x1 = _v;
    }
    void s( T _v )
    {
        x2 = _v;
    }
    void v( T _v )
    {
        x3 = _v;
    }

    void r( T v )
    {
        x1 = v;
    }
    void theta( T v )
    {
        x2 = v;
    }
    void phi( T v )
    {
        x3 = v;
    }

    void g( T _v )
    {
        x2 = _v;
    }
    void b( T _v )
    {
        x3 = _v;
    }
    void x( T v )
    {
        x1 = v;
    }
    void y( T v )
    {
        x2 = v;
    }
    void z( T v )
    {
        x3 = v;
    }

    T & r()
    {
        return x1;
    }
    T & theta()
    {
        return x2;
    }
    T & phi()
    {
        return x3;
    }

    T & x()
    {
        return x1;
    }
    T & y()
    {
        return x2;
    }
    T & z()
    {
        return x3;
    }

    T & g()
    {
        return x2;
    }
    T & b()
    {
        return x3;
    }

    T r()     const
    {
        return x1;
    }
    T theta() const
    {
        return x2;
    }
    T phi()   const
    {
        return x3;
    }

    T x() const
    {
        return x1;
    }
    T y() const
    {
        return x2;
    }
    T z() const
    {
        return x3;
    }

    T g() const
    {
        return x2;
    }
    T b() const
    {
        return x3;
    }

    T h() const
    {
        return x1;
    }

    T s() const
    {
        return x2;
    }
    T v() const
    {
        return x3;
    }
};

struct Vec4
{
    float r;
    float g;
    float b;
    float a;

    Vec4()
        : r(0.f), g(0.f), b(0.f), a(0.f) {}

    Vec4( const Vec3<float> & v3 )
        : r(v3.x()), g(v3.y()), b(v3.z()), a(1.f) {}

    Vec4( float _r, float _g, float _b, float _a  )
        : r(_r), g(_g), b(_b), a(_a) {}

    Vec4 operator=( const Vec3<float> & v3 )
    {
        r = v3.x();
        g = v3.y();
        b = v3.z();
        a = 1.f;

        return *this;
    }
};

struct I2
{

    int i1;
    int i2;

    I2( int _i1, int _i2 )
        : i1( _i1 ), i2( _i2 )
    {}

    I2() {}

    void offset( int amount )
    {
        i1 += amount;
        i2 += amount;
    }

    int a() const
    {
        return i1;
    }
    int b() const
    {
        return i2;
    }
};

struct I3
{
    unsigned int i1;
    unsigned int i2;
    unsigned int i3;

    I3( unsigned int _i1, unsigned int _i2, unsigned int _i3 )
        : i1( _i1 ), i2( _i2 ), i3( _i3 )
    {}

    I3() {}

    void offset( unsigned int amount )
    {
        i1 += amount;
        i2 += amount;
        i3 += amount;
    }

    unsigned int a() const
    {
        return i1;
    }
    unsigned int b() const
    {
        return i2;
    }
    unsigned int c() const
    {
        return i3;
    }
    bool operator == ( const I3 & other )
    {
        return i1 == other.i1 && i2 == other.i2 && i3 == other.i3;
    }
};

struct Triangle
{
    int64_t indices[ 3 ];
    Triangle() {}
    Triangle( int64_t i1, int64_t i2, int64_t i3 )
    {
        indices[ 0 ] = i1;
        indices[ 1 ] = i2;
        indices[ 2 ] = i3;
    }
    int64_t & operator[] ( int64_t i )
    {
        return indices[ i ];
    }

    const int64_t & operator[] ( int64_t i ) const
    {
        return indices[ i ];
    }
};

}

namespace std
{
template <>
template <typename T>
struct hash<TN::Vec3<T>>
{
    size_t operator ()( const TN::Vec3<float> & v ) const
    {
        return std::hash<T>()( v.x() ) ^ std::hash<T>()( v.y() ) ^ std::hash<T>()( v.z() );
    }
};
}

#endif
