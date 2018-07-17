#ifndef TN_GRID_BUILDER_HPP
#define TN_GRID_BUILDER_HPP

#include "Types/Vec.hpp"

#include <CGAL/Point_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/Triangulation_hierarchy_2.h>
#include <CGAL/Iso_rectangle_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>

#include <CGAL/Interpolation_gradient_fitting_traits_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/sibson_gradient_fitting.h>

#include <CGAL/Iterator_project.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/number_utils_classes.h>
#include <CGAL/utility.h>

#include <omp.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <vector>
#include <utility>

namespace TN
{

namespace
{

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K>             Vbb;
typedef CGAL::Triangulation_hierarchy_vertex_base_2<Vbb> Vb;
typedef CGAL::Triangulation_face_base_2<K>               Fb;
typedef CGAL::Triangulation_data_structure_2<Vb,Fb>      Tds;
typedef CGAL::Delaunay_triangulation_2<K,Tds>            Dt;

typedef CGAL::Delaunay_triangulation_adaptation_traits_2<Dt>                 At;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<Dt> Ap;
typedef CGAL::Voronoi_diagram_2<Dt,At,Ap>                                    Voronoi;

typedef Voronoi::Locate_result             Locate_result;
typedef Voronoi::Vertex_handle             Vertex_handle;
typedef Voronoi::Face_handle               Face_handle;
typedef Voronoi::Halfedge_handle           Halfedge_handle;
typedef Voronoi::Ccb_halfedge_circulator   Ccb_halfedge_circulator;

typedef K::Point_2 MeshType;

struct KeyHasher
{
    std::size_t operator()( const MeshType& m ) const
    {
        return std::hash<double>()( m[0] ) ^ std::hash<double>()( m[1] );
    }
};

typedef CGAL::Data_access< std::unordered_map< MeshType, std::int64_t, KeyHasher > > Value_Access;
}

class XGCGridBuilder
{
    Triangulation tri;
    Voronoi voronoi;
    // std::vector< MeshType > vertices;
    std::unordered_map< MeshType, std::int64_t, KeyHasher > idMap;

public:

    std::int64_t nearestNeighborIndex( const TN::Vec2< double > & point ) const
    {
        return idMap.at( tri.nearest_vertex( MeshType( point.x(), point.y() ) )->point() );
    }

    // void getBoundedVoronoiCells(
    //     std::vector< int64_t > & cellIndices,
    //     std::vector< float   > & cellAreas,
    //     std::vector< std::vector< float > > & voronoiCellBoundariesR,
    //     std::vector< std::vector< float > > & voronoiCellBoundariesZ )
    // {
    //     const auto ConvertToSeg = [&](const CGAL::Object seg_obj, bool outgoing) -> K::Segment_2
    //     {
    //         //One of these will succeed and one will have a NULL pointer
    //         const K::Segment_2 *dseg = CGAL::object_cast<K::Segment_2>(&seg_obj);
    //         const K::Ray_2     *dray = CGAL::object_cast<K::Ray_2>(&seg_obj);

    //         if (dseg)
    //         {
    //             //Okay, we have a segment
    //             return *dseg;
    //         }

    //         else
    //         {
    //             //Must be a ray
    //             const int RAY_LENGTH = 1000;
    //             const auto &source = dray->source();
    //             const auto dsx     = source.x();
    //             const auto dsy     = source.y();
    //             const auto &dir    = dray->direction();
    //             const auto tpoint  = K::Point_2(dsx+RAY_LENGTH*dir.dx(),dsy+RAY_LENGTH*dir.dy());

    //             if(outgoing)
    //             {
    //                 return K::Segment_2(
    //                     dray->source(),
    //                     tpoint
    //                 );
    //             }
    //             else
    //             {
    //                 return K::Segment_2(
    //                     tpoint,
    //                     dray->source()
    //                 );
    //             }
    //         }
    //     };

    //     const size_t NUM_VERTICIES = tri.number_of_vertices();

    //     cellAreas.resize(              NUM_VERTICIES );
    //     voronoiCellBoundariesR.resize( NUM_VERTICIES );
    //     voronoiCellBoundariesZ.resize( NUM_VERTICIES );
    //     cellIndices.resize(            NUM_VERTICIES );

    //     // std::cout << "\% complete: " << std::flush;

    //     int n_threads = 1;
    //     std::vector< Voronoi > per_thread_vornoi;

    //     #pragma omp parallel
    //     {
    //         n_threads = omp_get_num_threads();
    //     }

    //     per_thread_vornoi.resize( n_threads, Voronoi( voronoi ) );

    //     #pragma omp parallel for simd
    //     for( size_t index = 0; index < NUM_VERTICIES; ++index )
    //     {
    //         auto & my_voronoi = per_thread_vornoi[ omp_get_thread_num() ];
    //         auto & p = vertices[ index ];

    //         Locate_result lr = my_voronoi.locate( p );
    //         Face_handle* f = boost::get< Face_handle >( &lr );
    //         Ccb_halfedge_circulator ec_start = ( *f )->ccb();

    //         //Find a bounded edge to start on
    //         for(; ec_start->is_unbounded(); ec_start++) {}

    //         //Current location of the edge circulator
    //         Voronoi::Face::Ccb_halfedge_circulator ec = ec_start;

    //         CGAL::Polygon_2<K> pgon;

    //         do
    //         {
    //             //A half edge circulator representing a ray doesn't carry direction
    //             //information. To get it, we take the dual of the dual of the half-edge.
    //             //The dual of a half-edge circulator is the edge of a Delaunay triangle.
    //             //The dual of the edge of Delaunay triangle is either a segment or a ray.
    //             // const CGAL::Object seg_dual = rt.dual(ec->dual());
    //             const CGAL::Object seg_dual = my_voronoi.dual().dual(ec->dual());

    //             //Convert the segment/ray into a segment
    //             const auto this_seg = ConvertToSeg(seg_dual, ec->has_target());

    //             pgon.push_back(this_seg.source());

    //             //If the segment has no target, it's a ray. This means that the next
    //             //segment will also be a ray. We need to connect those two rays with a
    //             //segment. The following accomplishes this.
    //             if( ! ec->has_target() )
    //             {
    //                 const CGAL::Object nseg_dual = my_voronoi.dual().dual(ec->next()->dual());
    //                 const auto next_seg = ConvertToSeg(nseg_dual, ec->next()->has_target());
    //                 pgon.push_back(next_seg.target());
    //             }
    //         } while ( ++ec != ec_start ); //Loop until we get back to the beginning

    //         cellAreas[ index ] = pgon.area();
    //         for( auto vit = pgon.vertices_begin(); vit != pgon.vertices_end(); ++vit )
    //         {
    //             voronoiCellBoundariesR[ index ].push_back( vit->x() );
    //             voronoiCellBoundariesZ[ index ].push_back( vit->y() );
    //         }

    //         cellIndices[ index ] = index;

    //         // if( index % ( NUM_VERTICIES / 10 ) == 0 )
    //         // {
    //         //     std::cout << index / (double) ( NUM_VERTICIES - 1 ) * 100 << ", " << std::flush;
    //         // }
    //     }
    // }

    // void grid( SummaryGrid & summaryGrid,
    //            XGCPsinInterpolator   & psinInterpolator,
    //            XGCBFieldInterpolator & bFieldInterpolator,
    //            Vec2< float > poloidal_center )
    // {
    //     std::cout << "Bounding voronoi cells.\n";

    //     getBoundedVoronoiCells(
    //         summaryGrid.probeBoundaries.probeIndex,
    //         summaryGrid.probeBoundaries.area,
    //         summaryGrid.probeBoundaries.r,
    //         summaryGrid.probeBoundaries.z );

    //     std::cout << "Done.\n";

    //     size_t N_CELLS = summaryGrid.probeBoundaries.r.size();

    //     summaryGrid.probes.volume.resize( N_CELLS );
    //     summaryGrid.probeBoundaries.psin.resize(          N_CELLS );
    //     summaryGrid.probeBoundaries.poloidalAngle.resize( N_CELLS );
    //     summaryGrid.probeBoundaries.B.resize(             N_CELLS );
    //     summaryGrid.probeBoundaries.radius.resize(        N_CELLS );

    //     std::cout << "Interpolating mesh values onto voronoi cell boundary points.\n";
    //     std::cout << "\% complete: ";

    //     for( size_t i = 0; i < N_CELLS; ++i )
    //     {
    //         const size_t N_VERTS = summaryGrid.probeBoundaries.r[ i ].size();

    //         summaryGrid.probeBoundaries.psin[          i ].resize( N_VERTS );
    //         summaryGrid.probeBoundaries.poloidalAngle[ i ].resize( N_VERTS );
    //         summaryGrid.probeBoundaries.B[             i ].resize( N_VERTS );

    //         double radius = 0;

    //         for( size_t j = 0; j < N_VERTS; ++j )
    //         {
    //             summaryGrid.probeBoundaries.psin[ i ][ j ] =
    //                 psinInterpolator(
    //                     Vec2< double >(  summaryGrid.probeBoundaries.r[ i ][ j ], summaryGrid.probeBoundaries.z[ i ][ j ] ) );

    //             summaryGrid.probeBoundaries.poloidalAngle[ i ][ j ] =
    //                 ( Vec2< float >( summaryGrid.probeBoundaries.r[ i ][ j ], summaryGrid.probeBoundaries.z[ i ][ j ] )
    //                     - poloidal_center ).angle( Vec2< float >( 1.0, 0.0 ) );

    //             Vec3< double > b =
    //                 ( bFieldInterpolator(
    //                     Vec2< double >(  summaryGrid.probeBoundaries.r[ i ][ j ], summaryGrid.probeBoundaries.z[ i ][ j ] ) ) );

    //             summaryGrid.probeBoundaries.B[ i ][ j ] = std::sqrt( b.x()*b.x() + b.y()*b.y() + b.z()*b.z() );

    //             radius += summaryGrid.probeBoundaries.r[ i ][ j ];
    //         }

    //         summaryGrid.probeBoundaries.radius[ i ] = radius / ( double ) N_VERTS;
    //         summaryGrid.probes.volume[ i ] = 2.0 * M_PI * summaryGrid.probeBoundaries.area[ i ] * summaryGrid.probeBoundaries.radius[ i ];

    //         if( i % ( N_CELLS / 10 ) == 0 )
    //         {
    //             std::cout << i / (double) ( N_CELLS - 1 ) * 100 << ", " << std::flush;
    //         }
    //     }
    //     std::cout << "Done.\n";
    //     std::cout << "Copying the Vornoi duel Delaunay triangulation.\n";

    //     summaryGrid.probeTriangulation.resize( tri.number_of_faces() );
    //     size_t i = 0;
    //     for( auto face = tri.faces_begin(); face != tri.faces_end(); ++face )
    //     {
    //         TN::Vec2< double > v1( face->vertex(0)->point().x(), face->vertex(0)->point().y() );
    //         TN::Vec2< double > v2( face->vertex(1)->point().x(), face->vertex(1)->point().y() );
    //         TN::Vec2< double > v3( face->vertex(2)->point().x(), face->vertex(2)->point().y() );

    //         std::int64_t i1( idMap.find( face->vertex(0)->point() )->second );
    //         std::int64_t i2( idMap.find( face->vertex(1)->point() )->second );
    //         std::int64_t i3( idMap.find( face->vertex(2)->point() )->second );

    //         summaryGrid.probeTriangulation[ i ][ 0 ] = i1;
    //         summaryGrid.probeTriangulation[ i ][ 1 ] = i2;
    //         summaryGrid.probeTriangulation[ i ][ 2 ] = i3;
    //         ++i;
    //     }
    // }

    void save( const std::string & outpath, const std::string & ptype )
    {
        std::ofstream outFile( outpath + "/" + ptype + ".voronoi.bin.cin" );
        CGAL::set_binary_mode( outFile );
        outFile << voronoi;
        outFile.close();

        outFile.open( outpath + "/" + ptype + ".delaunay.bin.cin" );
        CGAL::set_binary_mode( outFile );
        outFile << tri;
        outFile.close();
    }

    void set( const std::vector< float > & r, const std::vector< float > & z )
    {
        tri = Triangulation();
        idMap.clear();

        const size_t SZ = r.size();
        // vertices.resize( SZ );

        for( size_t i = 0; i < SZ; ++i )
        {
            MeshType p( (double) r[ i ], (double) z[ i ] );
            tri.insert( p );
            idMap.insert( std::make_pair( p, i ) );
            // vertices[ i ] = p;
        }

        voronoi = Voronoi( tri );
    }
};

}

#endif // TN_GRID_BUILDER_HPP
