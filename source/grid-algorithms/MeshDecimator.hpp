#ifndef TN_MESH_DECIMATOR_HPP
#define TN_MESH_DECIMATOR_HPP

#include "../Types/Vec.hpp"

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

#include <CGAL/Delaunay_mesher_2.h>

#include <CGAL/Iterator_project.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/number_utils_classes.h>
#include <CGAL/utility.h>
#include <CGAL/config.h>
#include <CGAL/boost/graph/helpers.h>

#include <omp.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <vector>
#include <utility>
#include <set>

namespace TN
{

namespace
{

typedef CGAL::Exact_predicates_inexact_constructions_kernel   K;
typedef CGAL::Triangulation_vertex_base_2<K>                Vbb;
typedef CGAL::Triangulation_hierarchy_vertex_base_2<Vbb>     Vb;
typedef CGAL::Triangulation_face_base_2<K>                   Fb;
typedef CGAL::Triangulation_data_structure_2<Vb,Fb>         Tds;
typedef CGAL::Delaunay_triangulation_2<K,Tds>                Dt;

typedef CGAL::Delaunay_triangulation_adaptation_traits_2<Dt>                 At;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<Dt> Ap;
typedef CGAL::Voronoi_diagram_2<Dt,At,Ap>                                    Voronoi;

typedef Voronoi::Locate_result             Locate_result;
typedef Voronoi::Vertex_handle             Vertex_handle;
typedef Voronoi::Face_handle               Face_handle;
typedef Voronoi::Halfedge_handle           Halfedge_handle;
typedef Voronoi::Ccb_halfedge_circulator   Ccb_halfedge_circulator;

typedef K::Point_2 MeshType;

}

class MeshDecimator
{
    Dt tri;
    std::vector< MeshType > vertices;
    std::map< MeshType, std::int64_t, K::Less_xy_2 > idMap;

public:

    void save( const std::string & outpath, const std::string & ptype )
    {
        std::ofstream  outFile( outpath + "/" + ptype + ".delaunay.bin.cin" );
        CGAL::set_binary_mode( outFile );
        outFile << tri;
        outFile.close();
    }

    // void reduce()
    // {
    //     std::set< size_t > toRemove();
    //     for( auto face = tri.faces_begin(); face != tri.faces_end(); ++face )
    //     {
    //         TN::Vec2< double > v1( face->vertex(0)->point().x(), face->vertex(0)->point().y() );
    //         TN::Vec2< double > v2( face->vertex(1)->point().x(), face->vertex(1)->point().y() );
    //         TN::Vec2< double > v3( face->vertex(2)->point().x(), face->vertex(2)->point().y() );

    //         std::int64_t i1( idMap.find( face->vertex(0)->point() )->second );
    //         std::int64_t i2( idMap.find( face->vertex(1)->point() )->second );
    //         std::int64_t i3( idMap.find( face->vertex(2)->point() )->second );
  
    //         std::outfile << "f " << (i1+1) << " " << (i2+1) << " " << (i3+1) << "\n";
    //     }
    // }

    void writeObj( const std::string & outpath )
	{
	    std::ofstream outfile( outpath );

	    for( std::size_t i = 0, end = vertices.size(); i < end; ++i )
	    {
	        outfile << "v " << vertices[ i ].x() << " " << vertices[ i ].y() << " 0\n";
	    }
	    outfile << "\n";

        for( auto face = tri.faces_begin(); face != tri.faces_end(); ++face )
        {
            TN::Vec2< double > v1( face->vertex(0)->point().x(), face->vertex(0)->point().y() );
            TN::Vec2< double > v2( face->vertex(1)->point().x(), face->vertex(1)->point().y() );
            TN::Vec2< double > v3( face->vertex(2)->point().x(), face->vertex(2)->point().y() );

            std::int64_t i1( idMap.find( face->vertex(0)->point() )->second );
            std::int64_t i2( idMap.find( face->vertex(1)->point() )->second );
            std::int64_t i3( idMap.find( face->vertex(2)->point() )->second );
  
            outfile << "f " << (i1+1) << " " << (i2+1) << " " << (i3+1) << "\n";
        }

	    outfile.close();
	}

    void set( 
    	const std::vector< TN::Vec2< double > > & xy )
    {
        tri = Dt();
        idMap.clear();
        
        const size_t SZ = xy.size();
        vertices.clear();
        
        std::cout << "\n" << SZ;
        for( size_t i = 0, j = 0; i < SZ-1; ++i )
        {
            if( i % 2 == 0 )
            {
                MeshType p( xy[ i ].x(), xy[ i ].y() );
                tri.insert( p );
                idMap.insert( std::make_pair( p, j ) );
                vertices.push_back( p );
                ++j;
            }
        }
        std::cout << " " << vertices.size() << std::endl;

        writeObj("triangulation.obj");
    }
};

}

#endif // TN_MESH_DECIMATOR_HPP
