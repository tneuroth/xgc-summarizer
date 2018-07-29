#ifndef TN_SUMMARY_UTILS
#define TN_SUMMARY_UTILS

#include <vector>
#include <string>
#include <fstream>

inline void writeTriangularMeshObj(
    const std::vector< float > & r,
    const std::vector< float > & z,
    const std::vector< Triangle > & mesh,
    const std::string & outpath )
{
    std::ofstream outfile( outpath );
    for( size_t i = 0, end = r.size(); i < end; ++i )
    {
        outfile << "v " << r[ i ] << " " << z[ i ] << " 0\n";
    }
    outfile << "\n";

    for( size_t i = 0, end = mesh.size(); i < end; ++i )
    {
        outfile << "f";
        for( size_t j = 0; j < 3; ++j )
        {
            outfile << " " << mesh[ i ][ j ] + 1;
        }
        outfile << "\n";
    }
    outfile.close();
}

inline void writePolygonalMeshObj(
    const std::vector< std::vector< float > > & r,
    const std::vector< std::vector< float > > & z,
    const std::string & outpath )
{
    std::ofstream outfile( outpath );
    for( size_t i = 0, end = r.size(); i < end; ++i )
    {
        for( size_t j = 0; j < r[ i ].size(); ++ j )
        {
            outfile << "v " << r[ i ][ j ] << " " << z[ i ][ j ] << " 0\n";
        }
    }
    outfile << "\n";

    int64_t index = 1;
    for( size_t i = 0, end = r.size(); i < end; ++i )
    {
        outfile << "f";
        for( size_t j = 0; j < r[ i ].size(); ++j )
        {
            outfile << " " << index++;
        }
        outfile << "\n";
    }
    outfile.close();
}


#endif
