#ifndef TN_XGC_CONSTANT_READER
#define TN_XGC_CONSTANT_READER

#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace TN
{

template< typename FloatType >
inline void loadConstants(
    const std::string & units_path,
    std::map< std::string, FloatType > & constants )
{
    std::ifstream inFile;
    inFile.open( units_path );
    std::string line;

    if( ! inFile.is_open() )
    {
        std::cerr << "couldn't open " << units_path << std::endl;
    }
    while( inFile )
    {
        if ( ! std::getline( inFile, line ) ) break;
        line.erase( std::remove_if( line.begin(), line.end(), ::isspace ), line.end() );
        line.pop_back();
        std::stringstream ss( line );
        std::string name, valueStr;
        std::getline( ss, name, '=' );
        std::getline( ss, valueStr );

        constants.insert( { name, static_cast< FloatType >( std::stold( valueStr ) ) } );
    }

    inFile.close();

    if( constants.count( "eq_axis_r" ) <= 0 || constants.count( "eq_axis_z" ) <= 0 )
    {
        std::cerr << "units.m missing eq_axis_r, and eq_axis_z, need to compute poloidal angles" << std::endl;
    }
}

}

#endif