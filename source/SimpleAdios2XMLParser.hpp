#ifndef TN_SIMPLE_A2_XML_PARSER
#define TN_SIMPLE_A2_XML_PARSER

#include <string>
#include <map>
#include <fstream>
#include <iostream>

namespace TN
{

namespace XML
{

inline std::map< std::string, std::string > extractIoEngines( 
	const std::string & xmlPath )
{
    std::map< std::string, std::string > ioEngines;

    std::ifstream file( xmlPath );
    if( ! file.is_open() )
    {
        std::cerr << "couldn't open file " << xmlPath << std::endl;
    }

    std::string line;
    while( std::getline( file, line ) )
    {
        std::size_t pos1;
        std::string ioStr = "<io name=\"";
        if ( ( pos1 = line.find( ioStr ) ) != std::string::npos )
        {
            std::size_t pos2 = line.find( "\">" );
            std::string name( line.begin() + pos1 + ioStr.size(), line.begin() + pos2 );
	        std::getline( file, line );
            std::string enStr = "<engine type=\"";
	        if ( ( pos1 = line.find( enStr ) ) != std::string::npos )
            {
	            std::size_t pos2 = line.find( "\">" );
	            std::string engine( line.begin() + pos1 + enStr.size(), line.begin() + pos2 );

	            ioEngines.insert( { name, engine } );
            }
        }
    }

    return ioEngines;
}

}

}

#endif