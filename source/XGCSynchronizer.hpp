#ifndef TN_XGC_SYNCHRONIZER
#define TN_XGC_SYNCHRONIZER

#include <iostream>
#include <adios2.h>
#include <mpi.h>
#include <unistd.h>
#include <thread>
#include <chrono>

namespace TN
{

namespace Synchro
{
    inline void waitForFileExistence( const std::string & filePath, const int64_t TIMEOUT )
    {
        std::chrono::high_resolution_clock::time_point waitStart = std::chrono::high_resolution_clock::now();
        const int SLEEP_TIME = 1000;    	
        while( ! std::ifstream( filePath ).good() )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( SLEEP_TIME ) );
            auto timeWaited = std::chrono::duration_cast<std::chrono::milliseconds>( 
                    waitStart - std::chrono::high_resolution_clock::now() ).count();
            if( timeWaited > TIMEOUT )
            {
                std::cerr << "Summarizer timed out after " 
                          << TIMEOUT << " milliseconds waiting for " 
                          << filePath << " to exist" << std::endl;
                exit( 1 );
            }
        }
    }

    inline bool getNextStep( 
        int64_t & step, 
        std::vector< int64_t > & steps,
        std::vector< int64_t >::iterator & it,
        const std::string & filePath,
        const std::string & ptype,
        bool wait,
        int64_t timeoutMS )
    {
        adios2::ADIOS adios( MPI_COMM_WORLD, adios2::DebugOFF );
        adios2::IO bpIO = adios.DeclareIO( "Synchro-IO" );
        adios2::Engine bpReader = bpIO.Open( filePath, adios2::Mode::Read );
        const int SLEEP_TIME = 100;

        adios2::Variable<double> phase = bpIO.InquireVariable< double >(
            ptype == "ions" ? "iphase" : "ephase" );

        int64_t nSteps;

        if( wait )
        {
        	std::chrono::high_resolution_clock::time_point waitStart = std::chrono::high_resolution_clock::now();
		    int64_t timeWaited = 0;
		    while( ( nSteps = phase.Steps() ) <= step && timeWaited < timeoutMS )
		    {
		        std::this_thread::sleep_for( std::chrono::milliseconds( SLEEP_TIME ) );
		        timeWaited = std::chrono::duration_cast<std::chrono::milliseconds>( 
		        	waitStart - std::chrono::high_resolution_clock::now() ).count();
		    }
	        step = nSteps-1;
	        it = steps.end() - 1;
        }
        else
        {   ++it;
        	if( it == steps.end() )
        	{
                return false;
        	}
        	step = step + 1;
        }

        steps.push_back( step );
        it = steps.end() - 1;

        return true;
    }
}

}

#endif