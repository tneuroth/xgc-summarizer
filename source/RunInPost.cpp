template< typename ValueType >
void XGCAggregator< ValueType >::runInPost()
{
    double summaryStepTime = 0.0;
    int64_t outputStep     = 0;
    std::vector< int64_t > steps = { 200, 400 };
    SummaryStep2< ValueType > summaryStep;

    for( auto tstep : steps )
    {
        std::string tstepStr = std::to_string( tstep );
        std::chrono::high_resolution_clock::time_point readStartTime = std::chrono::high_resolution_clock::now();

        int64_t simstep;
        double  realtime;

        int64_t totalNumParticles = readBPParticleDataStep(
             m_phase,
             "ions",
             m_restartPath + "xgc.restart." + std::string( 5 - tstepStr.size(), '0' ) + tstepStr +  ".bp",
             m_rank,
             m_nranks,
             simstep,
             realtime );

        summaryStep.numParticles = totalNumParticles;
        summaryStep.setStep( outputStep, simstep, realtime );
        summaryStep.objectIdentifier = "ions";

        std::chrono::high_resolution_clock::time_point readStartEnd = std::chrono::high_resolution_clock::now();

        std::cout << "RANK: " << m_rank
                  << ", adios Read time took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( readStartEnd - readStartTime ).count()
                  << " std::chrono::milliseconds " << " for " << m_phase.size()/9 << " particles" << std::endl;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////

        std::unique_ptr< adios2::IO > io;
        std::unique_ptr< adios2::Engine > en;

        computeSummaryStep(
            m_phase,
            summaryStep,
            "ions",
            io,
            en );

        ++outputStep;
    }
}
