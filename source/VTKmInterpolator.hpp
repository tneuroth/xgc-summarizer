#ifndef TN_VTKM_INTERPOLATOR
#define TN_VTKM_INTERPOLATOR

#include <vtkm/cont/DeviceAdapterAlgorithm.h>

#include <vtkm/Math.h>
#include <vtkm/VectorAnalysis.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/cont/ArrayHandleReverse.h>

#include <vtkm/worklet/Keys.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/WorkletReduceByKey.h>
#include <vtkm/worklet/DispatcherReduceByKey.h>
#include <vtkm/worklet/internal/DispatcherBase.h>
#include <vtkm/worklet/internal/WorkletBase.h>

namespace TN
{

struct VTKmInterpolator2D
{
    struct InterpolationWorklet2D : public vtkm::worklet::WorkletMapField
    {
        using ControlSignature = void(
                                     FieldIn<> pcIn,
                                     FieldIn<> mnIn,
                                     WholeArrayIn<> qcIn,
                                     WholeArrayIn<> scIn,
                                     WholeArrayIn<> nbIn,
                                     WholeArrayIn<> nsIn,
                                     FieldOut<> rsOut );

        using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

        VTKM_CONT
        InterpolationWorklet2D() {}

        template <typename CoordsPortalType,
                  typename CoordVecType,
                  typename ScalarPortalType,
                  typename IdPortalType,
                  typename IndexType,
                  typename ScalarType >
        VTKM_EXEC void operator()(
            const CoordVecType     & myPos,
            const IndexType        & myNearestNeighbor,
            const CoordsPortalType & meshCoords,
            const ScalarPortalType & meshScalars,
            const IdPortalType     & meshNeighborhoods,
            const IdPortalType     & meshNeighborhoodSums,
            ScalarType             & myScalarOut ) const
        {
            if( meshScalars.GetNumberOfValues() != meshCoords.GetNumberOfValues() )
            {
                std::cerr << "interpolate error: 0" << std::endl;
            }

            if( meshNeighborhoodSums.GetNumberOfValues() != meshScalars.GetNumberOfValues() )
            {
                std::cerr << "interpolate error: 1" << std::endl;
            }

            if( myNearestNeighbor < 0 ||  myNearestNeighbor >= meshNeighborhoodSums.GetNumberOfValues() )
            {
                std::cerr << "interpolate error: 2" << std::endl;
            }

            const IndexType OFFSET = myNearestNeighbor > 0 ? meshNeighborhoodSums[ myNearestNeighbor - 1 ] : 0;
            const IndexType NUM_NEIGHBORS = meshNeighborhoodSums[ myNearestNeighbor ] - OFFSET;
            const ScalarType nearestDistance = vtkm::Magnitude( myPos - meshCoords[ myNearestNeighbor ] );

            if( OFFSET < 0 || OFFSET + NUM_NEIGHBORS > meshNeighborhoods.GetNumberOfValues() )
            {
                std::cerr << "interpolate error: 3" << std::endl;
                std::cerr << meshNeighborhoods.GetNumberOfValues() << " " << NUM_NEIGHBORS << " " << OFFSET << std::endl;
            }

            ScalarType sum = nearestDistance * meshScalars[ myNearestNeighbor ];
            ScalarType distanceSum = nearestDistance;

            for( IndexType i = OFFSET; i < OFFSET + NUM_NEIGHBORS; ++i )
            {
                const IndexType IDX = meshNeighborhoods[ i ];

                if( IDX < 0 || IDX >= meshCoords.GetNumberOfValues() )
                {
                    std::cerr << "interpolate error: 4" << std::endl;
                }

                const ScalarType dist = vtkm::Magnitude( myPos - meshCoords[ IDX ] );

                sum += dist * meshScalars[ IDX ];
                distanceSum += dist;
            }

            if( distanceSum <= 0 )
            {
                std::cerr << "interpolate error: 5" << std::endl;
            }

            myScalarOut = sum / distanceSum;
        }
    };

    template <typename CoordType,
              typename CoordStorageTag,
              typename ScalarType,
              typename ScalarStorageTag,
              typename IndexType,
              typename IndexStorageTag,
              typename DeviceAdapter>
    void run(
        const vtkm::cont::ArrayHandle< vtkm::Vec< CoordType, 2 >, CoordStorageTag > & particleCoords,
        const vtkm::cont::ArrayHandle< IndexType, IndexStorageTag >                 & particleNeighbors,
        const vtkm::cont::ArrayHandle< vtkm::Vec<CoordType, 2>, CoordStorageTag >   & meshCoords,
        const vtkm::cont::ArrayHandle< ScalarType, ScalarStorageTag >               & meshScalars,
        const vtkm::cont::ArrayHandle< IndexType, IndexStorageTag >                 & meshNeighborhoods,
        const vtkm::cont::ArrayHandle< IndexType, IndexStorageTag >                 & meshNeighborhoodSums,
        vtkm::cont::ArrayHandle< ScalarType, ScalarStorageTag >                     & result,
        DeviceAdapter device )
    {
        InterpolationWorklet2D interpolationWorklet;
        vtkm::worklet::DispatcherMapField< InterpolationWorklet2D, DeviceAdapter >
        interpDispatcher( interpolationWorklet );

        interpDispatcher.Invoke(
            particleCoords,
            particleNeighbors,
            meshCoords,
            meshScalars,
            meshNeighborhoods,
            meshNeighborhoodSums,
            result );
    }
};

}

#endif
