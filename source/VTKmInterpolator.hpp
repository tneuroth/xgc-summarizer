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

        using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7 );

        VTKM_EXEC
        vtkm::Id m_maxNeighbors;

        VTKM_CONT
        InterpolationWorklet2D( vtkm::Id maxNeighbors ) : m_maxNeighbors( maxNeighbors ){  }

        /* Based on the algorithm given by: http://www.mcs.anl.gov/~fathom/meshkit-docs/html/circumcenter_8cpp_source.html */
        template < typename CoordVecType >
        VTKM_EXEC CoordVecType getTriangleCircumenter(
            const CoordVecType & a,
            const CoordVecType & b,
            const CoordVecType & c )
        {
            double xba = b[0] - a[0];
            double yba = b[1] - a[1];
            double xca = c[0] - a[0];
            double yca = c[1] - a[1];
            double balength = xba * xba + yba * yba;
            double calength = xca * xca + yca * yca;
            double denominator = 0.5 / (xba * yca - yba * xca);
            CoordVecType circumcenter;
            circumcenter[0] = (yca * balength - yba * calength) * denominator;
            circumcenter[1] = (xba * calength - xca * balength) * denominator;
            return circumcenter;
        }

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
            const IndexType OFFSET = myNearestNeighbor > 0 ? meshNeighborhoodSums[ myNearestNeighbor - 1 ] : 0;
            const IndexType NUM_NEIGHBORS = meshNeighborhoodSums[ myNearestNeighbor ] - OFFSET;
            
            const CoordVecType voronoiSites[ NUM_NEIGHBORS ];

            for( IndexType i = OFFSET; i < OFFSET + NUM_NEIGHBORS - 1; ++i )
            {
                const IndexType IA = myNearestNeighbor;
                const IndexType IB = meshNeighborhoods[ i ];
                const IndexType IC = meshNeighborhoods[ i + 1 ];
                
                voronoiSites[ i - OFFSET ] = getTriangleCircumenter(  
                    meshCoords[ IA ],
                    meshCoords[ IB ],
                    meshCoords[ IC ] );
            }
            
            myScalarOut = meshScalars[ myNearestNeighbor ];
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
        const std::int64_t                                                          MAX_NEIGHBORS,
        vtkm::cont::ArrayHandle< ScalarType, ScalarStorageTag >                     & result,
        DeviceAdapter device )
    {
        InterpolationWorklet2D interpolationWorklet( static_cast< vtkm::Id >( MAX_NEIGHBORS )  );
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
