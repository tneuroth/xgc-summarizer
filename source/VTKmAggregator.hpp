#ifndef TN_VTKM_AGGREGATOR
#define TN_VTKM_AGGREGATOR

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

struct VTKmAggregator
{
    struct AggregateWorklet : public vtkm::worklet::WorkletReduceByKey
    {
        vtkm::Vec< vtkm::Int32,   2 > m_histDims;
        vtkm::Vec< vtkm::Float32, 2 > m_xRange;
        vtkm::Vec< vtkm::Float32, 2 > m_yRange;
        vtkm::Float32                 m_xWidth;
        vtkm::Float32                 m_yWidth;
        vtkm::Int64                   m_numHists;

        VTKM_CONT
        AggregateWorklet(
            const vtkm::Vec< vtkm::Int32, 2 > & histDims,
            const vtkm::Vec< vtkm::Float32, 2 > & xRange,
            const vtkm::Vec< vtkm::Float32, 2 > & yRange,
            const vtkm::Int64                   & numHists ) :
            m_histDims( histDims ),
            m_xRange( xRange ),
            m_yRange( yRange ),
            m_numHists( numHists )
        {
            m_xWidth = xRange[ 1 ] - xRange[ 0 ];
            m_yWidth = yRange[ 1 ] - yRange[ 0 ];
        }

        using ControlSignature = void(
                                     KeysIn              keys,
                                     ValuesIn<>          vxIn,
                                     ValuesIn<>          vyIn,
                                     ValuesIn<>          wIn,
                                     ReducedValuesOut<>  sumOut,
                                     ReducedValuesOut<>  squareSumOut,
                                     ReducedValuesOut<>  minOut,
                                     ReducedValuesOut<>  maxOut,
                                     ReducedValuesOut<>  countOut,
                                     ReducedValuesOut<>  histogram2DOut );

        using ExecutionSignature = void( _2, _3, _4, _5, _6, _7, _8, _9, _10 );
        using InputDomain = _1;


        template < typename OrigionalScalarVecType,
                   typename ReduceType,
                   typename HistReduceType  >
        VTKM_EXEC void operator()(
            const OrigionalScalarVecType  & vX,
            const OrigionalScalarVecType  & vY,
            const OrigionalScalarVecType  & w,
            ReduceType     & sum,
            ReduceType     & squareSum,
            ReduceType     & mn,
            ReduceType     & mx,
            ReduceType     & count,
            HistReduceType & histOut ) const
        {
            const vtkm::Int64 SZ = w.GetNumberOfComponents();
            const vtkm::Int32 N_ROWS = m_histDims[ 0 ];
            const vtkm::Int32 N_COLS = m_histDims[ 1 ];

            sum           = 0;
            squareSum     = 0;
            mn            = SZ > 0 ? w[ 0 ] : 0;
            mx            = mn;
            count         = SZ;

            histOut       = 0;
            HistReduceType _hc;

            if( SZ == 0 )
            {
                return;
            }

            vtkm::Float64 _sum  = 0;
            vtkm::Float64 _ssum = 0;

            vtkm::Float64 _c  = 0;
            vtkm::Float64 _cs = 0;

            // using the Kahan summation algorithm (compensated summatation)
            for( vtkm::Int64 i = 0; i < SZ; ++i )
            {
                const vtkm::Float64 v = w[ i ];

                // Weight Mean

                vtkm::Float64 _y = v - _c;
                vtkm::Float64 _t = _sum + _y;
                _c = ( _t - _sum ) - _y;
                _sum = _t;

                // Weight Variance

                vtkm::Float64 _ys = v - _cs;
                vtkm::Float64 _ts = _ssum + _ys;
                _c = ( _ts - _ssum ) - _ys;
                _ssum = _ts;

                mn = vtkm::Min( mn, v );
                mx = vtkm::Max( mx, v );

                // Weighted 2D Histogram

                vtkm::Int32 row = vtkm::Floor( ( ( vY[ i ] - m_yRange[ 0 ] ) / m_yWidth ) * N_ROWS );
                vtkm::Int32 col = vtkm::Floor( ( ( vX[ i ] - m_xRange[ 0 ] ) / m_xWidth ) * N_COLS );
                row = vtkm::Max( vtkm::Min( row, N_ROWS - 1 ), 0 );
                col = vtkm::Max( vtkm::Min( col, N_COLS - 1 ), 0 );
                const vtkm::Int64 index = row * N_COLS + col;
                const vtkm::Float32 _yh = v - _hc[ index ];
                const vtkm::Float32 _th = histOut[ index ] + _yh;
                _hc[ index ] = ( _th - histOut[ index ] ) - _yh;
                histOut[ index ] = _th;
            }

            sum = _sum;
            squareSum = _ssum;
        }
    };

    struct VarianceSumWorklet : public vtkm::worklet::WorkletReduceByKey
    {
        using ControlSignature = void(
                                     KeysIn keys,
                                     ValuesIn<> vIn,
                                     ReducedValuesIn<> meansIn,
                                     ReducedValuesOut<> sumOut );

        using ExecutionSignature = void( _2, _3, _4 );
        using InputDomain = _1;

        VTKM_CONT
        VarianceSumWorklet() {}

        template < typename ScalarVecType, typename ReduceType >
        VTKM_EXEC void operator()(
            const ScalarVecType & values,
            const ReduceType & mean,
            ReduceType & sum ) const
        {
            const vtkm::Int64 SZ = values.GetNumberOfComponents();

            vtkm::Float64 _sum = 0;
            vtkm::Float64 _c = 0;

            // using the Kahan summation algorithm (compensated summatation)
            for( vtkm::Int64 i = 0; i < SZ; ++i )
            {
                const vtkm::Float64 v = values[ i ] - mean;
                const vtkm::Float64 _y = v*v - _c;
                const vtkm::Float64 _t = _sum + _y;
                _c = ( _t - _sum ) - _y;
                _sum = _t;
            }
            sum = _sum;
        }
    };

    // struct Normalize1Worklet : public vtkm::worklet::WorkletMapField
    // {
    //     using ControlSignature = void(
    //                                  FieldInOut<>  vInOut,
    //                                  FieldIn<>    countIn );

    //     using ExecutionSignature = void( _1, _2, _3 );

    //     VTKM_CONT
    //     Normalize1Worklet() {}

    //     template < typename ScalarType >
    //     VTKM_EXEC void operator()(
    //         ScalarType  & v,
    //         const ScalarType  & count ) const
    //     {
    //         v = v / count;
    //     }
    // };

    // struct Normalize2Worklet : public vtkm::worklet::WorkletMapField
    // {
    //     using ControlSignature = void(
    //                                  FieldInOut<>  mInOut,
    //                                  FieldInOut<> msInOut,
    //                                  FieldIn<>    countIn );

    //     using ExecutionSignature = void( _1, _2, _3 );

    //     VTKM_CONT
    //     Normalize2Worklet() {}

    //     template < typename ScalarType >
    //     VTKM_EXEC void operator()(
    //         ScalarType  & m,
    //         ScalarType  & ms,
    //         const ScalarType  & count ) const
    //     {
    //         m = m / count;
    //         ms = vtkm::Sqrt( ms ) / count;
    //     }
    // };

    template < typename IntType,
               typename ScalarType,
               typename ScalarStorageTag,
               typename IndexType,
               typename ScalarHistType,
               typename ScalarHistTypeStorageTag,
               typename DeviceAdapter >
    void Run(
        const vtkm::Int64 nHists,
        const vtkm::Vec< IntType,     2 > & histDims,
        const vtkm::Vec< ScalarType,  2 > & xRange,
        const vtkm::Vec< ScalarType,  2 > & yRange,
        const vtkm::cont::ArrayHandle< ScalarType,     ScalarStorageTag >          & vX,
        const vtkm::cont::ArrayHandle< ScalarType,     ScalarStorageTag >          & vY,
        const vtkm::cont::ArrayHandle< ScalarType,     ScalarStorageTag >          & w,
        const vtkm::worklet::Keys < IndexType >                                    & keys,
        vtkm::cont::ArrayHandle< ScalarType,     ScalarStorageTag  >         & meanOut,
        vtkm::cont::ArrayHandle< ScalarType,     ScalarStorageTag  >         & rmsOut,
        vtkm::cont::ArrayHandle< ScalarType,     ScalarStorageTag  >         & varianceOut,
        vtkm::cont::ArrayHandle< ScalarType,     ScalarStorageTag  >         & minOut,
        vtkm::cont::ArrayHandle< ScalarType,     ScalarStorageTag  >         & maxOut,
        vtkm::cont::ArrayHandle< ScalarType,     ScalarStorageTag  >         & countOut,
        vtkm::cont::ArrayHandle< ScalarHistType, ScalarHistTypeStorageTag  > & histOut,
        DeviceAdapter device  )
    {
        AggregateWorklet aggregateWorklet( histDims, xRange, yRange, nHists );
        vtkm::worklet::DispatcherReduceByKey< AggregateWorklet, DeviceAdapter >
            aggregateDispatcher( aggregateWorklet );
        aggregateDispatcher.Invoke(
            keys,
            vX,
            vY,
            w,
            meanOut,
            rmsOut,
            minOut,
            maxOut,
            countOut,
            histOut
        );

        VarianceSumWorklet varianceSumWorklet;

        vtkm::worklet::DispatcherReduceByKey< VarianceSumWorklet, DeviceAdapter >
            varianceSumDispatcher( varianceSumWorklet );
        
        varianceSumDispatcher.Invoke(
            keys,
            w,
            meanOut,
            varianceOut
        );
    }
};

}

#endif