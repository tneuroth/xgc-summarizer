

template <typename CooriVecT, typename CooriT, typename IdPortalT, typename CoordiPortalT>
VTKM_EXEC
bool check( 
  CooriT & dis,
  vtkm::Id sIdx,
  vtkm::Id tIdx,
  const IdPortalT& treePortal,
  const CoordiPortalT& coordiPortal )
{
    if (tIdx - sIdx == 1)
    {
        ///// leaf node
        vtkm::Id leafNodeIdx = treePortal.Get(sIdx);

        CooriVecT leaf;

        for( int i =0; i < N_DIMS; ++i )
        {
            leaf[ i ] = coordiPortal.Get(leafNodeIdx)[ i ];
        }

        CooriT _dis = dist( leaf, qc );

        if (_dis < dis)
        {
            dis = _dis;
            nnpIdx = leafNodeIdx;
        }

        return true;
    }

    return false;
}

template <typename CooriVecT, typename CooriT, typename IdPortalT, typename CoordiPortalT>
VTKM_EXEC_CONT void NearestNeighborSearch(const CooriVecT& qc,
        CooriT& dis,
        vtkm::Id& nnpIdx,
        vtkm::Int32 level,
        vtkm::Id sIdx,
        vtkm::Id tIdx,
        const IdPortalT& treePortal,
        const IdPortalT& splitIdPortal,
        const CoordiPortalT& coordiPortal) const
{
    if ( ! check( dis, sIdx, tIdx, treePortal, coordiPortal ) )
    {
        ///// leaf node
        const vtkm::Id & leafNodeIdx = treePortal.Get(sIdx);
        
        CooriT _dis = dist( coordiPortal.Get(leafNodeIdx)[ i ], qc );

        if (_dis < dis)
        {
            dis = _dis;
            nnpIdx = leafNodeIdx;
        }
    }
    else
    {
        //normal Node
        vtkm::Id splitNodeLoc = static_cast<vtkm::Id>(vtkm::Ceil(double((sIdx + tIdx)) / 2.0));

        const CooriT & splitAxis = coordiPortal.Get(splitIdPortal.Get(splitNodeLoc))[ level % N_DIMS ];
        CooriT & queryCoordi = qc[ level % N_DIMS ];
        ///

        if (queryCoordi <= splitAxis)
        {
            //left tree first
            if (queryCoordi - dis <= splitAxis)
            {
                NearestNeighborSearch(  qc,
                                        dis,
                                        nnpIdx,
                                        level + 1,
                                        sIdx,
                                        splitNodeLoc,
                                        treePortal,
                                        splitIdPortal,
                                        coordiPortal);
            
            }
            if (queryCoordi + dis > splitAxis)
            {
                NearestNeighborSearch(qc,
                                      dis,
                                      nnpIdx,
                                      level + 1,
                                      splitNodeLoc,
                                      tIdx,
                                      treePortal,
                                      splitIdPortal,
                                      coordiPortal);
            }
        }
        else
        {
            //right tree first
            if (queryCoordi + dis > splitAxis)
            {
                NearestNeighborSearch(qc,
                                      dis,
                                      nnpIdx,
                                      level + 1,
                                      splitNodeLoc,
                                      tIdx,
                                      treePortal,
                                      splitIdPortal,
                                      coordiPortal);
            }
            if (queryCoordi - dis <= splitAxis)
            {
                NearestNeighborSearch(qc,
                                      dis,
                                      nnpIdx,
                                      level + 1,
                                      sIdx,
                                      splitNodeLoc,
                                      treePortal,
                                      splitIdPortal,
                                      coordiPortal);
            }
        }
    }
}