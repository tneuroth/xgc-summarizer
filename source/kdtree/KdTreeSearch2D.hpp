#ifndef TN_KD_TREE_SEARCH_HPP
#define TN_KD_TREE_SEARCH_HPP

#include "KDTree/KdTree.h"
#include <vector>

namespace TN
{

struct KdTreeSearch2D
{

private:

    vtkm::worklet::KdTree< 2 > m_kdTree;
    vtkm::cont::ArrayHandle< vtkm::Vec< vtkm::Float32, 2 > > m_gridHandle;
    std::vector< vtkm::Vec< vtkm::Float32, 2 > > m_gridPoints;

public:

	void setGrid( 
		const std::vector< float > & r,
	    const std::vector< float > & z );

	void run(
	    std::vector< int64_t > & result,
	    const std::vector< float > & r,
	    const std::vector< float > & z );

    KdTreeSearch2D();
};

}

#endif