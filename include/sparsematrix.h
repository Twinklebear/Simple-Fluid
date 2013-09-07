#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <string>
#include <vector>
#include <ostream>

/*
* Defines an element in the sparse matrix, its row, column and value
*/
struct MatrixElement {
	int row, col;
	float val;

	MatrixElement();
	//Create an element
	MatrixElement(int row, int col, float val);
	//Get a diagonal version of the element, ie. with row and col switched
	MatrixElement diagonal() const;
};
/*
* Comparison function for sorting elements by their row, returns true if lhs is a lower row
* than rhs, or if they're in the same row returns true if lhs is in a lower column
*/
bool rowMajor(const MatrixElement &lhs, const MatrixElement &rhs);
/*
* Comparison function for sorting elements by their column, returns true if lhs is a lower
* col than rhs, or if they're in the same col returns true if lhs is in a lower row
*/
bool colMajor(const MatrixElement &lhs, const MatrixElement &rhs);

/*
* A sparse matrix, supports loading coordinate symmetric matrices
* from matrix market files or specifying directly the sparse matrix elements
*/
class SparseMatrix {
public:
	/*
	* Load the matrix from a matrix market file, rowMaj true if we want to 
	* sort by row ascending, ie. row major, currently only support
	* loading from coordinate real symmetric matrices
	*/
	SparseMatrix(const std::string &file, bool rowMaj = true);
	/*
	* Load the matrix from data contained within the row, col, and val arrays
	* to setup a matrix of dim dimensions in the major order desired
	*/
	SparseMatrix(const int *row, const int *col, const float *vals, int nElems, int dim, bool rowMaj = true);
	/*
	* Load the matrix from a list of elements, specifying the dimensions and if it's symmetric or not
	*/
	SparseMatrix(const std::vector<MatrixElement> &elem, int dim, bool symmetric, bool rowMaj = true);
	/*
	* Get the underlying row, column and value arrays for use in passing to OpenCL
	* the row, col and val arrays must have been allocated previously and should have enough
	* room to contain elements.size() values
	*/
	void getRaw(int *row, int *col, float *val) const;

private:
	//Parse and load a matrix from a matrix market file
	void loadMatrix(const std::string &file, bool rowMaj = true);

public:
	std::vector<MatrixElement> elements;
	bool symmetric;
	int dim;
};
//Print the sparse matrix
std::ostream& operator<<(std::ostream &os, const SparseMatrix &mat);

#endif
