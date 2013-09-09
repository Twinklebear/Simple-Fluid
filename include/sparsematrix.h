#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <string>
#include <vector>
#include <ostream>
#include <algorithm>

/*
* Defines an element in the sparse matrix, its row, column and value
*/
template<class T>
struct MatrixElement {
	int row, col;
	T val;

	MatrixElement() : row(-1), col(-1), val(T{})
	{}
	//Create an element
	MatrixElement(int row, int col, T val) : row(row), col(col), val(val)
	{}
	//Get a diagonal version of the element, ie. with row and col switched
	MatrixElement<T> diagonal() const {
		return MatrixElement<T>(col, row, val);
	}
};
/*
* Comparison function for sorting elements by their row, returns true if lhs is a lower row
* than rhs, or if they're in the same row returns true if lhs is in a lower column
*/
template<class T>
bool rowMajor(const MatrixElement<T> &lhs, const MatrixElement<T> &rhs){
	return (lhs.row < rhs.row || (lhs.row == rhs.row && lhs.col < rhs.col));
}
/*
* Comparison function for sorting elements by their column, returns true if lhs is a lower
* col than rhs, or if they're in the same col returns true if lhs is in a lower row
*/
template<class T>
bool colMajor(const MatrixElement<T> &lhs, const MatrixElement<T> &rhs){
	return (lhs.col < rhs.col || (lhs.col == rhs.col && lhs.row < rhs.row));
}

/*
* A sparse matrix, supports loading coordinate symmetric matrices
* from matrix market files or specifying directly the sparse matrix elements
*/
template<class T>
class SparseMatrix {
public:
	/*
	* Load the matrix from a matrix market file, rowMaj true if we want to 
	* sort by row ascending, ie. row major, currently only support
	* loading from coordinate real symmetric matrices
	*/
	SparseMatrix(const std::string &file, bool rowMaj = true){
		loadMatrix(file, rowMaj);
	}
	/*
	* Load the matrix from data contained within the row, col, and val arrays
	* to setup a matrix of dim dimensions in the major order desired
	*/
	SparseMatrix(const int *row, const int *col, const T *vals, int nElems, int dim, bool rowMaj = true) 
		: dim(dim) 
	{
		elements.reserve(nElems);
		for (int i = 0; i < nElems; ++i)
			elements.push_back(MatrixElement<T>(row[i], col[i], vals[i]));
		//Select the appropriate sorting for the way we want to treat the matrix, row-maj or col-maj
		//If row major we'll sort by row, if column sort by col, both in ascending order
		if (rowMaj)
			std::sort(elements.begin(), elements.end(), rowMajor<T>);
		else
			std::sort(elements.begin(), elements.end(), colMajor<T>);
	}
	/*
	* Load the matrix from a list of elements, specifying the dimensions and if it's symmetric or not
	*/
	SparseMatrix(const std::vector<MatrixElement<T>> &elem, int dim, bool symmetric, bool rowMaj = true)
		: elements(elem), dim(dim), symmetric(symmetric)
	{
		if (rowMaj)
			std::sort(elements.begin(), elements.end(), rowMajor<T>);
		else
			std::sort(elements.begin(), elements.end(), colMajor<T>);
	}
	/*
	* Get the underlying row, column and value arrays for use in passing to OpenCL
	* the row, col and val arrays must have been allocated previously and should have enough
	* room to contain elements.size() values
	*/
	void getRaw(int *row, int *col, T *val) const {
		for (size_t i = 0; i < elements.size(); ++i){
			row[i] = elements.at(i).row;
			col[i] = elements.at(i).col;
			val[i] = elements.at(i).val;
		}
	}

private:
	//Parse and load a matrix from a matrix market file
	void loadMatrix(const std::string &file, bool rowMaj = true){
		if (file.substr(file.size() - 3, 3) != "mtx"){
			std::cout << "Error: Not a Matrix Market file: " << file << std::endl;
			return;
		}
		
		std::ifstream matFile(file.c_str());
		if (!matFile.is_open()){
			std::cout << "Error: Failed to open Matrix Market file: " << file << std::endl;
			return;
		}
		
		//The first line after end of comments is the M N L information
		bool readComment;
		std::string line;
		while (std::getline(matFile, line)){
			if (line.at(0) == '%'){
				readComment = true;
				//2 %% indicates header information
				if (line.at(1) == '%'){
					if (line.find("coordinate") == -1){
						std::cout << "Error: non-coordinate matrix is unsupported" << std::endl;
						return;
					}
					if (line.find("symmetric") == -1){
						std::cout << "Error: non-symmetric matrix is unsupported" << std::endl;
						return;
					}
					symmetric = true;
				}
			}
			//Non-comments will either be M N L info or matrix elements
			if (line.at(0) != '%'){
				//If this is the first line after comments it's the M N L info
				if (readComment){
					std::stringstream ss(line);
					int m, n, l;
					ss >> m >> n >> l;
					//Also account for # off diagonal elements
					if (symmetric)
						l += l - n;
					dim = m;
					elements.reserve(l);
				}
				//Otherwise it's element data in the form: row col value
				else {
					MatrixElement elem;
					std::stringstream ss(line);
					ss >> elem.row >> elem.col >> elem.val;
					//Matrix Market is 1-indexed, so subtract 1
					elem.row--;
					elem.col--;
					elements.push_back(elem);
					//If the row is symmetric we'll only be given the diagonal and lower-triangular implying that
					//if we're given an off-diagonal: i j v then a corresponding element: j i v should also be inserted
					if (symmetric && elem.row != elem.col)
						elements.push_back(elem.diagonal());
				}
				readComment = false;
			}
		}
		//Select the appropriate sorting for the way we want to treat the matrix, row-maj or col-maj
		//If row major we'll sort by row, if column sort by col, both in ascending order
		if (rowMaj)
			std::sort(elements.begin(), elements.end(), rowMajor);
		else
			std::sort(elements.begin(), elements.end(), colMajor);
	}

public:
	std::vector<MatrixElement<T>> elements;
	bool symmetric;
	int dim;
};
//Print the sparse matrix
template<class T>
std::ostream& operator<<(std::ostream &os, const SparseMatrix<T> &mat){
	for (MatrixElement<T> e : mat.elements)
		os << "element: " << e.row << ", " << e.col << " : " << e.val << "\n";
	return os;
}

#endif
