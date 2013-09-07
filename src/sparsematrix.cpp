#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "sparsematrix.h"

MatrixElement::MatrixElement() : row(-1), col(-1), val(-1)
{}
MatrixElement::MatrixElement(int row, int col, float val)
	: row(row), col(col), val(val)
{}
MatrixElement MatrixElement::diagonal() const {
	return MatrixElement(col, row, val);
}
bool rowMajor(const MatrixElement &lhs, const MatrixElement &rhs){
	return (lhs.row < rhs.row || (lhs.row == rhs.row && lhs.col < rhs.col));
}
bool colMajor(const MatrixElement &lhs, const MatrixElement &rhs){
		return (lhs.col < rhs.col || (lhs.col == rhs.col && lhs.row < rhs.row));
}
SparseMatrix::SparseMatrix(const std::string &file, bool rowMaj){
	loadMatrix(file, rowMaj);
}
SparseMatrix::SparseMatrix(const int *row, const int *col, const float *vals, int nElems, int dim, bool rowMaj)
	: dim(dim)
{
	elements.reserve(nElems);
	for (int i = 0; i < nElems; ++i)
		elements.push_back(MatrixElement(row[i], col[i], vals[i]));
	//Select the appropriate sorting for the way we want to treat the matrix, row-maj or col-maj
	//If row major we'll sort by row, if column sort by col, both in ascending order
	if (rowMaj)
		std::sort(elements.begin(), elements.end(), rowMajor);
	else
		std::sort(elements.begin(), elements.end(), colMajor);
}
SparseMatrix::SparseMatrix(const std::vector<MatrixElement> &elem, int dim, bool symmetric, bool rowMaj) 
	: elements(elem), dim(dim), symmetric(symmetric)
{
	if (rowMaj)
		std::sort(elements.begin(), elements.end(), rowMajor);
	else
		std::sort(elements.begin(), elements.end(), colMajor);
}
void SparseMatrix::getRaw(int *row, int *col, float *val) const {
	//It's assumed the appropriate amount of space is allocated for each array
	//ie. that row is a int[elements.size()] array, etc.
	for (size_t i = 0; i < elements.size(); ++i){
		row[i] = elements.at(i).row;
		col[i] = elements.at(i).col;
		val[i] = elements.at(i).val;
	}
}
void SparseMatrix::loadMatrix(const std::string &file, bool rowMaj){
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
std::ostream& operator<<(std::ostream &os, const SparseMatrix &mat){
	for (MatrixElement e : mat.elements)
		os << "element: " << e.row << ", " << e.col << " : " << e.val << "\n";
	return os;
}
