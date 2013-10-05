#include <iostream>
#include <array>
#include <glm/glm.hpp>
#include "tinycl.h"
#include "window.h"
#include "sparsematrix.h"
#include "simplefluid.h"

SimpleFluid::SimpleFluid(int dim, Window &win) 
	: context(tcl::DEVICE::GPU, true, false), dim(dim), window(win),
	interactionMat(createInteractionMatrix()), cgSolver(interactionMat, std::vector<float>(), context)
{}
void SimpleFluid::runTests(){
	std::vector<float> b;
	for (int i = 0; i < dim * dim; ++i){
		b.push_back(1);
	}
	cgSolver.updateB(b);
	cgSolver.solve();
	b = cgSolver.getResult();
	std::cout << "result of test solve on " << dim << "x" << dim << " fluid grid:\n";
	for (float &f : b){
		std::cout << f << ", ";
	}
	std::cout << std::endl;
}
void SimpleFluid::testVelocityField(){

}
SparseMatrix<float> SimpleFluid::createInteractionMatrix(){
	std::vector<MatrixElement<float>> elems;
	int nCells = dim * dim;
	for (int i = 0; i < nCells; ++i){
		//In the matrix all diagonal entires are 4 and neighbor cells are -1
		elems.push_back(MatrixElement<float>(i, i, 4));
		int x, y;
		cellPos(i, x, y);
		elems.push_back(MatrixElement<float>(i, cellNumber(x - 1, y), -1));
		elems.push_back(MatrixElement<float>(i, cellNumber(x + 1, y), -1));
		elems.push_back(MatrixElement<float>(i, cellNumber(x, y - 1), -1));
		elems.push_back(MatrixElement<float>(i, cellNumber(x, y + 1), -1));
	}
	return SparseMatrix<float>(elems, dim, true);
}
int SimpleFluid::cellNumber(int x, int y) const {
	if (x < 0){
		x += dim * (std::abs(x / dim) + 1);
	}
	if (y < 0){
		y += dim * (std::abs(y / dim) + 1);
	}
	return x % dim + (y % dim) * dim;
}
void SimpleFluid::cellPos(int n, int &x, int &y) const {
	x = n % dim;
	y = (n - x) / dim;
}
