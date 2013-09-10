#include <array>
#include <glm/glm.hpp>
#include "tinycl.h"
#include "window.h"
#include "sparsematrix.h"
#include "simplefluid.h"

const std::array<glm::vec3, 8> SimpleFluid::quad = {
	//Vertex positions
	glm::vec3(-1.0, -1.0, 0.0),
	glm::vec3(1.0, -1.0, 0.0),
	glm::vec3(-1.0, 1.0, 0.0),
	glm::vec3(1.0, 1.0, 0.0),
	//UV coords
	glm::vec3(0.0, 0.0, 0.0),
	glm::vec3(1.0, 0.0, 0.0),
	glm::vec3(0.0, 1.0, 1.0),
	glm::vec3(1.0, 1.0, 0.0)
};
const std::array<unsigned short, 6> SimpleFluid::quadElems = {
		0, 1, 2,
		1, 3, 2
};

SimpleFluid::SimpleFluid(int dim, Window &win) 
	: context(tcl::DEVICE::GPU, true, false), dim(dim), window(win), 
		interactionMat(createInteractionMatrix())
{}
void SimpleFluid::runTests(){

}
void SimpleFluid::testVelocityField(){

}
SparseMatrix<float> SimpleFluid::createInteractionMatrix(){
	std::vector<MatrixElement<float>> elems;
	int nCells = static_cast<float>(std::pow(dim, 2));
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
	//Wrap coordinates if needed, no x,y should be passed that's less than -dim - 1
	//so it's ok to just do += dim for negative x,y vals
	if (x < 0){
		x += dim;
	}
	else if (x >= dim){
		x %= dim;
	}
	if (y < 0){
		y += dim;
	}
	else if (y >= dim){
		y %= dim;
	}
	return (x + y * dim);
}
void SimpleFluid::cellPos(int n, int &x, int &y) const {
	x = n % dim;
	y = (n - x) / dim;
}
