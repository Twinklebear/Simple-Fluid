#include <iostream>
#include <cstdlib>
#include <ctime>
#include "sparsematrix.h"
#include "cgsolver.h"

int main(int argc, char **argv){
	//Create a basic identity matrix
	int rows[4] = { 0, 1, 2, 3 };
	int cols[4] = { 0, 1, 2, 3 };
	float vals[4] = { 1, 1, 1, 1 };
	SparseMatrix<float> mat(rows, cols, vals, 4, 4, true);

	std::vector<float> b;
	for (int i = 0; i < mat.dim; ++i){
		b.push_back(i + 1);
	}

	std::cout << "Simple 4x4 system solve with identity matrix\n";
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	CGSolver solver(mat, b, context);
	solver.solve();
	std::vector<float> result = solver.getResult();
	std::cout << "result is: ";
	for (float f : result){
		std::cout << f << ", ";
	}
	std::cout << "\nWill now change b to 4, 3, 2, 1 and re-solve\n";
	for (int i = 0; i < mat.dim; ++i){
		b.at(i) = mat.dim - i;
	}
	solver.updateB(b);
	solver.solve();
	result = solver.getResult();
	std::cout << "result is: ";
	for (float f : result){
		std::cout << f << ", ";
	}

	//Run another solver on the context to solve a larger system
	std::cout << "\nMore complicated 48x48 system using matrix bcsstk01 and random b values\n";
	SparseMatrix<float> mat2("../res/bcsstk01.mtx", true);

	std::srand(std::time(0));
	std::vector<float> b2;
	for (int i = 0; i < mat2.dim; ++i){
		b2.push_back(std::rand() / static_cast<float>(RAND_MAX / 100.f));
	}
	CGSolver solver2(mat2, b2, context);
	solver2.solve();
	result = solver2.getResult();
	std::cout << "result is: ";
	for (float f : result){
		std::cout << f << ", ";
	}
	std::cout << std::endl;

    return 0;
}
