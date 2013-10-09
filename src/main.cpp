#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <SOIL.h>
#include "util.h"
#include "simplefluid.h"
#include "tinycl.h"
#include "window.h"

void runCGTests();
//Test CG solve on the identity, just a sanity check
void testCGSolveIdentity();
//Test CG for how it'll be used in the simulation
void testCGSolveWiki();
//Test CG for consistency/larger matrices, ie. as it'll be used in the fluid sim
void testCGSim();
//Give a stress test to the solver with a large fluid grid and solve it multiple times
void testCGStress(int dim);
//Test the velocity divergence kernel
void testVelocityDivergence();
//Test the pressure subtraction to update the velocity field
void testSubtractPressureX();
void testSubtractPressureY();
//Test the field advection kernel
void testFieldAdvect();
//Test the x velocity field advection kernel
void testVXFieldAdvect();
//Test the y velocity field advection kernel
void testVYFieldAdvect();

int main(int argc, char **argv){
	testCGStress(16);

	SDL sdl(SDL_INIT_EVERYTHING);
	Window win("Fluid!", 640, 480);
	//16 is the dimensions of the textures we're loading
	SimpleFluid fluidSim(16, win);
	fluidSim.initSim();
	fluidSim.runSim();

    return 0;
}
void runCGTests(){
	std::cout << "Using CG to solve an identity system\n";
	testCGSolveIdentity();
	std::cout << "Using CG to solve an example from wikipedia\n";
	testCGSolveWiki();
	std::cout << "Using CG to solve an 16x16 fluid system\n";
	testCGSim();
	//Warning: be very wary of the memory usage of higher grid sizes.
	int dim = 32;
	std::cout << "Stress testing CG with multiple solves of a "
		<< dim << "x" << dim << " system\n";
	testCGStress(dim);
}
void testCGSolveIdentity(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	int row[] = { 0, 1, 2, 3 };
	int col[] = { 0, 1, 2, 3 };
	float val[] = { 1, 1, 1, 1 };
	SparseMatrix<float> matrix(row, col, val, 4, 4);
	std::vector<float> b;
	for (int i = 0; i < 4; ++i){
		b.push_back(i + 1);
	}

	//Make sure updateB works properly too
	CGSolver solver(matrix, std::vector<float>(), context, 10);
	solver.updateB(b);
	solver.solve();
	std::vector<float> x = solver.getResult();
	std::cout << "Identity result: ";
	for (float f : x){
		std::cout << f << ", ";
	}
	std::cout << std::endl;
}
void testCGSolveWiki(){
	/*
	* This solves the CG example on the wikipedia page so that we have
	* a known correct answer to check against.
	* Example: https://en.wikipedia.org/wiki/Conjugate_gradient#Numerical_example
	*/
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	/*
	* Matrix: 4 1    b: 1
	*		  1 3		2
	*
	* Expect x: .090909...
				.636363...
	*/
	int row[] = { 0, 0, 1, 1 };
	int col[] = { 0, 1, 0, 1 };
	float val[] = { 4, 1, 1, 3 };
	SparseMatrix<float> matrix(row, col, val, 4, 2);
	std::vector<float> b;
	b.push_back(1);
	b.push_back(2);

	CGSolver solver(matrix, b, context, 100);
	solver.solve();
	std::vector<float> x = solver.getResult();
	std::cout << "Wiki Result: ";
	for (float f : x){
		std::cout << f << ", ";
	}
	std::cout << std::endl;
}
int cellNumber(int x, int y, int dim){
	if (x < 0){
		x += dim * (std::abs(x / dim) + 1);
	}
	if (y < 0){
		y += dim * (std::abs(y / dim) + 1);
	}
	return x % dim + (y % dim) * dim;
}
void cellPos(int n, int &x, int &y, int dim){
	x = n % dim;
	y = (n - x) / dim;
}
SparseMatrix<float> createInteractionMatrix(int dim){
	std::vector<MatrixElement<float>> elems;
	int nCells = dim * dim;
	for (int i = 0; i < nCells; ++i){
		//In the matrix all diagonal entires are 4 and neighbor cells are -1
		elems.push_back(MatrixElement<float>(i, i, 4));
		int x, y;
		cellPos(i, x, y, dim);
		elems.push_back(MatrixElement<float>(i, cellNumber(x - 1, y, dim), -1));
		elems.push_back(MatrixElement<float>(i, cellNumber(x + 1, y, dim), -1));
		elems.push_back(MatrixElement<float>(i, cellNumber(x, y - 1, dim), -1));
		elems.push_back(MatrixElement<float>(i, cellNumber(x, y + 1, dim), -1));
	}
	return SparseMatrix<float>(elems, dim, true);
}
void testCGSim(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	SparseMatrix<float> matrix = createInteractionMatrix(16);
	std::vector<float> b;
	for (int i = 0; i < 16; ++i){
		b.push_back(i + 1);
	}
	CGSolver solver(matrix, b, context);
	solver.solve();
	std::vector<float> x = solver.getResult();
	for (float f : x){
		std::cout << f << ", ";
	}
	std::cout << std::endl;
}
void testCGStress(int dim){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	SparseMatrix<float> matrix = createInteractionMatrix(dim);
	std::vector<float> b;
	std::srand(std::time(NULL));
	for (int i = 0; i < dim; ++i){
		//Get random values between 150/-150
		b.push_back(static_cast<float>(std::rand()) / RAND_MAX * 300.f - 150.f);
	}
	CGSolver solver(matrix, b, context);
	std::cout << "Getting initial solve values to compare against\n";
	solver.solve();
	std::vector<float> baseline = solver.getResult();

	for (int i = 0; i < 50; ++i){
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		solver.solve();
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::duration duration = end - start;
		//We don't count reading the data off the GPU in the benchmark since in the sim it will
		//stay on the device and be used as the pressure vector
		std::vector<float> x = solver.getResult();
		for (int j = 0; j < baseline.size(); ++j){
			if (std::abs(baseline[j] - x[j]) > 1e-5){
				std::cout << "difference > 1e-5 in x vs. baseline for run: " << i << "\n";
			}
		}
		std::cout << "run " << i << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
			<< "ms\n";
	}
}
void testVelocityDivergence(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	//Test computation of the negative divergence of the velocity field
	cl::Kernel velocityDivergence(program, "velocity_divergence");
	//Velocity fields for a 2x2 MAC grid
	//For these velocity fields we expect
	//0,0: 2
	//1,0: 0
	//0,1: 0
	//1,1: 4
	float vxField[] = {
		1, 0, -1,
		2, 0, -2
	};
	float vyField[] = {
		1, -1,
		0, 0,
		2, -2
	};
	
	cl::Buffer vxBuf = context.buffer(tcl::MEM::READ_ONLY, 6 * sizeof(float), vxField);
	cl::Buffer vyBuf = context.buffer(tcl::MEM::READ_ONLY, 6 * sizeof(float), vyField);
	//Output for the negative divergence at each cell
	cl::Buffer negDiv = context.buffer(tcl::MEM::WRITE_ONLY, 4 * sizeof(float), nullptr);

	velocityDivergence.setArg(0, vxBuf);
	velocityDivergence.setArg(1, vyBuf);
	velocityDivergence.setArg(2, negDiv);

	context.runNDKernel(velocityDivergence, cl::NDRange(2, 2), cl::NullRange, cl::NullRange, false);

	float result[4] = {0};
	context.readData(negDiv, 4 * sizeof(float), result, 0, true);
	for (int i = 0; i < 4; ++i){
		std::cout << "Divergence at " << i % 2 << "," << i / 2
			<< " = " << result[i] << "\n";
	}
	std::cout << std::endl;
}
void testSubtractPressureX(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel subPressX(program, "subtract_pressure_x");

	float vxField[] = {
		0, 0, 0,
		0, 0, 0
	};
	float pressure[] = {
		1, 0,
		2, 0
	};
	//Just use 1 to make it easier to test
	float rho = 1.f, dt = 1.f;

	cl::Buffer vxBuff = context.buffer(tcl::MEM::READ_WRITE, 6 * sizeof(float), vxField);
	cl::Buffer pressBuff = context.buffer(tcl::MEM::READ_ONLY, 4 * sizeof(float), pressure);

	subPressX.setArg(0, rho);
	subPressX.setArg(1, dt);
	subPressX.setArg(2, vxBuff);
	subPressX.setArg(3, pressBuff);

	context.runNDKernel(subPressX, cl::NDRange(3, 2), cl::NullRange, cl::NullRange, false);
	context.readData(vxBuff, 6 * sizeof(float), vxField, 0, true);
	std::cout << "New velocity_x field:\n";
	for (int i = 0; i < 6; ++i){
		if (i != 0 && i % 3 == 0){
			std::cout << "\n";
		}
		std::cout << vxField[i] << " ";
	}
	std::cout << std::endl;
}
void testSubtractPressureY(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel subPressX(program, "subtract_pressure_y");

	float vyField[] = {
		0, 0,
		0, 0,
		0, 0
	};
	float pressure[] = {
		1, 2,
		0, 0
	};
	//Just use 1 to make it easier to test
	float rho = 1.f, dt = 1.f;

	cl::Buffer vxBuff = context.buffer(tcl::MEM::READ_WRITE, 6 * sizeof(float), vyField);
	cl::Buffer pressBuff = context.buffer(tcl::MEM::READ_ONLY, 4 * sizeof(float), pressure);

	subPressX.setArg(0, rho);
	subPressX.setArg(1, dt);
	subPressX.setArg(2, vxBuff);
	subPressX.setArg(3, pressBuff);

	context.runNDKernel(subPressX, cl::NDRange(2, 3), cl::NullRange, cl::NullRange, false);
	context.readData(vxBuff, 6 * sizeof(float), vyField, 0, true);
	std::cout << "New velocity_x field:\n";
	for (int i = 0; i < 6; ++i){
		if (i != 0 && i % 2 == 0){
			std::cout << "\n";
		}
		std::cout << vyField[i] << " ";
	}
	std::cout << std::endl;
}
void testFieldAdvect(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel advectField(program, "advect_field");

	int dim = 4;
	//The MAC grid 'values'
	float grid[] = {
		0, 0, 0, 1,
		0, 0, 0, 2,
		0, 0, 0, 3,
		7, 6, 5, 4,
	};
	//The velocity fields
	float vX[] = {
		1, 0, 0, 0, 0,
		1, 0, 0, 0, 0,
		1, 0, 0, 0, 0,
		1, 0, 0, 0, 0
	};
	float vY[] = {
		1, 1, 1, 1,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	float dt = 1.f;

	cl::Buffer gridA = context.buffer(tcl::MEM::READ_WRITE, dim * dim * sizeof(float), grid);
	cl::Buffer gridB = context.buffer(tcl::MEM::READ_WRITE, dim * dim * sizeof(float), nullptr);
	cl::Buffer vXBuf = context.buffer(tcl::MEM::READ_ONLY, dim * (dim + 1) * sizeof(float), vX);
	cl::Buffer vYBuf = context.buffer(tcl::MEM::READ_ONLY, dim * (dim + 1) * sizeof(float), vY);

	advectField.setArg(0, dt);
	advectField.setArg(1, gridA);
	advectField.setArg(2, gridB);
	advectField.setArg(3, vXBuf);
	advectField.setArg(4, vYBuf);

	context.runNDKernel(advectField, cl::NDRange(dim, dim), cl::NullRange, cl::NullRange, false);

	context.readData(gridB, dim * dim * sizeof(float), grid, 0, true);
	for (int i = 0; i < dim * dim; ++i){
		if (i != 0 && i % dim == 0){
			std::cout << "\n";
		}
		std::cout << std::setw(6) << std::setprecision(3) << grid[i] << " ";
	}
	std::cout << std::endl;
}
void testVXFieldAdvect(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel advectField(program, "advect_vx");

	int dim = 4;
	float vX[] = {
		1, 1, 0, 0, 0,
		1, 0, 0, 0, 1,
		0, 0, 0, 0, 2,
		0, 0, 5, 4, 3
	};
	float vY[] = {
		1, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	float dt = 1.f;

	cl::Buffer vxA = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), vX);
	cl::Buffer vxB = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), nullptr);
	cl::Buffer vyBuf = context.buffer(tcl::MEM::READ_ONLY, dim * (dim + 1) * sizeof(float), vY);

	advectField.setArg(0, dt);
	advectField.setArg(1, vxA);
	advectField.setArg(2, vxB);
	advectField.setArg(3, vyBuf);

	context.runNDKernel(advectField, cl::NDRange(dim + 1, dim), cl::NullRange, cl::NullRange, false);

	context.readData(vxB, dim * (dim + 1) * sizeof(float), vX, 0, false);
	for (int i = 0; i < dim * (dim + 1); ++i){
		if (i != 0 && i % (dim + 1) == 0){
			std::cout << "\n";
		}
		std::cout << std::setw(6) << std::setprecision(3) << vX[i] << " ";
	}
	std::cout << std::endl;
}
void testVYFieldAdvect(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel advectField(program, "advect_vy");

	int dim = 4;
	float vX[] = {
		1, 1, 0, 0, 0,
		1, 0, 0, 0, 1,
		0, 0, 0, 0, 2,
		0, 0, 5, 4, 3
	};
	float vY[] = {
		1, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	float dt = 1.f;

	cl::Buffer vyA = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), vY);
	cl::Buffer vyB = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), nullptr);
	cl::Buffer vxBuf = context.buffer(tcl::MEM::READ_ONLY, dim * (dim + 1) * sizeof(float), vX);

	advectField.setArg(0, dt);
	advectField.setArg(1, vyA);
	advectField.setArg(2, vyB);
	advectField.setArg(3, vxBuf);

	context.runNDKernel(advectField, cl::NDRange(dim, dim + 1), cl::NullRange, cl::NullRange, false);

	context.readData(vyB, dim * (dim + 1) * sizeof(float), vY, 0, false);
	for (int i = 0; i < dim * (dim + 1); ++i){
		if (i != 0 && i % dim == 0){
			std::cout << "\n";
		}
		std::cout << std::setw(6) << std::setprecision(3) << vY[i] << " ";
	}
	std::cout << std::endl;
}
