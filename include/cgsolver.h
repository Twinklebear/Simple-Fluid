#ifndef CGSOLVER_H
#define CGSOLVER_h

#include <array>
#include "tinycl.h"
#include "sparsematrix.h"

/*
* An OpenCL Conjuage Gradient solver, will perform
* a CG solve on the system passed and results can
* be retrieved in the form of a vector read from the 
* context or by getting the buffer containing the result
* The solver has some design decisions influenced by how it'll be
* used as the SimpleFluid solver
*/
class CGSolver {
public:
	/*
	* Give the solver the linear system to solve for x: Ax = b and the OpenCL context to
	* use for the computation. The matrix should be square and have equal dimensionality to the b vector
	* although an empty b vector is also valid if you want to upload everything but not solve yet
	* You can also specify the max iteration count (default 1000) and the length
	* to accept for convergence (default 0.01f)
	*/
	CGSolver(const SparseMatrix<float> &mat, const std::vector<float> &b, 
		tcl::Context &context, int iter = 1000, float len = 0.01f);
	/*
	* Run the solver until we converge or hit the max number of iterations
	*/
	void solve();
	/*
	* Load up a new b vector
	*/
	void updateB(const std::vector<float> &b);
	/*
	* Pass an existing CL buffer to be used as the b vector
	*/
	void updateB(cl::Buffer &b);
	/*
	* Get the result as a vector. This will read the result vector off the device and return it
	*/
	std::vector<float> getResult();
	/*
	* Get the memory buffer on the device containing the result, this will be a buffer
	* with matrix.dim floats
	*/
	cl::Buffer getResultBuffer();

private:
	/*
	* Load the program and the kernels
	*/
	void loadKernels();
	/*
	* Create and write the matrix buffers and other buffers needed for the computation
	* since we'll be running the CG solver many times it's faster to allocate the various
	* buffers needed for the compute once
	*/
	void createBuffers(const SparseMatrix<float> &mat, const std::vector<float> &b);
	/*
	* Initialize the unchanging arguments for the various kernels and perform
	* some initial calculations we need for the solve such as setting up initial vectors
	*/
	void initSolve();

private:
	//Meaningful names for the buffers in the matrix buffer
	enum MATRIX { ROW, COL, VAL };

	tcl::Context &context;
	int maxIterations, dimensions, matElems;
	float convergeLength;
	//The sparse matrix buffers
	std::array<cl::Buffer, 3> matrix;
	//Buffers for vectors and calculation data
	cl::Buffer bVec, result, r, p, aMultp, apDotp, alpha;
	std::array<cl::Buffer, 2> rDotr;
	//The program containing the various kernels
	cl::Program program;
	//The kernels to be used in running the solve
	cl::Kernel initVects, matVecMult, dot, updateXR, updateDir, updateAlpha;
};

#endif
