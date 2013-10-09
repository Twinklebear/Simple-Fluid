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
	* to accept for convergence (default 1e-5)
	*/
	CGSolver(const SparseMatrix<float> &mat, const std::vector<float> &b, 
		tcl::Context &context, int iter = 1000, float convergeLen = 1e-5);
	/*
	* Run the solver until we converge or hit the max number of iterations
	*/
	void solve();
	/*
	* Load up a new b vector
	*/
	void updateB(const std::vector<float> &bVec);
	/*
	* Pass an existing CL buffer to be used as the b vector
	*/
	void updateB(cl::Buffer &bBuf);
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
	void createBuffers(const SparseMatrix<float> &mat, const std::vector<float> &bVec);
	/*
	* Initialize unchanging parameters to kernels. To be called after createBuffers
	*/
	void initKernelArgs();
	/*
	* Initialize the unchanging arguments for the various kernels and perform
	* some initial calculations we need for the solve such as setting up initial vectors
	*/
	void initSolve();

private:
	//Meaningful names for the buffers in the matrix buffer
	enum MATRIX { ROW, COL, VAL };

	tcl::Context &context;
	int maxIterations, dimensions, matNVals;
	float convergeLen;
	//The sparse matrix buffers
	std::array<cl::Buffer, 3> matrix;
	//Buffers for vectors and calculation data
	//matP = Ap and pMatp = pAp
	cl::Buffer x, r, p, b, matP, pMatp, rDotr, dotPartial;
	//The program containing the various kernels
	cl::Program cgProgram;
	//The kernels to be used in running the solve
	//Kernel names here match the names in cg_kernels.cl to make it clearer who's who
	cl::Kernel sparse_mat_vec_mult, big_dot, sum_partial, update_xr, update_p;
};

#endif
