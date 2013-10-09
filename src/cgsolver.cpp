#include <iostream>
#include <array>
#include "tinycl.h"
#include "sparsematrix.h"
#include "cgsolver.h"

//Helper for debugging, print an array's values
void printArray(float *arr, int n){
	for (int i = 0; i < n; ++i){
		std::cout << arr[i] << ", ";
	}
	std::cout << "\n";
}

CGSolver::CGSolver(const SparseMatrix<float> &mat, const std::vector<float> &b, 
	tcl::Context &context, int iter, float convergeLen)
		: context(context), maxIterations(iter), dimensions(mat.dim), convergeLen(convergeLen),
		matNVals(mat.elements.size())
{
	loadKernels();
	createBuffers(mat, b);
	initKernelArgs();
}
void CGSolver::solve(){
	initSolve();

	//Compute initial r_dot_r_0
	big_dot.setArg(0, r);
	big_dot.setArg(1, r);
	context.runNDKernel(big_dot, cl::NDRange(dimensions), cl::NullRange, cl::NullRange);
	context.runNDKernel(sum_partial, cl::NDRange(1), cl::NullRange, cl::NullRange);
	context.mQueue.enqueueCopyBuffer(dotPartial, rDotr, 0, 0, sizeof(float));

	float rLen = 1000.f;
	int i = 0;
	for (i = 0; i < maxIterations && rLen > convergeLen; ++i){
		//find matP = Ap
		context.runNDKernel(sparse_mat_vec_mult, cl::NDRange(dimensions), cl::NullRange, cl::NullRange);

		//find pMatp = p dot Ap
		big_dot.setArg(0, p);
		big_dot.setArg(1, matP);
		context.runNDKernel(big_dot, cl::NDRange(dimensions), cl::NullRange, cl::NullRange);
		context.runNDKernel(sum_partial, cl::NDRange(1), cl::NullRange, cl::NullRange);
		context.mQueue.enqueueCopyBuffer(dotPartial, pMatp, 0, 0, sizeof(float));

		//find x_k+1 and r_k+1
		context.runNDKernel(update_xr, cl::NDRange(dimensions), cl::NullRange, cl::NullRange);

		//find r_dot_r_k+1
		big_dot.setArg(0, r);
		big_dot.setArg(1, r);
		context.runNDKernel(big_dot, cl::NDRange(dimensions), cl::NullRange, cl::NullRange);
		context.runNDKernel(sum_partial, cl::NDRange(1), cl::NullRange, cl::NullRange);
		context.mQueue.enqueueCopyBuffer(dotPartial, rDotr, 0, sizeof(float), sizeof(float));

		//find matP = Ap
		context.runNDKernel(sparse_mat_vec_mult, cl::NDRange(dimensions), cl::NullRange, cl::NullRange);

		//find p_k+1
		context.runNDKernel(update_p, cl::NDRange(dimensions), cl::NullRange, cl::NullRange);

		//copy r_dot_r_k+1 over to r_dot_r_k for next step
		context.mQueue.enqueueCopyBuffer(rDotr, rDotr, sizeof(float), 0, sizeof(float));

		//Read back residual length
		context.readData(rDotr, sizeof(float), &rLen, sizeof(float), true);
		rLen = std::sqrt(rLen);
	}
	std::cout << "solution took: " << i << " iterations, final residual length: " << rLen << std::endl;
}
void CGSolver::updateB(const std::vector<float> &bVec){
	b = context.buffer(tcl::MEM::READ_ONLY, dimensions * sizeof(float), &bVec[0]);
}
void CGSolver::updateB(cl::Buffer &bBuf){
	b = bBuf;
}
std::vector<float> CGSolver::getResult(){
	std::vector<float> res;
	res.resize(dimensions);
	float *xBuf = static_cast<float*>(context.mQueue.enqueueMapBuffer(x, CL_TRUE, CL_MAP_READ, 0, dimensions * sizeof(float)));
	std::memcpy(&res[0], xBuf, dimensions * sizeof(float));
	context.mQueue.enqueueUnmapMemObject(x, xBuf);
	return res;
}
cl::Buffer CGSolver::getResultBuffer(){
	return x;
}
void CGSolver::loadKernels(){
	cgProgram = context.loadProgram("../res/cg_kernels.cl");
	sparse_mat_vec_mult = cl::Kernel(cgProgram, "sparse_mat_vec_mult");
	big_dot = cl::Kernel(cgProgram, "big_dot");
	sum_partial = cl::Kernel(cgProgram, "sum_partial");
	update_xr = cl::Kernel(cgProgram, "update_xr");
	update_p = cl::Kernel(cgProgram, "update_p");
}
void CGSolver::createBuffers(const SparseMatrix<float> &mat, const std::vector<float> &bVec){
	matrix[MATRIX::ROW] = context.buffer(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		matNVals * sizeof(int), nullptr);
	matrix[MATRIX::COL] = context.buffer(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		matNVals * sizeof(int), nullptr);
	matrix[MATRIX::VAL] = context.buffer(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		matNVals * sizeof(float), nullptr);
	//Map the buffers and write the matrix over
	int *rows = static_cast<int*>(context.mQueue.enqueueMapBuffer(matrix[MATRIX::ROW], CL_FALSE,
		CL_MAP_WRITE, 0, matNVals * sizeof(int)));
	int *cols = static_cast<int*>(context.mQueue.enqueueMapBuffer(matrix[MATRIX::COL], CL_FALSE,
		CL_MAP_WRITE, 0, matNVals * sizeof(int)));
	//Block on the final map so that we can now start writing
	float *vals = static_cast<float*>(context.mQueue.enqueueMapBuffer(matrix[MATRIX::VAL], CL_TRUE,
		CL_MAP_WRITE, 0, matNVals * sizeof(float)));

	mat.getRaw(rows, cols, vals);
	
	context.mQueue.enqueueUnmapMemObject(matrix[MATRIX::ROW], rows);
	context.mQueue.enqueueUnmapMemObject(matrix[MATRIX::COL], cols);
	context.mQueue.enqueueUnmapMemObject(matrix[MATRIX::VAL], vals);

	//In the case that we want to upload everything but the b vector
	if (!bVec.empty()){
		b = context.buffer(CL_MEM_READ_ONLY, dimensions * sizeof(float), &bVec[0]);
	}
	x = context.buffer(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, dimensions * sizeof(float), nullptr);
	r = context.buffer(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, dimensions * sizeof(float), nullptr);
	p = context.buffer(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, dimensions * sizeof(float), nullptr);

	matP = context.buffer(CL_MEM_READ_WRITE, dimensions * sizeof(float), nullptr);
	pMatp = context.buffer(CL_MEM_READ_WRITE, dimensions * sizeof(float), nullptr);
	rDotr = context.buffer(CL_MEM_READ_WRITE, 2 * sizeof(float), nullptr);
	dotPartial = context.buffer(CL_MEM_READ_WRITE, dimensions * sizeof(float), nullptr);
}
void CGSolver::initKernelArgs(){
	sparse_mat_vec_mult.setArg(0, matNVals);
	for (int i = 0; i < 3; ++i){
		sparse_mat_vec_mult.setArg(i + 1, matrix[i]);
	}
	sparse_mat_vec_mult.setArg(4, p);
	sparse_mat_vec_mult.setArg(5, matP);

	big_dot.setArg(2, dotPartial);
	sum_partial.setArg(0, dotPartial);
	sum_partial.setArg(1, dimensions);

	update_xr.setArg(0, rDotr);
	update_xr.setArg(1, pMatp);
	update_xr.setArg(2, p);
	update_xr.setArg(3, matP);
	update_xr.setArg(4, x);
	update_xr.setArg(5, r);

	update_p.setArg(0, rDotr);
	update_p.setArg(1, r);
	update_p.setArg(2, p);
}
void CGSolver::initSolve(){
	context.mQueue.enqueueCopyBuffer(b, r, 0, 0, dimensions * sizeof(float));
	context.mQueue.enqueueCopyBuffer(b, p, 0, 0, dimensions * sizeof(float));
#ifdef CL_VERSION_1_2
	//Use FillBuffer to fill X with 0's but not have to transfer between host/device
	//cl::CommandQueue::enqueueFillBuffer(buffer, pattern, offset, size, events, event);
#else
	float *xBuf = static_cast<float*>(context.mQueue.enqueueMapBuffer(x, CL_TRUE, CL_MAP_WRITE, 0, dimensions * sizeof(float)));
	std::memset(xBuf, 0, dimensions * sizeof(float));
	context.mQueue.enqueueUnmapMemObject(x, xBuf);
#endif
}
