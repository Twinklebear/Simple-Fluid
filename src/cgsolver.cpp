#include <iostream>
#include <array>
#include "tinycl.h"
#include "sparsematrix.h"
#include "cgsolver.h"

CGSolver::CGSolver(const SparseMatrix<float> &mat, const std::vector<float> &b, 
	tcl::Context &context, int iter, float len)
		: context(context), maxIterations(iter), dimensions(mat.dim), matElems(mat.elements.size()), convergeLength(len)
{
	loadKernels();
	createBuffers(mat, b);
}
void CGSolver::solve(){
	initSolve();

	float rLen = 100;
	int i = 0;
	for (i = 0; i < maxIterations && rLen >= convergeLength; ++i){
		//Compute Ap
		context.runNDKernel(matVecMult, cl::NDRange(dimensions), cl::NullRange, cl::NullRange, false);
		//Compute p dot Ap
		dot.setArg(0, aMultp);
		dot.setArg(1, p);
		dot.setArg(2, apDotp);
		context.runNDKernel(dot, cl::NDRange(dimensions / 2), cl::NullRange, cl::NullRange, false);
		context.runNDKernel(updateAlpha, cl::NDRange(1), cl::NullRange, cl::NullRange, false);
		context.runNDKernel(updateXR, cl::NDRange(dimensions), cl::NullRange, cl::NullRange, false);
			
		//Find new r dot r
		dot.setArg(0, r);
		dot.setArg(1, r);
		dot.setArg(2, rDotr[1]);
		context.runNDKernel(dot, cl::NDRange(dimensions / 2), cl::NullRange, cl::NullRange, false);
		context.runNDKernel(updateDir, cl::NDRange(dimensions), cl::NullRange, cl::NullRange, false);

		//Update old r dot r and find rLen
		context.mQueue.enqueueCopyBuffer(rDotr[1], rDotr[0], 0, 0, sizeof(float));
		context.readData(rDotr[1], sizeof(float), &rLen, 0, true);
		rLen = std::sqrtf(rLen);
	}
	std::cout << "solution took: " << i << " iterations, final residual length: " << rLen << std::endl;
}
void CGSolver::updateB(const std::vector<float> &b){
	bVec = context.buffer(tcl::MEM::READ_ONLY, dimensions * sizeof(float), &b[0]);
}
void CGSolver::updateB(cl::Buffer &b){
	bVec = b;
}
std::vector<float> CGSolver::getResult(){
	std::vector<float> res;
	res.resize(dimensions);
	context.readData(result, dimensions * sizeof(float), &res[0], 0, true);
	return res;
}
cl::Buffer CGSolver::getResultBuffer(){
	return result;
}
void CGSolver::loadKernels(){
	program = context.loadProgram("../res/cg_solver.cl");
	initVects = cl::Kernel(program, "init_vects");
	matVecMult = cl::Kernel(program, "mat_vec_mult");
	dot = cl::Kernel(program, "big_dot");
	updateXR = cl::Kernel(program, "update_xr");
	updateDir = cl::Kernel(program, "update_dir");
	updateAlpha = cl::Kernel(program, "update_alpha");
}
void CGSolver::createBuffers(const SparseMatrix<float> &mat, const std::vector<float> &b){
	std::vector<int> rows, cols;
	std::vector<float> vals;
	rows.resize(mat.elements.size());
	cols.resize(mat.elements.size());
	vals.resize(mat.elements.size());
	mat.getRaw(&rows[0], &cols[0], &vals[0]);

	matrix[MATRIX::ROW] = context.buffer(tcl::MEM::READ_ONLY, rows.size() * sizeof(int), &rows[0]);
	matrix[MATRIX::COL] = context.buffer(tcl::MEM::READ_ONLY, cols.size() * sizeof(int), &cols[0]);
	matrix[MATRIX::VAL] = context.buffer(tcl::MEM::READ_ONLY, vals.size() * sizeof(float), &vals[0]);

	//In the case that we want to upload everything but the b vector, skip creating it
	if (!b.empty()){
		bVec = context.buffer(tcl::MEM::READ_ONLY, dimensions * sizeof(float), &b[0]);
	}
	result = context.buffer(tcl::MEM::READ_WRITE, dimensions * sizeof(float), nullptr);

	//Setup the other buffers we'll need for the computation
	r = context.buffer(tcl::MEM::READ_WRITE, dimensions * sizeof(float), nullptr);
	p = context.buffer(tcl::MEM::READ_WRITE, dimensions * sizeof(float), nullptr);
	aMultp = context.buffer(tcl::MEM::READ_WRITE, dimensions * sizeof(float), nullptr);
	apDotp = context.buffer(tcl::MEM::READ_WRITE, dimensions * sizeof(float), nullptr);
	alpha = context.buffer(tcl::MEM::READ_WRITE, sizeof(float), nullptr);
	for (int i = 0; i < 2; ++i){
		rDotr[i] = context.buffer(tcl::MEM::READ_WRITE, dimensions * sizeof(float), nullptr);
	}
}
void CGSolver::initSolve(){
	initVects.setArg(0, result);
	initVects.setArg(1, r);
	initVects.setArg(2, p);
	initVects.setArg(3, bVec);
	//We can run the intialization kernel while we setup more stuff, so start it now
	context.runNDKernel(initVects, cl::NDRange(dimensions), cl::NullRange, cl::NullRange, false);
	//We can do the same thing for finding the initial r dot r buffer
	dot.setArg(0, r);
	dot.setArg(1, r);
	dot.setArg(2, rDotr[0]);
	context.runNDKernel(dot, cl::NDRange(dimensions / 2), cl::NullRange, cl::NullRange, false);

	matVecMult.setArg(0, sizeof(int), &matElems);
	matVecMult.setArg(1, matrix[MATRIX::ROW]);
	matVecMult.setArg(2, matrix[MATRIX::COL]);
	matVecMult.setArg(3, matrix[MATRIX::VAL]);
	matVecMult.setArg(4, p);
	matVecMult.setArg(5, aMultp);

	updateXR.setArg(0, alpha);
	updateXR.setArg(1, p);
	updateXR.setArg(2, aMultp);
	updateXR.setArg(3, result);
	updateXR.setArg(4, r);

	updateDir.setArg(0, rDotr[1]);
	updateDir.setArg(1, rDotr[0]);
	updateDir.setArg(2, r);
	updateDir.setArg(3, p);

	updateAlpha.setArg(0, rDotr[0]);
	updateAlpha.setArg(1, apDotp);
	updateAlpha.setArg(2, alpha);
}
