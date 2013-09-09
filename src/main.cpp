#include <iostream>
#include <CL/cl.h>
#include "sparsematrix.h"

int main(int argc, char **argv){
    cl_platform_id test;
    cl_uint num;
    cl_uint ok = 1;
    clGetPlatformIDs(ok, &test, &num);
    std::cout << "there are: " << num << " platforms" << std::endl;

	int rows[3] = { 0, 1, 0 };
	int cols[3] = { 0, 0, 1 };
	int vals[3] = { 1, 2, 3 };
	SparseMatrix<int> mat(rows, cols, vals, 3, 2, false);
	std::cout << mat << std::endl;

    return 0;
}
