/*
* Kernels to manage running the Conjuage Gradient method
* also includes some math kernels that are needed during the
* method such as sparse matrix * vector kernel and a n-dim
* dot product kernel
*
* An Overview of the Conjugate Gradient algorithm here.
* The algorithm is split up at synchronization points to avoid
* being limited by max local work group sizes
* while (not_done)
*	find r_dot_r_k using big_dot and sum_partial
*	find Ap using sparse_mat_vec_mult
*	find pAp using big_dot and sum_partial
*	find x_k+1 & r_k+1 using update_xr
*	find r_dot_r_k+1 using big_dot and sum_partial
*	find p_k+1 using update_p
*/
/*
* Multiply a row in a sparse matrix and a vector. row, col and val should
* be the sparse matrix in row-major order and contain n_vals.
* The matrix should be n x n and the vectors vect and res should be n long
* where n is the global size. Each kernel will work on row id, where
* id is the kernel's global id
*/
__kernel void sparse_mat_vec_mult(int n_vals, __global int *row, __global int *col, 
	__global float *val, __global float *vect, __global float *res)
{
	int id = get_global_id(0);
	//Determine the indices we'll be working with for this row
	int2 indices = (int2)(-1, -1);
	//It's safe to assume there's at least one element per row
	//since we'll only work with fluid matrices or other matrices
	//with at least one item per row
	for (int i = 0; i < n_vals; ++i){
		if (row[i] == id && indices.x == -1){
			indices.x = i;
		}
		if (row[i] == id + 1 && indices.y == -1){
			indices.y = i - 1;
			break;
		}
		if (i == n_vals - 1 && indices.y == -1){
			indices.y = i;
		}
	}
	//Dot the sparse row with the vector
	float sum = 0.f;
	for (int i = indices.x; i <= indices.y; ++i){
		sum += val[i] * vect[col[i]];
	}
	res[id] = sum;
}
/*
* Dot two large vectors together with length n, kernel should be run
* with global size of n. Note that the partial is returned and must be
* summed up seperately. This is done to reduce dependencies between kernels
* ie. no need to wait for everyone to finish their multiplication
*/
__kernel void big_dot(__global float *a, __global float *b, __global float *partial){
	int id = get_global_id(0);
	partial[id] = a[id] * b[id];
}
/*
* Sum up the partial from the big_dot output. Only one kernel should be run,
* with n being the number of elements in the vector. The sum will be written
* to partial[0]
* I really should look up reduction kernel methods
*/
__kernel void sum_partial(__global float *partial, int n){
	float sum = 0.f;
	for (int i = 0; i < n; ++i){
		sum += partial[i];
	}
	partial[0] = sum;
}
/*
* Find x_k+1 and r_k+1. Kernel should be run with global size
* equal to the # of elements in the vectors (should be same dim)
* r_dot_r is a float[2] contining r_dot_r_k @ 0
* p_dot_mat_p is the result of pAp
*/
__kernel void update_xr(__global float *r_dot_r, __global float *p_dot_mat_p,
	__global float *p, __global float *mat_p, __global float *x, __global float *r)
{
	int id = get_global_id(0);
	float alpha = r_dot_r[0] / *p_dot_mat_p;
	x[id] += alpha * p[id];
	r[id] -= alpha * mat_p[id];
}
/*
* Find p_k+1. Kernel should be run with global size equal to the # of
* elements in the vectors (should be same dim)
* r_dot_r is a float[2] containing { r_dot_r_k, r_dot_r_k+1 }
*/
__kernel void update_p(__global float *r_dot_r, __global float *r, __global float *p){
	int id = get_global_id(0);
	float beta = r_dot_r[1] / r_dot_r[0];
	p[id] = r[id] + beta * p[id];
}

