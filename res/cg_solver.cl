/*
* Solve a sparse symmetric positive-definite matrix using CG method with various portions
* of the method split into their own kernels to allow for finer grained
* parallelism and not be limited by the devices work group size since 
* global synchronization is now handled by the host
* The matrix should be 2nx2n
* The kernels should be run in this order on the host:
*
* initVects(x, r, p)
* old_r_dot_r = big_dot(r, r)
* while (not done) (iteration condition and/or r length condition)
*	a_mult_p = mat_vec_mult(matrix, p)
*	apDotp = big_dot(a_mult_p, p)
*	alpha = old_r_dot_r / apDotp
*	update_xr(alpha, x, r)
*	r_dot_r = big_dot(r, r)
*	update_dir(r_dot_r, old_r_dot_r, r, p)
*	old_r_dot_r = r_dot_r
*/
/*
* Multiply a vectorf by a sparse matrixf and get the resultant vector.
* The global size should be equal to the number rows in the matrix
*/
__kernel void mat_vec_mult(int n_vals, __global int *rows, __global int *cols, 
	__global float *vals, __global float *vec, __global float *res)
{
	int id = get_global_id(0);
	//We use these to select the range of i to read our row's values from (row # == id)
	int startIdx = -1;
	int endIdx = -1;
	//Determine the indices containing the values for the row this kernel will be working on
	for (int i = id; i < n_vals; ++i){
		if (rows[i] == id && startIdx == -1)
			startIdx = i;
		if (rows[i] == id + 1 && startIdx != -1 && endIdx == -1){
			endIdx = i - 1;
			break;
		}
		if (i == n_vals - 1 && startIdx != -1 && endIdx == -1)
			endIdx = i;
	}
	float sum = 0.0f;
	for (int i = startIdx; i <= endIdx; ++i)
			sum += vals[i] * vec[cols[i]];

	res[id] = sum;
}
/*
* This kernel will perform a big dot product on arbitrary 2n size vectors
* the global size should be equal to n / 2
* note that the sum is computed by the first kernel which isn't so great
* I'd like to write a reduction kernel that will do this though
* for now the out vector should be size n / 2 but only the first
* value is needed by the host as that's the dot result
*/
__kernel void big_dot(__global float2 *a, __global float2 *b, __global float *out){
	int id = get_global_id(0);
	out[id] = dot(a[id], b[id]);

	float sum = 0.0f;
	if (id == 0){
		for (int i = 0; i < get_global_size(0); ++i)
			sum += out[i];

		out[id] = sum;
	}
}
/*
* Kernel to initialize x, r and p concurrently
* x will be set to 0, r and p will be set to b
*/
__kernel void init_vects(__global float *x, __global float *r, __global float *p, 
	__global float *b)
{
	int id = get_global_id(0);
	x[id] = 0.0f;
	r[id] = b[id];
	p[id] = b[id];
}
/*
* Update the value of alpha, we do it on the GPU to avoid slow read/writes 
* back and forth to the host
*/
__kernel void update_alpha(__global float *old_r_dot_r, __global float *apDotp, __global float *alpha){
	*alpha = (*old_r_dot_r) / (*apDotp);
}
/*
* Update the x and r guesses
*/
__kernel void update_xr(__global float *alpha, __global float *p, __global float *a_mult_p,
	__global float *x, __global float *r)
{
	int id = get_global_id(0);
	x[id] += (*alpha) * p[id];
	r[id] -= (*alpha) * a_mult_p[id];
}
/*
* Update the CG search direction
*/
__kernel void update_dir(__global float *r_dot_r, __global float *old_r_dot_r, 
	__global float *r, __global float *p)
{
	int id = get_global_id(0);
	p[id] = r[id] + (*r_dot_r) / (*old_r_dot_r) * p[id];
}
