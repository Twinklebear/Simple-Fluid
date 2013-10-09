#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct sparse_mat_t {
	int *row, *col;
	float *val;
	int n_vals;
} sparse_mat_t;

/*
 * Compute dot product of vectors a & b that are
 * n length vectors
 */
float dot(float *a, float *b, int n);
/*
 * Multiply a vector with a sparse matrix and put the
 * result in c. The matrix and vectors should all be of
 * dimensions n and the matrix contains n_val entries
 */
void sparse_mat_mult(sparse_mat_t *mat,	float *vect, float *res, int n);
/*
 * Solve the sparse linear system where mat is a sparse
 * symmetric positive-definite matrix and is
 * n x n dimensions and row-major and b is a n dimension vector
 * result will be written to x which should also be n dim
 */
void conjugate_gradient(sparse_mat_t *mat, float *b, float *x, int n);
//Test a simple case of the solver (identity matrix)
void solve_identity();
//Solve the simple example shown on wikipedia
void solve_wiki_ex();
//Print a vector of some length, debugging utility
void print_vector(float *v, int n);

int main(int argc, char **argv){
	solve_wiki_ex();

	return 0;
}
float dot(float *a, float *b, int n){
	float sum = 0.f;
	for (int i = 0; i < n; ++i){
		sum += a[i] * b[i];
	}
	return sum;
}
void sparse_mat_mult(sparse_mat_t *mat, float *vect, float *res, int n){
	for (int r = 0; r < n; ++r){
		//Determine the range of indices we'll be working with for this row
		int indices[] = { -1, -1 };
		for (int i = r; i < mat->n_vals; ++i){
			if (mat->row[i] == r && indices[0] == -1){
				indices[0] = i;
			}
			if (mat->row[i] == r + 1 && indices[1] == -1){
				indices[1] = i - 1;
				break;
			}
			if (i == mat->n_vals - 1 && indices[1] == -1){
				indices[1] = i;
			}
		}

		float sum = 0.f;
		for (int i = indices[0]; i <= indices[1]; ++i){
			sum += mat->val[i] * vect[mat->col[i]];
		}
		res[r] = sum;
	}
}
void conjugate_gradient(sparse_mat_t *mat, float *b, float *x, int n){
	float *r = malloc(n * sizeof(float));
	float *p = malloc(n * sizeof(float));
	memcpy(r, b, n * sizeof(float));
	memcpy(p, r, n * sizeof(float));
	memset(x, 0, n * sizeof(float));
	//Location to store Ap result
	float *mat_p = malloc(n * sizeof(float));
	float r_dot_r[] = { 0.f, 0.f };
	//Compute intial r_dot_r_0	
	r_dot_r[0] = dot(r, r, n);

	int step = 0;
	float r_len = 1000.f;
	float converge_len = 1e-5;
	for (; step < 10 && r_len > converge_len; ++step){
		sparse_mat_mult(mat, p, mat_p, n);
		float p_dot_mat_p = dot(p, mat_p, n);
		
		float alpha = r_dot_r[0] / p_dot_mat_p;
		//Compute new x and r
		for (int i = 0; i < n; ++i){
			x[i] += alpha * p[i];
			r[i] -= alpha * mat_p[i];
		}

		r_dot_r[1] = dot(r, r, n);

		//Update p vector
		float beta = r_dot_r[1] / r_dot_r[0];
		for (int i = 0; i < n; ++i){
			p[i] = r[i] + beta * p[i];
		}

		r_len = sqrtf(r_dot_r[1]);
		if (r_len < converge_len){
			printf("Solved!\n");
			break;
		}
		r_dot_r[0] = r_dot_r[1];
	}
	//step + 1 is the iteration count as 0 is the first iteration
	printf("Solution took %d iterations, residual length: %.2f\n", step + 1, r_len);
	free(r);
	free(p);
	free(mat_p);
}
void solve_identity(){
	int row[] = { 0, 1, 2, 3 };
	int col[] = { 0, 1, 2, 3 };
	float val[] = { 1, 1, 1, 1 };
	sparse_mat_t mat = { .row = row, .col = col, .val = val, .n_vals = 4 };

	float b[] = { 1, 2, 3, 4 };
	float x[4];
	conjugate_gradient(&mat, b, x, 4);

	//Since it's the identity matrix x == b
	for (int i = 0; i < 4; ++i){
		if (x[i] != b[i]){
			printf("solve_identity failed! x != b\n");
			return;
		}
	}
	printf("solve_identity passed\n");
}
void solve_wiki_ex(){
	int row[] = { 0, 0, 1, 1 };
	int col[] = { 0, 1, 0, 1 };
	float val[] = { 4, 1, 1, 3 };
	sparse_mat_t mat = { .row = row, .col = col, .val = val, .n_vals = 4 };

	float b[] = { 1, 2 };
	float x[2] = { 0 };
	conjugate_gradient(&mat, b, x, 2);

	printf("Wikipedia example result: [%.5f, %.5f]\n", x[0], x[1]);
}
void print_vector(float *v, int n){
	for (int i = 0; i < n; ++i){
		printf("%5.2f, ", v[i]);
	}
	printf("\n");
}

