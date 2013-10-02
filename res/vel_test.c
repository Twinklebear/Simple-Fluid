#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*
* Compute the index of the element in 1d buffer storing a row-major 2d grid
* x and y will be wrapped if they go out of bounds
*/
int elem_index(float x, float y, int n_row, int n_col){
	if (x < 0){
		x += n_col * (abs(x / n_col) + 1);
	}
	if (y < 0){
		y += n_row * (abs(y / n_row) + 1);
	}
	return (int)x % n_col + ((int)y % n_row) * n_col;
}
/*
 * Keep a coordinate in bounds on a grid with range [0, dim] by wrapping it
 */
float wrap(float a, int dim){
	if (a < 0){
		int offset = dim * (abs((int)a) / dim + 1);
		return a + offset;
	}
	else if (a > dim){
		int offset = dim * ((int)a / dim);
		return a - offset;
	}
	return a;
}
/*
* Compute the x,y grid coordinates of element i in a 1d buffer storing a row-major 2d grid
* with row_len elements per row
*/
void grid_pos(int i, int row_len, int *x, int *y){
	*x = i % row_len;
	*y = (i - *x) / row_len;
}
//A blending value
typedef struct blend_val_t {
	int idx, x, y;
} blend_val_t;
/*
 * Compute the velocity at x,y using bilinear interpolation of the values in the velocity field
 * f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
 * x,y will be adjusted to their equivalent location in the cell treating it as a unit square
 * n_row and n_col should be the dimensions of the velocity grid
 * I'm pretty sure this is correct, should implement a kernel now
 */
float bilinear_interpolate(float x, float y, float *v, int n_row, int n_col){
	//Handle wrapping coordinates that go beyond the edge case, ie. completely to the other side of the grid
	if (x < -1 || x > n_col){
		x = wrap(x, n_col);
	}
	if (y < -1 || y > n_row){
		y = wrap(y, n_row);
	}
	//Find the low corner of the unit square
	blend_val_t vals[4];
	printf("Blending at (%.2f, %.2f)\n", x, y);
	for (int i = 0; i < 4; ++i){
		vals[i].idx = elem_index(x + i % 2, y + i / 2, n_row, n_col);
		grid_pos(vals[i].idx, n_col, &vals[i].x, &vals[i].y);
	}
	//Translate positions into the unit square we're blending in
	//need to use val = ((old_val - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
	float x_range[2], y_range[2];
	if (x < 0){
		x_range[0] = -1.f;
		x_range[1] = 0.f;
	}
	//Wrapping over the right side
	else if (x > n_col - 1){
		x_range[0] = n_col - 1;
		x_range[1] = n_col;
	}
	else {
		x_range[0] = vals[0].x;
		x_range[1] = vals[1].x;
	}
	//Wrap y
	if (y < 0){
		y_range[0] = -1.f;
		y_range[1] = 0.f;
	}
	else if (y > n_row - 1){
		y_range[0] = n_row - 1;
		y_range[1] = n_row;
	}
	else {
		y_range[0] = vals[0].y;
		y_range[1] = vals[2].y;
	}
	printf("x range: [%.2f, %.2f]\ny range: [%.2f, %.2f]\n", 
		x_range[0], x_range[1], y_range[0], y_range[1]);	
	
	//Scale the x/y values into the unit range, we leave off the * (new_max - new_min) + new_min
	//because it's just * (1 - 0) + 0
	x = ((x - x_range[0]) / (x_range[1] - x_range[0]));
	y = ((y - y_range[0]) / (y_range[1] - y_range[0]));
	printf("Translated position: (%.2f, %.2f)\n", x, y);
	return v[vals[0].idx] * (1 - x) * (1 - y) + v[vals[1].idx] * x * (1 - y)
		+ v[vals[2].idx] * (1 - x) * y + v[vals[3].idx] * x * y;
}
int main(int argc, char **argv){
	int dim = 4;
	float v_x[] = {
		1, 0, 2, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1,
		1, 0, 0, 0, 0
	};
	float v_y[] = {
		1, 0, 2, 0,
		0, 0, 0, 1,
		0, 0, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 0
	};
	float x = -1.5, y = 1.5;
	//Expecting to wrap x to 3.5
	printf("testing wrap case @ (%.2f, %.2f) in the x grid, expect x wrap to 3.5\n", x, y);
	float v = bilinear_interpolate(x, y, v_x, dim, dim + 1);
	printf("interpolated v: %.2f\n", v);

	//Expecting cast to wrap x to 2.5
	x = -2.5 - dim - 1;
	printf("\ntesting wrap case @ (%.2f, %.2f) in the x grid expect x wrap to 2.5\n", x, y);
	v = bilinear_interpolate(x, y, v_x, dim, dim + 1);
	printf("interpolated v: %.2f\n", v);

	//Expecting x to wrap to 0.5
	x = 5.5;
	y = 0;
	printf("\ntesting wrap case @ (%.2f, %.2f) in x grid expect x wrap to 0.5\n", x, y);
	v = bilinear_interpolate(x, y, v_x, dim, dim + 1);
	printf("ninterpolated v: %.2f\n", v);

	//Expecting x to wrap to 1.5
	x = 6.5 + dim + 1;
	printf("\ntesting wrap case @ (%.2f, %.2f) in the x grid expect x wrap to 1.5\n", x, y);
	v = bilinear_interpolate(x, y, v_x, dim, dim + 1);
	printf("interpolated v: %.2f\n", v);

	/*
	for (int i = 0; i < dim; ++i){
		for (int j = 0; j < dim + 1; ++j){
			printf("x grid pos: (%d, %d)\n", j, i);
			float v = bilinear_interpolate(j - 0.5f, i + 0.5f, v_y, dim + 1, dim);
			printf("interpolated v: %.2f\n", v);
			v_x[j + i * (dim + 1)] = v;
		}
	}
	for (int i = 0; i < dim * (dim + 1); ++i){
		if (i != 0 && i % (dim + 1) == 0){
			printf("\n");
		}
		printf("%.2f ", v_x[i]);
	}
	printf("\n");
	*/
	return 0;
}
