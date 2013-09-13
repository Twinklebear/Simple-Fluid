/*
* Program containing various kernels and functions needed to run a
* simple 2D MAC grid fluid simulation such as reading/writing velocity
* values, updating the fluid and so one
*/
/*
* Compute the index of the element in 1d buffer storing a row-major 2d grid
* x and y will be wrapped if they go out of bounds
*/
int elem_index(int x, int y, int n_row, int n_col){
	if (x < 0){
		x += n_col * (abs(x / n_col) + 1);
	}
	if (y < 0){
		y += n_row * (abs(y / n_row) + 1);
	}
	return x % n_col + (y % n_row) * n_col;
}
/*
* Compute the x,y grid coordinates of element i in a 1d buffer storing a row-major 2d grid
* with row_len elements per row
*/
int2 grid_pos(int i, int row_len){
	return (int2)(i % row_len, (i - i % row_len) / row_len);
}
/*
* Compute the blend weight for the values at point b being interpolated to point a
* this function relies on our grid cells being of uniform size (1x1 in this case)
*/
float blend_weight(float2 a, float2 b){
	float2 weight = (float2)(1 - fabs(a.x - b.x), 1 - fabs(a.y - b.y));
	return weight.x * weight.y;
}
/*
* Interpolate the 4 nearest velocity values in the grid to find the velocity at x,y
* x,y should be the point in grid-space to blend at
* vel_grid is a 1d array representing a 2d grid with row length row_len
*/
float blend_velocity(float2 pos, __global float *vel_grid, int n_row, int n_col){
	float v_tot = 0.f;
	for (int i = 0; i < 4; ++i){
		int idx = elem_index(pos.x + i % 2, pos.y + i / 2, n_row, n_col);
		float2 v_pos = convert_float2(grid_pos(idx, n_col));
		float weight = blend_weight(pos, v_pos);
		v_tot += weight * vel_grid[idx];
	}
	return v_tot;
}
/*
* Kernel to test the velocity blending by sampling the centers of cells
* and outputting the velocities at the cells. Work group should be 2d and have
* dimensions equal to the fluid grid dimensions
* and the positions will be offset for reading from the x velocity grid
* grid_dim is the dimensionality of the square fluid grid
* Y velocities should be the same equation but just offsetting on the y axis
* and incrementing the y dimension instead of x
*/
__kernel void blend_test(__global float *v_x, __global float *v_cells){
	int2 id = (int2)(get_global_id(0), get_global_id(1));
	float2 pos = (float2)(id.x + 0.5f, id.y);
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	// +1 column b/c x velocities have an extra entry in the x axis
	v_cells[id.x + id.y * dim.x] = blend_velocity(pos, v_x, dim.y, dim.x + 1);
}
