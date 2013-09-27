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
* Compute the negative divergence of the velocity field at each cell
* One kernel should be run for each cell and as a 2d work group
* the output buffer (neg_div) should be n_cells in length and row-major
* delta_x is assumed to be one for now
*/
__kernel void velocity_divergence(__global float *v_x, __global float *v_y, __global float *neg_div){
	int2 id = (int2)(get_global_id(0), get_global_id(1));
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	//Find the x velocity difference
	int low = elem_index(id.x, id.y, dim.y, dim.x + 1);
	int hi = elem_index(id.x + 1, id.y, dim.y, dim.x + 1);
	float divergence = v_x[hi] - v_x[low];
	//Now for y
	low = elem_index(id.x, id.y, dim.y + 1, dim.x);
	hi = elem_index(id.x, id.y + 1, dim.y + 1, dim.x);
	divergence += v_y[hi] - v_y[low];
	neg_div[id.x + id.y * dim.x] = -divergence;
}
/*
* Subtract the pressure gradient off of the x velocity field
* Kernel should be run in 2d workgroup with dimensions equal to the velocity grid dimensions
* the cell size is assumed to be 1 (ie. delta_x = 1)
*/
__kernel void subtract_pressure_x(float rho, float dt, __global float *v, __global float *p){
	int2 id = (int2)(get_global_id(0), get_global_id(1));
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	//Find the low and high pressure value indices
	int low = elem_index(id.x - 1, id.y, dim.y, dim.x - 1);
	int hi = elem_index(id.x, id.y, dim.y, dim.x - 1);
	v[id.x + id.y * dim.x] -= (dt / rho) * (p[hi] - p[low]);
}
/*
* Subtract the pressure gradient off of the y velocity field
* Kernel should be run in a 2d workgroup with dimensions equal to the velocity grid dimensions
* the cell size is assumed to be 1 (ie. delta_x = 1)
*/
__kernel void subtract_pressure_y(float rho, float dt, __global float *v, __global float *p){
	int2 id = (int2)(get_global_id(0), get_global_id(1));
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	//Find the low and high pressure value indices
	int low = elem_index(id.x, id.y - 1, dim.y - 1, dim.x);
	int hi = elem_index(id.x, id.y, dim.y - 1, dim.x);
	v[id.x + id.y * dim.x] -= (dt / rho) * (p[hi] - p[low]);
}
//TODO: Write image and field advection kernels
/*
* Advect some MAC grid property using the x and y velocity fields over the desired timestep,
* kernel should be run with dimensions equal to those of the MAC grid
* The 
*/
__kernel void advect_field(float dt, __global float *in, __global float *out, 
	__global float *v_x, __global float *v_y)
{
	int2 id = (int2)(get_global_id(0), get_global_id(1));
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	float2 x_pos = (float2)(id.x + 0.5f, id.y);
	float2 y_pos = (float2)(id.x, id.y + 0.5f);
	//Find the velocity at this point and step back a half time step
	float2 vel = (float2)(blend_velocity(x_pos, v_x, dim.y, dim.x + 1),
		blend_velocity(y_pos, v_y, dim.y + 1, dim.x));
	//For testing just write the blended velocity at this point, change to test each dim
	out[id.x + id.y * dim.x] = vel.x;
}

