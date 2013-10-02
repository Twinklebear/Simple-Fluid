/*
* Program containing various kernels and functions needed to run a
* simple 2D MAC grid fluid simulation such as reading/writing velocity
* values, updating the fluid and so one
*/
/*
* Compute the index of the element in 1d buffer storing a row-major 2d grid
* x and y will be wrapped if they go out of bounds
*/
int elem_index(float x, float y, int n_row, int n_col){
	if (x < 0){
		x += n_col * (abs((int)x / n_col) + 1);
	}
	if (y < 0){
		y += n_row * (abs((int)y / n_row) + 1);
	}
	return (int)x % n_col + ((int)y % n_row) * n_col;
}
/*
* Compute the x,y grid coordinates of element i in a 1d buffer storing a row-major 2d grid
* with row_len elements per row
*/
int2 grid_pos(int i, int row_len){
	return (int2)(i % row_len, (i - i % row_len) / row_len);
}
/*
* Keep a coordinate in bounds on a grid with range [0, dim] by wrapping it over
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
* Compute the bilinear interpolated value of the field at some point in the field,
* it's assumed that the grid cells are all of equal w/h. n_row and n_col should be
* the number of rows and columns in the field grid.
*/
float bilinear_interpolate(float2 pos, __global float *field, int n_row, int n_col){
	//Wrap coordinates that go beyond the edge case, ie. that wrap the blending square
	//completely to the other side of the grid
	if (pos.x < -1 || pos.x > n_col){
		pos.x = wrap(pos.x, n_col);
	}
	if (pos.y < -1 || pos.y > n_row){
		pos.y = wrap(pos.y, n_row);
	}
	//Information about the positions and indices of the values being blended (w is idx)
	int4 vals[4];
	for (int i = 0; i < 4; ++i){
		vals[i].w = elem_index(pos.x + i % 2, pos.y + i / 2, n_row, n_col);
		vals[i].xy = grid_pos(vals[i].w, n_col);
	}
	//Translate position into a unit square to make the blending calculation easier, but be sure
	//to detect edge cases where the square is split across two sides of the grid, ie. the
	//square is wrapping around
	//*_range.x is the min value, *_range.y is the max value
	float2 x_range, y_range;
	if (pos.x < 0){
		x_range.x = -1.f;
		x_range.y = 0.f;
	}
	else if (pos.x > n_col - 1){
		x_range.x = n_col - 1;
		x_range.y = n_col;
	}
	else {
		x_range.x = vals[0].x;
		x_range.y = vals[1].x;
	}
	if (pos.y < 0){
		y_range.x = -1.f;
		y_range.y = 0.f;
	}
	else if (pos.y > n_row - 1){
		y_range.x = n_row - 1;
		y_range.y = n_row;
	}
	else {
		y_range.x = vals[0].y;
		y_range.y = vals[2].y;
	}
	pos.x = ((pos.x - x_range.x) / (x_range.y - x_range.x));
	pos.y = ((pos.y - y_range.x) / (y_range.y - y_range.x));
	return field[vals[0].w] * (1 - pos.x) * (1 - pos.y) + field[vals[1].w] * pos.x * (1 - pos.y)
		+ field[vals[2].w] * (1 - pos.x) * pos.y + field[vals[3].w] * pos.x * pos.y;
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
/*
* Advect some MAC grid property using the x and y velocity fields over the timestep
* The kernel should be run with dimensions equal to those of the MAC grid
*/
__kernel void advect_field(float dt, __global float *in, __global float *out, 
	__global float *v_x, __global float *v_y)
{
	int2 id = (int2)(get_global_id(0), get_global_id(1));
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	float2 pos = (float2)(id.x, id.y);
	float2 x_pos = (float2)(pos.x + 0.5f, pos.y);
	float2 y_pos = (float2)(pos.x, pos.y + 0.5f);
	//Find the velocity at this point and step back a half time step
	float2 vel = (float2)(bilinear_interpolate(x_pos, v_x, dim.y, dim.x + 1),
		bilinear_interpolate(y_pos, v_y, dim.y + 1, dim.x));

	//Take a half step and find the velocity to take the next
	pos -= 0.5f * dt * vel;
	x_pos = (float2)(pos.x + 0.5f, pos.y);
	y_pos = (float2)(pos.x, pos.y + 0.5f);
	vel = (float2)(bilinear_interpolate(x_pos, v_x, dim.y, dim.x + 1),
		bilinear_interpolate(y_pos, v_y, dim.y + 1, dim.x));
	//Take another step (is this RK2 function correct?)
	pos -= dt * vel;
	//Sample the grid at this pos and write it as the new value for this location
	out[id.x + id.y * dim.x] = bilinear_interpolate(pos, in, dim.y, dim.x);
}
/*
* Advect the MAC grid's x velocity field over the timestep, kernel should be run
* with dimensions equal to the x velocity field dimensions
*/
__kernel void advect_vx(float dt, __global float *v_x, __global float *v_x_out, __global float *v_y){
	int2 id = (int2)(get_global_id(0), get_global_id(1));
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	float2 pos = (float2)(id.x, id.y);
	float2 y_pos = (float2)(pos.x - 0.5f, pos.y + 0.5f);
	float2 vel = (float2)(bilinear_interpolate(pos, v_x, dim.y, dim.x),
		bilinear_interpolate(y_pos, v_y, dim.y + 1, dim.x - 1));

	//Take a half step and find the velocity for the next step
	pos -= 0.5f * dt * vel;
	y_pos = (float2)(pos.x - 0.5f, pos.y + 0.5f);
	vel = (float2)(bilinear_interpolate(pos, v_x, dim.y, dim.x),
		bilinear_interpolate(y_pos, v_y, dim.y + 1, dim.x - 1));
	pos -= dt * vel;
	//Sample the field at this pos and write it out as our new value for the location
	//v_x_out[id.x + id.y * dim.x] = bilinear_interpolate(pos, v_x, dim.y, dim.x);
	//For testing write the top-left corner of the v_x sampling square out
	v_x_out[id.x + id.y * dim.x] = elem_index(pos.x, pos.y, dim.y, dim.x);
}

