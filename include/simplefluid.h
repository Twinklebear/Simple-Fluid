#ifndef SIMPLEFLUID_H
#define SIMPLEFLUID_H

#include <array>
#include <glm/glm.hpp>
#include "tinycl.h"
#include "window.h"
#include "sparsematrix.h"
#include "cgsolver.h"

/*
* Handles running a simple 2d MAC grid fluid simulation
*/
class SimpleFluid {
public:
	/*
	* Create the simulator, specifying the dimensions for the simulation
	* grid and the window to draw to
	*/
	SimpleFluid(int dim, Window &win);
	//Clean up the OpenGL objects and other stuff
	~SimpleFluid();
	/*
	* Initialize the fluid simulation. This will upload the quad,
	* shaders, textures, kernels and buffers
	*/
	void initSim();
	/*
	* Run the simulation
	*/
	void runSim();

private:
	/*
	* Set up the OpenGL parts of the simulation, upload the quad, shaders
	* and textures
	*/
	void initGL();
	/*
	* Set up the OpenCL parts of the simulation, set up the kernels and
	* buffers. This must be done after initGL as it requires the textures
	* we'll be using as ImageGL's have been created
	*/
	void initCL();
	/*
	* Setup the OpenCL buffers
	*/
	void initCLBuffers();
	/*
	* Setup the OpenCL kernels, should be done after setting up buffers
	*/
	void initCLKernels();
	/*
	* Step the simulation forward over dt
	*/
	void stepSim(float dt);
	/*
	* For painting/pushing the fluid. Check if the mouse is clicked and
	* then paint the cells below it and apply forces base on the mouse motion
	* Note: CL-GL interop object must be acquired by CL prior to calling this
	* function so that the images can be worked with
	*/
	void clickFluid();
	/*
	* Generate the cell-cell interaction matrix for this simulation
	* where diagonal entries are 4 and neighbor cells are -1
	*/
	SparseMatrix<float> createInteractionMatrix();
	/*
	* Compute the cell number of a cell at the x,y coordinates
	*/
	int cellNumber(int x, int y) const;
	/*
	* Compute the x & y position of some cell in the grid, where
	* n is the absolute cell number (ie. [0, nCells])
	*/
	void cellPos(int n, int &x, int &y) const;

private:
	int dim;
	Window &window;
	SparseMatrix<float> interactionMat;
	//OpenCL components of the sim
	tcl::Context context;
	CGSolver cgSolver;
	cl::Program clProg;
	//Other kernels we'll need (names match kernel names in simple_fluid.cl)
	cl::Kernel velocity_divergence, subtract_pressure_x, subtract_pressure_y,
		advect_field, advect_vx, advect_vy, advect_img_field, set_pixel,
		apply_force;
	//velBuf[0] is v_x, 1 is v_y
	cl::Buffer velX[2], velY[2], velNegDivergence, brushColor, clickForce, gridDim;
#ifdef CL_VERSION_1_2
	cl::ImageGL fluid[2];
#else
	cl::Image2DGL fluid[2];
#endif
	std::vector<cl::Memory> clglObjs;
	//OpenGL components of the sim
	GLuint quadShader;
	//Quad VAO, VBO, EBO
	GLuint quad[3];
	GLuint textures[2];
	//The x/y range spanned by the quad
	//TODO: Should instead make a plane object that knows this stuff
	//and also manages the vao, vbo and ebo
	float quadRange[2];
	glm::vec3 eyePos;
	glm::mat4 view, projection;
	//For controlling if we want to paint on the fluid or not
	bool paintFluid;
};

#endif
