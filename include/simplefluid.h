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
	/*
	* Run some various tests on the solver as I need to test different
	* features
	*/
	void runTests();
	/*
	* Test the velocity field
	*/
	void testVelocityField();

private:
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
	tcl::Context context;
	int dim;
	Window &window;
	SparseMatrix<float> interactionMat;
	CGSolver cgSolver;
	//The quad to be used for drawing the fluid texture
	const static std::array<glm::vec3, 8> quad;
	//The quad's element buffer
	const static std::array<unsigned short, 6> quadElems;
};

#endif
