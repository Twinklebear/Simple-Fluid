#include <iostream>
#include <iomanip>
#include "simplefluid.h"
#include "tinycl.h"

int main(int argc, char **argv){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	//Test computation of the negative divergence of the velocity field
	cl::Kernel velocityDivergence(program, "velocity_divergence");
	//Velocity fields for a 2x2 MAC grid

	float vxField[] = {
		0, 0, 0,
		0, 0, 0
	};
	//For this vy field we expect negative divergences of:
	//0,0: 1
	//1,0: -1
	//0,1: 2
	//1,1: -2
	float vyField[] = {
		1, -1,
		0, 0,
		-2, 2
	};
	
	cl::Buffer vxBuf = context.buffer(tcl::MEM::READ_ONLY, 6 * sizeof(float), vxField);
	cl::Buffer vyBuf = context.buffer(tcl::MEM::READ_ONLY, 6 * sizeof(float), vyField);
	//Output for the negative divergence at each cell
	cl::Buffer negDiv = context.buffer(tcl::MEM::WRITE_ONLY, 4 * sizeof(float), nullptr);

	velocityDivergence.setArg(0, vxBuf);
	velocityDivergence.setArg(1, vyBuf);
	velocityDivergence.setArg(2, negDiv);

	context.runNDKernel(velocityDivergence, cl::NDRange(2, 2), cl::NullRange, cl::NullRange, false);

	float result[4] = {0};
	context.readData(negDiv, 4 * sizeof(float), result, 0, true);
	for (int i = 0; i < 4; ++i){
		std::cout << "Divergence at " << i % 2 << "," << i / 2
			<< " = " << result[i] << "\n";
	}
	std::cout << std::endl;

    return 0;
}
