#include <iostream>
#include <iomanip>
#include "simplefluid.h"
#include "tinycl.h"

int main(int argc, char **argv){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel blend_test(program, "blend_test");

	float velocity[6] = { 
		1, 2, 3, 
		4, 5, 6 
	};
	cl::Buffer velBuf = context.buffer(tcl::MEM::READ_ONLY, 6 * sizeof(float), velocity);
	cl::Buffer cellBuff = context.buffer(tcl::MEM::WRITE_ONLY, 4 * sizeof(float), nullptr);

	blend_test.setArg(0, velBuf);
	blend_test.setArg(1, cellBuff);
	context.runNDKernel(blend_test, cl::NDRange(2, 2), cl::NullRange, cl::NullRange, false);

	//Read out and see if the cell velocities have been blended in properly
	float cells[4] = { 0 };
	context.readData(cellBuff, 4 * sizeof(float), cells, 0, true);
	for (int i = 0; i < 4; ++i){
		if (i != 0 && i % 2 == 0){
			std::cout << "\n";
		}
		std::cout << std::setw(6) 
			<< std::setprecision(4) 
			<< std::left << cells[i];
	}
	std::cout << std::endl;

    return 0;
}
