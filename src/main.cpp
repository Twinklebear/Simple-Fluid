#include <iostream>
#include <iomanip>
#include "simplefluid.h"
#include "tinycl.h"

//Test the velocity divergence kernel
void testVelocityDivergence();
//Test the pressure subtraction to update the velocity field
void testSubtractPressureX();
void testSubtractPressureY();

int main(int argc, char **argv){
	testSubtractPressureX();

    return 0;
}
void testVelocityDivergence(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	//Test computation of the negative divergence of the velocity field
	cl::Kernel velocityDivergence(program, "velocity_divergence");
	//Velocity fields for a 2x2 MAC grid
	//For these velocity fields we expect
	//0,0: 2
	//1,0: 0
	//0,1: 0
	//1,1: 4
	float vxField[] = {
		1, 0, -1,
		2, 0, -2
	};
	float vyField[] = {
		1, -1,
		0, 0,
		2, -2
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
}
void testSubtractPressureX(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel subPressX(program, "subtract_pressure_x");

	float vxField[] = {
		0, 0, 0,
		0, 0, 0
	};
	float pressure[] = {
		1, 0,
		2, 0
	};
	//Just use 1 to make it easier to test
	float rho = 1.f, dt = 1.f;

	cl::Buffer vxBuff = context.buffer(tcl::MEM::READ_WRITE, 6 * sizeof(float), vxField);
	cl::Buffer pressBuff = context.buffer(tcl::MEM::READ_ONLY, 4 * sizeof(float), pressure);

	subPressX.setArg(0, rho);
	subPressX.setArg(1, dt);
	subPressX.setArg(2, vxBuff);
	subPressX.setArg(3, pressBuff);

	context.runNDKernel(subPressX, cl::NDRange(3, 2), cl::NullRange, cl::NullRange, false);
	context.readData(vxBuff, 6 * sizeof(float), vxField, 0, true);
	std::cout << "New velocity_x field:\n";
	for (int i = 0; i < 6; ++i){
		if (i != 0 && i % 3 == 0){
			std::cout << "\n";
		}
		std::cout << vxField[i] << " ";
	}
	std::cout << std::endl;
}
void testSubtractPressureY(){

}
