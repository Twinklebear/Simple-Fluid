#include <iostream>
#include <iomanip>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <SOIL.h>
#include "util.h"
#include "simplefluid.h"
#include "tinycl.h"
#include "window.h"

//Test the velocity divergence kernel
void testVelocityDivergence();
//Test the pressure subtraction to update the velocity field
void testSubtractPressureX();
void testSubtractPressureY();
//Test the field advection kernel
void testFieldAdvect();
//Test the x velocity field advection kernel
void testVXFieldAdvect();
//Test the y velocity field advection kernel
void testVYFieldAdvect();
//Test advection of an image using the velocity fields
void testImgAdvect();

int main(int argc, char **argv){
	testImgAdvect();

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
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel subPressX(program, "subtract_pressure_y");

	float vyField[] = {
		0, 0,
		0, 0,
		0, 0
	};
	float pressure[] = {
		1, 2,
		0, 0
	};
	//Just use 1 to make it easier to test
	float rho = 1.f, dt = 1.f;

	cl::Buffer vxBuff = context.buffer(tcl::MEM::READ_WRITE, 6 * sizeof(float), vyField);
	cl::Buffer pressBuff = context.buffer(tcl::MEM::READ_ONLY, 4 * sizeof(float), pressure);

	subPressX.setArg(0, rho);
	subPressX.setArg(1, dt);
	subPressX.setArg(2, vxBuff);
	subPressX.setArg(3, pressBuff);

	context.runNDKernel(subPressX, cl::NDRange(2, 3), cl::NullRange, cl::NullRange, false);
	context.readData(vxBuff, 6 * sizeof(float), vyField, 0, true);
	std::cout << "New velocity_x field:\n";
	for (int i = 0; i < 6; ++i){
		if (i != 0 && i % 2 == 0){
			std::cout << "\n";
		}
		std::cout << vyField[i] << " ";
	}
	std::cout << std::endl;
}
void testFieldAdvect(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel advectField(program, "advect_field");

	int dim = 4;
	//The MAC grid 'values'
	float grid[] = {
		0, 0, 0, 1,
		0, 0, 0, 2,
		0, 0, 0, 3,
		7, 6, 5, 4,
	};
	//The velocity fields
	float vX[] = {
		1, 0, 0, 0, 0,
		1, 0, 0, 0, 0,
		1, 0, 0, 0, 0,
		1, 0, 0, 0, 0
	};
	float vY[] = {
		1, 1, 1, 1,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	float dt = 1.f;

	cl::Buffer gridA = context.buffer(tcl::MEM::READ_WRITE, dim * dim * sizeof(float), grid);
	cl::Buffer gridB = context.buffer(tcl::MEM::READ_WRITE, dim * dim * sizeof(float), nullptr);
	cl::Buffer vXBuf = context.buffer(tcl::MEM::READ_ONLY, dim * (dim + 1) * sizeof(float), vX);
	cl::Buffer vYBuf = context.buffer(tcl::MEM::READ_ONLY, dim * (dim + 1) * sizeof(float), vY);

	advectField.setArg(0, dt);
	advectField.setArg(1, gridA);
	advectField.setArg(2, gridB);
	advectField.setArg(3, vXBuf);
	advectField.setArg(4, vYBuf);

	context.runNDKernel(advectField, cl::NDRange(dim, dim), cl::NullRange, cl::NullRange, false);

	context.readData(gridB, dim * dim * sizeof(float), grid, 0, true);
	for (int i = 0; i < dim * dim; ++i){
		if (i != 0 && i % dim == 0){
			std::cout << "\n";
		}
		std::cout << std::setw(6) << std::setprecision(3) << grid[i] << " ";
	}
	std::cout << std::endl;
}
void testVXFieldAdvect(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel advectField(program, "advect_vx");

	int dim = 4;
	float vX[] = {
		1, 1, 0, 0, 0,
		1, 0, 0, 0, 1,
		0, 0, 0, 0, 2,
		0, 0, 5, 4, 3
	};
	float vY[] = {
		1, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	float dt = 1.f;

	cl::Buffer vxA = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), vX);
	cl::Buffer vxB = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), nullptr);
	cl::Buffer vyBuf = context.buffer(tcl::MEM::READ_ONLY, dim * (dim + 1) * sizeof(float), vY);

	advectField.setArg(0, dt);
	advectField.setArg(1, vxA);
	advectField.setArg(2, vxB);
	advectField.setArg(3, vyBuf);

	context.runNDKernel(advectField, cl::NDRange(dim + 1, dim), cl::NullRange, cl::NullRange, false);

	context.readData(vxB, dim * (dim + 1) * sizeof(float), vX, 0, false);
	for (int i = 0; i < dim * (dim + 1); ++i){
		if (i != 0 && i % (dim + 1) == 0){
			std::cout << "\n";
		}
		std::cout << std::setw(6) << std::setprecision(3) << vX[i] << " ";
	}
	std::cout << std::endl;
}
void testVYFieldAdvect(){
	tcl::Context context(tcl::DEVICE::GPU, false, false);
	cl::Program program = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel advectField(program, "advect_vy");

	int dim = 4;
	float vX[] = {
		1, 1, 0, 0, 0,
		1, 0, 0, 0, 1,
		0, 0, 0, 0, 2,
		0, 0, 5, 4, 3
	};
	float vY[] = {
		1, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	float dt = 1.f;

	cl::Buffer vyA = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), vY);
	cl::Buffer vyB = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), nullptr);
	cl::Buffer vxBuf = context.buffer(tcl::MEM::READ_ONLY, dim * (dim + 1) * sizeof(float), vX);

	advectField.setArg(0, dt);
	advectField.setArg(1, vyA);
	advectField.setArg(2, vyB);
	advectField.setArg(3, vxBuf);

	context.runNDKernel(advectField, cl::NDRange(dim, dim + 1), cl::NullRange, cl::NullRange, false);

	context.readData(vyB, dim * (dim + 1) * sizeof(float), vY, 0, false);
	for (int i = 0; i < dim * (dim + 1); ++i){
		if (i != 0 && i % dim == 0){
			std::cout << "\n";
		}
		std::cout << std::setw(6) << std::setprecision(3) << vY[i] << " ";
	}
	std::cout << std::endl;
}
void testImgAdvect(){
	SDL sdl(SDL_INIT_EVERYTHING);
	const int winWidth = 640, winHeight = 480;
	Window win("Image Advection Test", winWidth, winHeight);

	//Maybe util::loadX should just throw an error
	GLint progStatus = util::loadProgram("../res/quad_v.glsl", "../res/quad_f.glsl");
	if (progStatus == -1){
		return;
	}
	//In the program position is set to location 0, and uv is set to position 1
	//Explicit uniform location is core in 4.3 so we still need to look up where the mvp is
	GLuint program = progStatus;
	GLint mvpUniform = glGetUniformLocation(program, "mvp");
	GLint texUniform = glGetUniformLocation(program, "tex");

	glm::mat4 model = glm::scale(1.5f, 1.5f, 1.f);
	glm::vec3 camPos = glm::vec3(0.f, 0.f, -5.f);
	glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.f, 0.f, 0.f),
		glm::vec3(0.f, 1.f, 0.f));
	glm::mat4 projection = glm::perspective(80.f, static_cast<float>(winWidth) / winHeight, 0.1f, 100.f);
	glm::mat4 mvp = projection * view * model;
	glUseProgram(program);
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, glm::value_ptr(mvp));

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	GLuint quad[2];
	glGenBuffers(2, quad);
	glBindBuffer(GL_ARRAY_BUFFER, quad[0]);
	glBufferData(GL_ARRAY_BUFFER, util::quadVerts.size() * sizeof(glm::vec3),
		&util::quadVerts[0], GL_STATIC_DRAW);
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quad[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, util::quadElems.size() * sizeof(GLushort),
		&util::quadElems[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	size_t uvOffset = util::quadVerts.size() / 2 * sizeof(glm::vec3);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(uvOffset));

	GLuint textures[2];
	textures[0] = SOIL_load_OGL_texture("../res/img_diag.png", SOIL_LOAD_AUTO,
		SOIL_CREATE_NEW_ID, SOIL_FLAG_INVERT_Y);
	glBindTexture(GL_TEXTURE_2D, textures[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glActiveTexture(GL_TEXTURE1);
	textures[1] = SOIL_load_OGL_texture("../res/img_diag.png", SOIL_LOAD_AUTO,
		SOIL_CREATE_NEW_ID, SOIL_FLAG_INVERT_Y);
	glBindTexture(GL_TEXTURE_2D, textures[1]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	//The images are 16x16
	size_t macDim = 16;

	//Now setup our OpenCL context and buffers
	tcl::Context context(tcl::DEVICE::GPU, true, false);
#ifdef CL_VERSION_1_2
	cl::ImageGL clImg[2];
#else
	cl::Image2DGL clImg[2];
#endif
	std::vector<cl::Memory> clglObjs;
	for (int i = 0; i < 2; ++i){
		clImg[i] = context.imageGL(tcl::MEM::READ_WRITE, textures[i]);
		clglObjs.push_back(clImg[i]);
	}
	const float vX[] = {
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	};
	const float vY[] = {
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	};
	//Block on the second write so we won't be waiting for the write while trying to do the first loop run
	cl::Buffer vXBuf = context.buffer(tcl::MEM::READ_ONLY, macDim * (macDim + 1) * sizeof(float), vX);
	cl::Buffer vYBuf = context.buffer(tcl::MEM::READ_ONLY, macDim * (macDim + 1) * sizeof(float), vY, 0, true);

	float dt = 1.f / 20.f;
	//Use to track which texture is storing the output of the advection
	//and which is being used as input.
	unsigned texIn = 0, texOut = 1;

	cl::Program clProg = context.loadProgram("../res/simple_fluid.cl");
	cl::Kernel kernel(clProg, "advect_img_field");
	kernel.setArg(0, dt);
	kernel.setArg(3, vXBuf);
	kernel.setArg(4, vYBuf);

	//Kernel and buffers for updating clicked pixels
	cl::Kernel setPixel(clProg, "set_pixel");
	//since the plane is square we can just use one range buffer
	float planeRange[] = { -1.5f, 1.5f };
	cl::Buffer planeRangeBuf = context.buffer(tcl::MEM::READ_ONLY, 2 * sizeof(float), planeRange);
	cl::Buffer planeClickPos = context.buffer(tcl::MEM::READ_ONLY, 2 * sizeof(float), nullptr);
	setPixel.setArg(0, planeClickPos);
	setPixel.setArg(1, planeRangeBuf);
	setPixel.setArg(2, planeRangeBuf);


	util::logGLError(std::cerr, "About to enter loop, final error check");

	//Track which run we're on, even or odd so we know which texture to read/write too
	//and which to draw
	bool quit = false;
	while (!quit){
		SDL_Event e;
		while (SDL_PollEvent(&e)){
			if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)){
				quit = true;
			}
			//We want to be able to click on and interact with the fluid
			//Later will switch to click & drag interaction but for now just draw dots on the grid
			if (SDL_GetMouseState(NULL, NULL) & SDL_BUTTON(1)){
				glm::vec4 ray((2.f * e.button.x) / winWidth - 1, 1 - (2.f * e.button.y) / winHeight, -1.f, 0.f);
				ray = glm::inverse(projection) * ray;
				ray.z = -1.f;
				ray.w = 0.f;
				ray = glm::normalize(glm::inverse(view) * ray);

				//Now intersect with a plane. Later put this in a function and make a ray struct
				//The plane normal is facing towards the camera
				float ndotr = glm::dot(ray, glm::vec4(0, 0, 1, 0));
				//if std::abs(ndotr) < 0.f) return
				float t = (glm::dot(glm::vec3(0, 0, 0), glm::vec3(0, 0, 1)) - glm::dot(camPos, glm::vec3(0, 0, 1))) / ndotr;
				glm::vec3 hit = camPos + glm::vec3(ray) * t;
				//The plane is from -1.5 to 1.5 in x & y
				if (std::abs(hit.x) < 1.5 && std::abs(hit.y) < 1.5){
					float hitPos[] = { hit.x, hit.y };
					context.writeData(planeClickPos, 2 * sizeof(float), hitPos);
					setPixel.setArg(3, clImg[texIn]);
					setPixel.setArg(4, clImg[texIn]);
					//Run the kernel to update the hit pixel
					glFinish();
					context.mQueue.enqueueAcquireGLObjects(&clglObjs);
					context.runNDKernel(setPixel, cl::NDRange(1), cl::NullRange, cl::NullRange, false);
					context.mQueue.enqueueReleaseGLObjects(&clglObjs);
					context.mQueue.finish();
				}
			}
		}
		glFinish();
		kernel.setArg(1, clImg[texIn]);
		kernel.setArg(2, clImg[texOut]);
		context.mQueue.enqueueAcquireGLObjects(&clglObjs);
		context.runNDKernel(kernel, cl::NDRange(macDim, macDim), cl::NullRange, cl::NullRange, false);
		context.mQueue.enqueueReleaseGLObjects(&clglObjs);
		context.mQueue.finish();

		glUniform1i(texUniform, texOut);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDrawElements(GL_TRIANGLES, util::quadElems.size(), GL_UNSIGNED_SHORT, 0);
		win.present();

		SDL_Delay(30);
		std::swap(texIn, texOut);
	}
	
	glDeleteTextures(2, textures);
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(2, quad);
	glDeleteProgram(program);
}
