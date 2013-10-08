#include <iostream>
#include <array>
#include <GL/glew.h>
#include <SOIL.h>
#include <SDL.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include "util.h"
#include "tinycl.h"
#include "window.h"
#include "sparsematrix.h"
#include "simplefluid.h"

SimpleFluid::SimpleFluid(int dim, Window &win) 
	: context(tcl::DEVICE::GPU, true, false), dim(dim), window(win),
	interactionMat(createInteractionMatrix()), cgSolver(interactionMat, std::vector<float>(), context)
{}
SimpleFluid::~SimpleFluid(){
	glDeleteProgram(quadShader);
	glDeleteVertexArrays(1, &quad[0]);
	glDeleteBuffers(2, &quad[1]);
	glDeleteTextures(2, textures);
}
void SimpleFluid::initSim(){
	initGL();
	initCL();
}
void SimpleFluid::runSim(){
	paintFluid = true;
	GLint texUnif = glGetUniformLocation(quadShader, "tex");
	//For buffers/images that flip the input/output each step we use
	//these vars to pick them, and swap the vars after each step
	int in = 0, out = 1;
	SDL_Event e;
	bool quit = false;
	while (!quit){
		while (SDL_PollEvent(&e)){
			if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)){
				quit = true;
			}
			//Controls: 1-4 will pick brush colors, q will toggle painting off/on
			if (e.type == SDL_KEYDOWN){
				bool updateBrush = false;
				float brush[3];
				switch (e.key.keysym.sym){
				case SDLK_1:
					brush[0] = 1.f;
					brush[1] = 0.f;
					brush[2] = 0.f;
					updateBrush = true;
					break;
				case SDLK_2:
					brush[0] = 0.f;
					brush[1] = 1.f;
					brush[2] = 0.f;
					updateBrush = true;
					break;
				case SDLK_3:
					brush[0] = 0.f;
					brush[1] = 0.f;
					brush[2] = 1.f;
					updateBrush = true;
					break;
				case SDLK_4:
					brush[0] = 1.f;
					brush[1] = 1.f;
					brush[2] = 1.f;
					updateBrush = true;
					break;
				case SDLK_q:
					paintFluid = !paintFluid;
					break;
				default:
					break;
				}
				if (updateBrush){
					context.writeData(brushColor, 3 * sizeof(float), brush);
				}

			}
		}
		//Update all the in/out kernel params
		//Velocity divergence and subtract pressure work on the outputs because
		//those are the output fields from the advection and force application stages
		velocity_divergence.setArg(0, velX[out]);
		velocity_divergence.setArg(1, velY[out]);
		subtract_pressure_x.setArg(2, velX[out]);
		subtract_pressure_y.setArg(2, velY[out]);
		//advect_field would be setup here if it was being used
		advect_vx.setArg(1, velX[in]);
		advect_vx.setArg(2, velX[out]);
		advect_vx.setArg(3, velY[in]);
		advect_vy.setArg(1, velY[in]);
		advect_vy.setArg(2, velY[out]);
		advect_vy.setArg(3, velX[in]);
		advect_img_field.setArg(1, fluid[in]);
		advect_img_field.setArg(2, fluid[out]);
		advect_img_field.setArg(3, velX[in]);
		advect_img_field.setArg(4, velY[in]);
		//We set pixels and apply forces to the outputs of the advection step
		set_pixel.setArg(1, fluid[out]);
		set_pixel.setArg(2, fluid[out]);
		apply_force.setArg(2, velX[out]);
		apply_force.setArg(3, velY[out]);
		
		stepSim(1 / 30.f);

		//Make sure OpenCL is done with our GL Objects
		context.mQueue.finish();
		//Update the texture unit and draw
		glUniform1i(texUnif, out);
		window.clear();
		glDrawElements(GL_TRIANGLES, util::quadElems.size(), GL_UNSIGNED_SHORT, 0);
		window.present();

		SDL_Delay(30);
		std::swap(in, out);
	}
}
void SimpleFluid::initGL(){
	GLint progStatus = util::loadProgram("../res/quad_v.glsl", "../res/quad_f.glsl");
	if (progStatus == -1){
		std::cout << "GL Shader program creation failed" << std::endl;
		throw std::runtime_error("Shader creation failed");
	}
	quadShader = progStatus;
	GLint mvpUnif = glGetUniformLocation(progStatus, "mvp");
	glm::mat4 model = glm::scale(1.5f, 1.5f, 1.5f);
	quadRange[0] = -1.5f;
	quadRange[1] = 1.5f;

	int width, height;
	window.getDim(width, height);
	eyePos = glm::vec3(0.f, 0.f, -5.f);
	view = glm::lookAt(eyePos, glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
	projection = glm::perspective(75.f, static_cast<float>(width) / height, 0.1f, 100.f);
	glm::mat4 mvp = projection * view * model;
	glUseProgram(quadShader);
	glUniformMatrix4fv(mvpUnif, 1, GL_FALSE, glm::value_ptr(mvp));

	glGenVertexArrays(1, &quad[0]);
	glBindVertexArray(quad[0]);
	
	glGenBuffers(2, &quad[1]);
	glBindBuffer(GL_ARRAY_BUFFER, quad[1]);
	glBufferData(GL_ARRAY_BUFFER, util::quadVerts.size() * sizeof(glm::vec3),
		&util::quadVerts[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quad[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, util::quadElems.size() * sizeof(GLushort),
		&util::quadElems[0], GL_STATIC_DRAW);

	//Pos is attrib 0, uv coords are attrib 1
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	//UV values are in the second half of the quad vbo
	size_t offset = util::quadVerts.size() / 2 * sizeof(glm::vec3);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(offset));

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
}
void SimpleFluid::initCL(){
	initCLBuffers();
	initCLKernels();
}
void SimpleFluid::initCLBuffers(){
	//Setup interop images
	for (int i = 0; i < 2; ++i){
		fluid[i] = context.imageGL(tcl::MEM::READ_WRITE, textures[i]);
		clglObjs.push_back(fluid[i]);
	}
	//Setup the velocity buffers
#ifdef CL_VERSION_1_2
	//I think this is correct for filling the buffers with pattern of 0.f
	//but I won't know for sure until I can test on my laptop with OpenCL 1.2
	for (int i = 0; i < 2; ++i){
		velX[i] = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), nullptr);
		velY[i] = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), nullptr);
		context.mQueue.enqueueFillBuffer(velX[i], 0.f, 0, sizeof(float));
		context.mQueue.enqueueFillBuffer(velY[i], 0.f, 0, sizeof(float));
	}
#else
	//is there a way to get new to zero out memory?
	float *zero_vel = static_cast<float*>(std::calloc(dim * (dim + 1), sizeof(float)));
	for (int i = 0; i < 2; ++i){
		velX[i] = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), nullptr);
		velY[i] = context.buffer(tcl::MEM::READ_WRITE, dim * (dim + 1) * sizeof(float), nullptr);
	}
	context.writeData(velX[0], dim * (dim + 1) * sizeof(float), zero_vel);
	//Make sure the entire zero_vel buffer is copied over before freeing it
	context.mQueue.finish();
	std::free(zero_vel);

	context.mQueue.enqueueCopyBuffer(velX[0], velX[1], 0, 0, dim * (dim + 1) * sizeof(float));
	context.mQueue.enqueueCopyBuffer(velX[0], velY[0], 0, 0, dim * (dim + 1) * sizeof(float));
	context.mQueue.enqueueCopyBuffer(velY[0], velY[1], 0, 0, dim * (dim + 1) * sizeof(float));
#endif

	velNegDivergence = context.buffer(tcl::MEM::READ_WRITE, dim * dim * sizeof(float), nullptr);
	cgSolver.updateB(velNegDivergence);

	float color[] = { 1.f, 1.f, 1.f, 1.f };
	int macDim[] = { dim, dim };
	brushColor = context.buffer(tcl::MEM::READ_ONLY, 4 * sizeof(float), color);
	clickForce = context.buffer(tcl::MEM::READ_ONLY, 2 * sizeof(float), nullptr);
	gridDim = context.buffer(tcl::MEM::READ_ONLY, 2 * sizeof(int), macDim);
}
void SimpleFluid::initCLKernels(){
	clProg = context.loadProgram("../res/simple_fluid.cl");
	velocity_divergence = cl::Kernel(clProg, "velocity_divergence");
	subtract_pressure_x = cl::Kernel(clProg, "subtract_pressure_x");
	subtract_pressure_y = cl::Kernel(clProg, "subtract_pressure_y");
	advect_field = cl::Kernel(clProg, "advect_field");
	advect_vx = cl::Kernel(clProg, "advect_vx");
	advect_vy = cl::Kernel(clProg, "advect_vy");
	advect_img_field = cl::Kernel(clProg, "advect_img_field");
	set_pixel = cl::Kernel(clProg, "set_pixel");
	apply_force = cl::Kernel(clProg, "apply_force");

	velocity_divergence.setArg(2, velNegDivergence);
	//Note: Some properties flip in/out buffers each step so those params aren't set here
	//TODO: Configurable rho values, should probably also effect force application
	float rho = 1.f;
	//TODO: Varying timestep?
	float dt = 1.f / 30.f;
	subtract_pressure_x.setArg(0, rho);
	subtract_pressure_x.setArg(1, dt);
	subtract_pressure_x.setArg(3, cgSolver.getResultBuffer());
	
	subtract_pressure_y.setArg(0, rho);
	subtract_pressure_y.setArg(1, dt);
	subtract_pressure_y.setArg(3, cgSolver.getResultBuffer());

	advect_field.setArg(0, dt);
	advect_vx.setArg(0, dt);
	advect_vy.setArg(0, dt);
	advect_img_field.setArg(0, dt);

	set_pixel.setArg(0, brushColor);
	apply_force.setArg(0, dt);
	apply_force.setArg(1, clickForce);
	apply_force.setArg(4, gridDim);
}
void SimpleFluid::stepSim(float dt){
	//Advect
	glFinish();
	context.mQueue.enqueueAcquireGLObjects(&clglObjs);
	//Should the fluid be advected first or the velocity? I think the fluid since
	//advecting the velocity field could break the incompressability we enforced in the Project step
	context.runNDKernel(advect_img_field, cl::NDRange(dim, dim), cl::NullRange, cl::NullRange);
	context.runNDKernel(advect_vx, cl::NDRange(dim + 1, dim), cl::NullRange, cl::NullRange);
	context.runNDKernel(advect_vy, cl::NDRange(dim, dim + 1), cl::NullRange, cl::NullRange);

	//Apply Forces
	//Click on the fluid and apply force. use SDL_GetMouseState to get position and if a button is down
	//then SDL_GetRelativeMouseState for force
	clickFluid();
	context.mQueue.enqueueReleaseGLObjects(&clglObjs);

	//Project
	//TODO: There is some kinda bug in the solver or somewhere in here
	//context.runNDKernel(velocity_divergence, cl::NDRange(dim, dim), cl::NullRange, cl::NullRange);
	//cgSolver.solve();
	//context.runNDKernel(subtract_pressure_x, cl::NDRange(dim + 1, dim), cl::NullRange, cl::NullRange);
	//context.runNDKernel(subtract_pressure_y, cl::NDRange(dim, dim + 1), cl::NullRange, cl::NullRange);
}
void SimpleFluid::clickFluid(){
	//Must call GetRelativeMouseState each frame to update the mouse deltas
	//even if we didn't click, otherwise we get spikes
	int delta[2];
	SDL_GetRelativeMouseState(&delta[0], &delta[1]);
	int pos[2];
	if (SDL_GetMouseState(&pos[0], &pos[1]) & SDL_BUTTON(1)){
		int winDim[2];
		window.getDim(winDim[0], winDim[1]);
		glm::vec4 ray((2.f * pos[0]) / winDim[0] - 1, 1 - (2.f * pos[1]) / winDim[1], -1.f, 0.f);
		ray = glm::inverse(projection) * ray;
		ray.z = -1.f;
		ray.w = 0.f;
		ray = glm::normalize(glm::inverse(view) * ray);

		//Find the ray-plane intersection point
		float ndotr = glm::dot(ray, glm::vec4(0, 0, 1, 0));
		if (std::abs(ndotr) < 0.f){
			return;
		}
		float t = (glm::dot(glm::vec3(0, 0, 0), glm::vec3(0, 0, 1)) - glm::dot(eyePos, glm::vec3(0, 0, 1))) / ndotr;
		glm::vec3 hit = eyePos + glm::vec3(ray) * t;
		
		//See if we hit within the quad
		if (std::abs(hit.x) < quadRange[1] && std::abs(hit.y) < quadRange[1]){
			//The coordinate system being used in the simulation is inverted
			//I suppose I could just rotate the plane?
			float force[] = { -delta[0], -delta[1] };
			context.writeData(clickForce, 2 * sizeof(float), force);

			int hitPixel[] = {
				((hit.x - quadRange[0]) / (quadRange[1] - quadRange[0])) * dim,
				((hit.y - quadRange[0]) / (quadRange[1] - quadRange[0])) * dim
			};
			context.runNDKernel(apply_force, cl::NDRange(1, 1), cl::NullRange, cl::NDRange(hitPixel[0], hitPixel[1]));
			if (paintFluid){
				context.runNDKernel(set_pixel, cl::NDRange(1, 1), cl::NullRange, cl::NDRange(hitPixel[0], hitPixel[1]));
			}
		}
	}
}
SparseMatrix<float> SimpleFluid::createInteractionMatrix(){
	std::vector<MatrixElement<float>> elems;
	int nCells = dim * dim;
	for (int i = 0; i < nCells; ++i){
		//In the matrix all diagonal entires are 4 and neighbor cells are -1
		elems.push_back(MatrixElement<float>(i, i, 4));
		int x, y;
		cellPos(i, x, y);
		elems.push_back(MatrixElement<float>(i, cellNumber(x - 1, y), -1));
		elems.push_back(MatrixElement<float>(i, cellNumber(x + 1, y), -1));
		elems.push_back(MatrixElement<float>(i, cellNumber(x, y - 1), -1));
		elems.push_back(MatrixElement<float>(i, cellNumber(x, y + 1), -1));
	}
	return SparseMatrix<float>(elems, dim, true);
}
int SimpleFluid::cellNumber(int x, int y) const {
	if (x < 0){
		x += dim * (std::abs(x / dim) + 1);
	}
	if (y < 0){
		y += dim * (std::abs(y / dim) + 1);
	}
	return x % dim + (y % dim) * dim;
}
void SimpleFluid::cellPos(int n, int &x, int &y) const {
	x = n % dim;
	y = (n - x) / dim;
}
