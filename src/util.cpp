#include <string>
#include <iostream>
#include <ostream>
#include <fstream>
#include <GL/glew.h>
#if defined(_MSC_VER)
#include <SDL.h>
#else
#include <SDL2/SDL.h>
#endif
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "util.h"

std::string util::readFile(const std::string &file){
	std::string content = "";
	std::ifstream fileIn(file.c_str());
	if (fileIn.is_open()){
		content = std::string(std::istreambuf_iterator<char>(fileIn),
			std::istreambuf_iterator<char>());
	}
	return content;
}
GLint util::loadShader(const std::string &file, GLenum shaderType){
	GLuint shader = glCreateShader(shaderType);
	std::string src = readFile(file);
	const char *csrc = src.c_str();
	glShaderSource(shader, 1, &csrc, 0);
	glCompileShader(shader);
	
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		switch (shaderType){
		case GL_VERTEX_SHADER:
			std::cerr << "Vertex shader: ";
			break;
		case GL_FRAGMENT_SHADER:
			std::cerr << "Fragment shader: ";
			break;
		default:
			std::cerr << "Unknown shader type: ";
		}
		std::cerr << file << " failed to compile. Compilation log:\n";
		GLint len;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
		char *log = new char[len];
		glGetShaderInfoLog(shader, len, 0, log);
		std::cerr << log << "\n";

		delete[] log;
		glDeleteShader(shader);
		return -1;
	}
	return shader;
}
GLint util::loadProgram(const std::string &vertfname, const std::string &fragfname){
	GLint vShader = loadShader(vertfname, GL_VERTEX_SHADER);
	GLint fShader = loadShader(fragfname, GL_FRAGMENT_SHADER);
	if (vShader == -1 || fShader == -1){
		std::cerr << "Program creation failed, a required shader failed to compile\n";
		return -1;
	}
	GLuint program = glCreateProgram();
	glAttachShader(program, vShader);
	glAttachShader(program, fShader);
	glLinkProgram(program);

	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		std::cerr << "Program linking failed. Link log:\n";
		GLint len;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
		char *log = new char[len];
		glGetProgramInfoLog(program, len, 0, log);
		std::cerr << log << "\n";

		delete[] log;
	}
	glDetachShader(program, vShader);
	glDetachShader(program, fShader);
	glDeleteShader(vShader);
	glDeleteShader(fShader);
	if (status == GL_FALSE){
		glDeleteProgram(program);
		return -1;
	}
	return program;
}
bool util::logSDLError(std::ostream &os, const std::string &msg){
	const char *err = SDL_GetError();
	if (err[0] == '\0'){
		return false;
	}
	os << "SDL Error! " << msg << " error: " << err << "\n";
	return true;
}
bool util::logGLError(std::ostream &os, const std::string &msg){
	GLint err = glGetError();
	if (err == GL_NO_ERROR){
		return false;
	}
	os << "OpenGL Error! " << msg << " error: #" << std::hex << err << std::dec
		<< " - " << gluErrorString(err) << "\n";
	return true;
}
void util::logCLError(std::ostream &os, const cl::Error &e, const std::string &msg){
	os << "OpenCL Error! " << msg << " at: " << e.what() 
		<< " error: # " << e.err() << " - " << clErrorString(e.err())
		<< "\n";
}
std::string util::clErrorString(int err){
	switch (err){
	case CL_SUCCESS:
		return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:
		return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:
		return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:
		return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:
		return "CL_OUT_OF_RESOURCES";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:
		return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:
		return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:
		return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:
		return "CL_MAP_FAILURE";
	case CL_INVALID_VALUE:
		return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:
		return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:
		return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:
		return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:
		return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:
		return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:
		return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:
		return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:
		return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:
		return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:
		return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:
		return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:
		return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:
		return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:
		return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:
		return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:
		return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:
		return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_SIZE:
		return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:
		return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:
		return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:
		return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:
		return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:
		return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:
		return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:
		return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:
		return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:
		return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:
		return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:
		return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:
		return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifdef CL_VERSION_1_2
	case CL_INVALID_PROPERTY:
		return "CL_INVALID_PROPERTY";
	case CL_INVALID_IMAGE_DESCRIPTOR:
		return "CL_INVALID_IMAGE_DESCRIPTOR";
	case CL_INVALID_COMPILER_OPTIONS:
		return "CL_INVALID_COMPILER_OPTIONS";
	case CL_INVALID_LINKER_OPTIONS:
		return "CL_INVALID_LINKER_OPTIONS";
	case CL_INVALID_DEVICE_PARTITION_COUNT:
		return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
	default:
		return "Unkown error";
	}
}
