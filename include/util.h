#ifndef UTIL_H
#define UTIL_H

#include <array>
#include <string>
#include <ostream>
#include <glm/glm.hpp>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

/*
* A namespace to contain various utility functions
*/
namespace util {
	/*
	* Vertices and element indices for a textured quad
	*/
	const static std::array<glm::vec3, 8> quadVerts = {
		//Vertex positions
		glm::vec3(-1.0, -1.0, 0.0),
		glm::vec3(1.0, -1.0, 0.0),
		glm::vec3(-1.0, 1.0, 0.0),
		glm::vec3(1.0, 1.0, 0.0),
		//UV coords
		glm::vec3(0.0, 0.0, 0.0),
		glm::vec3(1.0, 0.0, 0.0),
		glm::vec3(0.0, 1.0, 0.0),
		glm::vec3(1.0, 1.0, 0.0)
	};
	const static std::array<GLushort, 6> quadElems = {
		0, 1, 2,
		1, 3, 2
	};
	/*
	* Read the entire contents of a file into a string and return it
	*/
	std::string readFile(const std::string &file);
	/*
	* Load a GLSL shader from some file, will return -1 if loading failed
	*/
	GLint loadShader(const std::string &file, GLenum shaderType);
	/*
	* Simple GLSL program loader, just handles a basic vertex + fragment
	* shader program. vertfname and fragfname should be the paths to the shader files
	* will return -1 if loading failed
	*/
	GLint loadProgram(const std::string &vertfname, const std::string &fragfname);
	/*
	* Check if an SDL error occured and log it to the ostream of our choice
	* will return true if an error occured, false if no error
	* the message will be formated: msg error: sdl error \n
	*/
	bool logSDLError(std::ostream &os, const std::string &msg);
	/*
	* Check if a GL error occured and log it to the ostream of our choice
	* will return true if an error occured, false if no error
	* the message will be formated: msg error: gl error \n
	*/
	bool logGLError(std::ostream &os, const std::string &msg);
	/*
	* Log an OpenCL error and translate the error code into the error string
	*/
	void logCLError(std::ostream &os, const cl::Error &e, const std::string &msg);
	/*
	* Translate an OpenCL error code to the error string associated with it
	*/
	std::string clErrorString(int err);
}

#endif
