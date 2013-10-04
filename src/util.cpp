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

#include "util.h"

bool util::logSDLError(std::ostream &os, const std::string &msg){
	const char *err = SDL_GetError();
	if (err[0] == '\0'){
		return false;
	}
	os << msg << " error: " << err << "\n";
	return true;
}
bool util::logGLError(std::ostream &os, const std::string &msg){
	GLint err = glGetError();
	if (err == GL_NO_ERROR){
		return false;
	}
	os << msg << " error: #" << std::hex << err << std::dec
		<< " - " << gluErrorString(err) << "\n";
	return true;
}
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
