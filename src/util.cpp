#include <string>
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
