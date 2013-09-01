#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <ostream>

/*
* A namespace to contain various utility functions
*/
namespace util {
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
	* Read the entire contents of a file into a string and return it
	*/
	std::string readFile(const std::string &file);
}

#endif
