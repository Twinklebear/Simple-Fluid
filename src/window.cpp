#include <iostream>
#include <GL/glew.h>

#if defined(_MSC_VER)
#include <SDL.h>
#else
#include <SDL2/SDL.h>
#endif

#include "window.h"

SDL::SDL(int flags){
	if (SDL_Init(flags) != 0)
		std::cout << "Failed to init!" << std::endl;
}
bool SDL::initSubSystem(int flags){
	return (SDL_InitSubSystem(flags) != 0);
}
SDL::~SDL(){
	SDL_Quit();
}

Window::Window(const std::string &title, int width, int height)
	: mWindow(nullptr)
{
	//Setup SDL_GL attributes we want, this must be done before opening window/making context
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	//Turn on multisample
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    //Create our window
	mDim[0] = width;
	mDim[1] = height;
	mWindow = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
        mDim[0], mDim[1], SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
    //Make sure it created ok
    if (mWindow == nullptr){
        std::cout << "Window creation failed" << std::endl;
	}

    //Get the glcontext
    mContext = SDL_GL_CreateContext(mWindow);
	//Make sure GLEW gets running alright
	GLenum glewErr = glewInit();
	if (glewErr != GLEW_OK){
		std::cout << "GLEW init error" << std::endl;
	}

	//Setup some properites for the context
    glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << "\n"
        << "OpenGL Vendor: " << glGetString(GL_VENDOR) << "\n"
        << "OpenGL Renderer: " << glGetString(GL_RENDERER) << "\n"
        << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
}
Window::~Window(){
	close();
}
void Window::close(){
	SDL_GL_DeleteContext(mContext);
	SDL_DestroyWindow(mWindow);
}
void Window::clear(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
void Window::present(){
	glFlush();
    SDL_GL_SwapWindow(mWindow);
}
void Window::getDim(int &width, int &height){
	width = mDim[0];
	height = mDim[1];
}
