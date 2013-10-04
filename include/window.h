#ifndef WINDOW_H
#define WINDOW_H

#include <string>

#if defined(_MSC_VER)
#include <SDL.h>
#else
#include <SDL2/SDL.h>
#endif

/*
* Small class for managing SDL, destructor will quit SDL
* One instance of this class should be created at the start of
* the program and not destroyed until you're done with SDL
*/
class SDL {
public:
	//Initialize SDL with desired flags
	SDL(int flags);
	//Initialize a desired subsytem
	bool initSubSystem(int flags);
	//Quit SDL
	~SDL();
};

/*
*  Window management class, provides a simple wrapper around
*  the SDL_Window and SDL_Renderer functionalities
*/
class Window {
public:
	/*
	* Create a new window and a corresponding gl context and make it current
	* @param title window title
	* @param width window width
	* @param height window height
	*/
	Window(const std::string &title, int width, int height);
	//Close the window
	~Window();
	//Close the window & delete the context
	void close();
	//Clear the renderer
    void clear();
    //Present the renderer, ie. update screen
    void present();

private:
	SDL_Window *mWindow;
	SDL_GLContext mContext;
};

#endif
