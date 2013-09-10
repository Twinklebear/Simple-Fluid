#include <iostream>
#include "simplefluid.h"

int main(int argc, char **argv){
	SDL sdl(SDL_INIT_EVERYTHING);
	Window win("Simple Fluid", 640, 480);
	
	SimpleFluid fluid(8, win);
	fluid.runTests();

    return 0;
}
