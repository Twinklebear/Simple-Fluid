#include <iostream>
#include <CL/cl.h>

int main(int argc, char **argv){
    cl_platform_id test;
    cl_uint num;
    cl_uint ok = 1;
    clGetPlatformIDs(ok, &test, &num);
    std::cout << "there are: " << num << " platforms" << std::endl;
    return 0;
}
