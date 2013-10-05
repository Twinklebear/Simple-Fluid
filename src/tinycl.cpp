#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <GL/glew.h>
#include <CL/cl.hpp>
#include "util.h"
#include "tinycl.h"

tcl::Context::Context(DEVICE dev, bool interop, bool profile){
	if (interop){
		selectInteropDevice(dev, profile);
	}
	else {
		selectDevice(dev, profile);
	}
}
cl::Program tcl::Context::loadProgram(const std::string &file){
	cl::Program prog;
	try {
		std::string content = util::readFile(file);
		cl::Program::Sources src(1, std::make_pair(content.c_str(), content.size()));
		prog = cl::Program(mContext, src);
		prog.build(mDevices);
		return prog;
	}
	catch (const cl::Error &e){
		logCLError(e, "Context::loadProgram");
		if (e.err() == CL_BUILD_PROGRAM_FAILURE){
			std::cout << "Building program failed, error log:\n"
				<< prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(mDevices.at(0))
				<< "\n";
		}
		throw e;
	}
}
cl::Buffer tcl::Context::buffer(MEM mem, size_t size, const void *data, size_t offset, bool blocking,
	const std::vector<cl::Event> *depends, cl::Event *notify)
{
	try {
		cl::Buffer buf(mContext, static_cast<cl_mem_flags>(mem), size);
		if (data != nullptr){
			mQueue.enqueueWriteBuffer(buf, blocking, offset, size, data, depends, notify);
		}
		return buf;
	}
	catch (const cl::Error &e){
		logCLError(e, "Context::buffer");
		throw e;
	}
}
cl::BufferGL tcl::Context::bufferGL(MEM mem, GLuint buf){
	try {
		return cl::BufferGL(mContext, static_cast<cl_mem_flags>(mem), buf);
	}
	catch (const cl::Error &e){
		logCLError(e, "Context::bufferGL");
		throw e;
	}
}
#ifdef CL_VERSION_1_2
cl::ImageGL tcl::Context::imageGL(MEM mem, GLuint tex){
	try {
		return cl::ImageGL(mContext, static_cast<cl_mem_flags>(mem), 0, tex);
	}
	catch (const cl::Error &e){
		logCLError(e, "Context::imageGL");
		throw e;
	}
}

#else
cl::Image2DGL tcl::Context::imageGL(MEM mem, GLuint tex){
	try {
		return cl::Image2DGL(mContext, static_cast<cl_mem_flags>(mem), GL_TEXTURE_2D, 0, tex);
	}
	catch (const cl::Error &e){
		logCLError(e, "Context::imageGL");
		throw e;
	}
}

#endif
void tcl::Context::writeData(cl::Buffer &buf, size_t size, const void *data, size_t offset, bool blocking,
	const std::vector<cl::Event> *depends, cl::Event *notify)
{
	try {
		mQueue.enqueueWriteBuffer(buf, blocking, offset, size, data, depends, notify);
	}
	catch (const cl::Error &e){
		logCLError(e, "Context::writeData to buffer");
		throw e;
	}
}
void tcl::Context::readData(const cl::Buffer &buf, size_t size, void *data, size_t offset,
	bool blocking, const std::vector<cl::Event> *depends, cl::Event *notify)
{
	try {
		mQueue.enqueueReadBuffer(buf, blocking, offset, size, data, depends, notify);
	}
	catch (const cl::Error &e){
		logCLError(e, "Context::readData from buffer");
		throw e;
	}
}
void tcl::Context::runNDKernel(cl::Kernel &kernel, cl::NDRange global, cl::NDRange local,
	cl::NDRange offset, bool blocking, const std::vector<cl::Event> *depends,
	cl::Event *notify)
{
	try {
		mQueue.enqueueNDRangeKernel(kernel, offset, global, local, depends, notify);
	}
	catch (const cl::Error &e){
		logCLError(e, "Context::runNDKernel");
		throw e;
	}
}
void tcl::Context::selectDevice(DEVICE dev, bool profile){
	try {
		cl::Platform::get(&mPlatforms);
		//Find the first device that is the type we want
		for (int i = 0; mDevices.empty(); ++i){
			try {
				mPlatforms.at(i).getDevices(static_cast<cl_device_type>(dev), &mDevices);
			}
			catch (const cl::Error &e){
				if (e.err() == CL_DEVICE_NOT_FOUND){
					continue;
				}
				else {
					throw e;
				}
			}
		}
		//Still being kind of lazy and assuming that only one device is on each platform,
		//or at least that the first device is the one we want
		std::cout << "Device info--\n" << "Name: " << mDevices.at(0).getInfo<CL_DEVICE_NAME>()
			<< "\nVendor: " << mDevices.at(0).getInfo<CL_DEVICE_VENDOR>() 
			<< "\nDriver Version: " << mDevices.at(0).getInfo<CL_DRIVER_VERSION>() 
			<< "\nDevice Profile: " << mDevices.at(0).getInfo<CL_DEVICE_PROFILE>() 
			<< "\nDevice Version: " << mDevices.at(0).getInfo<CL_DEVICE_VERSION>()
			<< "\nMax Work Group Size: " << mDevices.at(0).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
			<< std::endl;
		mContext = cl::Context(mDevices);
		if (profile){
			mQueue = cl::CommandQueue(mContext, mDevices.at(0), CL_QUEUE_PROFILING_ENABLE);
		}
		else {
			mQueue = cl::CommandQueue(mContext, mDevices.at(0));
		}
	}
	catch (const cl::Error &e){
		logCLError(e, "Context::selectDevice");
		throw e;
	}
}
void tcl::Context::selectInteropDevice(DEVICE dev, bool profile){
	try {
		//We assume only the first device and platform will be used
		//This is after all a lazy implementation
		cl::Platform::get(&mPlatforms);
		//Find the first device that is the type we want
		for (int i = 0; mDevices.empty(); ++i){
			try {
				mPlatforms.at(i).getDevices(static_cast<cl_device_type>(dev), &mDevices);
			}
			catch (const cl::Error &e){
				if (e.err() == CL_DEVICE_NOT_FOUND){
					continue;
				}
				else {
					throw e;
				}
			}
		}
		//Some different stuff for linux/mac should be used for these properties
		cl_context_properties properties[] = {
			CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
			CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
			CL_CONTEXT_PLATFORM, (cl_context_properties)(mPlatforms[0])(),
			0
		};
		mContext = cl::Context(mDevices, properties);
		//Grab the OpenGL device
		mDevices = mContext.getInfo<CL_CONTEXT_DEVICES>();
		if (profile)
			mQueue = cl::CommandQueue(mContext, mDevices.at(0), CL_QUEUE_PROFILING_ENABLE);
		else
			mQueue = cl::CommandQueue(mContext, mDevices.at(0));

		std::cout << "OpenCL Interop Device Info:" 
			<< "\nName: " << mDevices.at(0).getInfo<CL_DEVICE_NAME>()
			<< "\nVendor: " << mDevices.at(0).getInfo<CL_DEVICE_VENDOR>() 
			<< "\nDriver Version: " << mDevices.at(0).getInfo<CL_DRIVER_VERSION>() 
			<< "\nDevice Profile: " << mDevices.at(0).getInfo<CL_DEVICE_PROFILE>() 
			<< "\nDevice Version: " << mDevices.at(0).getInfo<CL_DEVICE_VERSION>()
			<< "\nMax Work Group Size: " << mDevices.at(0).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
			<< std::endl;
	}
	catch (const cl::Error &e){
		logCLError(e, "Context::selectInteropDevice");
		throw e;
	}
}
void tcl::Context::logCLError(const cl::Error &e, const std::string &msg) const {
	std::cout << msg << " error: " << e.what() << ", code: " << e.err() << "\n";
}
