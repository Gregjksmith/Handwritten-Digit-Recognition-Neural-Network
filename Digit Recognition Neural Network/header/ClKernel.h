#pragma once

#include <fstream>
#include "ClContext.h"
#include <stdio.h>
#include <vector>

#define MAX_SOURCE_SIZE 10000

/*
class ClKernel
Base Kernel class. Contains kernel handles and functions used to enqueue kernels and add arguments.
Loads the kernel source directory 'kernelSource' and builds the source. 

contructor :
@param const char* kernel source. 
@param ClContext* context.
*/
class ClKernel
{
public:
	ClKernel(const char* kernelSource, ClContext* context);
	virtual ~ClKernel();
	size_t totalWorkItems;
protected:
	/*
	int createKernel
	creates a kernel object of 'kernelName'. 'kerneName' must be located in the source.

	@param const char* kernelName
	@return int, index of the kernel created.
	*/
	int createKernel(const char* kernelName);
	
	/*
	void addKernelArg 
	adds an argument to the kernel in the kernel vector, indexed by kernelIndex.

	@param size_t kernelIndex: Index of the kernel in the kernel vector.
	@param int argId : argument index
	@param unsigned int bufferSize
	@param void* buffer.
	*/
	void addKernelArg(size_t kernelIndex, int argId, unsigned int bufferSize, void* buffer);
	
	/*
	void enqueueKernel
	enqueues and runs the specified kernel with 'totalWorkItems' work items.

	@param size_t kernelIndex.
	*/
	void enqueueKernel(size_t kernelIndex);

	cl_program __program;
	std::vector<cl_kernel> __kernel;
	ClContext* __context;
	std::vector<cl_mem> __memoryObjects;
private:
	size_t __globalWorkSize;
	size_t __localWorkSize;

};