#include "../header/ClKernel.h"

ClKernel::ClKernel(const char* kernelSource, ClContext* context)
{
	this->__context = context;
	FILE *fp;
	char *source_str;
	size_t source_size;

	fopen_s(&fp, kernelSource, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}

	fseek(fp, 0, SEEK_END);
	long fSize = ftell(fp);
	rewind(fp);

	source_str = (char*)malloc(fSize);
	source_size = fread(source_str, 1, fSize, fp);
	fclose(fp);
	cl_int ret;

	__program = NULL;
	__program = clCreateProgramWithSource(context->context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
	switch (ret)
	{
		case CL_INVALID_CONTEXT:
		{
			printf("ERROR, context is invalid.\n");
			break;
		}
		case CL_INVALID_VALUE:
		{
			printf("ERROR, invalid value.\n");
			break;
		}
		case CL_OUT_OF_HOST_MEMORY:
		{
			printf("ERROR, out of host memory. \n");
			break;
		}
		case CL_SUCCESS:
		{
			printf("Success, program created. \n");
		}
	}

	ret = clBuildProgram(__program, 1, &context->deviceId, NULL, NULL, NULL);
	switch (ret)
	{
		case CL_INVALID_PROGRAM:
		{
			printf("ERROR, program is invalid. \n");
			break;
		}
		case CL_INVALID_VALUE:
		{
			printf("ERROR, invalid value.\n");
			break;
		}
		case CL_INVALID_DEVICE:
		{
			printf("ERROR, invalid device.\n");
			break;
		}
		case CL_INVALID_BINARY:
		{
			printf("ERROR, invalid binary program.\n");
			break;
		}
		case CL_INVALID_BUILD_OPTIONS:
		{
			printf("ERROR, invalid build options.\n");
			break;
		}
		case CL_INVALID_OPERATION:
		{
			printf("ERROR, invalid build operation.\n");
			break;
		}
		case CL_COMPILER_NOT_AVAILABLE:
		{
			printf("ERROR, compiler not available.\n");
			break;
		}
		case CL_OUT_OF_HOST_MEMORY:
		{
			printf("ERROR, out of host memory.\n");
			break;
		}
		case CL_BUILD_PROGRAM_FAILURE:
		{
			printf("ERROR, build failed.\n");
			break;
		}
		case CL_SUCCESS:
		{
			printf("Success, build successful.\n");
			break;
		}

	}

	size_t buildLogSize;
	clGetProgramBuildInfo(__program, context->deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
	char* buildLog = new char[buildLogSize];
	clGetProgramBuildInfo(__program, context->deviceId, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);

	printf(buildLog);
	printf("\n");
	delete buildLog;

	free(source_str);

	__globalWorkSize = 0;
	__localWorkSize = 0;
}
ClKernel::~ClKernel()
{
	for (int i = 0; i < __kernel.size(); i++)
	{
		clReleaseKernel(__kernel[i]);
	}
	__kernel.clear();

	clReleaseProgram(__program);
	for (int i = 0; i < __memoryObjects.size(); i++)
	{
		clReleaseMemObject(__memoryObjects[i]);
	}
	__memoryObjects.clear();
}


int ClKernel::createKernel(const char* kernelName)
{
	cl_int ret;
	cl_kernel k = clCreateKernel(__program, kernelName, &ret);
	int kernelPos = -1;

	switch (ret)
	{
	case CL_INVALID_PROGRAM:
		printf("ERROR, program is not a valid program object. \n");
		break;

	case CL_INVALID_PROGRAM_EXECUTABLE:
		printf("ERROR, there is no successfully built executable for program. \n");
		break;

	case CL_INVALID_KERNEL_NAME:
		printf("ERROR, 'kernelName' is not found in the program. \n");
		break;

	case CL_INVALID_KERNEL_DEFINITION:
		printf("ERROR, invalid kernel definition. \n");
		break;

	case CL_INVALID_VALUE:
		printf("ERROR, 'kernelName' is NULL. \n");
		break;

	case CL_OUT_OF_HOST_MEMORY:
		printf("ERROR, out of host memory. \n");
		break;

	case CL_SUCCESS:
		printf("Success, kernel created. \n");
		__kernel.push_back(k);
		kernelPos = __kernel.size();
		break;
	}

	char kernelInfo[1000];
	size_t kernelInfoSize;

	clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 1000, kernelInfo, &kernelInfoSize);
	printf("Kernel function name: ");
	printf(kernelInfo);
	printf("\n");

	cl_int kernelIntParam;
	clGetKernelInfo(k, CL_KERNEL_NUM_ARGS, 1, &kernelIntParam, &kernelInfoSize);
	printf("Number of kernel parameters: %i \n", kernelIntParam);

	clGetKernelInfo(k, CL_KERNEL_REFERENCE_COUNT, 1, &kernelIntParam, &kernelInfoSize);
	printf("Kernel reference count: %i \n", kernelIntParam);

	return kernelPos;
}

void ClKernel::addKernelArg(size_t kernelIndex, int argId, unsigned int bufferSize, void* buffer)
{
	if (__kernel.size() <= kernelIndex)
	{
		printf("ERROR, kernel out of index. \n");
		return;
	}
	cl_int ret = clSetKernelArg(__kernel[kernelIndex], argId, bufferSize, buffer);

	switch (ret)
	{
	case CL_INVALID_KERNEL:
		printf("ERROR, invalid kernel object. \n");
		break;

	case CL_INVALID_ARG_INDEX:
		printf("ERROR, 'argId' is not a valid argument index. \n");
		break;

	case CL_INVALID_ARG_VALUE:
		printf("ERROR, 'bufer' is NULL. \n");
		break;

	case CL_INVALID_MEM_OBJECT:
		printf("ERROR, memory buffer is not a valid memory object. \n");
		break;

	case CL_INVALID_SAMPLER:
		printf("ERROR, sampler buffer is not a valid sampler object. \n");
		break;

	case CL_INVALID_ARG_SIZE:
		printf("ERROR, invalid argument size. \n");
		break;

	case CL_SUCCESS:
		//printf("Success, kernel argument added. \n");
		break;
	
	}
}

void ClKernel::enqueueKernel(size_t kernelIndex)
{
	//compute the number of iterations, offset, global work items and local work items

	if (__kernel.size() <= kernelIndex)
	{
		printf("ERROR, kernel out of index. \n");
		return;
	}

	size_t returnSize;
	size_t maxWorkItemSize[3];
	maxWorkItemSize[0] = 0;
	maxWorkItemSize[1] = 0;
	maxWorkItemSize[2] = 0;
	clGetDeviceInfo(__context->deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(cl_int), &maxWorkItemSize, &returnSize);

	__localWorkSize = 1;
	size_t workItemsPerIteration = maxWorkItemSize[0];
	size_t iterations = (totalWorkItems / workItemsPerIteration) + 1;
	size_t offset = 0;

	size_t* workItemIteration = new size_t[iterations];
	int i;
	for (i = 0; i < iterations - 1; i++)
	{
		workItemIteration[i] = workItemsPerIteration;
	}
	workItemIteration[i] = totalWorkItems % workItemsPerIteration;

	for (int i = 0; i < iterations; i++)
	{
		__globalWorkSize = workItemIteration[i];

		cl_int ret = clEnqueueNDRangeKernel(__context->commandQueue, __kernel[kernelIndex], 1, &offset, &__globalWorkSize, &__localWorkSize, 0, NULL, NULL);
		switch (ret)
		{
		case CL_INVALID_PROGRAM_EXECUTABLE:
			printf("ERROR, no successfully built program executable available for device. \n");
			break;

		case CL_INVALID_COMMAND_QUEUE:
			printf("ERROR, invalid command queue. \n");
			break;

		case CL_INVALID_KERNEL:
			printf("ERROR, invalid kernel. \n");
			break;

		case CL_INVALID_CONTEXT:
			printf("ERROR, invalid context. \n");
			break;

		case CL_INVALID_KERNEL_ARGS:
			printf("ERROR, the kernel arguments have not been specified. \n");
			break;

		case CL_INVALID_WORK_DIMENSION:
			printf("ERROR, invalid work dimensions. \n");
			break;

		case CL_INVALID_GLOBAL_WORK_SIZE:
			printf("ERROR, invalid work size. \n");
			break;

		case CL_INVALID_GLOBAL_OFFSET:
			printf("ERROR, invalid global offset. \n");
			break;

		case CL_INVALID_WORK_GROUP_SIZE:
			printf("ERROR, invalid work group size. \n");
			break;

		case CL_INVALID_WORK_ITEM_SIZE:
			printf("ERROR, invalid work item size. \n");
			break;

		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			printf("ERROR, misaligned sub buffer offset. \n");
			break;


		case CL_INVALID_IMAGE_SIZE:
			printf("ERROR, invalid image size. \n");
			break;

		case CL_OUT_OF_RESOURCES:
			printf("ERROR, out of resources. \n");
			break;

		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			printf("ERROR, memory object allocation failure. \n");
			break;

		case CL_INVALID_EVENT_WAIT_LIST:
			printf("ERROR, invalid wait list. \n");
			break;

		case CL_OUT_OF_HOST_MEMORY:
			printf("ERROR, out of host memory. \n");
			break;

		case CL_SUCCESS:
			//printf("Success, kernel enqueued. \n");
			break;
		}

		clFinish(__context->commandQueue);
		offset = offset + __globalWorkSize;
	}

	delete workItemIteration;
}