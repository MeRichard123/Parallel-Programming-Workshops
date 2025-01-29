#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>

int main(int argc, char** argv)
{
	// Get Information about Platforms and Devices

	cl_platform_id platform_id;
	clGetPlatformIDs(1, &platform_id, NULL);
	char platformName[128];
	clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL); 

	cl_device_id device_id;
	clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	char deviceName[128];
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);

	std::cout << "Current Device is: " << deviceName << std::endl;
	
	std::cout << "Current Platform is: " << platformName << std::endl;


	// Create a Context
	// - Contexts manage objects in the OpenCL Runtime:
	// - Objects lile Command Queues, Memory, Program, Kernel
	
	cl::Device device(device_id);
	cl::Context context = cl::Context({ device });


	// Create a Command Queue
	// - This is how we push commands onto the device
	// - These are called Streams in Cuda
	cl::CommandQueue queue(context, device);


	// Next we define the source of code which will run on the device
	// This is will store the Kernel
	cl::Program::Sources sources;

	// Allocating Some Memory - _h stands for host (it is on the host)
	int SIZE = 10;
	int A_h[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int B_h[] = { 11, 12, 13, 15, 16, 17, 18, 19, 20 };
	// We need to allocate a memory buffer on the device itself (_d is device)
	// A bugger can be of several types:
	// - CL_MEM_READ_ONLY
	// - CL_MEM_WRITE_ONLY
	cl::Buffer A_d(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE);
	cl::Buffer B_d(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE);	
	cl::Buffer C_d(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE);	

	// Writing into the Device Memory
	// - we can execute commands on a device 
	// - we can initialise vectors A_d using values from A_h
	queue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(int) * SIZE, A_h);
	queue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(int) * SIZE, B_h);


	// Building the Kernel 
	// - A Kernel must return void
	// - global: means pointing to global memory
	std::ifstream file("kernels.cl");
	std::string kernel_code((std::istreambuf_iterator<char>(file)),
			        (std::istreambuf_iterator<char>()));
	
	// Add the Kernel to the Sources
	sources.push_back({ kernel_code.c_str(), kernel_code.length() });
	// Then we create a program which links the OpenCL code to the context
	cl::Program program(context, sources);

	// Building the Code
	if (program.build({ device }) != CL_SUCCESS) 
	{
		std::cout << "Error Building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}

	
	// Create a Kernel from the program
	cl::Kernel simple_add = cl::Kernel(program, "simple_add");
	// Add all the data as arguments to the kernel
	simple_add.setArg(0, A_d);
	simple_add.setArg(1, B_d);
	simple_add.setArg(2, C_d);
	
	// Execute the Kernel by Enqueueing it 
	// - We want to run the simple_add kernel
	// - Global Offset: start kernel from 0 (NullRange)
	// - Global Work Size: How many work-items or threads we run 
	// - Local Work Size: work-group size -> OpenCL chooses the size (NullRange)
	queue.enqueueNDRangeKernel(simple_add, cl::NullRange, cl::NDRange(SIZE), cl::NullRange);
	
	int C_h[SIZE];
	// Read the final buffer
	// - Read from C_d 
	// - Set Blocking or Non-Blocking
	// - Start Reading at 0
	// - Read 4 Bytes * SIZE
	// - Destination pointer (read into)
	queue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(int) * SIZE, C_h);


	std::cout << " result: \n";
	for (int i = 0; i<10; i++) {
		std::cout << C_h[i] << " ";
	}
	std::cout << std::endl;
}
