//g++ -std=c++0x tutorial1.cpp -o tutorial1 -lOpenCL
// 778201
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include "Utils.h"

#include <iostream>
#include <vector>

void print_help() {
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");
		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		//catch (const cl::Error& err) {
		catch (...) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			//throw err;
		}

		//Part 3 - memory allocation
		//host - input
		
		std::vector<float> A = { 0.0f, 1.5f, 2.5f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f }; //C++11 allows this type of initialisation
		std::vector<float> B = { 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f };
		//std::vector<int> A(1000000);
		//std::vector<int> B(1000000);
		

		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size()*sizeof(float);//size in bytes

		//host - output
		std::vector<float> C(vector_elements);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Part 4 - device operations

		//4.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_multadd = cl::Kernel(program, "multadd");
		kernel_multadd.setArg(0, buffer_A);
		kernel_multadd.setArg(1, buffer_B);
		kernel_multadd.setArg(2, buffer_C);
		
		cl::Kernel kernel_add = cl::Kernel(program, "add");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, buffer_C);

		cl::Kernel kernel_mul = cl::Kernel(program, "mul");
		kernel_mul.setArg(0, buffer_C);
		kernel_mul.setArg(1, buffer_B);
		kernel_mul.setArg(2, buffer_C);
	
		cl::Kernel kernel_addf = cl::Kernel(program, "addf");
		kernel_addf.setArg(0, buffer_A);
		kernel_addf.setArg(1, buffer_B);
		kernel_addf.setArg(2, buffer_C);

		cl::Kernel kernel_add2d = cl::Kernel(program, "add2d");
		kernel_add2d.setArg(0, buffer_A);
		kernel_add2d.setArg(1, buffer_B);
		kernel_add2d.setArg(2, buffer_C);
		
		cl::Event prof_event;
		/*
		queue.enqueueNDRangeKernel(kernel_mul, cl::NullRange, 
				cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);


		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, 
				cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		queue.enqueueNDRangeKernel(kernel_multadd, cl::NullRange, 
				cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);
		*/
		queue.enqueueNDRangeKernel(kernel_add2d, cl::NullRange,cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);

		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		std::cout << "C = " << C << std::endl;


		std::cout << "Kernel Execution Time [ns]: " <<
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		
		// Get info about the program execution, enqueue, prep time etc...
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
	}
	//catch (cl::Error err) {
	catch (...) {
		//std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
