#include <iostream>
#include <vector>
#include <climits>

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//------- handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0;}
	}

	//detect any potential exceptions
	try {
		//------ host operations
		//Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;

		//----------- memory allocation
		//host - input
		std::vector<mytype> A(10, 1);//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
		//std::vector<mytype> A = {-5, 1,1,1,1,1,5,5,5,5,5, 12, 15, 11, 12, 2, 3,4,5, 6};

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 10;
		int neutral_element = -2;
		int min_value = -1;
		int max_value = 10;

		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, neutral_element);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;
			

		std::vector<int> min_value_vec(1, INT_MAX);
		std::vector<int> max_value_vec(1, INT_MIN);

		//host - output
		int nr_bins = 10;
		std::vector<mytype> B(nr_bins, 0);
		size_t output_size = B.size() * sizeof(int); //size in bytes
		
		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

	   	cl::Buffer buffer_min(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_max(context, CL_MEM_READ_WRITE, output_size);

		//------------ device operations

		//copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueWriteBuffer(buffer_min, CL_TRUE, 0, output_size, &min_value_vec[0]);
    		queue.enqueueWriteBuffer(buffer_max, CL_TRUE, 0, output_size, &max_value_vec[0]);


		//Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_3");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		
		cl::Kernel simple_hist = cl::Kernel(program, "hist_simple");
		simple_hist.setArg(0, buffer_A);
		simple_hist.setArg(1, buffer_B);
		simple_hist.setArg(2, nr_bins);
		simple_hist.setArg(3, neutral_element);
		simple_hist.setArg(4, min_value);
		simple_hist.setArg(5, max_value);

		// Compute min and max
		cl::Kernel reduce_min_kernel(program, "reduce_min");
		reduce_min_kernel.setArg(0, buffer_A);
		reduce_min_kernel.setArg(1, buffer_min);
		reduce_min_kernel.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		//queue.enqueueNDRangeKernel(reduce_min_kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		cl::Kernel reduce_max_kernel(program, "reduce_max");
		reduce_max_kernel.setArg(0, buffer_A);
		reduce_max_kernel.setArg(1, buffer_max);
		reduce_max_kernel.setArg(2, cl::Local(local_size*sizeof(mytype)));
		//queue.enqueueNDRangeKernel(reduce_max_kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		
		// Read min and max values back to host
    		//queue.enqueueReadBuffer(buffer_min, CL_TRUE, 0, output_size, &min_value_vec[0]);
    		//queue.enqueueReadBuffer(buffer_max, CL_TRUE, 0, output_size, &max_value_vec[0]);
			
		//std::cout << min_value_vec[0] << std::endl;
		//std::cout << max_value_vec[0] << std::endl;

		// Run histogram kernel
		cl::Kernel hist_kernel(program, "hist_complex");
		hist_kernel.setArg(0, buffer_A);
		hist_kernel.setArg(1, buffer_B);
		hist_kernel.setArg(2, nr_bins);
		hist_kernel.setArg(3, min_value_vec[0]);
		hist_kernel.setArg(4, max_value_vec[0]);
			
		cl::Kernel scan_add_kernel(program, "scan_add");
		scan_add_kernel.setArg(0, buffer_A);
		scan_add_kernel.setArg(1, buffer_B);
		scan_add_kernel.setArg(2, cl::Local(input_size*sizeof(float)));
		scan_add_kernel.setArg(3, cl::Local(input_size*sizeof(float))); 


		// Perf Events 
		cl::Event prof_event;

		//call all kernels in a sequence
		//queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		//queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(CL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), NULL, &prof_event);
		//queue.enqueueNDRangeKernel(simple_hist, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
	
		queue.enqueueNDRangeKernel(scan_add_kernel, cl::NullRange, 
				cl::NDRange(input_elements), cl::NDRange(local_size), 
				NULL, &prof_event
				);

		//Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		
		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		
#if 0
		for (int i = 0; i < nr_bins; ++i) {
        		std::cout << "Bin " << i << ": " << B[i] << std::endl;
		}
#endif

		std::cout << "Kernel Execution Time [ns]: " <<
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
