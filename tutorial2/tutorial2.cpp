#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"


using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//---------- handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test_large.ppm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");

		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9,
							1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9,
							1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9 };
		

		std::vector<int> sobel_Gx = {
			-1, 0, 1,
			-2, 0, 2,
			-1, 0, +1
		};
		std::vector<int> sobel_Gy = {
			1, 2, 1,
			0, 0, 0,
			-1, -2, -1
		};
		float gamma_val = 1.5f;
		int mask_size = 2;
		int conv_size = 5;

		//-----------host operations
		//Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

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

		//--------device operations

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size()*sizeof(float));
		cl::Buffer dev_sobel_gx(context, CL_MEM_READ_ONLY, sobel_Gx.size()*sizeof(int));
		cl::Buffer grad_x(context, CL_MEM_READ_WRITE, sobel_Gx.size()*sizeof(int));


		//Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueWriteBuffer(dev_convolution_mask, CL_TRUE, 0, convolution_mask.size()*sizeof(float), &convolution_mask[0]);
		queue.enqueueWriteBuffer(dev_sobel_gx, CL_TRUE, 0, sobel_Gx.size()*sizeof(int), &sobel_Gx[0]);

		int width = image_input.width();
		int height = image_input.height();
		int channels = image_input.spectrum();
		

		//Setup and execute the kernel (i.e. device code)
		cl::Kernel blur_kernel = cl::Kernel(program, "avg_filterND");
		blur_kernel.setArg(0, dev_image_input);
		blur_kernel.setArg(1, dev_image_output);
		blur_kernel.setArg(2, mask_size);

		
		cl::Kernel gray_kernel = cl::Kernel(program, "rgb2gray");
		gray_kernel.setArg(0, dev_image_input);
		gray_kernel.setArg(1, dev_image_output);

		cl::Kernel conv_kernel = cl::Kernel(program, "convolutionND");
		conv_kernel.setArg(0, dev_image_input);
		conv_kernel.setArg(1, dev_image_output);
		conv_kernel.setArg(2, dev_convolution_mask);
		conv_kernel.setArg(3, conv_size);

		cl::Kernel edgingX = cl::Kernel(program, "convolutionND");
		edgingX.setArg(0, dev_image_input);
		edgingX.setArg(1, dev_image_output);
		edgingX.setArg(2, dev_sobel_gx);

		cl::Kernel gamma_correct = cl::Kernel(program, "gamma_transform");
		gamma_correct.setArg(0, dev_image_input);
		gamma_correct.setArg(1, dev_image_output);
		gamma_correct.setArg(2, gamma_val);

		// Profiling
		cl::Event prof_event;

		std::cout << std::to_string(image_input.size()) << '\n';
		// std::cout << std::to_string(image_output.size()) << '\n';
		
		queue.enqueueNDRangeKernel(conv_kernel, cl::NullRange, cl::NDRange(width, height, channels), cl::NullRange, NULL, &prof_event);
		//queue.enqueueNDRangeKernel(blur_kernel, cl::NullRange, cl::NDRange(width, height, channels), cl::NullRange, NULL, &prof_event);
		//queue.enqueueNDRangeKernel(gray_kernel, cl::NullRange, cl::NDRange(image_input.size()/3), cl::NullRange, NULL, &prof_event);
		
		//queue.enqueueNDRangeKernel(gamma_correct, cl::NullRange, cl::NDRange(image_input.size()/3), cl::NullRange, NULL, &prof_event);
		//queue.enqueueNDRangeKernel(edgingX, cl::NullRange, cl::NDRange(width, height, channels), cl::NullRange, NULL, &prof_event);
		
		vector<unsigned char> output_buffer(image_input.size());
		//Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);
		
		std::cout << "Kernel Execution Time [ns]: " <<
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		
		// Get info about the program execution, enqueue, prep time etc...
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;


		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");
		
 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
