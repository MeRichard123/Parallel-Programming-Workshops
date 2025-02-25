//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	if (colour_channel == 0)
	{ 
		B[id] = A[id];
	}
	else 
	{
		B[id] = 0;
	}
}

// Invert Kernel 
__kernel void invert(global const uchar* A, global uchar* B) {

	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels	
	int image_size = width * height; 
	int x = get_global_id(0);
	int y = get_global_id(1);
	int c = get_global_id(2);
	
	int id = x + y*width + c*image_size;
	B[id] = 255 - A[id];

}

// Rgb2Gray
__kernel void rgb2gray(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = (width*height); //image size in pixels
	
	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); // current channel	
	
	
	int id = x + y * width;
	// Get RBG
	uchar r = A[id + (0 * image_size)];
	uchar g = A[id + (1 * image_size)];
	uchar b = A[id + (2 * image_size)];


	uchar gray = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);

	B[id + (0 * image_size)] = gray;
	B[id + (1 * image_size)] = gray;
	B[id + (2 * image_size)] = gray;
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int z = get_global_id(2); //current z coord.
	int c = get_global_id(3); //current colour channel

	int id = x + y*width + z*width * height + c*image_size; //global id in 3D space

	B[id] = A[id];
}


// Gamma Correction
__kernel void gamma_transform(global const uchar* A, global uchar* B, const float gamma)
{
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(3); //current colour channel
	
	int id = x + y * width;
	// Get RBG
	float r = ((float)A[id + (0 * image_size)]) / 255.0;
	float g = ((float)A[id + (1 * image_size)]) / 255.0;
	float b = ((float)A[id + (2 * image_size)]) / 255.0;
		

	float norm_pixel_r = pow(r, gamma);
	float norm_pixel_g = pow(g, gamma);
	float norm_pixel_b = pow(b, gamma);

	B[id + (0 * image_size)] = norm_pixel_r * 255.0;
	B[id + (1 * image_size)] = norm_pixel_g * 255.0;
	B[id + (2 * image_size)] = norm_pixel_b * 255.0;
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B, const int range) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;
	int w_range = 4;

	for (int i = max(0, x-w_range); i <= min(width - 1, x + w_range); i++)
	{
		for (int j = max(0, y-w_range); j <= min(height - 1, y + w_range); j++)
		{
			result += A[i + j*width + c*image_size];
		}
	}

	int kernel_size = (2 * w_range + 1) * (2 * w_range + 1);
	result /= kernel_size;

	B[id] = (uchar)result;
}


//2D 3x3 convolution kernel
__kernel void convolutionND(__global const uchar* A, global uchar* B, __constant const float* mask, const int mask_size) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;
	int offset = mask_size/ 2;

   	// Iterate over the mask region
   	for (int i = -offset; i <= offset; i++) {
		 for (int j = -offset; j <= offset; j++) {
			  int xi = x + i;
			  int yi = y + j;

			  if (xi >= 0 && xi < width && yi >= 0 && yi < height) {
			  	int mask_x = i + offset;  // Translate mask coordinates to the linear index
				int mask_y = j + offset;
				int mask_index = mask_x + mask_y * mask_size; // Linear index in the mask
				result += A[xi + yi * width + c * image_size] * mask[mask_index];
		    	  }
		 }
	}
	// Clamping the result before storing it in B
	result = clamp(result, 0.0f, 255.0f);  // Ensure the result is within valid range for uchar

	B[id] = (uchar)result;
}

