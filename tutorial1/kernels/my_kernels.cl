//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
__kernel void add(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

__kernel void mul(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] * B[id];
}


__kernel void multadd(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = (A[id] * B[id]) + B[id];
}

__kernel void addf(global const float* A, global const float* B, global float* C) {
	int id = get_global_id(0);	
	C[id] = A[id] + B[id];

}

//a simple smoothing kernel averaging values in a local window (radius 1)
__kernel void avg_filter(global const int* A, global int* B) {
	int id = get_global_id(0);
	B[id] = (A[id - 5] + A[id] + A[id + 5])/3;
}

//a simple 2D kernel
__kernel void add2D(global const int* A, global const int* B, global int* C) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int id = x + y*width;

	printf("id = %d x = %d y = %d w = %d h = %d\n", id, x, y, width, height);

	C[id]= A[id]+ B[id];
}
