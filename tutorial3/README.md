# Reductions in OpenCL
- Reduce combines all elements of the input data into a single output element.
```c
//flexible step reduce 
__kernel void reduce_add_2(global const int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	// this one loops the above but still only works for powers of 2

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N)) 
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
```
- This code runs on the Global Memory Space within 1 workgroup. 
- It is limited to the size of the workgroup, since we specify the size to be 10 it wil be 10 long.
```cpp
std::vector<mytype> A(20, 1);//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
size_t local_size = 10;
size_t padding_size = A.size() % local_size;
```

## Local Memory + Large Vectors 
- Operating directly on global memory is slow, and affects performance. 
- Commonly we use local memory (a form of cache) to speed it up.
	- This is called a *Privatisation techniqiue* 
- This time we store partial sums in local memory to make this faster.
```cpp
cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_3");
kernel_1.setArg(0, buffer_A);
kernel_1.setArg(1, buffer_B);
kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
```
And the kernel uses a scratch buffer:
```c
//reduce using local memory (so called privatisation)
kernel void reduce_add_3(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}
```
- For larget datasets we combine results from individual work groups.
- There are many ways to do this, but one way is via atomic functions.
```cpp
kernel void reduce_add_4(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}
```
