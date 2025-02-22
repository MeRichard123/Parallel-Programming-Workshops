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
# Scatter
- The scatter pattern writes data into output locations indicated by an index array.
- This is similar to gather, where an index array is used for looking at the input location.

## Histograms 
- One of the simplest examples of scatter patterns is a histogram. 
- It uses a set of value ranges (bins) and counts the number of inputs falling into the bin. 

```c
//a very simple histogram implementation
__kernel void hist_simple(global const int* A, global int* H, const int nr_bins, const int neutral_element) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index
	
	if (bin_index == neutral_element) return;

	if (bin_index >= nr_bins) 
	{
		bin_index = nr_bins - 1; // add to last bin
	}

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}
```
- This function takes an array of values we want to run the histogram on
- The array H is the output where each item in there is a bin. 
- `nr_bins` we specify the number of bins to check if the bin overflows 
	- in which case we just set that number to the final bin
- and a neutral element which is used for padding and memory alignment which is ignored.
- After all the checks we use an atomic operation to increment that bin without corrupting threads.
### Atomics
- Lowest-level independent operations which can run without interrupting other operations.
- Low-Level and light weight version of a mutex (Mutexes tend to be built from atomics)
- Incremented by a Thread

# Scan
- Similar to reduction but it keeps all the partial results.
- This pattern can be used to solve seemingly sequential probelms such as sorting or searching.
```c
//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}
```
- Uses two local memory buffers which are swapped after each reduction step to avoid data being overwritten. 
