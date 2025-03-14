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
- It is limited to the size of the workgroup; since we specify the size to be 10, it will be 10 long.
```cpp
std::vector<mytype> A(10, 1); //allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
size_t local_size = 10; // work-group size
size_t padding_size = A.size() % local_size;
```

## Local Memory + Large Vectors 
- Operating directly on global memory is slow and affects performance. 
- Commonly, we use local memory (a cache form) to speed it up.
	- This is called a *Privatisation technique* 
- This time, we store partial sums in local memory to make this faster.
```cpp
cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_3");
kernel_1.setArg(0, buffer_A);
kernel_1.setArg(1, buffer_B);
kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
```
And the kernel uses a scratch buffer:
```c
//reduce using local memory (so-called privatisation)
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
- For large datasets, we combine results from individual work groups.
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
- In scatter each thread (work-item) writes to a different memory location.
	- Writes tend to be non-contiguous and depend on input data.
- This is similar to gather, where an index array is used for looking at the input location.
	- Each work-item reads from a different location in memory. 

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

	// Scatter - writing to a computed location `bin_index`
	// - Writes to H are scattered across the array
	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}
```
- This function takes an array of values we want to run the histogram on
- The array H is the output where each item is in a bin. 
- `nr_bins`, we specify the number of bins to check if the bin overflows 
	- in which case,e we just set that number to the final bin
- and a neutral element, which is used for padding and memory alignment, which is ignored.
- After all the checks, we use an atomic operation to increment that bin without corrupting threads.
### Atomics
- Lowest-level independent operations which can run without interrupting other operations.
- Low-level and lightweight version of a mutex (Mutexes tend to be built from atomics)
- Incremented by a Thread

# Scan
- Similar to reduction, but it keeps all the partial results.
- This pattern can be used to solve seemingly sequential problems such as sorting or searching.
## Hillis-Steele Scan (Parallel Prefix Scan)
- This is a cumulative sum, which is Span-Efficient
- An extra buffer is needed to avoid overwriting data.
- This is the inclusive version because it accounts for the current element.
- The *Prefix Sum* of an Array `A`, is an array `B`, where each element `B[I]` is the sum of all elements from `A[0]` to `A[i]`. 
```c
kernel void scan_hs(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		// each work-item writes its current value to B
		B[id] = A[id];
		// if the current work-item id surpasses the stride
		if (id >= stride)
		{
			// add the value from A[id - stride] to B
			// this will propagate the sums
			B[id] += A[id - stride];
		}
		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
		
		// updated values in B become the input for the next step
		C = A; A = B; B = C; //swap A & B between steps
	}
}
```
- We can also do a double-buffered version.
- This uses local memory and improves efficiency.
- The issue with the standard approach is that we assume that there are as many PEs as data elements, which often isn't the case for large arrays, so instead, we use a double buffer approach.
```c
//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments, which correspond to two local buffers
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
- It uses two local memory buffers swapped after each reduction step to avoid overwriting the data. 

## Blelloch Scan
- The Hillis scan would work poorly on large arrays due to its work-inefficiency.
- Operations
- The Blelloch Scan consists of two phases
  	1. Reduce Phase (Up-Sweep): traverse the tree from leaves to root, computing the partial sums (parallel reduction). 
  	2. Down-Sweep Phase: traverse back down the tree from root using the partial sums to build the scan in place.
  		- Start by inserting a 0 at the root
		- Move down the tree, distributing the partial sums to the left and right children.
		- At each step, the left child keeps its value, and the right child adds the left child's value to its own.
```c
//Blelloch basic exclusive scan
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N-1] = 0; //exclusive scan so set last to 0

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	// moving down the tree in halves
	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			// add the value of the left child to the current element
			// propagate down the tree
			A[id] += A[id - stride]; //reduce
			// move the original value to the left child.
			A[id - stride] = t;	 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}
```
The [Cuda Version](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) can be found here.

## Blelloch Large Vector 
