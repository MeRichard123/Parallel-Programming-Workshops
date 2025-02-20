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
