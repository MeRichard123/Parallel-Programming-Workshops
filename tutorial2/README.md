# Kernel Execution Model
- Devices run code on computating units, which conisist of processing elements. 
- Each separate kernel execution is performed on a PE, and is called a "work-item"
- We can think about these items as separate threads.
- For 10 vectors, the kernel will launch 10 work-items.
- Each Processing unit consists of a fixed number of PEs.
	- This limits the number of items which might be executed at a time. 
	- Work-items often need to communicate with each other, and share resources.
- OpenCL introduces work-groups, which are collections of work-items executed by a single CU
- The device launches as many work-groups as possible, depending on the number of CUs.

## OpenCL IDs 
- `get_global_id` - gets the current id for each work-items
	- A unique identifier for a work item in the grid.
   	- Specifies the work item position in the global index space
- Work items are executed arbitrarily, so one cannot make any assumptions about which work item is executed first.
- `get_local_id` returns a work item id within a specific work group. 
	- The values vary from 0 to M1, where M is the size of a work group.
   	- ID for work items within a work group.
IDs are based on the dimensions of the workgroup.
- In 1D we have $$N$$ work items (global size) and $$M$$ sized work groups (local size)
  	- The global id for a work group item is: `get_global_id(0)`
  	- The local id for a work item is `get_local_id(0)`
- In 2D, let's say we have $$N_z \times N_y$$ (global size) then
  	- `global_id_x = get_global_id(0);`
  	- `global_id_y = get_global_id(1);`  
- We can define the work-group size manually when calling the kernel from the host using the last parameter
```cpp
size_t local_size[2] = {16,16};
queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
```
The OpenCL kernel uses them like this:
```c
__kernel void myKernel(__global float* input, __global float* output) {
    // Get global and local IDs
    int global_id_x = get_global_id(0); // Global ID in x-dimension
    int global_id_y = get_global_id(1); // Global ID in y-dimension
    int local_id_x = get_local_id(0);   // Local ID in x-dimension
    int local_id_y = get_local_id(1);   // Local ID in y-dimension

    // Perform computation
    output[global_id_y * N + global_id_x] = input[global_id_y * N + global_id_x] * 2.0f;
}
```
Imagine a 2D grid of work-items
- The grid is divided into work groups (4x4 work groups)
- Each work group contains a smaller grid of work items (8x8 work items)
- The global id identifies a work item's position in the entire grid.
- The local id, identifies the work item's position in its work group.
   
In OpenCL 1.2 the data input size should be divisible by the work group size.
- Global Size has to be divisible by the local size.
- Different kernels will have different preferred work-group sizes (due to architecture)
```cpp
cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; // get device
cerr << kernel_add.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (device) << endl; // get info
```
- It is suggested to use the value with the smallest work group size.
- But we can specify a different size:
```cpp
kernel_add.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device)
```
Let's define group sizes:
```cpp
    // Define global and local sizes
    size_t global_size[2] = {1024, 1024}; // 1024x1024 work items
    size_t local_size[2] = {16, 16};      // 16x16 work items per work group

    // Create memory buffers
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 1024 * 1024 * sizeof(float), NULL, NULL);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1024 * 1024 * sizeof(float), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);

    // Enqueue the kernel
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
```

# Map Pattern
- The map pattern describes the same computation performed on different data without the need for communication between work items.
- It is an independent element-wise operation. 
- Often used in conjunction with reduce.

# Stencil Pattern
- Takes mutliple data inputs from a pre-defined neighbourhood and combines them into a single value. 
	- Like running a convolution using a kernel
- Like Map, each task/work-item is independent.
- This is used in signal processing for noise filtering, blurring and sharpening.
## 1D Stencil
```cpp
// a simple smoothing kernel averaging values in a local window (radius 1)
__kernel void avg_filter(global const int* A, global int* B) {
        int id = get_global_id(0);
        B[id] = (A[id - 1] + A[id] + A[id + 1])/3;
}

```
- This operation is equivalent to a simple smoothing filter, which can be applied to filter out noise from data (e.g. audio signal).

## 2D Stencil 
- In Images we often store the pixels in 2D
- The kernel dimensionality is specified within a Kernel execution command.
- In 2D, the first parameter of NDRange specifes the number of columns (width), and the second the rows.
- For a 5x2 array we use:
```cpp
queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(5, 2), cl::NullRange);
```
- Now that the global indices are arranged in two dimensions, the `get_global_size(d)` function.
	- returns the number of elements along the dth dimension.
