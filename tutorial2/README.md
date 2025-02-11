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

## OpenCL Function 
- `get_global_id` - gets the current id for each work-items
- We can define the work-group size manually when calleding the kernel from the host
```cpp
queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
```
- In OpenCL 1.2 the data input size should be divisible by the work group size.
- Different kernels will have different perferred work-group sizes (due to architecture)
```cpp
cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; // get device
cerr << kernel_add.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (device) << endl; // get info
```
- It is suggested to use the value with the smallest work group size.
- But we can specify a different size:
```cpp
kernel_add.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device)
```

# Map Pattern
- The map pattern describes the same computation performed on different data without the need for communication between work items. 
