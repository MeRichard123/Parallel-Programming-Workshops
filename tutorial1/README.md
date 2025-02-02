# OpenCL

- **Platform**, The OpenCL platform is a specific implementionof the standard by individual hardware vendors. 
 	- One Single Computer may serve several OpenCL platforms/implementations.
	- e.g CPU supported by the Intel Platform and GPU by AMD. 
- **Device** (or kernel), which is an actual hardware device supporting OpenCL.
	- A Device will run its code on 'Computing Units'.
	- e.g a single core of a modern CPU, which consists of 'Processing Elements'
	- Each Platform can support mutiple devices.
- **Contexts**, Mutiple devices from the same platform can then be assembled into different 'Contexts' which enable flexible configurations on machines with multiple devices (e.g 4 GPUs).
- **Queue**, is associated with a specific context (i.e. specific device) and allows a programmer to interact with the device.
	- We send commands to the queue, a command can request a copy operatoin bewtween host and device.
- **Events** - OpenCL enables profiling using events, these are attached to different queue commands and collect info about timings.

## Testing
Looking at the execution time on different devices:

| Device | Platform | Execution Time | Vec Size |
| -------|----------|----------------|----------|
|0 (GPU) | 0 (CUDA) |  10240ns       |  10      |
|0 (CPU) |1 (OpenCL)|  10698ns       |  10      |
|0 (GPU) | 0 (CUDA) |  9216ns        | 100000   |
|0 (CPU) |1 (OpenCL)|  18825ns       | 100000   |

Execution Varies on subsequent runs. But GPU tends to do better for longer vectors.

## Kernels 

```c
kernel void add(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}
```
- Almost a regualar C function except it is marked as a `kernel` 
- The `global` label indicates that the parameters are in main memory. 
- The Kernel is launched in parallel on as many CUs as possible as many times as needed.
	- Each launch gets a separate ID which is obtained by the `get_global_id(0)` function.
	- The total number of kernels launched will be equal to the vector length.
	- This is specified in `enqueueNDRangeKernel` using an `NDRange`
- We can have multiple kernels in the .cl file, we specify the one we want to use using the kernel initialiser `cl::Kernel(program, "mult")`.
