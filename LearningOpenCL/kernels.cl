void kernel simple_add(global const int* A, global const int* B, global int* C) 
{
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
};
