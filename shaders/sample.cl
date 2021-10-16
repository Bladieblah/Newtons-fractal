__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__constant float laplace_kernel[3][3] = {{0.05, 0.2, 0.05}, {0.2, -1., 0.2}, {0.05, 0.2, 0.05}};

__kernel void rdKernel(global float *map, int nColours, global unsigned int *data)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	const int W = get_global_size(0);
	const int H = get_global_size(1);
	
	int index, index2;
	
	index = 3 * (W * y + x);
	index2 = 3 * (int)(nColours * (x + y) / (float)(H + W));
	
	for (int i=0; i<3; i++)
	    data[index + i] = map[index2 + i] * 4294967295;
}
