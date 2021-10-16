__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__constant float laplace_kernel[3][3] = {{0.05, 0.2, 0.05}, {0.2, -1., 0.2}, {0.05, 0.2, 0.05}};

//2 component vector to hold the real and imaginary parts of a complex number:
typedef double2 cfloat;

#define I ((cfloat)(0.0, 1.0))


inline float cmod(cfloat a){
    return (sqrt(a.x*a.x + a.y*a.y));
}

inline cfloat m2(cfloat a, cfloat b){
    return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline cfloat m3(cfloat a, cfloat b, cfloat c){
    return m2(a, m2(b, c));
}

inline cfloat m4(cfloat a, cfloat b, cfloat c, cfloat d){
    return m2(m2(a, b), m2(c, d));
}

inline cfloat cdiv(cfloat a, cfloat b){
    return (cfloat)((a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y));
}

inline cfloat func3(cfloat x, cfloat z0, cfloat z1, cfloat z2) {
    return m3(x - z0, x - z1, x - z2);
}

inline cfloat deriv3(cfloat x, cfloat z0, cfloat z1, cfloat z2) {
    return m2(x - z0, x - z1) + m2(x - z1, x - z2) + m2(x - z2, x - z0);
}

inline cfloat step3(cfloat x, cfloat z0, cfloat z1, cfloat z2) {
    return x - cdiv(func3(x, z0, z1, z2), deriv3(x, z0, z1, z2));
}

// inline func5(cfloat x, cfloat z0, cfloat z1, cfloat z2, cfloat z3, cfloat z4) {
//     return m3(x - z0)
// }

__constant float cols[15] = {
    109, 158, 133,
    15,  169, 230,
    229, 109, 178,
    242, 125, 57,
    37,  78,  112
};

__kernel void newton3(global float *roots, global float *map, int nColours, global unsigned int *data, 
    float scale, float dx, float dy)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	const int W = get_global_size(0);
	const int H = get_global_size(1);
	
	int index, index2;
	index = 3 * (W * y + x);
	
	float scale2 = 1. / H * scale;
	
	cfloat z = ((cfloat)(x * scale2 + dx - scale * 0.5 * W / H, y * scale2 + dy - scale * 0.5));
	
	if (cmod(z) < (scale / 200.)) {
	    data[index] = 0.8 * 4294967295;
	    data[index + 1] = 0;
	    data[index + 2] = 0;
	    
	    return;
	}
	
	if (cmod(z - ((cfloat)(dx, dy))) < (scale / 200.)) {
	    data[index] = 0;
	    data[index + 1] = 0.8 * 4294967295;
	    data[index + 2] = 0;
	    
	    return;
	}
	
	cfloat z0 = ((cfloat)(roots[0], roots[1]));
	cfloat z1 = ((cfloat)(roots[2], roots[3]));
	cfloat z2 = ((cfloat)(roots[4], roots[5]));
	
	cfloat croots[3] = {z0, z1, z2};
	
	int i;
	double dist;
	double thr = 1e-10;
	
	double minDist = 1000;
	int minLoc = 0;
	
	for (i=0; i<20000; i++) {
	    z = step3(z, z0, z1, z2);
	    
	    for (int j=0; j<3; j++) {
	        dist = cmod(z - croots[j]);
	        
	        if (dist < minDist) {
	            minDist = dist;
	            minLoc = j;
	        }
	    }
	        
        if (minDist < thr) {
            break;
        }
	}
	
	index2 = 3 * (int)(minLoc);
	float corr = 1. / sqrt(log((float)i + 1 + log(thr) / log(minDist)));
// 	corr = 1;
	
	for (int i=0; i<3; i++) {
	    data[index + i] = cols[index2 + i] / 255. * 4294967295 * corr;
	}
}
