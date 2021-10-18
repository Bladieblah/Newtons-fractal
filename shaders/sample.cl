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

inline cfloat func4(cfloat x, cfloat z0, cfloat z1, cfloat z2, cfloat z3) {
    return m4(x - z0, x - z1, x - z2, x - z3);
}

inline cfloat deriv4(cfloat x, cfloat z0, cfloat z1, cfloat z2, cfloat z3) {
    return m3(x - z0, x - z1, x - z2) + m3(x - z1, x - z2, x - z3) + 
        m3(x - z2, x - z3, x - z0) + m3(x - z3, x - z0, x - z1);
}

inline cfloat step4(cfloat x, cfloat z0, cfloat z1, cfloat z2, cfloat z3) {
    return x - 
//     ((cfloat)(0.6, 0.6)) * 
    cdiv(func4(x, z0, z1, z2, z3), deriv4(x, z0, z1, z2, z3));
}

// inline func5(cfloat x, cfloat z0, cfloat z1, cfloat z2, cfloat z3, cfloat z4) {
//     return m3(x - z0)
// }

inline cfloat funcn(cfloat x, cfloat *z, int n) {
    cfloat result = x - z[0];
    
    for (int i=1; i<n; i++) {
        result = m2(result, x - z[i]);
    }
    
    return result;
}

inline cfloat derivn(cfloat x, cfloat *z, int n) {
    cfloat result = ((cfloat)(0,0));
    
    for (int i=0; i<n; i++) {
        cfloat _result = x - z[i];
    
        for (int j=i+1; j<i+n-1; j++) {
            _result = m2(_result, x - z[j % n]);
        }
        
        result = result + _result;
    }
    
    return result;
}

inline cfloat stepn(cfloat x, cfloat *z, int n) {
    return x - cdiv(funcn(x, z, n), derivn(x, z, n));
}

__constant float cols[15] = {
    109, 158, 133,
    15,  169, 230,
    229, 109, 178,
    242, 125, 57,
    37,  78,  112
};

struct Matrix {
    float m00, m01, m02;
    float m10, m11, m12;
    float m20, m21, m22;
};

struct Color {
    float r, g, b;
};

inline struct Matrix hueRotation(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    
    float c1 = (1 - c) / 3.;
    float s1 = sqrt(1./3.) * s;
    
    struct Matrix mat;
    
    mat.m00 = c + c1;
    mat.m01 = c1 - s1;
    mat.m02 = c1 + s1;
    mat.m10 = mat.m02;
    mat.m11 = mat.m00;
    mat.m12 = mat.m01;
    mat.m20 = mat.m01;
    mat.m21 = mat.m02;
    mat.m22 = mat.m00;
    
    return mat;
}

inline struct Color applyMat(struct Matrix mat, struct Color col) {
    struct Color newCol;
    
    newCol.r = mat.m00 * col.r + mat.m01 * col.g + mat.m02 * col.b;
    newCol.g = mat.m10 * col.r + mat.m11 * col.g + mat.m12 * col.b;
    newCol.b = mat.m20 * col.r + mat.m21 * col.g + mat.m22 * col.b;
    
    return newCol;
}

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
	double thr = 1e-20;
	
	double minDist = 1000;
	int minLoc = 0;
	
	for (i=0; i<2000000; i++) {
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
	
// 	index2 = 3 * (int)(minLoc);
// 	float corr = 1. / sqrt(log((float)i + 1 + log(thr) / log(minDist)));
// // 	corr = 1;
// 	
// 	for (int i=0; i<3; i++) {
// 	    data[index + i] = cols[index2 + i] / 255. * 4294967295 * corr;
// 	}
	
	index2 = 3 * (int)(i) * nColours;
	
	for (int i=0; i<3; i++)
	    data[index + i] = map[index2 + i] * 4294967295;
}

__kernel void newton4(global float *roots, global float *map, int nColours, global unsigned int *data, 
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
	cfloat z3 = ((cfloat)(roots[6], roots[7])); 
	
	cfloat croots[4] = {z0, z1, z2, z3};
	
	int i;
	double dist, prevDist;
	double thr = 1e-8;
	
	double minDist = 1000;
	int minLoc = 0;
	
	for (i=0; i<2000000; i++) {
	    z = step4(z, z0, z1, z2, z3);
	    
	    for (int j=0; j<4; j++) {
	        dist = cmod(z - croots[j]);
	        
	        if (dist < minDist) {
                prevDist = minDist;
	            minDist = dist;
	            minLoc = j;
	        }
	    }
	        
        if (minDist < thr) {
            break;
        }
	}
	
	index2 = 3 * ((int)(10. * (i - 2. * (log(thr) - log(minDist)) / (log(prevDist) - log(minDist)))) % nColours);
	
	struct Color col;
	
	col.r = map[index2 + 0];
	col.g = map[index2 + 1];
	col.b = map[index2 + 2];
	
	float theta = 0.2 * M_PI * minLoc + 0.3;
	struct Matrix mat = hueRotation(theta);
	col = applyMat(mat, col);
	
	data[index + 0] = col.r * 4294967295;
	data[index + 1] = col.g * 4294967295;
	data[index + 2] = col.b * 4294967295;
}

__kernel void newtonn(global float *roots, global float *map, int nColours, global unsigned int *data, 
    float scale, float dx, float dy, int nRoots)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	const int W = get_global_size(0);
	const int H = get_global_size(1);
	
	int i;
	int index, index2;
	index = 3 * (W * y + x);
	
	float scale2 = 1. / H * scale;
	
	cfloat z = ((cfloat)(x * scale2 + dx - scale * 0.5 * W / H, y * scale2 + dy - scale * 0.5));
	
	cfloat croots[nRoots];
	
	for (i=0; i<nRoots; i++) {
	    croots[i] = ((cfloat)(roots[2*i], roots[2*i+1]));
	}
	
	double dist, prevDist;
	double thr = 1e-8;
	
	double minDist = 1000;
	int minLoc = 0;
	
	for (i=0; i<10; i++) {
	    z = stepn(z, croots, nRoots);
	    
	    for (int j=0; j<4; j++) {
	        dist = cmod(z - croots[j]);
	        
	        if (dist < minDist) {
                prevDist = minDist;
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
	corr = 1;
	
	for (int i=0; i<3; i++) {
	    data[index + i] = cols[index2 + i] / 255. * 4294967295 * corr;
	}
	
	return;
	
	index2 = 3 * ((int)(10. * (i - 2. * (log(thr) - log(minDist)) / (log(prevDist) - log(minDist)))) % nColours);
	index2 = (int)(i) % nColours;
	
	struct Color col;
	
	col.r = map[index2 + 0];
	col.g = map[index2 + 1];
	col.b = map[index2 + 2];
	
	float theta = 0.2 * M_PI * minLoc + 0.3;
	struct Matrix mat = hueRotation(theta);
// 	col = applyMat(mat, col);
	
	data[index + 0] = col.r * 4294967295;
	data[index + 1] = col.g * 4294967295;
	data[index + 2] = col.b * 4294967295;
}
