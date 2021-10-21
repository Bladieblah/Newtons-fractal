__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

//2 component vector to hold the real and imaginary parts of a complex number:
typedef double2 cfloat;

#define I ((cfloat)(0.0, 1.0))


inline float cmod(cfloat a){
    return (sqrt(a.x*a.x + a.y*a.y));
}

inline cfloat m2(cfloat a, cfloat b){
    return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline cfloat cdiv(cfloat a, cfloat b){
    return (cfloat)((a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y));
}

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

inline cfloat deriv2n(cfloat x, cfloat *z, int n) {
    cfloat result = ((cfloat)(0,0));
    
    for (int i=0; i<n; i++) {
        for (int j=i+1; j<n; j++) {
            cfloat _result = x - z[i];
            for (int k=0; k<n; k++) {
                if (k != i && k != j) {
                    _result = m2(_result, x - z[j % n]);
                }
            }
            result = result + _result;
        }
    }
    
    return ((cfloat)(2, 2)) * result;
}

inline cfloat stepn(cfloat x, cfloat *z, int n) {
//     double stepsize = 0.9;
//     return x - ((cfloat)(stepsize, stepsize)) * cdiv(funcn(x, z, n), derivn(x, z, n));
    return x - cdiv(funcn(x, z, n), derivn(x, z, n));
}

inline cfloat step2(cfloat x, cfloat num, cfloat denom) {
    return x - cdiv(num, denom);
}

struct Matrix {
    double m00, m01, m02;
    double m10, m11, m12;
    double m20, m21, m22;
};

struct Color {
    double r, g, b;
};

inline struct Matrix hueRotation(double theta) {
    double c = cos(theta);
    double s = sin(theta);
    
    double c1 = (1 - c) / 3.;
    double s1 = sqrt(1./3.) * s;
    
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

__kernel void newtonn(global float *roots, global float *map, int nColours, global int *data, 
    double scale, double dx, double dy, int _nRoots)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	const int W = get_global_size(0);
	const int H = get_global_size(1);
	
	int nRoots = (int)_nRoots;
	
	int i;
	int index, index2;
	index = 3 * (W * y + x);
	
	double scale2 = 1. / H * scale;
	
	cfloat z = ((cfloat)(x * scale2 + dx - scale * 0.5 * W / H, y * scale2 + dy - scale * 0.5));
	cfloat zprev;
	cfloat croots[20];
	
	for (i=0; i<nRoots; i++) {
	    croots[i].x = roots[2*i];
	    croots[i].y = roots[2*i+1];
	}
	
	double dist, prevDist;
	double thr = 1e-6;
	
	double minDist = 1000;
	int minLoc = 0;
	
	for (i=0; i<2000; i++) {
        zprev = z;
        z = stepn(z, croots, nRoots);
	    
	    for (int j=0; j<nRoots; j++) {
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
	
	float theta = 0.1 * M_PI * minLoc + 0.;
	struct Matrix mat = hueRotation(theta);
	col = applyMat(mat, col);
	
	data[index + 0] = col.r * 4294967295;
	data[index + 1] = col.g * 4294967295;
	data[index + 2] = col.b * 4294967295;
}
