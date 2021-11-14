__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

//2 component vector to hold the real and imaginary parts of a complex number:
typedef double2 cfloat;

#define I ((cfloat)(0.0, 1.0))


inline double cmod(cfloat a){
    return (sqrt(a.x*a.x + a.y*a.y));
}

inline double carg(cfloat a){
    return fmod(atan2(a.y, a.x) + 4 * M_PI, M_PI);
}

inline cfloat m2(cfloat a, cfloat b){
    return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline cfloat cdiv(cfloat a, cfloat b){
    return (cfloat)((a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y));
}

inline cfloat csin(cfloat z) {
    double epz = exp(z.y);
    double emz = exp(-z.y);
    
    return ((cfloat)(0.5 * sin(z.x) * (epz + emz), 0.5 * cos(z.x) * (epz - emz)));
}

inline cfloat funcn(cfloat x, cfloat *z, char n) {
    cfloat result = x - z[0];
    
    for (char i=1; i<n; i++) {
        result = m2(result, x - z[i]);
    }
    
    return result;
}

inline cfloat derivn(cfloat x, cfloat *z, char n) {
    cfloat result = ((cfloat)(0,0));
    
    for (char i=0; i<n; i++) {
        cfloat _result = x - z[i];
    
        for (char j=i+1; j<i+n-1; j++) {
            _result = m2(_result, x - z[j % n]);
        }
        
        result = result + _result;
    }
    
    return result;
}

// inline cfloat deriv2n(cfloat x, cfloat *z, char n) {
//     cfloat result = ((cfloat)(0,0));
//     
//     char kstart = 2;
//     
//     for (char i=0; i<n; i++) {
//         for (char j=i+1; j<n; j++) {
//             cfloat _result = x - z[kstart];
//             for (char k=kstart+1; k<n; k++) {
//                 if (k != i && k != j) {
//                     _result = m2(_result, x - z[j % n]);
//                 }
//             }
//             result = result + _result;
//             kstart = 1;
//         }
//         kstart = 0;
//     }
//     
//     return ((cfloat)(2, 2)) * result;
// }

inline cfloat deriv2n(cfloat x, cfloat *z, char n) {
    cfloat result = ((cfloat)(0,0));
    
    for (char i=0; i<n; i++) {
        for (char j=0; j<n; j++) {
            if (j == i) {
                continue;
            }
            
            cfloat _result = ((cfloat)(1,0));
            
            for (char k=0; k<n; k++) {
                if (k != i && k != j) {
                    _result = m2(_result, x - z[j]);
                }
            }
            
            result = result + _result;
        }
    }
    
    return result;
}

inline cfloat stepn(cfloat x, cfloat *z, char n) {
//     double stepsize = 0.9;
//     return x - cdiv(funcn(x, z, n), derivn(x, z, n));
    return x - cdiv(funcn(x, z, n), derivn(x, z, n));
}

inline cfloat mandelStep(cfloat z, cfloat c) {
    return m2(z, z) + c;
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

inline struct Color applyShade(struct Color col, cfloat z, cfloat dz, cfloat v, double h) {
    struct Color newCol;
    cfloat u = cdiv(z, dz);
    double norm = cmod(u);
    
    u.x = u.x / norm;
    u.y = u.y / norm;
    
//     double scale = 0.;
//     double highlight = scale + (1. - scale) * (1. - (1. - u.x * v.x - u.y * v.y) / (1. + h));
    double highlight = (u.x * v.x + u.y * v.y + h) / (1. + h);
    
//     newCol.r = 1. - (1. - col.r) * highlight;
//     newCol.g = 1. - (1. - col.g) * highlight;
//     newCol.b = 1. - (1. - col.b) * highlight;
    
//     newCol.r = col.r * highlight;
//     newCol.g = col.g * highlight;
//     newCol.b = col.b * highlight;
    
    newCol.r = highlight;
    newCol.g = highlight;
    newCol.b = highlight;
    
    return newCol;
}

inline struct Color RGBtoHSL(struct Color rgb) {
    struct Color hsl;
    double cmin, cmax, delta;
    
    cmin = fmin(rgb.r, fmin(rgb.g, rgb.b));
    cmax = fmax(rgb.r, fmax(rgb.g, rgb.b));
    
    delta = cmax - cmin;
    
    // Hue
    if (delta == 0.) {
        hsl.r = 0.;
    } else if (cmax == rgb.r) {
        hsl.r = 60. * (rgb.g - rgb.b) / delta;
    } else if (cmax == rgb.g) {
        hsl.r = 60. * ((rgb.b - rgb.r) / delta + 2.);
    } else {
        hsl.r = 60. * ((rgb.r - rgb.g) / delta + 4.);
    }
    
    // Lightness
    hsl.b = (cmax + cmin) * 0.5;
    
    // Saturation
    if (delta == 0.) {
        hsl.g = 0.;
    } else {
        hsl.g = delta / (1 - fabs(2 * hsl.b - 1));
    }
    
    return hsl;
}

inline struct Color HSLtoRGB(struct Color hsl) {
    struct Color rgb;
    double C, X, m;
    
    hsl.r = fmod(fmod(hsl.r, 360.) + 360, 360);
    
    C = (1 - fabs(2. * hsl.b - 1.)) * hsl.g;
    X = C * (1 - fabs(fmod(hsl.r / 60., 2) - 1));
    m = hsl.b - C * 0.5;
    
    if (hsl.r < 60) {
        rgb.r = C + m;
        rgb.g = X + m;
        rgb.b = m;
    } else if (hsl.r < 120) {
        rgb.r = X + m;
        rgb.g = C + m;
        rgb.b = m;
    } else if (hsl.r < 180) {
        rgb.r = m;
        rgb.g = C + m;
        rgb.b = X + m;
    } else if (hsl.r < 240) {
        rgb.r = m;
        rgb.g = X + m;
        rgb.b = C + m;
    } else if (hsl.r < 300) {
        rgb.r = X + m;
        rgb.g = m;
        rgb.b = C + m;
    } else {
        rgb.r = C + m;
        rgb.g = m;
        rgb.b = X + m;
    }
    
    return rgb;
}

inline struct Color shade2(struct Color col, cfloat z, cfloat dz, cfloat v, double h) {
    struct Color hsl;
    cfloat u = cdiv(z, dz);
    double norm = cmod(u);
    
    u.x = u.x / norm;
    u.y = u.y / norm;
    
    double highlight = (u.x * v.x + u.y * v.y + h + 2) / (1. + h) - 1.1;
    hsl = RGBtoHSL(col);
    
    hsl.b = fmax(0., fmin(1., hsl.b + 0.5 * highlight));
    
    return HSLtoRGB(hsl);
}

inline struct Color rotateHue(struct Color col, double theta) {
    struct Color hsl;
    
    hsl = RGBtoHSL(col);
    
    hsl.r = fmod(fmod(hsl.r + theta, 360) + 360, 360);
    
    return HSLtoRGB(hsl);
}

__kernel void newtonn(global float *roots, global float *map, int nColours, global unsigned int *data, 
    double scale, double dx, double dy, int _nRoots, double __)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	const int W = get_global_size(0);
	const int H = get_global_size(1);
	
	char nRoots = (char)_nRoots;
	
	int i;
	int index, index2;
	index = 3 * (W * y + x);
	
	double scale2 = 1. / H * scale;
	
	cfloat z = ((cfloat)(x * scale2 + dx - scale * 0.5 * W / H, y * scale2 + dy - scale * 0.5));
	cfloat croots[20];
	
	for (i=0; i<nRoots; i++) {
	    croots[i].x = roots[2*i];
	    croots[i].y = roots[2*i+1];
	}
	
	double dist, prevDist;
	double thr = 1e-6;
	
	double minDist = 1000;
	int minLoc = 2;
	
	for (i=0; i<2000; i++) {
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

__kernel void mandel(global float *roots, global float *map, int nColours, global unsigned int *data, 
    double scale, double dx, double dy, int _nRoots, double __)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	const int W = get_global_size(0);
	const int H = get_global_size(1);
	
	int i;
	int index, index2;
	index = 3 * (W * y + x);
	
	double scale2 = 1. / H * scale;
	
	cfloat c = ((cfloat)(x * scale2 + dx - scale * 0.5 * W / H, y * scale2 + dy - scale * 0.5));
	cfloat z = ((cfloat)(0, 0));
	cfloat one = ((cfloat)(1, 0));
	cfloat two = ((cfloat)(2, 2));
	
	double minDist = cmod(c);
	
	cfloat dz = one;
	
	double dist = 0.;
	double thr = 20;
	double maxIter = _nRoots;
    
    double starFactor = 2 * 0.000012786458333;
    starFactor = 1.;
    
    double h = 6.;
    double phi = 45./360.;
    cfloat v = ((cfloat)(cos(2 * M_PI * phi), sin(2 * M_PI * phi)));
	
	for (i=0; i<maxIter; i++) {
	    dz = two * m2(z, dz) + one;
        z = m2(z, z) + c;
        dist = cmod(z);
        
        minDist = fmin(minDist, dist);
        
        if (dist > thr) {
            break;
        }
	}
	
	if (i > maxIter - 2) {	
        data[index + 0] = 0;
        data[index + 1] = 0;
        data[index + 2] = 0;
        
	    return;
	}
	
	index2 = 3 * ((int)(minDist / starFactor + fabs((i + 1 - log(log(sqrt(dist)))/log(thr))) / 2) % nColours);
	
	struct Color col;
	
	col.r = map[index2 + 0];
	col.g = map[index2 + 1];
	col.b = map[index2 + 2];
	
	double theta = 0.1 * carg(z) / M_PI * 180;
	col = rotateHue(col, theta);
	col = shade2(col, z, dz, v, h);
	
	data[index + 0] = col.r * 4294967295;
	data[index + 1] = col.g * 4294967295;
	data[index + 2] = col.b * 4294967295;
}

__kernel void wave(global float *roots, global float *map, int nColours, global unsigned int *data, 
    double scale, double dx, double dy, int _nRoots, double thr)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	const int W = get_global_size(0);
	const int H = get_global_size(1);
	
	int i;
	int index, index2;
	index = 3 * (W * y + x);
	
	double scale2 = scale / H;
	
	cfloat z = ((cfloat)(x * scale2 + dx - scale * 0.5 * W / H, y * scale2 + dy - scale * 0.5));
	cfloat sz, zPrev;
	
	double minDist = cmod(z);
	
	cfloat dz = ((cfloat)(1, 0));
	
	double dist = cmod(z);
	double prevDist = dist;
// 	double thr = 10;
	double maxIter = _nRoots;
    
    double starFactor = 2 * 0.012786458333;
//     starFactor = 1.;
    
    double h = 6.;
    double phi = 45./360.;
    cfloat v = ((cfloat)(cos(2 * M_PI * phi), sin(2 * M_PI * phi)));
    
    double arg = 0;
	
	for (i=0; i<maxIter; i++) {
	    zPrev = z;
        if (i % 2 == 0) {
            sz = csin(m2(z, z));
            z = z + cdiv(z - sz, z + sz);
        } else {
            z = m2(z, z);
        }
        
        arg += fabs((carg(z) - carg(zPrev)) / pow(cmod(z), 2));
        prevDist = dist;
        dist = cmod(z);
        minDist = fmin(minDist, dist);
        
        if (dist > thr) {
            break;
        }
        
        if (z.x != z.x) {
            z = zPrev;
            dist = prevDist;
            break;
        }
	}
	
	if (i > maxIter - 2) {	
        data[index + 0] = 0;
        data[index + 1] = 0;
        data[index + 2] = 0;
        
	    return;
	}
	
	index2 = 0;
// 	index2 += minDist / starFactor + fabs((i + 1 - 0.1*log(log(sqrt(dist)))/log(thr))) / 2 + (carg(z) + M_PI) * 30) % nColours);
	index2 += 8 * i;
	index2 += fabs(log(minDist)) / starFactor;
	index2 += log(log(sqrt(dist))) / sqrt(thr) * 150;
	index2 += sqrt(arg / M_PI * 180.);
	
	index2 = 3 * ((int)index2 % nColours);
	
	struct Color col;
	
	col.r = map[index2 + 0];
	col.g = map[index2 + 1];
	col.b = map[index2 + 2];
	
// 	double theta = carg(z) / M_PI * 180;
// 	col = rotateHue(col, theta);
// 	col = shade2(col, z, dz, v, h);
	
	data[index + 0] = col.r * 4294967295;
	data[index + 1] = col.g * 4294967295;
	data[index + 2] = col.b * 4294967295;
}

// __kernel void newtonns(global float *roots, global float *map, int nColours, global unsigned int *data, 
//     double scale, double dx, double dy, int _nRoots)
// {
// 	const int x = get_global_id(0);
// 	const int y = get_global_id(1);
// 	
// 	const int W = get_global_size(0);
// 	const int H = get_global_size(1);
// 	
// 	char nRoots = (char)_nRoots;
// 	
// 	int i;
// 	int index, index2;
// 	index = 3 * (W * y + x);
// 	
// 	double scale2 = 1. / H * scale;
// 	
// 	cfloat z = ((cfloat)(x * scale2 + dx - scale * 0.5 * W / H, y * scale2 + dy - scale * 0.5));
// 	cfloat dz = ((cfloat)(1, 0));
// 	cfloat croots[20];
// 	
// 	double theta = 0. * M_PI / 4.;
// 	cfloat v = ((cfloat)(cos(theta), sin(theta)));
// 	double h = 1;
// 	
// 	cfloat f0, f1, f2;
// 	
// 	for (i=0; i<nRoots; i++) {
// 	    croots[i].x = roots[2*i];
// 	    croots[i].y = roots[2*i+1];
// 	}
// 	
// 	double dist, prevDist;
// 	double thr = 1e-6;
// 	
// 	double minDist = 1000;
// 	int minLoc = 2;
// 	
// 	for (i=0; i<2; i++) {
//         f0 = funcn(z, croots, nRoots);
//         f1 = derivn(z, croots, nRoots);
//         f2 = deriv2n(z, croots, nRoots);
//         
//         z = step2(z, f0, f1);
//         dz = m2(dz, cdiv(m2(f0, f2), m2(f1, f1)));
// 	    
// 	    for (int j=0; j<nRoots; j++) {
// 	        dist = cmod(z - croots[j]);
// 	        
// 	        if (dist < minDist) {
//                 prevDist = minDist;
// 	            minDist = dist;
// 	            minLoc = j;
// 	        }
// 	    }
// 	        
//         if (minDist < thr) {
//             break;
//         }
// 	}
// 	
// 	index2 = 3 * (nColours - 1);
// // 	index2 = 3 * ((int)(10. * (i - 0. * (log(thr) - log(minDist)) / (log(prevDist) - log(minDist)))) % nColours);
// // 	index2 = 3 * (int)(minLoc / (double)(nRoots) * nColours);
// 	
// 	struct Color col;
// 	
// 	col.r = map[index2 + 0];
// 	col.g = map[index2 + 1];
// 	col.b = map[index2 + 2];
// 	
// 	theta = 0.1 * M_PI * minLoc + 0.;
// 	struct Matrix mat = hueRotation(theta);
// 	col = applyMat(mat, col);
// 	col = applyShade(col, z, dz, v, h);
// 	
// 	data[index + 0] = col.r * 4294967295;
// 	data[index + 1] = col.g * 4294967295;
// 	data[index + 2] = col.b * 4294967295;
// }
