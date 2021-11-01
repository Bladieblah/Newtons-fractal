#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <ctime>
#include <chrono>
#include <math.h>
#include <vector>
#include <complex.h>
#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

#include "colour.hpp"

// For loading shader files
#define MAX_SOURCE_SIZE (0x100000)

// Window size
#define size_x 2400
#define size_y 1300

// OpenCL initialisation
cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue command_queue = NULL;

cl_mem rootmobj = NULL;
cl_mem mapmobj = NULL;
cl_mem datamobj = NULL;

cl_program program = NULL;
cl_kernel kernel = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;

size_t source_size;
char *source_str;

// Array to be drawn
unsigned int data[size_y*size_x*3];

// Colourmap stuff
float *colourMap;
float *pointMap;
int nColours = 384/2;

// Kernel size for parallelisation
size_t global_item_size[2] = {(size_t)size_x, (size_t)size_y};
size_t local_item_size[2] = {(size_t)size_x, (size_t)size_y};

// Roots
float *roots;
int nRoots = 3;
int rootIndex;

// Positioning
double scale = 2;
double scale2;
double dx = 0.;
double dy = 0.;

int drawRoots = 0;

const char fractal[3][10] = {"newtonn", "mandel", "wave"};
int frindex = 2;

double threshold = 10.;

// Mouse
int mouse_x, mouse_y;
int zoom = 1;
int mouseState;

std::complex<double> getRoot(int i) {
    std::complex<double> root (roots[2*i], roots[2*i + 1]);
    return root;
}

std::complex<double> newtonStep(std::complex<double> z) {
    std::complex<double> fn;
    std::complex<double> df, df_;
    
    fn = z - getRoot(0);
    for (int i=1; i<nRoots; i++) {
        fn *= z - getRoot(i);
    }
    
    df = 0;
    for (int i=0; i<nRoots; i++) {
        df_ = z - getRoot(i);
        for (int j=0; j<nRoots-1; j++) {
            df_ *= z - getRoot((i + j) % nRoots);
        }
        df += df_;
    }
    
    return z - fn / df;
    
}

std::complex<double> mandelStep(std::complex<double> z, std::complex<double> c) {
    return z*z + c;
}

std::complex<double> waveStep(std::complex<double> z, int i) {
    if (i % 2 == 0) {
        std::complex<double> sz = sin(z*z);
        return z + (z - sz) / (z + sz);
    }
    
    return z * z;
}

std::complex<double> fractalStep(std::complex <double> z, std::complex<double> c, int i) {
    switch (frindex) {
        case 0:
            return newtonStep(z);
            break;
        case 1:
            return mandelStep(z, c);
            break;
        case 2:
            return waveStep(z, i);
            break;
    }
    
    return waveStep(z, i);
}

// Functions

void makeColourmap() {
    std::vector<float> x = {0., 0.2, 0.4, 0.6, 0.8, 1.};
    std::vector< std::vector<float> > y = {
        {26,17,36},
        {33,130,133},
        {26,17,36},
        {200,40,187},
        {241, 249, 244},
        {26,17,36}
    };
    Colour bluePink(x, y, nColours);
    
    // Inferno
    x = {
        0.000,
        0.004,
        0.008,
        0.012,
        0.016,
        0.020,
        0.024,
        0.028,
        0.032,
        0.036,
        0.040,
        0.043,
        0.047,
        0.051,
        0.055,
        0.059,
        0.063,
        0.067,
        0.071,
        0.075,
        0.079,
        0.083,
        0.086,
        0.090,
        0.094,
        0.098,
        0.102,
        0.106,
        0.110,
        0.114,
        0.118,
        0.122,
        0.125,
        0.129,
        0.133,
        0.137,
        0.141,
        0.145,
        0.149,
        0.153,
        0.157,
        0.161,
        0.165,
        0.168,
        0.172,
        0.176,
        0.180,
        0.184,
        0.188,
        0.192,
        0.196,
        0.200,
        0.204,
        0.208,
        0.211,
        0.215,
        0.219,
        0.223,
        0.227,
        0.231,
        0.235,
        0.239,
        0.243,
        0.247,
        0.250,
        0.254,
        0.258,
        0.262,
        0.266,
        0.270,
        0.274,
        0.278,
        0.282,
        0.286,
        0.290,
        0.293,
        0.297,
        0.301,
        0.305,
        0.309,
        0.313,
        0.317,
        0.321,
        0.325,
        0.329,
        0.333,
        0.336,
        0.340,
        0.344,
        0.348,
        0.352,
        0.356,
        0.360,
        0.364,
        0.368,
        0.372,
        0.375,
        0.379,
        0.383,
        0.387,
        0.391,
        0.395,
        0.399,
        0.403,
        0.407,
        0.411,
        0.415,
        0.418,
        0.422,
        0.426,
        0.430,
        0.434,
        0.438,
        0.442,
        0.446,
        0.450,
        0.454,
        0.458,
        0.461,
        0.465,
        0.469,
        0.473,
        0.477,
        0.481,
        0.485,
        0.489,
        0.493,
        0.497,
        0.500,
        0.504,
        0.508,
        0.512,
        0.516,
        0.520,
        0.524,
        0.528,
        0.532,
        0.536,
        0.540,
        0.543,
        0.547,
        0.551,
        0.555,
        0.559,
        0.563,
        0.567,
        0.571,
        0.575,
        0.579,
        0.583,
        0.586,
        0.590,
        0.594,
        0.598,
        0.602,
        0.606,
        0.610,
        0.614,
        0.618,
        0.622,
        0.625,
        0.629,
        0.633,
        0.637,
        0.641,
        0.645,
        0.649,
        0.653,
        0.657,
        0.661,
        0.665,
        0.668,
        0.672,
        0.676,
        0.680,
        0.684,
        0.688,
        0.692,
        0.696,
        0.700,
        0.704,
        0.708,
        0.711,
        0.715,
        0.719,
        0.723,
        0.727,
        0.731,
        0.735,
        0.739,
        0.743,
        0.747,
        0.750,
        0.754,
        0.758,
        0.762,
        0.766,
        0.770,
        0.774,
        0.778,
        0.782,
        0.786,
        0.790,
        0.793,
        0.797,
        0.801,
        0.805,
        0.809,
        0.813,
        0.817,
        0.821,
        0.825,
        0.829,
        0.833,
        0.836,
        0.840,
        0.844,
        0.848,
        0.852,
        0.856,
        0.860,
        0.864,
        0.868,
        0.872,
        0.875,
        0.879,
        0.883,
        0.887,
        0.891,
        0.895,
        0.899,
        0.903,
        0.907,
        0.911,
        0.915,
        0.918,
        0.922,
        0.926,
        0.930,
        0.934,
        0.938,
        0.942,
        0.946,
        0.950,
        0.954,
        0.958,
        0.961,
        0.965,
        0.969,
        0.973,
        0.977,
        0.981,
        0.985,
        0.989,
        0.993,
        0.997,
    };
    y = {
        {0,0,3},
        {0,0,4},
        {0,0,6},
        {1,0,7},
        {1,1,9},
        {1,1,11},
        {2,1,14},
        {2,2,16},
        {3,2,18},
        {4,3,20},
        {4,3,22},
        {5,4,24},
        {6,4,27},
        {7,5,29},
        {8,6,31},
        {9,6,33},
        {10,7,35},
        {11,7,38},
        {13,8,40},
        {14,8,42},
        {15,9,45},
        {16,9,47},
        {18,10,50},
        {19,10,52},
        {20,11,54},
        {22,11,57},
        {23,11,59},
        {25,11,62},
        {26,11,64},
        {28,12,67},
        {29,12,69},
        {31,12,71},
        {32,12,74},
        {34,11,76},
        {36,11,78},
        {38,11,80},
        {39,11,82},
        {41,11,84},
        {43,10,86},
        {45,10,88},
        {46,10,90},
        {48,10,92},
        {50,9,93},
        {52,9,95},
        {53,9,96},
        {55,9,97},
        {57,9,98},
        {59,9,100},
        {60,9,101},
        {62,9,102},
        {64,9,102},
        {65,9,103},
        {67,10,104},
        {69,10,105},
        {70,10,105},
        {72,11,106},
        {74,11,106},
        {75,12,107},
        {77,12,107},
        {79,13,108},
        {80,13,108},
        {82,14,108},
        {83,14,109},
        {85,15,109},
        {87,15,109},
        {88,16,109},
        {90,17,109},
        {91,17,110},
        {93,18,110},
        {95,18,110},
        {96,19,110},
        {98,20,110},
        {99,20,110},
        {101,21,110},
        {102,21,110},
        {104,22,110},
        {106,23,110},
        {107,23,110},
        {109,24,110},
        {110,24,110},
        {112,25,110},
        {114,25,109},
        {115,26,109},
        {117,27,109},
        {118,27,109},
        {120,28,109},
        {122,28,109},
        {123,29,108},
        {125,29,108},
        {126,30,108},
        {128,31,107},
        {129,31,107},
        {131,32,107},
        {133,32,106},
        {134,33,106},
        {136,33,106},
        {137,34,105},
        {139,34,105},
        {141,35,105},
        {142,36,104},
        {144,36,104},
        {145,37,103},
        {147,37,103},
        {149,38,102},
        {150,38,102},
        {152,39,101},
        {153,40,100},
        {155,40,100},
        {156,41,99},
        {158,41,99},
        {160,42,98},
        {161,43,97},
        {163,43,97},
        {164,44,96},
        {166,44,95},
        {167,45,95},
        {169,46,94},
        {171,46,93},
        {172,47,92},
        {174,48,91},
        {175,49,91},
        {177,49,90},
        {178,50,89},
        {180,51,88},
        {181,51,87},
        {183,52,86},
        {184,53,86},
        {186,54,85},
        {187,55,84},
        {189,55,83},
        {190,56,82},
        {191,57,81},
        {193,58,80},
        {194,59,79},
        {196,60,78},
        {197,61,77},
        {199,62,76},
        {200,62,75},
        {201,63,74},
        {203,64,73},
        {204,65,72},
        {205,66,71},
        {207,68,70},
        {208,69,68},
        {209,70,67},
        {210,71,66},
        {212,72,65},
        {213,73,64},
        {214,74,63},
        {215,75,62},
        {217,77,61},
        {218,78,59},
        {219,79,58},
        {220,80,57},
        {221,82,56},
        {222,83,55},
        {223,84,54},
        {224,86,52},
        {226,87,51},
        {227,88,50},
        {228,90,49},
        {229,91,48},
        {230,92,46},
        {230,94,45},
        {231,95,44},
        {232,97,43},
        {233,98,42},
        {234,100,40},
        {235,101,39},
        {236,103,38},
        {237,104,37},
        {237,106,35},
        {238,108,34},
        {239,109,33},
        {240,111,31},
        {240,112,30},
        {241,114,29},
        {242,116,28},
        {242,117,26},
        {243,119,25},
        {243,121,24},
        {244,122,22},
        {245,124,21},
        {245,126,20},
        {246,128,18},
        {246,129,17},
        {247,131,16},
        {247,133,14},
        {248,135,13},
        {248,136,12},
        {248,138,11},
        {249,140,9},
        {249,142,8},
        {249,144,8},
        {250,145,7},
        {250,147,6},
        {250,149,6},
        {250,151,6},
        {251,153,6},
        {251,155,6},
        {251,157,6},
        {251,158,7},
        {251,160,7},
        {251,162,8},
        {251,164,10},
        {251,166,11},
        {251,168,13},
        {251,170,14},
        {251,172,16},
        {251,174,18},
        {251,176,20},
        {251,177,22},
        {251,179,24},
        {251,181,26},
        {251,183,28},
        {251,185,30},
        {250,187,33},
        {250,189,35},
        {250,191,37},
        {250,193,40},
        {249,195,42},
        {249,197,44},
        {249,199,47},
        {248,201,49},
        {248,203,52},
        {248,205,55},
        {247,207,58},
        {247,209,60},
        {246,211,63},
        {246,213,66},
        {245,215,69},
        {245,217,72},
        {244,219,75},
        {244,220,79},
        {243,222,82},
        {243,224,86},
        {243,226,89},
        {242,228,93},
        {242,230,96},
        {241,232,100},
        {241,233,104},
        {241,235,108},
        {241,237,112},
        {241,238,116},
        {241,240,121},
        {241,242,125},
        {242,243,129},
        {242,244,133},
        {243,246,137},
        {244,247,141},
        {245,248,145},
        {246,250,149},
        {247,251,153},
        {249,252,157},
        {250,253,160},
        {252,254,164},
    };
    Colour inferno(x, y, nColours);
    
//     std::vector<float> x = {0., 0.3333333, 0.6666666, 1.};
//     std::vector< std::vector<float> > y = {
//         {0, 0, 64},
//         {0, 255, 192},
//         {64, 192, 0},
//         {0, 0, 64}
//     };
    std::vector<float> x2 = {0., 0.5, 1.};
    std::vector< std::vector<float> > y2 = {
        {255, 100, 0},
        {0, 255, 100},
        {100, 0, 255}
    };

//     std::vector<float> x = {0., 1.};
//     std::vector< std::vector<float> > y = {
//         {100, 0, 0},
// //         {255,255,255},
//         {255,0,0}
//     };

    Colour col(x, y, nColours);
    Colour col2(x2, y2, nColours);
    
    colourMap = (float *)malloc(3 * nColours * sizeof(float));
    pointMap = (float *)malloc(3 * nColours * sizeof(float));
    
    inferno.apply(colourMap);
    col2.apply(pointMap);
    
    // Write colourmap to GPU
    ret = clEnqueueWriteBuffer(command_queue, mapmobj, CL_TRUE, 0, 3*nColours*sizeof(float), colourMap, 0, NULL, NULL);
}

void setKernelArgs() {
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&rootmobj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mapmobj);
    ret = clSetKernelArg(kernel, 2, sizeof(int), &nColours);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&datamobj);
    
    ret = clSetKernelArg(kernel, 4, sizeof(double), &scale);
    ret = clSetKernelArg(kernel, 5, sizeof(double), &dx);
    ret = clSetKernelArg(kernel, 6, sizeof(double), &dy);
    
    ret = clSetKernelArg(kernel, 7, sizeof(int), &nRoots);
    ret = clSetKernelArg(kernel, 8, sizeof(double), &threshold);
}

void initData() {
//     float offset[10] = {0.1, 0, -0.1, 0.2, 0.4, 0, 0.1, 0.2, -0.1, 0.3};
//     float theta;
//     
//     for (int i=0; i<nRoots; i++) {
//         theta = 2 * M_PI / nRoots * i;// + offset[i];
//         roots[2*i] = cos(theta) / 2.;
//         roots[2*i + 1] = sin(theta) / 2.;
//     }

//     float a = 0.5;
//     float b = 0.1;
//     float c = 0.25;
//     float d = 0.1;
//     
//     for (int i=0; i<nRoots; i++) {
//         roots[2*i + 0] = c * i + d;
//         roots[2*i + 1] = c * (a * i * i * i + b);
//     }

    roots[0] = 1; roots[1] = 0;
    roots[2] = 0; roots[3] = 1;
    roots[4] = -1; roots[5] = 0;
    
    for (int i=0; i<nRoots; i++) {
        fprintf(stderr, "(%.2f, %.2f)\n", roots[2*i], roots[2*i+1]);
    }
    
    scale = 2; dx = 0; dy = 0;
    
    if (frindex != 0) {
        nRoots = 200;
    }

//     scale = 0.000000125169754;
//     dx = 0.374839144945144;
//     dy = 0.095639694956283;
//     
//     dx = 0.374839140798777;
//     dy = 0.095639681189519; 
//     scale = 0.000000002115452;
//     
    dx = 0.824615384615;
    dy = -0.243076923077; 
    scale = 0.5;
}

void prepare() {
    FILE *fp;
    const char fileName[] = "./shaders/sample.cl";
    
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    
    srand(time(NULL));
    
    roots = (float *)malloc(2 * nRoots * sizeof(float));
    
    initData();
    
    /* Get Platform/Device Information */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    /* Create OpenCL Context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS)
      fprintf(stderr, "Failed on function clCreateContext: %d\n", ret);

    /* Create command queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Create Buffer Object */
    rootmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*nRoots*sizeof(float), NULL, &ret);
    mapmobj  = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*nColours*sizeof(float), NULL, &ret);
    datamobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*size_x*size_y*sizeof(unsigned int), NULL, &ret);

    /* Create kernel program from source file*/
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    if (ret != CL_SUCCESS)
      fprintf(stderr, "Failed on function clCreateProgramWithSource: %d\n", ret);
    
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS)
      fprintf(stderr, "Failed on function clBuildProgram: %d\n", ret);
    
    size_t len = 10000;
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char *buffer = (char *)calloc(len, sizeof(char));
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    fprintf(stderr, "%s\n", buffer);

    /* Create data parallel OpenCL kernel */
//     kernel = clCreateKernel(program, "newtonn", &ret);
    kernel = clCreateKernel(program, fractal[frindex], &ret);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "Failed on function clCreateKernel: %d\n", ret);
    
    setKernelArgs();
}

void cleanup() {
    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    
    ret = clReleaseMemObject(rootmobj);
    ret = clReleaseMemObject(mapmobj);
    ret = clReleaseMemObject(datamobj);
    
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(roots);
    free(colourMap);
    
    free(source_str);
}

void step() {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    ret = clEnqueueWriteBuffer(command_queue, rootmobj, CL_TRUE, 0, 2*nRoots*sizeof(float), roots, 0, NULL, NULL);
    
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Failed on function clEnqueueNDRangeKernel: %d\n", ret);
    }
    
    ret = clEnqueueReadBuffer(command_queue, datamobj, CL_TRUE, 0, 3*size_x*size_y*sizeof(unsigned int), data, 0, NULL, NULL);
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    fprintf(stderr, "step time = %.3g\n", time_span.count());
}

void setRoot(int x, int y) {
    scale2 = 1. / size_y * scale;
    double xpos = x * scale2 + dx - scale * 0.5 * size_x / size_y;
    double ypos = (size_y - y) * scale2 + dy - scale * 0.5;
    
    if (rootIndex != -1) {
        roots[2*rootIndex] = xpos;
        roots[2*rootIndex + 1] = ypos;
        
        step();
    }
}

void drawPath() {
    int k;
    double x, y;
    scale2 = 1. / size_y * scale;
    double xpos = mouse_x * scale2 + dx - scale * 0.5 * size_x / size_y;
    double ypos = (size_y - mouse_y) * scale2 + dy - scale * 0.5;
    
    int N = nRoots;
    
    std::complex<double> z;
    std::complex<double> c (xpos, ypos);
    
    if (frindex == 1) {
        z = 0;
    } else {
        z = c;
    }
    
    x = 2 * (c.real() - dx) / scale * size_y / (double)size_x;
    y = 2 * (c.imag() - dy) / scale;
    
    glColor4f(1.,1.,1.,1.);
    glBegin(GL_LINE_STRIP);
    
    for (k=0; k<N; k++) {
        glVertex2f(x, y);
        x = 2 * (z.real() - dx) / scale * size_y / (double)size_x;
        y = 2 * (z.imag() - dy) / scale;
        glVertex2f(x, y);
        
        z = fractalStep(z, c, k);
    }
    fprintf(stderr, "\n");
    
    glEnd();
    
    z = {xpos, ypos};
    
    glPointSize(3);
    glEnable(GL_POINT_SMOOTH);
    
    glBegin(GL_POINTS);
    
    for (k=0; k<nColours; k++) {
        x = 2 * (z.real() - dx) / scale * size_y / (float)size_x;
        y = 2 * (z.imag() - dy) / scale;
                
        glColor4f(pointMap[3*k + 0],pointMap[13*k + 1],pointMap[3*k + 2],1.);
        glVertex2f(x, y);
        
        z = fractalStep(z, c, k);
    }
    
    glEnd();
    
    glColor4f(1,1,1,1);
}

void display() {
    setRoot(mouse_x, mouse_y);
    
    glClearColor( 0, 0, 0, 1 );
    glClear( GL_COLOR_BUFFER_BIT );

    glEnable (GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D (
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        size_x,
        size_y,
        0,
        GL_RGB,
        GL_UNSIGNED_INT,
        &data[0]
    );

    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0,  1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0,  1.0);
    glEnd();
    
    glDisable (GL_TEXTURE_2D);

    glFlush();
    
    // Draw points at roots
    if (drawRoots && frindex == 0) {
        glPointSize(10);
        glColor4f(1.,1.,1.,1.);
        glEnable(GL_POINT_SMOOTH);
    
        glBegin(GL_POINTS);
        float x, y;
        for (int i=0; i<nRoots; i++) {
            x = 2 * (roots[2*i] - dx) / scale * size_y / (float)size_x;
            y = 2 * (roots[2*i+1] - dy) / scale;
            glVertex2f (x, y);
        }
    
        glEnd();
    }
    
    if (!zoom) {
        drawPath();
    }
    
    glutSwapBuffers();
}

void key_pressed(unsigned char key, int x, int y) {
    switch (key)
    {
        case 'w':
            scale /= 2.;
            ret = clSetKernelArg(kernel, 4, sizeof(double), &scale);
            break;
        case 's':
            scale *= 2.;
            ret = clSetKernelArg(kernel, 4, sizeof(double), &scale);
            break;
        case 'p':
            fprintf(stderr, "(xc, xy) = (%.12g, %.12g), scale = %.12g, thr = %.12g\n", dx, dy, scale, threshold);
            break;
        case 'r':
            initData();
            setKernelArgs();
            break;
        case 'm':
            drawRoots = 1 - drawRoots;
            break;
        case 'd':
            nRoots *= 2;
            ret = clSetKernelArg(kernel, 7, sizeof(int), &nRoots);
            break;
        case 'a':
            nRoots /= 2;
            ret = clSetKernelArg(kernel, 7, sizeof(int), &nRoots);
            break;
        case 'q':
        	cleanup();
        	fprintf(stderr, "\n");
            exit(0);
            break;
        case 'f':
            threshold /= 1.01;
            ret = clSetKernelArg(kernel, 8, sizeof(double), &threshold);
            break;
        case 'g':
            threshold *= 1.01;
            ret = clSetKernelArg(kernel, 8, sizeof(double), &threshold);
            break;
        case 'z':
            zoom = 1 - zoom;
            break;
        default:
            break;
    }
    
    step();
}

void mouseFunc(int button, int state, int x,int y) {
    scale2 = 1. / size_y * scale;
    double xpos = x * scale2 + dx - scale * 0.5 * size_x / size_y;
    double ypos = (size_y - y) * scale2 + dy - scale * 0.5;
    
    rootIndex = -1;
    mouseState = state;
    
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
	    fprintf(stderr, "(x, y) = (%.3g, %.3g)\n", xpos, ypos);
	    float dist;
	    
	    for (int i=0; i<nRoots; i++) {
	        dist = sqrt(pow(xpos - roots[2*i], 2) + pow(ypos - roots[2*i+1], 2));
	        if (dist / scale * size_y < 20) {
	            rootIndex = i;
	            break;
	        }
	    }
	    
	    if ((frindex != 0 || rootIndex == -1) && zoom) {
        
            dx = xpos;
            dy = ypos;
        
            ret = clSetKernelArg(kernel, 5, sizeof(double), &dx);
            ret = clSetKernelArg(kernel, 6, sizeof(double), &dy);
        
            step();
        }
	}
}

void motionFunc(int x, int y) {
    mouse_x = x;
    mouse_y = y;
}

int main(int argc, char **argv) {
    prepare();
    makeColourmap();
    
	glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( size_x, size_y );
    glutCreateWindow( "Hello World" );
    glutDisplayFunc( display );
    
    glutDisplayFunc(&display);
    glutIdleFunc(&display);
    glutKeyboardUpFunc(&key_pressed);
    glutMouseFunc(mouseFunc);
    glutMotionFunc(&motionFunc);
    
    mouse_x = 0; mouse_y = 0;
    rootIndex = -1;
    
    step();
    display();
    glutMainLoop();

    return 0;
}	