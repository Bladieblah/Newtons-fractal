#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <ctime>
#include <chrono>
#include <math.h>
#include <vector>

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
#define size_x 1920
#define size_y 1080

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
int nColours = 384;

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

// Mouse
int mouse_x, mouse_y;

// Functions

void makeColourmap() {
//     std::vector<float> x = {0., 0.2, 0.4, 0.6, 0.8, 1.};
//     std::vector< std::vector<float> > y = {
//         {26,17,36},
//         {33,130,133},
//         {26,17,36},
//         {200,40,187},
//         {241, 249, 244},
//         {26,17,36}
//     };
    
//     std::vector<float> x = {0., 0.3333333, 0.6666666, 1.};
//     std::vector< std::vector<float> > y = {
//         {0, 0, 64},
//         {0, 255, 192},
//         {64, 192, 0},
//         {0, 0, 64}
//     };

    std::vector<float> x = {0., 1.};
    std::vector< std::vector<float> > y = {
        {0, 0, 0},
//         {255,255,255},
        {255,255,255}
    };

    Colour col(x, y, nColours);
    
    colourMap = (float *)malloc(3 * nColours * sizeof(float));
    
    col.apply(colourMap);
    
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
    kernel = clCreateKernel(program, "newtonn", &ret);
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
    if (drawRoots) {
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
    
    glutSwapBuffers();
}

void key_pressed(unsigned char key, int x, int y) {
    switch (key)
    {
        case 'w':
            scale /= 2.;
            ret = clSetKernelArg(kernel, 4, sizeof(double), &scale);
            step();
            break;
        case 's':
            scale *= 2.;
            ret = clSetKernelArg(kernel, 4, sizeof(double), &scale);
            step();
            break;
        case 'p':
//             glutIdleFunc(&display);
            fprintf(stderr, "(xc, xy) = (%.12g, %.12g), scale = %.12g\n", dx, dy, scale);
            break;
        case 'e':
            glutPostRedisplay();
            break;
        case 'r':
            initData();
            step();
            break;
        case 'm':
            drawRoots = 1 - drawRoots;
            break;
        case 'q':
        	cleanup();
        	fprintf(stderr, "\n");
            exit(0);
            break;
        default:
            break;
    }
}

void mouseFunc(int button, int state, int x,int y) {
    scale2 = 1. / size_y * scale;
    double xpos = x * scale2 + dx - scale * 0.5 * size_x / size_y;
    double ypos = (size_y - y) * scale2 + dy - scale * 0.5;
    
    rootIndex = -1;
    
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
	    float dist;
	    
	    for (int i=0; i<nRoots; i++) {
	        dist = sqrt(pow(xpos - roots[2*i], 2) + pow(ypos - roots[2*i+1], 2));
	        if (dist / scale * size_y < 20) {
	            rootIndex = i;
	            break;
	        }
	    }
	    
	    if (rootIndex == -1) {
        
            dx = xpos;
            dy = ypos;
        
            ret = clSetKernelArg(kernel, 5, sizeof(double), &dx);
            ret = clSetKernelArg(kernel, 6, sizeof(double), &dy);
        }
        
        step();
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