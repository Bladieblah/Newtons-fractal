#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
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
#define size_x 2560
#define size_y 1440

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
int nColours = 255;

// Kernel size for parallelisation
size_t global_item_size[2] = {(size_t)size_x, (size_t)size_y};
size_t local_item_size[2] = {(size_t)size_x, (size_t)size_y};

// Roots
float *roots;
int nRoots = 4;

// Positioning
float scale = 0.5;
float scale2;
float dx = 0.5;
float dy = 0.5;

// Functions

void makeColourmap() {
    std::vector<float> x = {0., 0.2, 0.4, 0.7, 1.};
    std::vector< std::vector<float> > y = {
        {26,17,36},
        {33,130,133},
        {26,17,36},
        {200,40,187},
        {241, 249, 244}
    };
    
//     std::vector<float> x = {0., 1.};
//     std::vector< std::vector<float> > y = {
//         {0,0,0},
//         {200, 200, 200}
//     };

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
    
    ret = clSetKernelArg(kernel, 4, sizeof(float), &scale);
    ret = clSetKernelArg(kernel, 5, sizeof(float), &dx);
    ret = clSetKernelArg(kernel, 6, sizeof(float), &dy);
}

void initData() {
    float offset[4] = {0.1, 0, -0.1, 0.2};
    float theta;
    
    for (int i=0; i<nRoots; i++) {
        theta = 2 * M_PI / nRoots * i + offset[i];
        roots[2*i] = cos(theta) / 2.3 + 0.5;
        roots[2*i + 1] = sin(theta) / 2.2 + 0.5;
    }
    
    for (int i=0; i<nRoots; i++) {
        fprintf(stderr, "(%.2f, %.2f)\n", roots[2*i], roots[2*i+1]);
    }
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
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    
    size_t len = 10000;
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char *buffer = (char *)calloc(len, sizeof(char));
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    fprintf(stderr, "%s\n", buffer);

    /* Create data parallel OpenCL kernel */
    kernel = clCreateKernel(program, "newton4", &ret);
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
    ret = clEnqueueWriteBuffer(command_queue, rootmobj, CL_TRUE, 0, 2*nRoots*sizeof(float), roots, 0, NULL, NULL);
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, datamobj, CL_TRUE, 0, 3*size_x*size_y*sizeof(unsigned int), data, 0, NULL, NULL);
}

void display() {
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

    glFlush();
    glutSwapBuffers();
    
    step();
}

void key_pressed(unsigned char key, int x, int y) {
    switch (key)
    {
        case 'w':
            scale /= 2.;
            ret = clSetKernelArg(kernel, 4, sizeof(float), &scale);
            break;
        case 's':
            scale *= 2.;
            ret = clSetKernelArg(kernel, 4, sizeof(float), &scale);
            break;
        case 'p':
            glutIdleFunc(&display);
            break;
        case 'e':
            glutPostRedisplay();
            break;
        case 'r':
            initData();
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

// x * scale2 + dx - scale * 0.5 * W / H, 
// y * scale2 + dy - scale * 0.5

void mouseFunc(int button, int state, int x,int y) {
    scale2 = 1. / size_y * scale;
    float xpos = x * scale2 + dx - scale * 0.5 * size_x / size_y;
    float ypos = (size_y - y) * scale2 + dy - scale * 0.5;
    
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
	    fprintf(stderr, "%f, %f\n", xpos, ypos);
		
		dx = xpos;
		dy = ypos;
		
        ret = clSetKernelArg(kernel, 5, sizeof(float), &dx);
        ret = clSetKernelArg(kernel, 6, sizeof(float), &dy);
	}
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
    
    display();
    glutMainLoop();

    return 0;
}	