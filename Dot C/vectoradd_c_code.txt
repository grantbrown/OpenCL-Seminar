// This program implements a vector addition using OpenCL
// Error checking added by Kate Cowles

// System includes
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#include <CL/cl.h>

// Simple OpenCL error checking function
void chk(cl_int status, const char* cmd) {

   if(status != CL_SUCCESS) {
      printf("%s failed (%d)\n", cmd, status);
      exit(-1);
   }
}


// OpenCL kernel to perform an element-wise addition 
const char* programSource =
"__kernel                                            \n"
"void vecadd(__global int *A,                        \n"
"            __global int *B,                        \n"
"            __global int *C)                        \n"
"{                                                   \n"
"                                                    \n"
"   // Get the work-item’s unique ID                 \n"
"   int idx = get_global_id(0);                      \n"
"                                                    \n"
"   // Add the corresponding locations of            \n"
"   // 'A' and 'B', and store the result in 'C'.     \n"
"   C[idx] = A[idx] + B[idx];                        \n"
"}                                                   \n"
;

void execute(int* A, int* B, int* C,const int* elements) {
    // This code executes on the OpenCL host    
    // Compute the size of the data 
    size_t datasize = sizeof(int)*(*elements);

    // Use this to check the output of each API call
    cl_int status;  
     
    // Retrieve the number of platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    chk(status, "clGetPlatformIDs");

 
    // Allocate enough space for each platform
    cl_platform_id *platforms = NULL;
    platforms = (cl_platform_id*)malloc(
        numPlatforms*sizeof(cl_platform_id));
 
    // Fill in the platforms
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    // Retrieve the number of devices
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, 
        NULL, &numDevices);
    chk(status, "clGetDeviceIDs");

    // Allocate enough space for each device
    cl_device_id *devices;
    devices = (cl_device_id*)malloc(
        numDevices*sizeof(cl_device_id));

    // Fill in the devices 
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,        
        numDevices, devices, NULL);
    chk(status, "clGetDeviceIDs");

    // Create a context and associate it with the devices
    cl_context context;
    context = clCreateContext(NULL, numDevices, devices, NULL, 
        NULL, &status);
    chk(status, "clCreateContext");

    // Create a command queue and associate it with the device 
    cl_command_queue cmdQueue;
    cmdQueue = clCreateCommandQueue(context, devices[0], 0, 
        &status);
    chk(status, "clCreateCommandQueue");


    // Create a buffer object that will contain the data 
    // from the host array A
    cl_mem bufA;
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize,                       NULL, &status);
    chk(status, "clCreateBuffer");

    // Create a buffer object that will contain the data 
    // from the host array B
    cl_mem bufB;
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize,                        NULL, &status);
    chk(status, "clCreateBuffer");

    // Create a buffer object that will hold the output data
    cl_mem bufC;
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize,
        NULL, &status); 
    chk(status, "clCreateBuffer");
    
    // Write input array A to the device buffer bufferA
    status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 
        0, datasize, A, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer");    
    // Write input array B to the device buffer bufferB
    status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 
        0, datasize, B, 0, NULL, NULL);

    // Create a program with source code
    cl_program program = clCreateProgramWithSource(context, 1, 
        (const char**)&programSource, NULL, &status);
    chk(status, "clCreateProgramWithSource");

    // Build (compile) the program for the device
    // Show log if errors occur

    if (clBuildProgram(program, numDevices, devices, NULL, NULL, NULL) 
        != CL_SUCCESS) 
        {
         // Shows the log
         size_t log_size;
         // First call to get the proper size
         clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, 
            NULL, &log_size);
         char build_log[log_size+1] ;
         // Second call to get the log
         clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 
            log_size, build_log, NULL);
         build_log[log_size] = '\0';
         printf("Compile error: %s \n", build_log) ;
         free(build_log) ;
         exit(-1);
        }
  

    // Create the vector addition kernel
    cl_kernel kernel;
    kernel = clCreateKernel(program, "vecadd", &status);
    chk(status, "clCreateKernel");


    // Associate the input and output buffers with the kernel 
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    chk(status, "clSetKernelArg");

    // Define an index space (global work size) of work 
    // items for execution. A workgroup size (local work size) 
    // is not required, but can be used.
    size_t globalWorkSize[1];   
 
    // There are 'elements' work-items 
    globalWorkSize[0] = *elements;

    // Enqueue the kernel for execution
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, 
        globalWorkSize, NULL, 0, NULL, NULL);
    chk(status, "clEnqueueNDRangeKernel");


    // Read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, 
        datasize, C, 0, NULL, NULL);
    chk(status, "clEnqueueReadBuffer");


    // Verify the output
    //int result = 1;
    //int i;
    //for(i = 0; i < *elements; i++) {
    //    if(C[i] != i+i) {
    //        result = 0;
    //        break;
    //    }
    //}

    // Free OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseContext(context);

    // Free host resources
    //free(A);
    //free(B);
    //free(C);
    free(platforms);
    free(devices);

    
}
