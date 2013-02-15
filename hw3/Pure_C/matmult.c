// This program implements a vector addition using OpenCL
// Error checking added by Kate Cowles

// System includes
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#include <CL/cl.h>
#include <time.h>

#define BLOCKSIZE 32

// Simple OpenCL error checking function
void chk(cl_int status, const char* cmd) {

   if(status != CL_SUCCESS) {
      printf("%s failed (%d)\n", cmd, status);
      exit(-1);
   }
}

char* readSource(char* kernelPath)
{
    cl_int status;
    FILE* fp;
    char* source;
    long int size;
    printf("Kernel file is: %s\n", kernelPath);
    fp = fopen(kernelPath, "rb");
    if (!fp)
    {
        printf("Couldn't Open Kernel File\n");
        exit(-1);
    }
    status = fseek(fp, 0, SEEK_END);
    if (status != 0)
    {
        printf("Error seeking end of file\n");
        exit(-1);
    }
    size = ftell(fp);
    if (size<0)
    {
        printf("Error getting file position\n");
        exit(-1);
    }

    rewind(fp);

    source = (char *) malloc(size + 1);
    int i;
    for (i = 0; i< size + 1; i++)
    {
        source[i] = '\0';

    }
    if (source == NULL)
    {
        printf("Error allocating space for kernel source");
        exit(-1);
    }
    fread(source,1,size, fp);
    source[size] = '\0';
    return(source);
}

float* readDataFile(char fn[], int *mnum, int *nnum ){
      /* first row of input file contains two ints: num rows and num cols */
      /* rest of input file is data vals in row major order */
      int status;
      int size;
      float *data;
      int m,n,i;
      FILE* fp;
      fp=fopen(fn,"r");
      if (!fp)
      {
          printf("Error opening file\n");
      }
      status = fseek(fp, 0, SEEK_END);
      if (status != 0)
      {
          printf("Error seeking end of file\n");
          exit(-1);
      }
      size = ftell(fp);
      if (size<0)
      {
          printf("Error getting file position\n");
          exit(-1);
      }

      rewind(fp);

      fscanf(fp,"%d %d",&m, &n);
      printf("Rows: %d, Columns:  %d\n", m, n);
      data=malloc(sizeof(float)*m*n); //
      for (i=0;i<m*n;i++)
      fscanf(fp,"%f",&data[i]);
      fclose(fp);
      *mnum = m;  // store 'm' where the caller can see it
      *nnum = n;  // store 'n' where the caller can see it
      return(data);
}

// CPU-based matrix multiplication
//
void simpleMultiplyCPU( float *C, int widthA, int heightA, int widthB,
    int heightB, float *A, float *B)
{
    int i, j, k;
    for (i=0; i < heightA; i++)
    {
        for (j = 0; j<  widthB; j++)
        {
            C[i * widthB + j] = 0.0 ;  
            for( k = 0 ; k < widthA; k++)
            {
                C[i * widthB + j] += A[i * widthA + k] * B[k * widthB + j] ;
            }
        }
    }
}

// OpenCL kernel to perform an element-wise addition

int main(int argc, char** argv) {
    // This code executes on the OpenCL host
    
    const char* programSource = readSource("./matmult.kernel");
    // Host data
    int* Arows = (int *) malloc(sizeof(int));
    int* Acols = (int *) malloc(sizeof(int));
    int* Brows = (int *) malloc(sizeof(int));
    int* Bcols = (int *) malloc(sizeof(int));
;

    float* A = readDataFile(argv[1], Arows, Acols);  // Input array
    float* B = readDataFile(argv[2], Brows, Bcols);  // Input array
    int CPU;
    int VERIFY;
    printf("%d\n", argc);
    if (argc <= 3)
    {

        CPU = 1;
        VERIFY = 1;
    }
    else if (argc == 4)
    {

        CPU = ((*argv[3]) == 'T');
        VERIFY = 1;
    }
    else
    {

        CPU = 1*( *argv[3] == 'T');
        VERIFY = 1*( *argv[4] == 'T');
    }
    if (CPU)
    {
        printf("CPU On\n");
        if (VERIFY)
        {
            printf("Verify on\n");
        }
    }

    int Adatasize = sizeof(float)*(*Arows)*(*Acols);
    int Bdatasize = sizeof(float)*(*Brows)*(*Bcols);
    int Cdatasize = sizeof(float)*(*Arows)*(*Bcols);


    float* C = (float*) malloc(Cdatasize);  // Output array
    float* C_cpu = (float*) malloc(Cdatasize);  // Output array

    
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
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, Adatasize, NULL, &status);
    chk(status, "clCreateBuffer");

    // Create a buffer object that will contain the data 
    // from the host array B
    cl_mem bufB;
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, Bdatasize, NULL, &status);
    chk(status, "clCreateBuffer");

    // Create a buffer object that will hold the output data
    cl_mem bufC;
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, Cdatasize,
        NULL, &status); 
    chk(status, "clCreateBuffer");
    
    // Write input array A to the device buffer bufferA
    status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 
        0, Adatasize, A, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer");    
    // Write input array B to the device buffer bufferB
    status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 
        0, Bdatasize, B, 0, NULL, NULL);

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
    kernel = clCreateKernel(program, "matmult", &status);
    chk(status, "clCreateKernel");


    // Associate the input and output buffers with the kernel 
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufC);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufA);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufB);
    status |= clSetKernelArg(kernel, 3, sizeof(int), Arows);
    status |= clSetKernelArg(kernel, 4, sizeof(int), Brows);
    status |= clSetKernelArg(kernel, 5, sizeof(int), Acols);
    status |= clSetKernelArg(kernel, 6, sizeof(int), Bcols);




    chk(status, "clSetKernelArg");

    // Define an index space (global work size) of work 
    // items for execution. A workgroup size (local work size) 
    // is not required, but can be used.
    size_t globalWorkSize[2] ;   
 
    // There are 'elements' work-items 
    globalWorkSize[0] = *Bcols;
    globalWorkSize[1] = *Arows;

    // Enqueue the kernel for execution
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, 
        globalWorkSize, NULL, 0, NULL, NULL);
    chk(status, "clEnqueueNDRangeKernel");


    // Read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, 
        Cdatasize, C, 0, NULL, NULL);
    chk(status, "clEnqueueReadBuffer");


    // Verify the output
    if (CPU)
    {
        printf("Doing cpu multiplication...\n");
        simpleMultiplyCPU(C_cpu, *Acols, *Arows, *Bcols,*Brows, A, B);
        printf("...done\n");
        if (VERIFY)
        {
            int result = 1;
            int idx;
            double diff;
            for (idx = 0; idx < (*Arows)*(*Bcols); idx ++)
            {
                diff = C[idx] - C_cpu[idx];
                if (diff < 0) diff *= -1;
                if (diff > 0.0001)
                {
                    result = 0;
                    printf("Breaking, index = %d\n", idx);
                    printf("Total Data size is = %d\n", (*Arows)*(*Bcols));
                    printf("OCL: %f\n", C[idx]);
                    printf("CPU: %f\n", C_cpu[idx]);

                    break;
                }
            }
            if(result) {
                printf("Output is correct\n");
            } else {
                printf("Output is incorrect\n");
            }
        }
    }

    // Free OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseContext(context);

    // Free host resources
    free(A);
    free(B);
    free(C);
    free(platforms);
    free(devices);

    return 0;
}
