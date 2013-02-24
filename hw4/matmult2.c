#define _CRT_SECURE_NO_WARNINGS
//What does this do?
#define NUM_KERNELS 1
#define PROGRAM_FILE "./matmult.kernel"

#include <math.h>
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <CL/cl.h>


cl_device_id create_device()
{
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        perror("Couldn't identify platform.");
        exit(1);
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
    {
        printf("Couldn't find GPU!\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err<0)
    {
        perror("Couldn't access any devices.\n");
        exit(1);
    }
    return(dev);
}


cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;
    /*Read in program*/
    program_handle = fopen(filename, "r");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*) malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);


    program = clCreateProgramWithSource(ctx,1, 
            (const char**)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't Create Program");
        exit(1);
    }
    free(program_buffer);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL,
                &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }
    return(program);
}

float* transposeDataMatrix(float* inData, int nrow, int ncol)
{
    float* newvector = (float*) malloc(sizeof(float)*nrow*ncol);
    int rw;
    int col;
    int i = 0;
    for (col = 0; col < ncol; col ++)
    {
        for (rw = 0; rw < nrow; rw ++)
        {
            newvector[i] = inData[rw*ncol + col];
            i += 1;
        }
    }
    return(newvector);
}
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


void chk(cl_int status, const char* cmd)
{
    if (status != CL_SUCCESS)
    {
        printf("%s failed (%d)\n", cmd, status);
        exit(-1);
    }
}



int main()
{

    int* Arows = (int *) malloc(sizeof(int));
    int* Acols = (int *) malloc(sizeof(int));
    int* Brows = (int *) malloc(sizeof(int));
    int* Bcols = (int *) malloc(sizeof(int));

    float* A = readDataFile("A.txt", Arows, Acols);  // Input array
    float* B1 = readDataFile("B.txt", Brows, Bcols);  // Input array
    float* B = transposeDataMatrix(B1, *Brows, *Bcols);  // Input array
   


    int Adatasize = sizeof(float)*(*Arows)*(*Acols);
    int Bdatasize = sizeof(float)*(*Brows)*(*Bcols);
    int Cdatasize = sizeof(float)*(*Arows)*(*Bcols);

    float* C = (float*) malloc(Cdatasize);  // Output array
    float* scalar_sum;
    int i;
    for (i = 0; i < (*Arows)*(*Acols); i++)
    {
        C[i] = 0.0;
    }

    cl_int status;  
     
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel[NUM_KERNELS];
    cl_command_queue cmdQueue;
    cl_event prof_event;
    cl_int j, err, num_groups;
    size_t local_size, global_size;
    // Perhaps incorporate reduction kernel?
    char kernel_names[NUM_KERNELS][20] = {"matmult"};
    int BLOCKSIZE; 
    device = create_device();
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(local_size),
            &local_size, NULL);
    if (err < 0)
    {
        perror("Couldn't obtain device information");
        exit(1);

    }

    scalar_sum = (float*) malloc(sizeof(float)*local_size);
    BLOCKSIZE = sqrt(local_size);
    num_groups = (*Arows)*(*Bcols)/BLOCKSIZE;
    //if ((num_groups % BLOCKSIZE) == 0)
    //{
    //    num_groups = num_groups/BLOCKSIZE;
    //}
    //else
    //{
    //    num_groups = num_groups/BLOCKSIZE + 1;
    //}
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    if (err < 0)
    {
        perror("Couldn't create context.\n");
        exit(1);
    }
    cmdQueue = clCreateCommandQueue(context, device, 
            CL_QUEUE_PROFILING_ENABLE,&status);
    chk(status, "create cmd queue");
    program = build_program(context, device, PROGRAM_FILE);

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


    //Create local data buffer
    cl_mem partialSumBuf;
    partialSumBuf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR
            , local_size*sizeof(float), scalar_sum, &status);
    chk(status, "clCreateBuffer");


    // Write input array A to the device buffer bufferA
    status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 
        0, Adatasize, A, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer");    

    // Write input array B to the device buffer bufferB
    status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 
        0, Bdatasize, B, 0, NULL, NULL);
    chk(status, "clCreateBuffer");

    kernel[0] = clCreateKernel(program, "matmult", &status);
    chk(status, "clCreateKernel");

    // associate the input and output buffers with the kernel 
    status  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &bufC);
    status |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &bufA);
    status |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &bufB);
    //status |= clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &partialSumBuf);
    //status |= clSetKernelArg(kernel[0], 3, sizeof(cl_mem), NULL);

    status |= clSetKernelArg(kernel[0], 3, sizeof(int), Arows);
    status |= clSetKernelArg(kernel[0], 4, sizeof(int), Brows);
    status |= clSetKernelArg(kernel[0], 5, sizeof(int), Acols);
    status |= clSetKernelArg(kernel[0], 6, sizeof(int), Bcols);

    chk(status, "clSetKernelArg");

    // define an index space (global work size) of work 
    // items for execution. a workgroup size (local work size) 
    // is not required, but can be used.

    size_t globalworksize[2] ;   
    //size_t localWorkSize[2] = {BLOCKSIZE, BLOCKSIZE} ;   
    size_t localWorkSize[2];
    //localWorkSize[0] = local_size;
    // Choose local size appropriately. 
    int ls;
    int totSize = (*Arows)*(*Bcols);
    for (ls = sqrt(local_size); totSize % ls != 0; ls --)
    {
        //Do nothing
    }
    localWorkSize[0] = ls;
    localWorkSize[1] = ls;
    // there are 'elements' work-items 
    globalworksize[0] = *Bcols;
    globalworksize[1] = *Arows;
    //
    //

    /* size_t globalWorkSize[2] = {widthB, heightA};  // dims of outputC  */
    //int wB1 = (*Bcols) / BLOCKSIZE + ((*Bcols) % BLOCKSIZE==0? 0:1);
    //int hA1 = (*Arows) / BLOCKSIZE  + ((*Arows) % BLOCKSIZE==0? 0:1);
    //size_t globalWorkSize[2] = {wB1 * BLOCKSIZE, hA1 * BLOCKSIZE};  
    //
    //

    // enqueue the kernel for execution

    printf("1\n");
    status = clEnqueueNDRangeKernel(cmdQueue, kernel[0], 2, NULL, 
        globalworksize, localWorkSize, 0, NULL, NULL);
    chk(status, "clenqueuendrangekernel");
    printf("2\n");




    // read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufC, 1, 0, 
        Cdatasize, C, 0, NULL, NULL);
    chk(status, "clenqueuereadbuffer");

    //Verification code goes here
        float* C_cpu = (float*) malloc(sizeof(float)*Cdatasize);
        simpleMultiplyCPU(C_cpu, *Acols, *Arows, *Bcols,*Brows, A, B1);



        int result = 1;
        int idx;
        double diff;
        for (idx = 0; idx < (*Arows)*(*Bcols); idx ++)
        {
            diff = C[idx] - C_cpu[idx];
            if (diff < 0) diff *= -1;
            if (diff > 0.001)
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




    // Free OpenCL resources
    clReleaseKernel(kernel[0]);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseMemObject(partialSumBuf);
    clReleaseContext(context);

    // Free host resources
    free(A);
    free(B);
    free(C);
    free(B1);

    free(scalar_sum);



    return 0;


}
