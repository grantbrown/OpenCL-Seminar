#define _CRT_SECURE_NO_WARNINGS
//What does this do?
#define NUM_KERNELS 1
#define PROGRAM_FILE "./matmult_partitioning.kernel"

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


cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename, char* definestr, int lendefinestr)
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
    program_buffer = (char*) malloc(program_size + lendefinestr + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);

    int i;
    for (i = 0; i < lendefinestr; i ++)
    {
        program_buffer[i] = definestr[i];
    }
    //printf("%s", program_buffer);
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

float* padDataMatrix(float* inData, int nrow, int ncol, int padcol, int padrow)
{
    float* paddedMatrix = (float*) malloc(sizeof(float)*(nrow + padrow)*(ncol + padcol));
    int rw = 0;
    int col = 0;
    int idx = 0;
    for (rw = 0; rw < nrow; rw ++)
    {
        for (col = 0; col < ncol; col ++)
        {
            paddedMatrix[idx] = inData[col + rw*ncol];
            idx += 1;
        }
        for (col = 0; col < padcol; col ++)
        {
            paddedMatrix[idx] = 0.0;
            idx += 1;
        }
    }
    for (rw = 0; rw < padrow; rw ++)
    {
        for (col = 0; col < (ncol + padcol); col ++)
        {
            paddedMatrix[idx] = 0.0;
            idx += 1;
        }
    }
    return(paddedMatrix);
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
void stoptime( clock_t start, char msg[] )
{
    clock_t end ;
    double cpu_time_used;
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used for %s = %.3lf \n", msg, cpu_time_used) ;
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
void simpleMultiplyCPU_fp64( double *C, int widthA, int heightA, int widthB,
    int heightB, double *A, double *B)
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

double* readDataFileDouble(char fn[], int *mnum, int *nnum ){
    
      /* first row of input file contains two ints: num rows and num cols */
      /* rest of input file is data vals in row major order */
      int status;
      int size;
      double *data;
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
      data=malloc(sizeof(double)*m*n); //
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

int supportsDouble()
{
    cl_int err;
    cl_device_id device;

    device = create_device();

    char name_data[48], ext_data[4096];

    int supports_double = 0;
    char fp64_ext[] = "cl_khr_fp64";
    cl_uint addr_data;
    size_t ext_size;
    char options[20] = "";
    if (clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS,sizeof(addr_data), &addr_data, NULL) < 0)
    {
        perror("Couldn't read extension data");
        exit(1);
    }
    printf("Address width: %u\n", addr_data);

    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 
            sizeof(ext_data), ext_data, NULL);

    if (strstr(ext_data, fp64_ext) != NULL)
    {
        printf("The %s extension is supported.\n", fp64_ext);
        strcat(options, "-DFP_64 ");
        return(1);
    }
    else 
    {
        printf("The %s extension is not supported.\n", fp64_ext);
    }
    return(0);
}

int main_fp()
{

    int i;
    cl_int status;  
     
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel[NUM_KERNELS];
    cl_command_queue cmdQueue;
    cl_event prof_event;
    cl_int j, err, num_groups;
    size_t local_size, global_size;

    char kernel_names[NUM_KERNELS][20] = {"matmult"};
    int BLOCKSIZE; 
    device = create_device();
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(local_size),
            &local_size, NULL);
    printf("Max work group size: %d\n", local_size);

    if (err < 0)
    {
        perror("Couldn't obtain device information");
        exit(1);
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    if (err < 0)
    {
        perror("Couldn't create context.\n");
        exit(1);
    }

    int* Arows = (int *) malloc(sizeof(int));
    int* Acols = (int *) malloc(sizeof(int));
    int* Brows = (int *) malloc(sizeof(int));
    int* Bcols = (int *) malloc(sizeof(int));

    float* A = readDataFile("A.txt", Arows, Acols);  // Input array
    float* B = readDataFile("B.txt", Brows, Bcols);  // Input array

    clock_t start;

    int Adatasize = sizeof(float)*(*Arows)*(*Acols);
    int Bdatasize = sizeof(float)*(*Brows)*(*Bcols);
    int Cdatasize = sizeof(float)*(*Arows)*(*Bcols);

    float* C = (float*) malloc(Cdatasize);  // Output array

    cmdQueue = clCreateCommandQueue(context, device, 
            CL_QUEUE_PROFILING_ENABLE,&status);
    chk(status, "create cmd queue");


    // define an index space (global work size) of work 
    // items for execution. a workgroup size (local work size) 
    // is not required, but can be used.

    size_t globalworksize[2] ;   
    size_t localWorkSize[2];

    // Choose local size appropriately. 
    int ls;
    ls = sqrt(local_size)/2;

    localWorkSize[0] = ls;
    localWorkSize[1] = ls;

    globalworksize[0] = (*Bcols % ls == 0 ? *Bcols : (*Bcols/ls + 1)*ls);
    globalworksize[1] = (*Arows % ls == 0 ? *Arows : (*Arows/ls + 1)*ls);

    int Apad_rows = globalworksize[1] - *Arows;
    int Apad_cols = (*Acols % ls == 0 ? *Acols : (*Acols/ls + 1) * ls) - *Acols;

    int Bpad_rows = Apad_cols;
    int Bpad_cols = globalworksize[0] - *Bcols;

    Bdatasize = sizeof(float)*((*Brows + Bpad_rows)*(*Bcols + Bpad_cols));
    Adatasize = sizeof(float)*((*Arows + Apad_rows)*(*Acols + Apad_cols));
    Cdatasize = sizeof(float)*(globalworksize[0]*globalworksize[1]);

    C = (float*) realloc(C, Cdatasize);
    status = (C == NULL);

    chk(status, "Reallocation.");

 
    for (i = 0; i < Cdatasize/sizeof(float); i++)
    {
        C[i] = 0.0;
    }

    printf("Local work block size is: %d\n", ls);
    printf("Global work size is: %d by %d\n", globalworksize[0], globalworksize[1]);

    int lendefstr = 13;
    char defstr[lendefstr];
    sprintf(defstr, "             ");

    program = build_program(context, device, PROGRAM_FILE, defstr,  lendefstr);

    // Create a buffer object that will contain the data 
    // from the host array A

    start = clock();
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
        0, sizeof(float)*(*Acols)*(*Arows), A, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer");    

    // Write input array B to the device buffer bufferB
    status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 
        0, sizeof(float)*(*Bcols)*(*Brows), B, 0, NULL, NULL);
    chk(status, "clCreateBuffer");

    kernel[0] = clCreateKernel(program, "matmult", &status);
    chk(status, "clCreateKernel");

    // associate the input and output buffers with the kernel 
    status  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &bufC);
    status |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &bufA);
    status |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &bufB);
    status |= clSetKernelArg(kernel[0], 3, sizeof(int), Arows);
    status |= clSetKernelArg(kernel[0], 4, sizeof(int), Brows);
    status |= clSetKernelArg(kernel[0], 5, sizeof(int), Acols);
    status |= clSetKernelArg(kernel[0], 6, sizeof(int), Bcols);
    status |= clSetKernelArg(kernel[0], 7, ls*ls*sizeof(float), NULL);
    status |= clSetKernelArg(kernel[0], 8, ls*ls*sizeof(float), NULL);

    chk(status, "clSetKernelArg");

    // enqueue the kernel for execution

    status = clEnqueueNDRangeKernel(cmdQueue, kernel[0], 2, NULL, 
        globalworksize, localWorkSize, 0, NULL, NULL);
    chk(status, "clenqueuendrangekernel");

    clFinish(cmdQueue);
    stoptime(start,"OCL: Move data to device and multiply matrices.");

    // read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufC, 1, 0, 
        Cdatasize, C, 0, NULL, NULL);
    chk(status, "clenqueuereadbuffer");
    //clFinish(cmdQueue);

    float* C_cpu = (float*) malloc(Cdatasize);
    start = clock();
    simpleMultiplyCPU(C_cpu, *Acols, *Arows, *Bcols,*Brows, A, B);
    stoptime(start, "CPU: Multiply Matrices");

    int result = 1;
    int idx;
    double diff;

    for (idx = 0; idx < (*Arows)*(*Bcols); idx ++)
    {
        diff = C[idx] - C_cpu[idx];
        if (diff < 0) diff *= -1;
        if (diff > 0.01)
        {
            result = 0;
            printf("Breaking, index = %d\n", idx);
            printf("Total Data size is = %d\n", (*Arows)*(*Bcols));
            printf("OCL: %f\n", C[idx]);
            printf("CPU: %f\n", C_cpu[idx]);

            //break;
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

    clReleaseContext(context);

    // Free host resources
    free(A);
    free(B);
    free(C);

    return 0;
}

int main_fp64()
{

    int i;
    cl_int status;  
     
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel[NUM_KERNELS];
    cl_command_queue cmdQueue;
    cl_event prof_event;
    cl_int j, err, num_groups;
    size_t local_size, global_size;

    char kernel_names[NUM_KERNELS][20] = {"matmult_fp64"};
    int BLOCKSIZE; 
    device = create_device();
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(local_size),
            &local_size, NULL);
    printf("Max work group size: %d\n", local_size);
    if (err < 0)
    {
        perror("Couldn't obtain device information");
        exit(1);
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    if (err < 0)
    {
        perror("Couldn't create context.\n");
        exit(1);
    }

    int* Arows = (int *) malloc(sizeof(int));
    int* Acols = (int *) malloc(sizeof(int));
    int* Brows = (int *) malloc(sizeof(int));
    int* Bcols = (int *) malloc(sizeof(int));

    double* A = readDataFileDouble("A.txt", Arows, Acols);  // Input array
    double* B = readDataFileDouble("B.txt", Brows, Bcols);  // Input array

    clock_t start;

    int Adatasize = sizeof(double)*(*Arows)*(*Acols);
    int Bdatasize = sizeof(double)*(*Brows)*(*Bcols);
    int Cdatasize = sizeof(double)*(*Arows)*(*Bcols);

    double* C = (double*) malloc(Cdatasize);  // Output array

    cmdQueue = clCreateCommandQueue(context, device, 
            CL_QUEUE_PROFILING_ENABLE,&status);
    chk(status, "create cmd queue");


    // define an index space (global work size) of work 
    // items for execution. a workgroup size (local work size) 
    // is not required, but can be used.

    size_t globalworksize[2] ;   
    size_t localWorkSize[2];

    // Choose local size appropriately. 
    int ls;
    ls = sqrt(local_size)/2;
    //ls = 1;
    localWorkSize[0] = ls;
    localWorkSize[1] = ls;

    globalworksize[0] = (*Bcols % ls == 0 ? *Bcols : (*Bcols/ls + 1)*ls);
    globalworksize[1] = (*Arows % ls == 0 ? *Arows : (*Arows/ls + 1)*ls);

    int Apad_rows = globalworksize[1] - *Arows;
    int Apad_cols = (*Acols % ls == 0 ? *Acols : (*Acols/ls + 1) * ls) - *Acols;

    int Bpad_rows = Apad_cols;
    int Bpad_cols = globalworksize[0] - *Bcols;

    Bdatasize = sizeof(double)*((*Brows + Bpad_rows)*(*Bcols + Bpad_cols));
    Adatasize = sizeof(double)*((*Arows + Apad_rows)*(*Acols + Apad_cols));
    Cdatasize = sizeof(double)*(globalworksize[0]*globalworksize[1]);

    C = (double*) realloc(C, Cdatasize);
    status = (C == NULL);

    chk(status, "Reallocation.");

 
    for (i = 0; i < Cdatasize/sizeof(double); i++)
    {
        C[i] = 0.0;
    }

    printf("Local work block size is: %d\n", ls);
    printf("Global work size is: %d by %d\n", globalworksize[0], globalworksize[1]);

    int lendefstr = 15;
    char defstr[lendefstr];
    sprintf(defstr, "#define FP_64 1"); 
    printf("%s\n", defstr);
    
    program = build_program(context, device, PROGRAM_FILE, defstr,  lendefstr);

    // Create a buffer object that will contain the data 
    // from the host array A

    start = clock();
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
        0, sizeof(double)*(*Acols)*(*Arows), A, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer");    

    // Write input array B to the device buffer bufferB
    status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 
        0, sizeof(double)*(*Bcols)*(*Brows), B, 0, NULL, NULL);
    chk(status, "clCreateBuffer");

    kernel[0] = clCreateKernel(program, "matmult", &status);
    chk(status, "clCreateKernel");

    // associate the input and output buffers with the kernel 
    status  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &bufC);
    status |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &bufA);
    status |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &bufB);
    status |= clSetKernelArg(kernel[0], 3, sizeof(int), Arows);
    status |= clSetKernelArg(kernel[0], 4, sizeof(int), Brows);
    status |= clSetKernelArg(kernel[0], 5, sizeof(int), Acols);
    status |= clSetKernelArg(kernel[0], 6, sizeof(int), Bcols);
    status |= clSetKernelArg(kernel[0], 7, ls*ls*sizeof(double), NULL);
    status |= clSetKernelArg(kernel[0], 8, ls*ls*sizeof(double), NULL);

    chk(status, "clSetKernelArg");

    // enqueue the kernel for execution

    status = clEnqueueNDRangeKernel(cmdQueue, kernel[0], 2, NULL, 
        globalworksize, localWorkSize, 0, NULL, NULL);
    chk(status, "clenqueuendrangekernel");

    clFinish(cmdQueue);
    stoptime(start,"OCL: Move data to device and multiply matrices.");

    // read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufC, 1, 0, 
        Cdatasize, C, 0, NULL, NULL);
    chk(status, "clenqueuereadbuffer");
    //clFinish(cmdQueue);

    double* C_cpu = (double*) malloc(Cdatasize);
    start = clock();
    simpleMultiplyCPU_fp64(C_cpu, *Acols, *Arows, *Bcols,*Brows, A, B);
    stoptime(start, "CPU: Multiply Matrices");

    int result = 1;
    int idx;
    double diff;

    for (idx = 0; idx < (*Arows)*(*Bcols); idx ++)
    {
        diff = C[idx] - C_cpu[idx];
        if (diff < 0) diff *= -1;
        if (diff > 0.01)
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

    clReleaseContext(context);

    // Free host resources
    free(A);
    free(B);
    free(C);

    return 0;
}

int main()
{
    int supports_double = supportsDouble();
    if (supports_double) return(main_fp64());
    return(main_fp());
}
