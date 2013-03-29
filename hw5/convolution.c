#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h> 
#include <time.h>
#include "bmpfuncs.h"

#define WGX 16
#define WGY 16

// Uncomment each of these (one at a time) to run with the corresponding 
// optimization
#define NON_OPTIMIZED
//#define READ_ALIGNED
//#define READ4 

// This function takes a positive integer and rounds it up to
// the nearest multiple of another provided integer
unsigned int roundUp(unsigned int value, unsigned int multiple) {
	
  // Determine how far past the nearest multiple the value is
  unsigned int remainder = value % multiple;
  
  // Add the difference to make the value a multiple
  if(remainder != 0) {
          value += (multiple-remainder);
  }
  
  return value;
}

void stoptime (clock_t start, char msg[])
{
    clock_t end;
    double cpu_time_used;
    end = clock();
    cpu_time_used = ((double) (end - start))/CLOCKS_PER_SEC;
    printf("CPU time used for %s =  %.3lf \n", msg, cpu_time_used);
}

// This function reads in a text file and stores it as a char pointer
char* readSource(char* kernelPath) {

   cl_int status;
   FILE *fp;
   char *source;
   long int size;

   printf("Program file is: %s\n", kernelPath);

   fp = fopen(kernelPath, "rb");
   if(!fp) {
      printf("Could not open kernel file\n");
      exit(-1);
   }
   status = fseek(fp, 0, SEEK_END);
   if(status != 0) {
      printf("Error seeking to end of file\n");
      exit(-1);
   }
   size = ftell(fp);
   if(size < 0) {
      printf("Error getting file position\n");
      exit(-1);
   }

   rewind(fp);

   source = (char *)malloc(size + 1);
   
   int i;
   for (i = 0; i < size+1; i++) {
      source[i]='\0';
   } 

   if(source == NULL) {
      printf("Error allocating space for the kernel source\n");
      exit(-1);
   }

   fread(source, 1, size, fp);
   source[size] = '\0';

   return source;
}

int main(int argc, char** argv) {

   // Set up the data on the host	
   clock_t start, start0;
   start0 = clock();
   start = clock();
   // Rows and columns in the input image
   int imageHeight;
   int imageWidth;

   const char* inputFile = "input.bmp";
   const char* outputFile = "output.bmp";



   // Homegrown function to read a BMP from file
   float* inputImage = readImage(inputFile, &imageWidth, 
      &imageHeight);

   // Size of the input and output images on the host
   int dataSize = imageHeight*imageWidth*sizeof(float);

   // Pad the number of columns 
#ifdef NON_OPTIMIZED
   int deviceWidth = imageWidth;
#else  // READ_ALIGNED || READ4
   int deviceWidth = roundUp(imageWidth, WGX);
#endif
   int deviceHeight = imageHeight;
   // Size of the input and output images on the device
   int deviceDataSize = imageHeight*deviceWidth*sizeof(float);

   // Output image on the host
   float* outputImage = NULL;
   outputImage = (float*)malloc(dataSize);
   int i, j;
   for(i = 0; i < imageHeight; i++) {
       for(j = 0; j < imageWidth; j++) {
           outputImage[i*imageWidth+j] = 0;
       }
   }

   // 45 degree motion blur
   float filter[49] = 
      {0,      0,      0,      0,      0, 0.0145,      0,
       0,      0,      0,      0, 0.0376, 0.1283, 0.0145,
       0,      0,      0, 0.0376, 0.1283, 0.0376,      0,
       0,      0, 0.0376, 0.1283, 0.0376,      0,      0,
       0, 0.0376, 0.1283, 0.0376,      0,      0,      0,
  0.0145, 0.1283, 0.0376,      0,      0,      0,      0,
       0, 0.0145,      0,      0,      0,      0,      0};
 
   int filterWidth = 7;
   int paddingPixels = (int)(filterWidth/2) * 2; 
   stoptime(start, "set up input, output.");
   start = clock();
   // Set up the OpenCL environment

   // Discovery platform
   cl_platform_id platform;
   clGetPlatformIDs(1, &platform, NULL);

   // Discover device
   cl_device_id device;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device,
      NULL);

    size_t time_res;
    clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION,
            sizeof(time_res), &time_res, NULL);
    printf("Device profiling timer resolution: %zu ns.\n", time_res);

   // Create context
   cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 
       (cl_context_properties)(platform), 0};
   cl_context context; 
   context = clCreateContext(props, 1, &device, NULL, NULL, 
      NULL);

   // Create command queue
   cl_ulong time_start, time_end, exec_time;
   cl_event timing_event;
   cl_command_queue queue;
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

   // Create memory buffers
   cl_mem d_inputImage;
   cl_mem d_outputImage;
   cl_mem d_filter;
   d_inputImage = clCreateBuffer(context, CL_MEM_READ_ONLY, 
       deviceDataSize, NULL, NULL);
   d_outputImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
       deviceDataSize, NULL, NULL);
   d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, 
       49*sizeof(float),NULL, NULL);
   
   // Write input data to the device
#ifdef NON_OPTIMIZED
   clEnqueueWriteBuffer(queue, d_inputImage, CL_TRUE, 0, deviceDataSize,
       inputImage, 0, NULL, NULL);
#else // READ_ALIGNED || READ4
   size_t buffer_origin[3] = {0,0,0};
   size_t host_origin[3] = {0,0,0};
   size_t region[3] = {deviceWidth*sizeof(float), 
      imageHeight, 1};
   clEnqueueWriteBufferRect(queue, d_inputImage, CL_TRUE, 
      buffer_origin, host_origin, region, 
      deviceWidth*sizeof(float), 0, imageWidth*sizeof(float), 0,
      inputImage, 0, NULL, NULL);
#endif
	
   // Write the filter to the device
   clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0, 
      49*sizeof(float), filter, 0, NULL, NULL);
	
   // Read in the program from file
   char* source = readSource("convolution.cl");

   // Create the program
   cl_program program;
	
   // Create and compile the program
   program = clCreateProgramWithSource(context, 1, 
       (const char**)&source, NULL, NULL);
   cl_int build_status;
   build_status = clBuildProgram(program, 1, &device, NULL, NULL,
      NULL);
      
   // Create the kernel
   cl_kernel kernel;
#if defined NON_OPTIMIZED || defined READ_ALIGNED
   // Only the host-side code differs for the aligned reads
   kernel = clCreateKernel(program, "convolution", NULL);
#else // READ4
   kernel = clCreateKernel(program, "convolution_read4", NULL);
#endif
	
   // Selected work group size is 16x16
   int wgWidth = WGX;
   int wgHeight = WGY;

   // When computing the total number of work items, the 
   // padding work items do not need to be considered
   int totalWorkItemsX = roundUp(imageWidth-paddingPixels, 
      wgWidth);
   int totalWorkItemsY = roundUp(imageHeight-paddingPixels, 
      wgHeight);

   // Size of a work group
   size_t localSize[2] = {wgWidth, wgHeight};
   // Size of the NDRange
   size_t globalSize[2] = {totalWorkItemsX, totalWorkItemsY};

   // The amount of local data that is cached is the size of the
   // work groups plus the padding pixels
#if defined NON_OPTIMIZED || defined READ_ALIGNED
   int localWidth = localSize[0] + paddingPixels;
#else // READ4
   // Round the local width up to 4 for the read4 kernel
   int localWidth = roundUp(localSize[0]+paddingPixels, 4);
#endif
   int localHeight = localSize[1] + paddingPixels;

   // Compute the size of local memory (needed for dynamic 
   // allocation)
   size_t localMemSize = (localWidth * localHeight * 
      sizeof(float));

   // Set the kernel arguments
   clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_inputImage);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_outputImage);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_filter);
   clSetKernelArg(kernel, 3, sizeof(int), &deviceHeight);
   clSetKernelArg(kernel, 4, sizeof(int), &deviceWidth); 
   clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);
   clSetKernelArg(kernel, 6, localMemSize, NULL);
   clSetKernelArg(kernel, 7, sizeof(int), &localHeight); 
   clSetKernelArg(kernel, 8, sizeof(int), &localWidth);

   stoptime(start, "set up kernel");
   start = clock();
   // Execute the kernel
   clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, 
      localSize, 0, NULL, &timing_event);

   // Wait for kernel to complete
   clFinish(queue);
   stoptime(start, "run kernel");
   clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_START,
           sizeof(time_start), &time_start, NULL);
   clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_END,
           sizeof(time_end), &time_end, NULL);
   exec_time = time_end-time_start;
   printf("Profile execution time = %.3lf sec.\n", (double) exec_time/1000000000);

   // Read back the output image
#ifdef NON_OPTIMIZED
   clEnqueueReadBuffer(queue, d_outputImage, CL_TRUE, 0, 
      deviceDataSize, outputImage, 0, NULL, NULL);
#else // READ_ALIGNED || READ4
   // Begin reading output from (3,3) on the device 
   // (for 7x7 filter with radius 3)
   buffer_origin[0] = 3*sizeof(float);
   buffer_origin[1] = 3;
   buffer_origin[2] = 0;

   // Read data into (3,3) on the host
   host_origin[0] = 3*sizeof(float);
   host_origin[1] = 3;
   host_origin[2] = 0;
	
   // Region is image size minus padding pixels
   region[0] = (imageWidth-paddingPixels)*sizeof(float);
   region[1] = (imageHeight-paddingPixels);
   region[2] = 1;
	
	// Perform the read
   clEnqueueReadBufferRect(queue, d_outputImage, CL_TRUE, 
      buffer_origin, host_origin, region, 
      deviceWidth*sizeof(float), 0, imageWidth*sizeof(float), 0, 
      outputImage, 0, NULL, NULL);
#endif
  
   // Homegrown function to write the image to file
   storeImage(outputImage, outputFile, imageHeight, 
      imageWidth, inputFile);
   
   // Free OpenCL objects
   clReleaseMemObject(d_inputImage);
   clReleaseMemObject(d_outputImage);
   clReleaseMemObject(d_filter);
   clReleaseKernel(kernel);
   clReleaseProgram(program);
   clReleaseCommandQueue(queue);
   clReleaseContext(context);

   return 0;
}

