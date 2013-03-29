__kernel
void convolution(__global float* imageIn,
                 __global float* imageOut, 
               __constant float* filter, 
                            int  rows,
                            int  cols,
                            int  filterWidth,
                  __local float* localImage,
                            int  localHeight,
                            int  localWidth) {
    
    
    // Determine the amount of padding for this filter
    int filterRadius = (filterWidth/2);
    int padding = filterRadius * 2;
    
    // Determine the size of the work group output region
    int groupStartCol = get_group_id(0)*get_local_size(0);
    int groupStartRow = get_group_id(1)*get_local_size(1);
    
    // Determine the local ID of each work item
    int localCol = get_local_id(0);
    int localRow = get_local_id(1);
    
    // Determine the global ID of each work item.  Work items
    // representing the output region will have a unique global
    // ID
    int globalCol = groupStartCol + localCol;
    int globalRow = groupStartRow + localRow;   
    
    // Cache the data to local memory
    
    // Step down rows
    for(int i = localRow; i < localHeight; i += 
        get_local_size(1)) {
        
        int curRow = groupStartRow+i;
        
        // Step across columns
        for(int j = localCol; j < localWidth; j += 
            get_local_size(0)) {
            
            int curCol = groupStartCol+j;
            
            // Perform the read if it is in bounds
            if(curRow < rows && curCol < cols) {
                localImage[i*localWidth + j] = 
                    imageIn[curRow*cols+curCol];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform the convolution
    if(globalRow < rows-padding && globalCol < cols-padding) {
        
        // Each work item will filter around its start location 
        //(starting from the filter radius left and up)
        float sum = 0.0f;
        int filterIdx = 0;
        
         // Not unrolled
         for(int i = localRow; i < localRow+filterWidth; i++) {
            int offset = i*localWidth;
            for(int j = localCol; j < localCol+filterWidth; j++){
                sum += localImage[offset+j] * 
                   filter[filterIdx++];
            }
         }
        
         /*
         // Inner loop unrolled
         for(int i = localRow; i < localRow+filterWidth; i++) {
            int offset = i*localWidth+localCol;
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
         }
         */
         
        // Write the data out
        imageOut[(globalRow+filterRadius)*cols + 
           (globalCol+filterRadius)] = sum;
    }
    
    return;
}

__kernel
void convolution_read4(__global float4* imageIn,
                        __global float* imageOut, 
                      __constant float* filter, 
                                   int  rows,
                                   int  cols,
                                   int  filterWidth,
                         __local float* localImage,
                                   int  localHeight,
                                   int  localWidth) {
    
    // Vector pointer that will be used to cache data
    // scalar memory
    __local float4* localImage4;
    
    // Determine the amount of padding for this filter
    int filterRadius = (filterWidth/2);
    int padding = filterRadius * 2;
    
    // Determine where each work group begins reading
    int groupStartCol = get_group_id(0)*get_local_size(0)/4;
    int groupStartRow = get_group_id(1)*get_local_size(1);
    
    // Flatten the localIds 0-255
    int localId = get_local_id(1)*get_local_size(0) + 
        get_local_id(0);
    // There will be localWidth/4 work items reading per row
    int localRow = (localId / (localWidth/4));
    // Each work item is reading 4 elements apart
    int localCol = (localId % (localWidth/4)); 
    
    // Determine the row and column offset in global memory 
    // assuming each element reads 4 floats
    int globalRow = groupStartRow + localRow;
    int globalCol = groupStartCol + localCol;
    
    // Set the vector pointer to the correct scalar location
    // in local memory
    localImage4 = (__local float4*) 
        &localImage[localRow*localWidth+localCol*4];
    
    // Perform all of the reads with a single load
    if(globalRow < rows && globalCol < cols/4 &&
       localRow < localHeight) {
        
        localImage4[0] = imageIn[globalRow*cols/4+globalCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reassign local IDs based on each work item processing 
    // one output element
    localCol = get_local_id(0);
    localRow = get_local_id(1);
    
    // Reassign global IDs for unique output locations
    globalCol = get_group_id(0)*get_local_size(0) + localCol;
    globalRow = get_group_id(1)*get_local_size(1) + localRow;   
    
    // Perform the convolution
    if(globalRow < rows-padding && globalCol < cols-padding) {
        
        // Each work item will filter around its start location 
        // (starting from half the filter size left and up)
        float sum = 0.0f;
        int filterIdx = 0;

        // Not unrolled
        for(int i = localRow; i < localRow+filterWidth; i++) {
            int offset = i*localWidth;
            for(int j = localCol; j < localCol+filterWidth; j++){
                sum += localImage[offset+j] * 
                    filter[filterIdx++];
            }
        }
        
        /*
        // Inner loop unrolled
        for(int i = localRow; i < localRow+filterWidth; i++) {
            int offset = i*localWidth+localCol;
         
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
            sum += localImage[offset++] * filter[filterIdx++];
        }
        */

        /*
        // Completely unrolled
        int offset = localRow*localWidth+localCol;
        
        sum += localImage[offset+0] * filter[filterIdx++];
        sum += localImage[offset+1] * filter[filterIdx++];
        sum += localImage[offset+2] * filter[filterIdx++];
        sum += localImage[offset+3] * filter[filterIdx++];
        sum += localImage[offset+4] * filter[filterIdx++];
        sum += localImage[offset+5] * filter[filterIdx++];
        sum += localImage[offset+6] * filter[filterIdx++];
        
        offset += localWidth;
        
        sum += localImage[offset+0] * filter[filterIdx++];
        sum += localImage[offset+1] * filter[filterIdx++];
        sum += localImage[offset+2] * filter[filterIdx++];
        sum += localImage[offset+3] * filter[filterIdx++];
        sum += localImage[offset+4] * filter[filterIdx++];
        sum += localImage[offset+5] * filter[filterIdx++];
        sum += localImage[offset+6] * filter[filterIdx++];
        
        offset += localWidth;
        
        sum += localImage[offset+0] * filter[filterIdx++];
        sum += localImage[offset+1] * filter[filterIdx++];
        sum += localImage[offset+2] * filter[filterIdx++];
        sum += localImage[offset+3] * filter[filterIdx++];
        sum += localImage[offset+4] * filter[filterIdx++];
        sum += localImage[offset+5] * filter[filterIdx++];
        sum += localImage[offset+6] * filter[filterIdx++];
        
        offset += localWidth;
        
        sum += localImage[offset+0] * filter[filterIdx++];
        sum += localImage[offset+1] * filter[filterIdx++];
        sum += localImage[offset+2] * filter[filterIdx++];
        sum += localImage[offset+3] * filter[filterIdx++];
        sum += localImage[offset+4] * filter[filterIdx++];
        sum += localImage[offset+5] * filter[filterIdx++];
        sum += localImage[offset+6] * filter[filterIdx++];
        
        offset += localWidth;
        
        sum += localImage[offset+0] * filter[filterIdx++];
        sum += localImage[offset+1] * filter[filterIdx++];
        sum += localImage[offset+2] * filter[filterIdx++];
        sum += localImage[offset+3] * filter[filterIdx++];
        sum += localImage[offset+4] * filter[filterIdx++];
        sum += localImage[offset+5] * filter[filterIdx++];
        sum += localImage[offset+6] * filter[filterIdx++];
        
        offset += localWidth;
        
        sum += localImage[offset+0] * filter[filterIdx++];
        sum += localImage[offset+1] * filter[filterIdx++];
        sum += localImage[offset+2] * filter[filterIdx++];
        sum += localImage[offset+3] * filter[filterIdx++];
        sum += localImage[offset+4] * filter[filterIdx++];
        sum += localImage[offset+5] * filter[filterIdx++];
        sum += localImage[offset+6] * filter[filterIdx++];
        
        offset += localWidth;
        
        sum += localImage[offset+0] * filter[filterIdx++];
        sum += localImage[offset+1] * filter[filterIdx++];
        sum += localImage[offset+2] * filter[filterIdx++];
        sum += localImage[offset+3] * filter[filterIdx++];
        sum += localImage[offset+4] * filter[filterIdx++];
        sum += localImage[offset+5] * filter[filterIdx++];
        sum += localImage[offset+6] * filter[filterIdx++];
        */
        
        // Write the data out
        imageOut[(globalRow+filterRadius)*cols + 
           (globalCol+filterRadius)] = sum;
    }
    
    return;
}
