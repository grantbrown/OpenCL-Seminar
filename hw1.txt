# This is a modified version of the example code in the documentation
# of the oclRun function in the OpenCL package
# Kate Cowles 12/21/12
# Modified by Grant Brown for HW 1 2/1/2013

library(OpenCL)
p = oclPlatforms()      #retrieves all available OpenCL platforms
d = oclDevices(p[[1]])  #retrieves a list of OpenCL devices for the given
		     #platform

# stores the kernel code in a character array
code1 = c(
"__kernel void vadd(\n",
"  __global float* output,\n",
" const unsigned int count,\n",
"  __global float* vector1,\n",
"  __global float* vector2)\n",
"{\n",
"  int i = get_global_id(0);\n",
"  if(i < count)\n",
"      output[i] = vector1[i] + vector2[i] ;\n", 
"};")
code2 = c(
"__kernel void matmult(\n",
"  __global float* output,\n",
" const unsigned int count,\n",
"  __global float* A,\n",
"  __global float* B,\n",

" const unsigned int Arows,\n",
" const unsigned int Brows,\n",
" const unsigned int Acols,\n",
" const unsigned int Bcols)\n",
"{\n",
"  int i = (get_global_id(0));\n",
"  if(i< count)\n",
"  {\n",
"      unsigned int OutputRow = (i/Bcols);\n",
"      unsigned int OutputCol = i - OutputRow*Bcols;\n",
"      float tot = 0;\n",
"      unsigned int j;\n",

"      for (j = 0; j < Acols; j++)\n",
"      {\n",
"         tot += (A[OutputRow + Arows*j] * B[OutputCol*Brows + j]);\n",
"      }\n",
"      output[i] = tot ;\n", 
"  }\n",
"};")



# creates a kernel object by compiling the supplied code
k.vadd <- oclSimpleKernel(d[[1]], "vadd", code1, "single")
k.matmult = oclSimpleKernel(d[[1]], "matmult", code2, "single")


# R function to supply arguments to the kernel and run it
oclVadd <- function(v1,v2, ...)
{
	oclRun(k.vadd, length(v1), clFloat(v1),clFloat(v2), ...)
}

rVadd = function(v1, v2, ...)
{
	return(v1 + v2)
}

oclMatmult = function(A, B)
{
	if (ncol(A) != nrow(B))
	{
		print("Invalid Dimensions Silly!")
		return(1)
	}
	count = nrow(A)*ncol(B)
	return(matrix(oclRun(k.matmult, count, clFloat(A), clFloat(B), nrow(A), nrow(B), ncol(A), ncol(B)), nrow = nrow(A), byrow = TRUE))
}

rMatmult = function(A, B)
{
	return(A %*% B)
}

runVaddSim = function()
{
	cat("###   Running Comparison of Vector Addition Functions.   ### \n")
	for (vsize in seq(50000, 100000, 10000))
	{
		cat("    ###   Now Adding Vectors of Size ", as.character(vsize), "   ###\n")
		t1 = 0
		t2 = 0
		for (trial in 1:10)
		{
			v1 = 1:vsize
			v2 = v1
			t1 = t1 + system.time(oclOut <- oclVadd(v1, v2))
			t2 = t2 + system.time(rOut <- rVadd(v1, v2))
		}
		cat("    ###   Total Open CL Vector Addition Time (10 trials):   ###\n")
		print(t1)
		cat("    ###   Total Native R Vector Addition Time (10 trials):   ###\n")
		print(t2)
		cat("    ###   Summary of Vector Addition Output Differences:   ###\n")
		output = summary(oclOut - rOut)
		print(output)
		cat("\n")
	}
	cat("\n\n")

}
runVaddSim()


runMatmultSim = function()
{
	cat("\n\n\n###   Running Comparison of Matrix Multiplication Functions.   ### \n")

	for (matsize in c(1000,10000,50000,100000))
	{
		cat("    ###   Now Multiplying Matrices With ", as.character(matsize), "Elements   ###\n")
		t1 = 0
		t2 = 0
		maxdiff = 0
		for (trial in 1:10)
		{
			A = matrix(rnorm(matsize), nrow = 100)
			B = matrix(rnorm(matsize), ncol = 100)


			t1 = t1 + system.time(oclOut <- oclMatmult(A,B))
			t2 = t2 + system.time(rOut <- rMatmult(A, B))
			maxdiff = max(maxdiff, max(abs(oclOut-rOut)))

		}
		cat("    ###   Matrix Multiplication Results:   ###\n")

		cat("    ### Total Open CL Matrix Multiplication Time (10 trials):   ###\n")
		print(t1)
		cat("    ### Total Native R Matrix Multiplication Time (10 trials):   ###\n")
		print(t2)
		cat("    ### Maximum Absolute Element-Wise Difference Between Output Marices:   ###\n")
		print(maxdiff)
		cat("\n")
	}
	cat("\n\n")
}
runMatmultSim()



###### Output Below ###### 



####   Running Comparison of Vector Addition Functions.   ### 
#    ###   Now Adding Vectors of Size  50000    ###
#    ###   Total Open CL Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#   0.08    0.01    0.03 
#    ###   Total Native R Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#  0.010   0.000   0.003 
#    ###   Summary of Vector Addition Output Differences:   ###
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#      0       0       0       0       0       0 
#
#    ###   Now Adding Vectors of Size  60000    ###
#    ###   Total Open CL Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#  0.090   0.020   0.029 
#    ###   Total Native R Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#  0.010   0.000   0.005 
#    ###   Summary of Vector Addition Output Differences:   ###
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#      0       0       0       0       0       0 
#
#    ###   Now Adding Vectors of Size  70000    ###
#    ###   Total Open CL Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#  0.080   0.000   0.033 
#    ###   Total Native R Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#  0.010   0.000   0.003 
#    ###   Summary of Vector Addition Output Differences:   ###
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#      0       0       0       0       0       0 
#
#    ###   Now Adding Vectors of Size  80000    ###
#    ###   Total Open CL Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#  0.080   0.030   0.032 
#    ###   Total Native R Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#  0.000   0.000   0.003 
#    ###   Summary of Vector Addition Output Differences:   ###
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#      0       0       0       0       0       0 
#
#    ###   Now Adding Vectors of Size  90000    ###
#    ###   Total Open CL Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#  0.010   0.010   0.037 
#    ###   Total Native R Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#  0.000   0.000   0.005 
#    ###   Summary of Vector Addition Output Differences:   ###
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#      0       0       0       0       0       0 
#
#    ###   Now Adding Vectors of Size  1e+05    ###
#    ###   Total Open CL Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#   0.07    0.02    0.04 
#    ###   Total Native R Vector Addition Time (10 trials):   ###
#   user  system elapsed 
#  0.010   0.000   0.006 
#    ###   Summary of Vector Addition Output Differences:   ###
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#      0       0       0       0       0       0 






####   Running Comparison of Matrix Multiplication Functions.   ### 
#    ###   Now Multiplying Matrices With  1000 Elements   ###
#    ###   Matrix Multiplication Results:   ###
#    ### Total Open CL Matrix Multiplication Time (10 trials):   ###
#   user  system elapsed 
#   0.07    0.00    0.02 
#    ### Total Native R Matrix Multiplication Time (10 trials):   ###
#   user  system elapsed 
#  0.010   0.000   0.002 
#    ### Maximum Absolute Element-Wise Difference Between Output Marices:   ###
#[1] 2.527348e-06
#
#    ###   Now Multiplying Matrices With  10000 Elements   ###
#    ###   Matrix Multiplication Results:   ###
#    ### Total Open CL Matrix Multiplication Time (10 trials):   ###
#   user  system elapsed 
#  0.060   0.020   0.025 
#    ### Total Native R Matrix Multiplication Time (10 trials):   ###
#   user  system elapsed 
#  0.020   0.000   0.011 
#    ### Maximum Absolute Element-Wise Difference Between Output Marices:   ###
#[1] 2.105845e-05
#
#    ###   Now Multiplying Matrices With  50000 Elements   ###
#    ###   Matrix Multiplication Results:   ###
#    ### Total Open CL Matrix Multiplication Time (10 trials):   ###
#   user  system elapsed 
#  0.070   0.040   0.049 
#    ### Total Native R Matrix Multiplication Time (10 trials):   ###
#   user  system elapsed 
#  0.060   0.000   0.047 
#    ### Maximum Absolute Element-Wise Difference Between Output Marices:   ###
#[1] 9.186667e-05
#
#    ###   Now Multiplying Matrices With  1e+05 Elements   ###
#    ###   Matrix Multiplication Results:   ###
#    ### Total Open CL Matrix Multiplication Time (10 trials):   ###
#   user  system elapsed 
#  0.110   0.050   0.074 
#    ### Total Native R Matrix Multiplication Time (10 trials):   ###
#   user  system elapsed 
#  0.100   0.000   0.096 
#    ### Maximum Absolute Element-Wise Difference Between Output Marices:   ###
# [1] 0.000160629


