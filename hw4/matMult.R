

simulateMatrix = function(nrow, ncol)
{
    return(matrix(rnorm(nrow*ncol, 0, 10), ncol = ncol))
}

writeMatrixToFile = function(mat, filename)
{
    cat("", file = filename)
    cat(nrow(mat), ncol(mat), "\n",file=filename, append = TRUE)
    for (i in 1:(nrow(mat)))
    {
         cat(mat[i,],"\n" , file =filename, append = TRUE)
    }

}

mat1 = simulateMatrix(500,500)
mat2 = simulateMatrix(500,800)
writeMatrixToFile(mat1, "./A.txt")
writeMatrixToFile(mat2, "./B.txt")


