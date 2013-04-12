

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

mat1 = simulateMatrix(250,250)
mat2 = simulateMatrix(250,400)
writeMatrixToFile(mat1, "./A.txt")
writeMatrixToFile(mat2, "./B.txt")


