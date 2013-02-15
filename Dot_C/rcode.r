

dyn.load("vectoradd.so")


oclVectorAdd = function(A, B, C)
{
	out = .C("execute", as.integer(A), as.integer(B), as.integer(C), as.integer(length(C)))
	return(out)
}
A = c(1,2,3,4)
B = A
C = rep(0, 4)

print(oclVectorAdd(A,B,C))

