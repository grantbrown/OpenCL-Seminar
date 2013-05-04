import re
import numpy as np

def dot(vec1, vec2):
    if len(vec1) != len(vec2):
        raise Exception("Vectors have different lengths")
    return(float(sum([vec1[i]*vec2[i] for i in xrange(len(vec1))])))

def minus(vec1, vec2):
    if len(vec1) != len(vec2):
        raise Exception("Vectors have different lengths")
    return([vec1[i] - vec2[i] for i in xrange(len(vec1))])

def mult(scalar1, vec1):
    return([vec1[i]*scalar1 for i in xrange(len(vec1))])

def reg(resp, covlist):
    # Test Exact Match First
    coefflist = []
    i = 0
    for cov in covlist:
        if resp == cov:
            coefflist.append(1)
            return([1, coefflist + [0]*(len(covlist)-(i + 1))])
        i += 1
        coefflist.append(0)
    # No exact match, try regression
    Y = np.array(resp).transpose()
    X = np.array(covlist).transpose()

    coeffs = np.round(np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(Y), 8)
    SSR = np.sum((X.dot(coeffs) - Y)**2)
    return([1*(abs(SSR) < 0.0001), list(coeffs)])
    
        
class Matrix():
    def __init__(self, name,nrow, ncol, populate = True):
        # Constructor: build the matrix object
        self.name = name
        self.nrow = nrow
        self.ncol = ncol
        self.patterns = None
        self._tmpIndexCounter = -1
        self._referencePatterns = None
        if populate:
            self.data = ([[name + "[" + str(j) + "," + str(i) + "]"
                       for i in xrange(ncol)] for j in xrange(nrow)])
    def __repr__(self):
        # make the object representation intelligible
        return("Matrix: " + self.name + ", dim [%d x %d]" % (self.nrow, self.ncol))

    def __str__(self):
        return("\n".join([str(x) for x in self.data]))

    def transpose(self):
        # Build a transposed matrix
        outMatrix = Matrix("(" + self.name + ")_t", self.ncol, self.nrow, populate = False)
        outMatrix.data = ([[self.data[i][j] for i in xrange(self.nrow)]
                 for j in xrange(self.ncol)])
        return(outMatrix)
	
    def matrixAdd(self, Mat2):
        if (self.nrow != Mat2.nrow or self.ncol != Mat2.ncol):
            raise Exception("Matrix Dimensions Differ")
        else:
            outMatrix = Matrix(self.name + "+" + Mat2.name,self.nrow, self.ncol, populate=False)
            outMatrix.data = ([[self.data[i][j] + "+" + Mat2.data[i][j]
                      for j in xrange(self.ncol)] for i in xrange(self.nrow)])
            return(outMatrix)

    def dotProduct(self, vec1, vec2):
        if (len(vec1) != len(vec2)):
            raise Exception("Vector Lengths Differ")
        return("+".join(["(" + vec1[i] + ") * (" + vec2[i] + ")" for i in xrange(len(vec1))]))

    def matrixMult(self, Mat2):
        if (self.ncol != Mat2.nrow):
            raise Exception("Matrix Dimensions Differ")
        else:
            outMatrix = Matrix("(" + self.name + ")X("+Mat2.name+")",
                               self.nrow, Mat2.ncol, populate = False)
            Mat2Trans = Mat2.transpose()
            outMatrix.data = ([[self.dotProduct(self.data[i] , Mat2Trans.data[j])
                               for j in xrange(Mat2Trans.nrow)] for i in xrange(self.nrow) ])
            return(outMatrix)
        
    def scalarAdd(self, scalar):
        strScalar = str(scalar)
        outMatrix = Matrix(self.name + "+" + strScalar, self.nrow, self.ncol)
        outMatrix.data = ([[self.data[i][j] +"+"+ strScalar
                  for j in xrange(self.ncol)] for i in xrange(self.nrow)])
        return(outMatrix)
    
    def scalarMult(self, scalar):
        strScalar = str(scalar)
        outMatrix = Matrix("(" + self.name + ")*" + strScalar, self.nrow, self.ncol)
        outMatrix.data = ([["(" + self.data[i][j] +")*"+ strScalar
                  for j in xrange(self.ncol)] for i in xrange(self.nrow) ])
        return(outMatrix)

    def exp(self):
        outMatrix = Matrix("exp(" + self.name + ")", self.nrow, self.ncol)
        outMatrix.data = ([["exp(" + self.data[i][j] +")"
                  for j in xrange(self.ncol)] for i in xrange(self.nrow) ])
        return(outMatrix)
    
    def log(self):
        outMatrix = Matrix("log(" + self.name + ")", self.nrow, self.ncol)
        outMatrix.data = ([["log(" + self.data[i][j] +")"
                  for j in xrange(self.ncol)] for i in xrange(self.nrow) ])
        return(outMatrix)

    def findReferencePatterns(self):
        outData = ([[re.sub(r"\[\d+,\d+\]","[,]", self.data[i][j]) for j in xrange(self.ncol)] for i in xrange(self.nrow)])
        patterns = set()
        [patterns.add(outData[i][j]) for i in xrange(self.nrow) for j in xrange(self.ncol)]
        print(("Matrix has %d unique pattern" %len(patterns)) + "s"*(len(patterns) > 1))
        patterns = list(patterns)
        outData = [[patterns.index(outData[i][j]) for j in xrange(self.ncol)] for i in xrange(self.nrow)]
        self.patterns = patterns
        return((len(patterns), outData))

    def _findIndexPredictors(self,i,j):
        pairs = ([x[1:].strip("[").strip("]").split(",") for x in re.findall(r"\[\d+,\d+\]", self.data[i][j])])
        outlist = []
        for pair in pairs:
            for item in pair:
                outlist.append(int(item))
        return(outlist)

    def _populateReferencePatterns(self,i,j):
        outData = ([x[0] for x in re.findall(r"\w\[\d+,\d+\]", self.data[i][j])])
        self._referencePatterns = outData
        return(outData)
    
    def findIndexPredictors(self):
        self.findReferencePatterns()
        if len(self.patterns) > 1:
            raise NotImplementedError("This matrix has more than one pattern. This hasn't been implemented yet, but should be very feasible!")
        byrow = [[i,j] + (self._findIndexPredictors(i, j)) for i in xrange(self.nrow) for j in xrange(self.ncol)]
        
        bycol = ([[byrow[i][j] for i in xrange(len(byrow))]
                 for j in xrange(len(byrow[0]))])
        return(bycol)

    
    def findCommonPatternString(self):
        indexPredictors = self.findIndexPredictors()
        preds = [[1]*len(indexPredictors[0])] + indexPredictors[0:2]
        indices = indexPredictors[2:]
        parsed = []

        for index in indices:
            appendstr = ""
            coeffs = reg(index, preds)
            if coeffs[0] == 0:
                raise(Exception("Imperfect Index Prediction"))
            if coeffs[1][1] == coeffs[1][2] == 0:
                appendstr = "0"
            elif coeffs[1][1] == 0:
                appendstr = "global_id_1" if (coeffs[1][2] == 1) else str(coeffs[1][2]) + "*global_id_1"
            elif coeffs[1][2] == 0:     
                appendstr = "global_id_0" if (coeffs[1][1] == 1) else str(coeffs[1][1]) + "*global_id_0"
            else: 
                appendstr1 = "global_id_0" if (coeffs[1][1] == 1) else str(coeffs[1][1]) + "*global_id_0"
                appendstr2 = "global_id_1" if (coeffs[1][2] == 1) else str(coeffs[1][2]) + "*global_id_1"
                appendstr = appendstr1 + " + " + appendstr2
            if coeffs[1][0] != 0:
                if appendstr == "0":
                    appendstr = str(coeffs[1][0])
                else:                    
                    appendstr += (" + " + str(coeffs[1][0]))
            
            parsed.append(appendstr)

        #print parsed
        pairs = []
        i = 0
        self._populateReferencePatterns(0,0)
        #while i <= (len(parsed) -1):
        #    pairs.append("[" + parsed[i] + "," + parsed[i + 1] + "]")
        #    i += 2
        
        while i <= (len(parsed) -1):
            if parsed[i] == parsed[i+1] == "0":
                pairs.append("[0]")
            elif parsed[i] == "0":
                pairs.append("[(" + parsed[i + 1] + ")]")
            elif parsed[i+1] == "0":
                pairs.append("[(" + parsed[i] + ")*" + (self._referencePatterns[i/2] + "cols ") + "]")
            else:                            
                pairs.append("[(" + parsed[i] + ")*" + (self._referencePatterns[i/2] + "cols ") + " + (" + parsed[i + 1] + ")]")
            i += 2            
                
        #return(pairs)
        # The following will need to be generalized for matrices of more than one pattern
        
        def f(x):
            self._tmpIndexCounter += 1
            return(pairs[self._tmpIndexCounter])
        
        outData = re.sub(r"\[\d+,\d+\]",f, self.data[0][0])
        outData = re.sub(r"\]\)\+","])\n+", outData)
        self._tmpIndexCounter = -1
        return(outData)

class Kernel():
    def __init__(self, kernelName ,outMatrix, matrixList):
        patternstr = outMatrix.findCommonPatternString()
        defstr = "".join(["#define %s %d\n#define %s %d\n" % (x.name + "rows", x.nrow, x.name + "cols", x.ncol) for x in matrixList])
        defstr += "#define %s %d\n#define %s %d\n\n" % ("outputrows", outMatrix.nrow, "outputcols", outMatrix.ncol)
        self.source = defstr+"""
__kernel void %s(""" %kernelName
        for matrix in matrixList:
            self.source += "__global float* " + matrix.name + ",\n"
        self.source += "__global float* output)\n{\n"
        self.source += "    int global_id_0 = get_global_id(0);\n"
        self.source += "    int global_id_1 = get_global_id(1);\n"
        self.source += "    output[global_id_0*outputrows + global_id_1] = \n" + patternstr + ";\n}"

    def writeToFile(self, fname):
        f = open(fname, "w")
        f.write(self.source)
        f.close()
            

if __name__ == "__main__":
    A = Matrix("A", 10, 10)
    B = Matrix("B", 10, 10)
    C =A.matrixMult(B).scalarAdd(10).transpose()
    K = Kernel("mmult", C, [A, B])
    print K.source
    K.writeToFile("matmultTestKernel.kernel")





