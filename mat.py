from vec import Vec


def getitem(M, k):
    """
    Input: Matrix M and value-pair k
    Returns: the value of entry k in M.

    >>> M = Mat( ({0,1,2},{0,1}), {(0,0):1,(0,1):2,(1,0):3,(1,1):4,(2,0):10,(2,1):0})
    >>> getitem(M, (1,0))
    3
    
    >>> M = Mat( ({0,1,2},{0,1}), {(0,0):1,(0,1):2,(1,0):3,(1,1):4,(2,0):10,(2,1):0})
    >>> M[1,0]
    3
    """
    assert k[0] in M.D[0] and k[1] in M.D[1]

    return M.f.get(k,0)


def setitem(M, k, val):
    """
    input: matrix M, label-pair k and value k
    Sets the element of v with label k to be val.  The value of k should be a pair

    >>> M = Mat( ({0,1,2},{0,1}), {(0,0):1,(0,1):2,(1,0):3,(1,1):4,(2,0):10,(2,1):0})
    >>> M[1,0] = 17
    >>> M == Mat( ({0,1,2},{0,1}), {(0,0):1,(0,1):2,(1,0):17,(1,1):4,(2,0):10,(2,1):0})
    True
    """
    assert k[0] in M.D[0] and k[1] in M.D[1]
    M.f[k] = val


def add(A, B):
    """
    Input: matrices A and B
    Returns the sum of A and B

    >>> A = Mat( ({0,1,2},{0,1}), {(0,0):1,(0,1):2,(1,0):3,(1,1):4,(2,0):10,(2,1):0})
    >>> B = Mat( ({0,1,2},{0,1}), {(0,0):6,(0,1):-13,(1,0):-8,(1,1):40,(2,0):1,(2,1):2})
    >>> A_plus_B = add(A, B)
    >>> A_plus_B == Mat(({0,1,2},{0,1}), {(0,0):7,(0,1):-11,(1,0):-5,(1,1):44,(2,0):11,(2,1):2})
    True
    """
    assert A.D == B.D

    return Mat(A.D, {(r,c):(A[r,c]+B[r,c]) for r in A.D[0] for c in A.D[1]})


def scalar_mul(M, alpha):
    "Returns the product of scalar alpha with M" 

    return Mat(M.D, {(r,c):alpha*M[r,c] for r in M.D[0] for c in M.D[1]})


def equal(A, B):
    "Returns true iff A is equal to B"
    assert A.D == B.D

    return all({A[r,c]==B[r,c] for r in A.D[0] for c in A.D[1]})


def transpose(M):
    """
    Returns the transpose of M

    >>> A = Mat( ({0,1,2},{0,1}), {(0,0):1,(0,1):2,(1,0):3,(1,1):4,(2,0):10,(2,1):0})
    >>> B = transpose(A)
    >>> B == Mat(({0,1},{0,1,2}),{(0,0):1,(0,1):3,(0,2):10,(1,0):2,(1,1):4,(1,2):0})
    True
    """
    return Mat((M.D[1],M.D[0]), {(c,r):M[r,c] for c in M.D[1] for r in M.D[0]})


def vector_matrix_mul(v, M):
    """
    Returns: a vector, the product of vector v and matrix M

    >>> M = Mat( ({0,1,2},{0,1}), {(0,0):1,(0,1):2,(1,0):3,(1,1):4,(2,0):10,(2,1):0})
    >>> v = Vec({0,1,2},{0:1,1:7,2:2})
    >>> v_M = v*M
    >>> v_M == Vec({0,1},{0:42,1:30})
    True
    """
    assert M.D[0] == v.D

    return Vec(M.D[1], {c:sum(v[r]*M[r,c] for r in M.D[0]) for c in M.D[1]})


def matrix_vector_mul(M, v):
    "Returns the product of matrix M and vector v"
    assert M.D[1] == v.D

    return Vec(M.D[0], {r:sum(M[r,c]*v[c] for c in M.D[1]) for r in M.D[0]})


def matrix_matrix_mul(A, B):
    """
    Returns the product of A and B

    >>> A = Mat(({0, 1, 2}, {0, 1}), {(0, 1): 2, (0, 0): 1, (2, 1): 0, (2, 0): 10, (1, 0): 3, (1, 1): 4})
    >>> B = Mat(({0, 1}, {0, 1, 2}), {(0, 1): -8, (1, 2): 2, (0, 0): 6, (1, 1): 40, (1, 0): -13, (0, 2): 1})
    >>> AB = matrix_matrix_mul(A,B)
    >>> AB == Mat(({0, 1, 2},{0, 1, 2}), {(0,0):-20, (0,1):72, (0,2):5, (1,0):-34, (1,1):136, (1,2):11, (2,0):60, (2,1):-80, (2,2):10})
    True
    
    """
    assert A.D[1] == B.D[0]
    
    D = (A.D[0], B.D[1])

    return Mat(D, {(r,d): sum(A[r,c]*B[c,d] for c in A.D[1]) for r in D[0] for d in D[1]})


################################################################################

class Mat:
    def __init__(self, labels, function):
        self.D = labels
        self.f = function

    __getitem__ = getitem
    __setitem__ = setitem
    transpose = transpose

    def __neg__(self):
        return (-1)*self

    def __mul__(self,other):
        if Mat == type(other):
            return matrix_matrix_mul(self,other)
        elif Vec == type(other):
            return matrix_vector_mul(self,other)
        else:
            return scalar_mul(self,other)
            #this will only be used if other is scalar (or not-supported). mat and vec both have __mul__ implemented

    def __rmul__(self, other):
        if Vec == type(other):
            return vector_matrix_mul(other, self)
        else:  # Assume scalar
            return scalar_mul(self, other)

    __add__ = add

    def __sub__(a,b):
        return a+(-b)

    __eq__ = equal

    def copy(self):
        return Mat(self.D, self.f.copy())

    def __str__(M, rows=None, cols=None):
        "string representation for print()"
        if rows == None:
            try:
                rows = sorted(M.D[0])
            except TypeError:
                rows = sorted(M.D[0], key=hash)
        if cols == None:
            try:
                cols = sorted(M.D[1])
            except TypeError:
                cols = sorted(M.D[1], key=hash)
        separator = ' | '
        numdec = 3
        pre = 1+max([len(str(r)) for r in rows])
        colw = {col:(1+max([len(str(col))] + [len('{0:.{1}G}'.format(M[row,col],numdec)) if isinstance(M[row,col], int) or isinstance(M[row,col], float) else len(str(M[row,col])) for row in rows])) for col in cols}
        s1 = ' '*(1+ pre + len(separator))
        s2 = ''.join(['{0:>{1}}'.format(c,colw[c]) for c in cols])
        s3 = ' '*(pre+len(separator)) + '-'*(sum(list(colw.values())) + 1)
        s4 = ''.join(['{0:>{1}} {2}'.format(r, pre,separator)+''.join(['{0:>{1}.{2}G}'.format(M[r,c],colw[c],numdec) if isinstance(M[r,c], int) or isinstance(M[r,c], float) else '{0:>{1}}'.format(M[r,c], colw[c]) for c in cols])+'\n' for r in rows])
        return '\n' + s1 + s2 + '\n' + s3 + '\n' + s4

    def pp(self, rows, cols):
        print(self.__str__(rows, cols))

    def __repr__(self):
        "evaluatable representation"
        return "Mat(" + str(self.D) +", " + str(self.f) + ")"

if __name__ == '__main__':
    import doctest
    print(doctest.testmod(verbose=False))
