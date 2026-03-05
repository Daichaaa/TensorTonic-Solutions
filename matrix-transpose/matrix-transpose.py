import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns) without .T or np.transpose()
    """
    #Define transpose matrix with dimensions (m,n)
    A = np.array(A)
    n,m = A.shape
    A_transpose = np.zeros((m,n))

    #Fill the values of the transpose matrix
    for i in range(n):
        for j in range(m):
            A_transpose[j][i] = A[i][j]
    return A_transpose
