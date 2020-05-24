import numpy as np
pi = np.pi

'''
>>> np.kron(I, X)
array([[0, 1, 0, 0],
       [1, 0, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]])
>>> np.kron(X, I)
array([[0, 0, 1, 0],
       [0, 0, 0, 1],
       [1, 0, 0, 0],
       [0, 1, 0, 0]])
>>> a = np.array([[1, 0],
...               [0, 1]])
>>> b = np.array([[4, 1],
...               [2, 2]])
>>> np.matmul(a, b)
array([[4, 1],
       [2, 2]])
'''


def U(u, q, Q):
    ''' The states:
            |xQ, xQ-1, ..., xq, xq-1, ..., x0>
    '''
    if q>1 and q<Q:
        I1 = np.identity(2**(q-1))
        I2 = np.identity(2**(Q-q))
        return np.kron(np.kron(I1, u), I2)
    elif q==1:
        return np.kron(u, np.identity(2**(Q-1)))
    elif q==Q:
        return np.kron(np.identity(2**(Q-1)), u)


def H(q, Q):
    h = 1/np.sqrt(2) *np.array([[1, 1],
                                [1, -1]])
    return U(h, q, Q)


def X(q, Q):
    x = np.array([[0, 1],
                  [1, 0]])
    return U(x, q, Q)


def Ry(theta, q, Q):
    ry = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                   [np.sin(theta/2),  np.cos(theta/2)]])
    return U(ry, q, Q)


def CNOT(ctrl, trgt, Q):
    assert Q==2, 'CNOT is not implemented for Q={}'.format(Q)
    if ctrl==1 and trgt==2:
        cnot = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]])
    elif ctrl==2 and trgt==1:
        cnot = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0]])
    return cnot


def CH(ctrl, trgt, Q):
    assert Q==2, 'CNOT is not implemented for Q={}'.format(Q)
    if ctrl==1 and trgt==2:
        cnot = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)],
                         [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]])
    elif ctrl==2 and trgt==1:
        cnot = np.array([[1, 0, 0, 0],
                         [0, 1/np.sqrt(2), 0, 1/np.sqrt(2)],
                         [0, 0, 1, 0],
                         [0, 1/np.sqrt(2), 0, -1/np.sqrt(2)]])
    return cnot


def twoq(t1, t2, t3, t4):
    return mmul([Ry(t1, 1, 2), Ry(t2, 2, 2),
                 CNOT(1, 2, 2),
                 Ry(t3, 1, 2), Ry(t4, 2, 2)])


def mmul(matarr):
    ''' the firts applyed is the first'''
    A = np.identity(len(matarr[0]))
    for mat in matarr:
        A = np.matmul(mat, A)
    return A
    
    
