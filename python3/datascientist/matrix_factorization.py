import numpy

def get_rating_error(r, p, q):
    return r - numpy.dot(p, q)

def get_error(R, P, Q, beta):
    error = 0.0
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] == 0:
                continue
            error += pow(get_rating_error(R[i][j], P[:,i], Q[:,j]), 2)
    error += beta/2.0 * (numpy.linalg.norm(P) + numpy.linalg.norm(Q))
    return error

def matrix_factorization(R, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
    P = numpy.random.rand(K, len(R))
    Q = numpy.random.rand(K, len(R[0]))
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] == 0:
                    continue
                err = get_rating_error(R[i][j], P[:, i], Q[:, j])
                for k in range(K):
                    P[k][i] += alpha * (2 * err * Q[k][j])
                    Q[k][j] += alpha * (2 * err * P[k][i])
        
        error = get_error(R, P, Q, beta)
        if error < threshold:
            break
    return P, Q

R = numpy.array([
    [5, 3, 0, 0],
    [4, 0, 4, 1],
    [1, 1, 0, 5],
    [0, 0, 4, 4],
    [0, 1, 5, 4],
])
nP, nQ = matrix_factorization(R, 2)
nR = numpy.dot(nP.T, nQ)
print(nR)
