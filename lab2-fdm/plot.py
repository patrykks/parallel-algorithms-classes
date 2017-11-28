from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 3:
    print('Incorrect number of arguments. Usage: python plot.py <INPUT_FILE> <OUTPUT_FILE>')

input = sys.argv[1]
output = sys.argv[2]

matrix_2d = data = np.loadtxt(input)

X = []
Y = []
Z = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for row in xrange(matrix_2d.shape[0]):
    for col in xrange(matrix_2d.shape[1]):
        X.append(row)
        Y.append(col)
        Z.append(matrix_2d[row][col])

ax.scatter(X, Y, Z, c='r', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.savefig(output)