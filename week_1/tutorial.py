###################################
# Introduction to PYTHON
# This tutorial was adapted from Matlab Tutorial from here: https://people.cs.pitt.edu/~kovashka/cs1674_sp22/tutorial.m
#
#
###################################


# Basics

# (a) Commands

# PYTHON is an interpreted language. Hence, you can use it as a calculator
# (I sometimes do!)

print(' ----- Section 1 - calculator ----- ')
print(3 + 5)

# Python's command line is a little like a standard shell:
# - Use the up arrow to recall commands without retyping them (and
#   down arrow to go forward in the command history).

# The symbol "#" is used to indicate a comment (for the remainder of
# the line).


# (b) Printing

# IMPORTANT: To manipulate matrices in python, we use numpy package.

wait = input("Press Enter to continue.")
print(' ----- Section 2 - printing/matrices ----- ')
import numpy as np

A = np.array([[1,2], [2,3]]);

# ways to print A include:

print(A)

# Other ways

print(A[0,0], A[0,1], A[1,0], A[1,1])

# (note you have to specify how many numbers (%u's) you want to print and order matters).

# Printing other things:

s = 'friend'

print('Hello',s)

m = 5;
print('See you in ',m,' minutes');
print('See you in ', str(m),' minutes');

# To get the type of a variable, which can be helpful when debugging:

print(type(A))

# or to see if it's Inf, NaN, etc.

print(np.isinf(m))
print(np.isnan(m))
print(np.isnan(np.NaN))

# (c) Functions and scripts

# Python scripts are files with ".py" extension containing Python
# commands. This file is a script.
# Variables in a script file are global and will change the
# value of variables of the same name in the environment of the current
# Python session.  A script with name "script1.py" can be invoked by
# typing "python3 script1.py" in the shell window for Linux.

# Functions are also python-files or can be written inside a same file. The first line in a function file must be
# of this form:

# def function(inarg_1, ..., inarg_m):
#
#
#   ...........
#
#   return outarg_1, ..., outarg_m

# Functions are executed using local workspaces: there is no risk of
# conflicts with the variables in the main workspace. At the end of a
# function execution only the output arguments will be visible in the
# main workspace.
wait = input("Press Enter to continue.")
print(' ----- Section 3 - Functions ----- ')

def myfunction(x):
    a = 42
    print('The value of a in myfunction is:', str(a));
    return a+x

from myotherfunction import myotherfunction

a = 1674;                    # Global variable a
b = myfunction(2 * a);       # Call myfunction which has local variable a

print('The value of a in the script is:', a);

c, d = myotherfunction(a, b);     # Call myotherfunction with two return values

print(c,d)

# (d) Reserved/special words:

# for, if, while, def, return, else, elif, case
# continue, else, try, except, global, break, as


# (e) Simple debugging

# To start debugging within the program just insert import pdb, pdb.set_trace() commands.  
# Run your script normally, and execution will stop where we have introduced a breakpoint. 
# So basically we are hard coding a breakpoint on a line below where we call set_trace()

# (f) Documentation

# Python source has official documentation as well their packages.

# (h) Paths
# To add a path to some directory whose functions/scripts you're calling:

# import sys
# sys.path.append(YOUR_PATH)

# (2) Basic types in PYTHON

# (a) The basic types in PYTHON are scalars (usually double-precision
# floating point), vectors, and matrices:

wait = input("Press Enter to continue.")
print(' ----- Section 4a - arrays ----- ')
A = np.array([[1,2],[3,4]]) # Creates a 2x2 matrix

B = np.array([[1,2],[3,4]])  # The simplest way to create a matrix is
                             # to list its entries in square brackets.
                             # The "],[" symbol separates rows;
                             # the "," separates columns.

N = 5                        # A scalar
print('scalar: ', N)
v = np.array([[1,0,0]])      # A row vector
print('row vector: ', v)
v = np.array([[1],[0],[0]])  # A column vector
print('column vector: ', v)
v = v.T                      # Transpose a vector (row to column or
                             #   column to row)
print('transpose vector: ', v)

v = np.linspace(1,3,5)       # A vector filled in a specified range:
print('spaced vector: ', v)
v = []                       # Empty vector
print('empty vector: ', v)

# Creating special matrices: 1ST parameter is ROWS, 2ND parameter is COLS 
wait = input("Press Enter to continue.")
print(' ----- Section 4b - special matrices----- ')
m = np.zeros((2, 3))         # Creates a 2x3 matrix of zeros
print('zeros matrix: ', m)
v = np.ones((1, 3))          # Creates a 1x3 matrix (row vector) of ones
print('ones matrix: ', m)
m = np.eye(3)                # Identity matrix (3x3)
print('eye matrix: ', m)
v = np.random.rand(3, 1)     # Randomly filled 3x1 matrix (column
                             # vector); see also randn
print('rand vector: ', v)

                             # But watch out:
m = np.zeros(3)              # Creates a (3,) vector (!) of zeros
print('zeros vector: ', m)

# (c) Indexing vectors and matrices.
# Warning: Indices always start at 0
wait = input("Press Enter to continue.")
print(' ----- Section 4c - indexing----- ')

v = np.array([1,2,3])

print(v[2])                  # Access a vector element 
print(v[-1])                 # Access the last element
print('Number of scalars: ', len(v))                # Number of scalars in vector. Compare to:
print('Matrix Shape: ', v.shape)               # Returns the size of a matrix

M = np.array([[1,2,3,4],[5,7,8,8],[9,10,11,12],[13,14,15,16]])
print(M[0,2])                # Access a matrix element
                             # matrix(ROW #, COLUMN #)
print('Access matrix row: ', M[1,:])                # Access a whole matrix row (2nd row)
print('Access matrix column: ', M[:,0])                # Access a whole matrix column (1st column)

print('Access subcells: ', M[0,0:3])              # Access elements 1 through 3 of the 1st row


print('Access subcells: ', M[1:3, 1])             # Access elements 2 through 3 of the
                             #   2nd column
print('Access subcells: ', M[1:, 3])              # Keyword ":" accesses the remainder of a
                             #   column or row

C = np.random.rand(4, 3, 2)*10            # A three-dimensional array
D = np.random.rand(2, 1, 3)*10            # A three-dimensional array... but annoying

M = np.array([[1,2,3],[4, 5, 6]])   

print(M.shape)               # Returns the size of a matrix
print('# of rows: ', M.shape[0])            # Number of rows
print('# of columns: ', M.shape[1])            # Number of columns

M1 = np.ones(M.shape)       # Create a new matrix with the size of m
print(M1)
                             
# (d) Cell arrays and structures
wait = input("Press Enter to continue.")
print(' ----- Section 4d - cell/structures ----- ')

# Creates a cell array 
A = [np.array([[1, 4, 3],[0, 5, 8],[7, 2, 9]]), 'Anne Smith', 3]

print(A[0])         # Return cell
print(A[0][0,0])    # Return first element of first cell

# 1-minute break! 

# (3) Simple operations on vectors and matrices

# (a) Element-wise operations:

# These operations are done "element by element".  If two
# vectors/matrices are to be added, subtracted, or element-wise
# multiplied or divided, they must have the same size.
wait = input("Press Enter to continue.")
print(' ----- Section 5a - operations vectors/matrices ----- ')

a = np.array([1,2,3,4]) # A row vector
print('a: ', a)

print('2 * a: ', 2 * a)
print('a + 5: ', a + 5)
print('5 + a: ', 5 + a)
print('a / 4: ', a / 4)
print('4 / a: ', 4 / a)

b = np.array([5,6,7,8]) # Another row vector
print('Operations with vectors')
print('a: ', a)
print('b: ', b)

print('a+b: ', a+b)          # Vector addition
print('a+b.T: ', a+b.T)        # What does this return?
print('a-b: ', a-b)          # Vector subtraction
print('a**2: ', a**2)         # Element-wise squaring
print('a*b: ', a*b)          # Element-wise multiplication
print('a/b: ', a/b)          # Element-wise division

print('log: ', np.log(a))     # Element-wise base 2 logarithm
print('power: ', np.power(a,3)) # Element-wise cubing
print('round: ', np.round([1.4, 2.6])) # Element-wise rounding to nearest integer

# ... and plenty more intuitive examples like this.
# Other element-wise arithmetic operations include e.g. 
#   floor, ceil, ...

# (b) Vector Operations
wait = input("Press Enter to continue.")
print(' ----- Section 5b - vector operations----- ')

a = np.array([1,4,6,3]) # A row vector

print('sum: ', np.sum(a))        # Sum of vector elements
print('mean: ', np.mean(a))       # Mean of vector elements
print('std: ', np.std(a))        # Standard deviation
print('max: ', np.max(a))        # Maximum

print('sort: ', np.sort(a))       # Sorting

# If a matrix is given, then these functions will operate on each column
#   of the matrix and return a row vector as result

A = np.array([[1,2,3],[4,5,6]])  # A matrix

print('mean: ', np.mean(A))               # Mean of each column
print('Column mean: ', np.mean(A,0))             # Same thing (second argument specifies dimension along which operation is taken)
print('Row mean: ', np.mean(A,1))             # Mean of each *row*


# Inner and outer products - STOP HERE.

A = np.array([[1,2,3]])
print('Dot/inner product: ',A@A.T)                 #   1x3 row vector times a 3x1 column vector
                             #   results in a scalar.  Known as dot product
                             #   or inner product.  Note the use of '@'

print('Outer product: ',A.T@A)                 # 3x1 column vector times a 1x3 row vector
                             # results in a 3x3 matrix.  Known as outer
                             # product.  Note the use of '@'

# (c) Matrix Operations:
wait = input("Press Enter to continue.")
print(' ----- Section 5c - Matrix Operations----- ')

A = np.random.rand(3,2)      # A 3x2 matrix of random numbers
B = np.random.rand(2,4)      # A 2x4 matrix of random numbers
C = A@B                      # Matrix product results in a 3x4 matrix
print('A: ', A)
print('B: ', B)
print('A*B [matrix product]: ', C)


A = np.array([[1,2],[3,4],[5,6]])   # A 3x2 matrix
B = np.array([[5,6,7]])             # A 1x3 row vector
print('A: ', A)
print('B: ', B)
print('B*A [matrix product]: ', B@A) # Vector-matrix product results in
                                    # a 1x2 row vector

C = np.array([[8],[9]])
print('A*C: ', A@C)                          # Matrix-vector product results in a 3x1 column vector


A = np.array([[1,3,2],[6,5,4],[7,8,9]]) # a 3x3 matrix
print('A: ', A)

print('Inverse Matrix of A: ', np.linalg.inv(A))                    # Matrix inverse of a
print('Eigen values of A: ', np.linalg.eig(A))                    # Vector of eigenvalues of a


# Commutativity 
print('Commutativity')

B = np.random.rand(3, 3) * 7

print('A*B: ', A@B)                          # Matrix product
print('B*A: ', B@A)                          # Does this result equal the previous line?
                                    # How to check for equality? 

print('Check equality: ', A@B == B@A)                   # Element-wise result
print('Conmutative comparison: ', np.all(A@B == B@A))           # Global result

# Distributivity
print('Distributivity')

C = np.random.rand(2, 3)

r1 = C @ (A + B)
r2 = C @ A + C @ B                  # Are r1 and r2 equal? 

print(r1)
print(r2)


# Associativity
print('Associativity')

r1 = (C @ A) @ B
r2 = C @ (A @ B)

print(r1)
print(r2)

# See more: https://en.wikipedia.org/wiki/Matrix_multiplication

# (d) Reshaping and assembling matrices:
wait = input("Press Enter to continue.")
print(' ----- Section 5d - reshaping/assembling matrices----- ')

A = np.array([[1,2],[3,4],[5,6]])   #   A 3x2 matrix
B = A.reshape((6,1))                #   Make 6x1 column vector by stacking 
                                    #   up columns of a: in column-major order

print(A)
print(B)

A = np.array([1,2])
B = np.array([3,4])                 #  Two row vectors

C = np.hstack((A,B))                 # Horizontal concatenation

print('Horizontal concatenation: ', C)


A = np.array([[1],[2],[3]])

C = np.vstack((A,np.array([[4]])))       # Vertical concatenation

print('Vertical concatenation: ', C)

# THE SAME FOR MATRICES

# 1-minute break!

# (4) Control statements & vectorization

# Syntax of control flow statements:

# for i in range_value:
#     STATEMENT
#      ...
#     STATEMENT
#

# while EXPRESSION:
#     STATEMENTS
#
# if EXPRESSION:
#     STATEMENTS
# elif EXPRESSION:
#     STATEMENTS
# else:
#     STATEMENTS
# end
#
#   The operators are <, >, <=, >=, ==, ~=  (almost like in C++)

# Warning, IMPORTANT:
#   Loops run very slowly in PYTHON, because of interpretation overhead.
#   you should
#   nevertheless try to avoid them by "vectorizing" the computation,
#   i.e. by rewriting the code in form of matrix operations.  This is
#   illustrated in some examples below.

# Examples:
wait = input("Press Enter to continue.")
print(' ----- Section 6 - for loops----- ')

for i in range(10):
    print(i)

for i in np.linspace(1,7,7):
    print(i)


for i in [5, 13, -1]:
    if i > 10:
        print('Larger than 10')
    elif i < 0:
        print('Negative value')
    else:
        print('Something else')


m = 50
n = 10
A = np.ones((m,n))
v = 2 * np.random.rand(1,n)

# Implementation using loops:

for i in range(m):
    A[i,:] = A[i,:] - v

#print(A)

# We can compute the same thing using only matrix operations. How? 
# This version of the code runs, much faster !!!

A2 = np.ones((m,n)) - v

print('Loop vs vectorization: ', np.all(A2==A))

# We can vectorize the computation even when loops contain conditional statements.

# Example: given an mxn matrix A, create a matrix B of the same size containing all zeros, and then copy into B the elements of A that are greater than zero.

# Implementation using loops:

B = np.zeros((m,n))

for i in range(m):
    for j in range(n):
        if A[i,j] > 0 :
            B[i,j] = A[i,j]

# All this can be computed w/o any loop! How? 

B2 = np.zeros((m,n))
ind = A>0   # Find indices of positive elements of A
B2[ind] = A[ind] # Copies into B only the elements of A that are > 0

print('Loop vs vectorization example 2:', np.all(B2==B))


#  How to do inner product in a slow way? 

row_vector    = np.array([[1,2,3,4]])
column_vector = np.array([[4],[5],[6],[7]])

# inner product
scalar_multiplication = row_vector @ column_vector

print('scalar_multiplication: ', scalar_multiplication)

# ugly inner product
manual_scalar_mult = 0
for i in range(row_vector.shape[1]):
    manual_scalar_mult = manual_scalar_mult + (row_vector[0,i] * column_vector[i,0])

print('manual_scalar_mult: ', manual_scalar_mult)

# (5) Plotting (useful later)
wait = input("Press Enter to continue.")
print(' ----- Section 7 - Ploting ----- ')

import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4]) # Basic plotting

plt.plot(x) # Plot x versus its index values
plt.title('Plot 1')
plt.show()  # show the diagram

plt.plot(x, 2*x) # Plot 2*x versus x
plt.title('Plot 2')
plt.show()

x = np.pi * np.linspace(-24,24,48)/24;
plt.plot(x, np.sin(x))
plt.xlabel('radians')
plt.ylabel('sin value')
plt.title('Plot 3: dummy')
plt.show()

figs, axs = plt.subplots(1, 2) # Multiple functions in separate graphs
axs[0].plot(x,np.sin(x))
axs[1].plot(x,2*np.cos(x))
plt.title('Plot 4')
plt.show()

plt.plot(x, np.sin(x)) # Multiple functions in single graph
plt.plot(x, 2*np.cos(x))
plt.plot(x, 3*np.cos(x))
plt.legend(["sin", "2cos", "3cos"], loc ="lower right")
plt.title('Plot 5')
plt.show()


# (6) Working with images
wait = input("Press Enter to continue.")
print(' ----- Section 8 - images ----- ')

import cv2
I = cv2.imread('pittsburgh.png') # Read a PNG image
cv2.imshow('image',I)            # Display the image
cv2.waitKey(0)
print(I)

I_resize = cv2.resize(I, (I.shape[0]//4,I.shape[1]//4), interpolation = cv2.INTER_AREA) # Resize to 50% using bilinear interpolation
cv2.imshow('resize image', I_resize)
cv2.waitKey(0)

I_crop =  cv2.rotate(I, cv2.ROTATE_90_COUNTERCLOCKWISE) # Rotate 90 degrees and crop to  original size
cv2.imshow('crop image', I_crop)
cv2.waitKey(0)
	
I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) # Grayscale
cv2.imshow('gray image', I_gray)
cv2.waitKey(0)


I = I.astype(np.float32) # to float32

I = I - np.mean(I, axis = (0,1))
I = np.where(I<0, 0, I)
I = np.where(I>255, 255, I)
I = I.astype(np.uint8)

cv2.imshow('mean image', I)
cv2.waitKey(0)