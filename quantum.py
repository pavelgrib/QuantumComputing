#!/usr/local/bin/python
import cmath
import math
import numpy as np


class BlochState(object):
	def __init__(self, phi, theta):
		self.phi = phi
		self.theta = theta

	def zeroState(self):
		return math.sin(self.theta)

	def oneState(self):
		return cmath.exp(complex(0, math.pi * 0.25)) * math.cos(self.theta)

	def printState(self):
		print str(self.zeroState()) + '|0>' + ' + ' + \
			  str(self.oneState()) + '|1>'

	def xyz(self):
		x = math.cos(self.phi) * math.sin(self.theta *2)
		y = math.sin(2*self.theta) * math.sin(self.phi)
		z = math.cos(self.theta)
		return (x, y, z)

	def probability(self):
		pOne = math.abs( zeroState() )



def complexConjugate(V):
	""" returns new array """
	conjV = np.array(V)
	if V.dtype.name.find('complex') >= 0:
		for v in np.nditer(conjV, op_flags=['readwrite']):
			v[...] = complex(np.real(v), -np.imag(v))
	return conjV

def transpose(V):
	""" returns new array with shape transposed from the shape of V """
	assert( len(V.shape) == 2 )
	transV = np.zeros(shape=V.shape[::-1], dtype=complex)
	for i in range(transV.shape[0]):
		for j in range(transV.shape[1]):
			transV[i,j] = V[j,i]
	return transV

def adjoint(M):
	return transpose( complexConjugate( M ) )

def innerProduct(V1, V2):
	assert(V1.shape == V2.shape and V1.shape[1] == 1)
	adjV1 = adjoint(V1)
	result = 0.0;
	for i in range(V1.shape[0]):
		result += adjV1[0,i] * V2[i]
	return result

def norm(V):
	return np.sqrt(np.real(innerProduct(V, V)))

def distance(V1, V2):
	return norm( V1 - V2 )

def outerProduct(V1, V2):
	assert( V1.shape[1] == V2.shape[0] )
	result = np.zeros( (V1.shape[0], V2.shape[1]), dtype=complex )
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			for k in range(V1.shape[1]):
				result[i,j] = V1[i,k] * V2[k,j]
	return result

def isHermitian(M):
	""" Hermitian matrices equal their complex conjugate transpose (adjoint) """
	if M.shape[0] != M.shape[1]:
		return False
	else:
		return np.allclose( adjoint(M), M )

def isUnitary(M):
	"""unitary matrices have inverses which  equal their adjoint """
	if M.shape[0] != M.shape[1]:
		return False
	else:
		return np.allclose( M.dot(adjoint(M)), np.identity(M.shape[0]) )

def tensor(M1, M2):
	result = np.zeros((M1.shape[0]*M2.shape[0], M1.shape[1]*M2.shape[1]), dtype=complex)
	for i in range(M1.shape[0]):
		for j in range(M1.shape[1]):
			# print ((i+1)*M1.shape[0])
			result[(i*M1.shape[0]):((i+1)*M1.shape[0]), 
				   (j*M2.shape[0]):((j+1)*M2.shape[1])] = M1[i,j] * M2
	return result

def commutator(Operator1, Operator2):
	assert(Operator1.shape == Operator2.shape and Operator1.shape[0] == Operator1.shape[1])
	return Operator1.dot(Operator2) - Operator2.dot(Operator1)

def expectedValue(Operator):
	def innerFunc(State):
		return innerProduct(Operator.dot(State), State)
	return innerFunc

def variance(Operator):
	expVal = expectedValue(Operator)
	def innerFunc(State):
		temp = Operator - expVal(State) * np.identity(State.shape[0])
		return adjoint(State).dot(temp.dot(temp.dot(State)))
	return innerFunc 

def ControlledPhase(phi):
	phase = cmath.exp(complex(0, phi))
	return np.hstack( (np.vstack( (np.identity(3) , np.zeros((1,3))) ), \
					   np.vstack( (np.zeros((3,1)), phase          ) ) ) )

def meanInverter(n):
	return 2.0/n * np.ones((n, n)) - np.identity( n )

def faN(a, N):
	def innerFunc(fan_x_prior):
		return (fan_x_prior * a) % N

	def returnFunc(x):
		result = 1
		for i in range(x):
			result = innerFunc(result)
		return result
	return returnFunc

def euclid(a, b):
	""" returns GCD of a and b assuming a < b """
	if b==0:
		return a
	else:
		return euclid(b, a % b)

def tests():
	m1 = np.array([[1,0,0,0],[0,1,0,0], [0,0,0,1], [0,0,1,0]])
	# print adjoint(m1)

	v1 = np.ones((5,1), dtype=complex)
	v2 = np.ones((5,1), dtype=complex)
	
	v1[3] = -2.5 + 1j
	v2[1] = 3+2j
	# print innerProduct(v1, v2)
	# print distance(v1, v2)
	# print outerProduct(v1, v2)

	theta = math.pi/8.0
	m2 = np.array([[math.cos(theta), -math.sin(theta), 0], 
				   [math.sin(theta), math.cos(theta), 0],
				   [0, 0, 1]])
	# print m2
	# print isHermitian(m1)
	
	m3 = 3*np.ones((2,2))
	m4 = np.array([[1,2],[2j,3j]])
	# print tensor(m3, m4)


	## example 4.2.5
	Omega = np.array([[1,-1j], [1j, 2]])
	sqrt2Over2 = math.sqrt(2.0)/2.0
	phi = sqrt2Over2 * np.array([1, complex(0, 1)]).reshape(2,1)
	OmegaE = expectedValue(Omega)
	print OmegaE(phi)
	OmegaVar = variance(Omega)
	print OmegaVar(phi)
	# end of example 4.2.5

	Sx = np.array([[0,1],[1,0]])
	Sy = np.array([[0,-1j], [1j, 0]])
	Sz = np.array([[1,0],[0,-1]])
	Hadamard = sqrt2Over2 * np.array( [[1, 1], [1, -1]] )
	print isUnitary(Hadamard)
	print isUnitary(Sx.dot(Hadamard))
	print isUnitary(Hadamard.dot(Sx))

	print commutator(Sx, Sy)
	print commutator(Sx, Sz)
	print commutator(Sy, Sz)

	V = np.array([5+3j, 6j]).reshape(2,1)

	CNOT = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])

	I4 = np.identity(4)
	Z4 = np.zeros((4,4))
	F4 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
	TOFFOLI = np.hstack( (np.vstack((I4, Z4)), np.vstack((Z4, CNOT))) )
	FREDKIN = np.hstack( (np.vstack((I4, Z4)), np.vstack((Z4, F4))) )
	S = np.array([[1,0,], [0,1j]])
	T = np.array([[1,0,], [0, cmath.exp(complex(0,0.25*math.pi))]])

	# normal states
	N00 = np.array([1,0,0,0]).reshape(4,1)
	N01 = np.array([0,1,0,0]).reshape(4,1)
	N10 = np.array([0,0,1,0]).reshape(4,1)
	N11 = np.array([0,0,0,1]).reshape(4,1)

	newState = BlochState(math.pi/2, math.pi/6)
	newState.printState()

	# revCNOT = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
	# print CNOT.dot(revCNOT.dot(CNOT))


	print CNOT.dot( tensor( Hadamard, np.identity(2) ) ).dot(N11)

	someArray = np.random.randint(low=0, high=100, size=(10,1))
	print meanInverter(10).dot(someArray).reshape(1,10)

	f_6_371 = faN(6, 371)
	print f_6_371(13)

	print euclid( 24, 666)

	# bell states
	# Bphi_plus = 
tests()
