# Code for computing crystal electric fields

import numpy as np
from scipy import optimize
import scipy.linalg as LA
from scipy.special import wofz



class Operator():
    def __init__(self, J):
        self.O = np.zeros((int(2*J+1), int(2*J+1)))
        self.m = np.arange(-J,J+1,1)
        self.j = J

    @staticmethod
    def Jz(J):
        obj = Operator(J)
        for i in range(len(obj.O)):
            for k in range(len(obj.O)):
                if i == k:
                    obj.O[i,k] = (obj.m[k])
        return obj

    @staticmethod
    def Jplus(J):
        obj = Operator(J)
        for i in range(len(obj.O)):
            for k in range(len(obj.O)):
                if k+1 == i:
                    obj.O[i,k] = np.sqrt((obj.j-obj.m[k])*(obj.j+obj.m[k]+1))
        return obj

    @staticmethod
    def Jminus(J):
        obj = Operator(J)
        for i in range(len(obj.O)):
            for k in range(len(obj.O)):
                if k-1 == i:
                    obj.O[i,k] = np.sqrt((obj.j+obj.m[k])*(obj.j-obj.m[k]+1))
        return obj

    @staticmethod
    def Jx(J):
        objp = Operator.Jplus(J)
        objm = Operator.Jminus(J)
        return 0.5*objp + 0.5*objm

    @staticmethod
    def Jy(J):
        objp = Operator.Jplus(J)
        objm = Operator.Jminus(J)
        return -0.5j*objp + 0.5j*objm

    def __add__(self,other):
        newobj = Operator(self.j)
        if isinstance(other, Operator):
           newobj.O = self.O + other.O
        else:
           newobj.O = self.O + other*np.identity(int(2*self.j+1))
        return newobj

    def __radd__(self,other):
        newobj = Operator(self.j)
        if isinstance(other, Operator):
            newobj.O = self.O + other.O
        else:
            newobj.O = self.O + other*np.identity(int(2*self.j+1))
        return newobj

    def __sub__(self,other):
        newobj = Operator(self.j)
        if isinstance(other, Operator):
            newobj.O = self.O - other.O
        else:
            newobj.O = self.O - other*np.identity(int(2*self.j+1))
        return newobj

    def __mul__(self,other):
        newobj = Operator(self.j)
        if (isinstance(other, int) or isinstance(other, float) or isinstance(other, complex)):
           newobj.O = other * self.O
        else:
           newobj.O = np.dot(self.O, other.O)
        return newobj

    def __rmul__(self,other):
        newobj = Operator(self.j)
        if (isinstance(other, int) or isinstance(other, float)  or isinstance(other, complex)):
           newobj.O = other * self.O
        else:
           newobj.O = np.dot(other.O, self.O)
        return newobj

    def __pow__(self, power):
        newobj = Operator(self.j)
        newobj.O = self.O
        for i in range(power-1):
            newobj.O = np.dot(newobj.O,self.O)
        return newobj

    def __neg__(self):
        newobj = Operator(self.j)
        newobj.O = -self.O
        return newobj

    def __repr__(self):
        return repr(self.O)



def StevensOp(J,n,m):
	"""generate stevens operator for a given total angular momentum
	and a given n and m state"""
	Jz = Operator.Jz(J=J)
	Jp = Operator.Jplus(J=J)
	Jm = Operator.Jminus(J=J)
	X = J*(J+1.)

	if [n,m] == [0,0]:
		return np.zeros((int(2*J+1), int(2*J+1)))
	elif [n,m] == [1,0]:
		matrix = Jz
	elif [n,m] == [1,1]:
		matrix = 0.5 *(Jp + Jm)
	elif [n,m] == [1,-1]:
		matrix = -0.5j *(Jp - Jm)

	elif [n,m] == [2,2]:
		matrix = 0.5 *(Jp**2 + Jm**2)
	elif [n,m] == [2,1]:
		matrix = 0.25*(Jz*(Jp + Jm) + (Jp + Jm)*Jz)
	elif [n,m] == [2,0]:
		matrix = 3*Jz**2 - X
	elif [n,m] == [2,-1]:
		matrix = -0.25j*(Jz*(Jp - Jm) + (Jp - Jm)*Jz)
	elif [n,m] == [2,-2]:
		matrix = -0.5j *(Jp**2 - Jm**2)

	elif [n,m] == [4,4]:
		matrix = 0.5 *(Jp**4 + Jm**4)
	elif [n,m] == [4,3]:
		matrix = 0.25 *((Jp**3 + Jm**3)*Jz + Jz*(Jp**3 + Jm**3))
	elif [n,m] == [4,2]:
		matrix = 0.25 *((Jp**2 + Jm**2)*(7*Jz**2 -X -5) + (7*Jz**2 -X -5)*(Jp**2 + Jm**2))
	elif [n,m] == [4,1]:
		matrix = 0.25 *((Jp + Jm)*(7*Jz**3 -(3*X+1)*Jz) + (7*Jz**3 -(3*X+1)*Jz)*(Jp + Jm))
	elif [n,m] == [4,0]:
		matrix = 35*Jz**4 - (30*X -25)*Jz**2 + 3*X**2 - 6*X
	elif [n,m] == [4,-4]:
		matrix = -0.5j *(Jp**4 - Jm**4)
	elif [n,m] == [4,-3]:
		matrix = -0.25j *((Jp**3 - Jm**3)*Jz + Jz*(Jp**3 - Jm**3))
	elif [n,m] == [4,-2]:
		matrix = -0.25j *((Jp**2 - Jm**2)*(7*Jz**2 -X -5) + (7*Jz**2 -X -5)*(Jp**2 - Jm**2))
	elif [n,m] == [4,-1]:
		matrix = -0.25j *((Jp - Jm)*(7*Jz**3 -(3*X+1)*Jz) + (7*Jz**3 -(3*X+1)*Jz)*(Jp - Jm))

	elif [n,m] == [6,6]:
		matrix = 0.5 *(Jp**6 + Jm**6)
	elif [n,m] == [6,5]:
		matrix = 0.25*((Jp**5 + Jm**5)*Jz + Jz*(Jp**5 + Jm**5))
	elif [n,m] == [6,4]:
		matrix = 0.25*((Jp**4 + Jm**4)*(11*Jz**2 -X -38) + (11*Jz**2 -X -38)*(Jp**4 + Jm**4))
	elif [n,m] == [6,3]:
		matrix = 0.25*((Jp**3 + Jm**3)*(11*Jz**3 -(3*X+59)*Jz) + (11*Jz**3 -(3*X+59)*Jz)*(Jp**3 + Jm**3))
	elif [n,m] == [6,2]:
		matrix = 0.25*((Jp**2 + Jm**2)*(33*Jz**4 -(18*X+123)*Jz**2 +X**2 +10*X +102) +\
				 (33*Jz**4 -(18*X+123)*Jz**2 +X**2 +10*X +102)*(Jp**2 + Jm**2))
	elif [n,m] == [6,1]:
		matrix = 0.25*((Jp +Jm)*(33*Jz**5 -(30*X-15)*Jz**3 +(5*X**2 -10*X +12)*Jz) +\
				 (33*Jz**5 -(30*X-15)*Jz**3 +(5*X**2 -10*X +12)*Jz)*(Jp+ Jm))
	elif [n,m] == [6,0]:
		matrix = 231*Jz**6 - (315*X-735)*Jz**4 + (105*X**2 -525*X +294)*Jz**2 -\
				 5*X**3 + 40*X**2 - 60*X
	elif [n,m] == [6,-6]:
		matrix = -0.5j *(Jp**6 - Jm**6)
	elif [n,m] == [6,-5]:
		matrix = -0.25j*((Jp**5 - Jm**5)*Jz + Jz*(Jp**5 - Jm**5))
	elif [n,m] == [6,-4]:
		matrix = -0.25j*((Jp**4 - Jm**4)*(11*Jz**2 -X -38) + (11*Jz**2 -X -38)*(Jp**4 - Jm**4))
	elif [n,m] == [6,-3]:
		matrix = -0.25j*((Jp**3 - Jm**3)*(11*Jz**3 -(3*X+59)*Jz) + (11*Jz**3 -(3*X+59)*Jz)*(Jp**3 - Jm**3))
	elif [n,m] == [6,-2]:
		matrix = -0.25j*((Jp**2 - Jm**2)*(33*Jz**4 -(18*X+123)*Jz**2 +X**2 +10*X +102) +\
				 (33*Jz**4 -(18*X+123)*Jz**2 +X**2 +10*X +102)*(Jp**2 - Jm**2))
	elif [n,m] == [6,-1]:
		matrix = -0.25j*((Jp - Jm)*(33*Jz**5 -(30*X-15)*Jz**3 +(5*X**2 -10*X +12)*Jz) +\
				 (33*Jz**5 -(30*X-15)*Jz**3 +(5*X**2 -10*X +12)*Jz)*(Jp - Jm))

	return matrix.O
#	return matrix








Jion = {}   # [S, L, J]
Jion['Ce3+'] = [0.5, 3., 2.5]
Jion['Pr3+'] = [1., 5., 4.]
Jion['Nd3+'] = [1.5, 6., 4.5]
Jion['Pm3+'] = [2., 6., 4.]
Jion['Sm3+'] = [2.5, 5, 2.5]
Jion['Tb3+'] = [3., 3., 6.]
Jion['Dy3+'] = [2.5, 5., 7.5]
Jion['Ho3+'] = [2., 6., 8.]
Jion['Er3+'] = [1.5, 6., 7.5]
Jion['Tm3+'] = [1., 5., 6.]
Jion['Yb3+'] = [0.5, 3., 3.5]
# def ionJ(ion):
#     return Jion[ion]

def LandeGFactor(ion):
    s, l, j = Jion[ion]
    return 1.5 + (s*(s+1.) - l*(l+1.))/(2.*j*(j+1.))


