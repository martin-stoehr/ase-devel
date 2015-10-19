import numpy as np
from matplotlib import pyplot as plt

def calculate_double_well(A=1.00, B=1.00, C=0.05, D=0.0):


        import matplotlib.pyplot as plt
        x=np.linspace(-6,6,200)
        V11 = A*(x- B)*(x- B)
        V22 =  A*((x+ B)*(x+ B) + D)
        V12 =  C * np.ones_like(x)
        e = A*A* (2.0*B*x+D/2.0)**2+ C* C
        ee = A *(x*x+B*B+D/2.0)
        e1 = ee - np.sqrt(e)
        e2 = ee + np.sqrt(e)
        d = - (2*A*B*C)/(2*C*C+(A*B*x+A*D/4)**2 )
        F = (8*A*A*B*B*x + 2*B*D) / (np.sqrt(e))
        F1 = -(2*A*x - F)
        F2 = -(2*A*x + F)

        return x, V11, V22, V12, e1, e2, -d/4


x, V11, V22, V12, e1, e2, d = calculate_double_well(A=1.00, B=1.00, C=0.05, D=-1.5)

plt.plot(x,V11,x,V22,x,V12)

plt.show()

plt.plot(x,e1,x,e2,x,d)

plt.show()
