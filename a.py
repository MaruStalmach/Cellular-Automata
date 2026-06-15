import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl



# 1 = p0+p3 +4p p -> [0;0.25]
p = np.linspace(0,0.25,50)

# assumption !!!
#p0 = a*p3

a=np.linspace(0,7.5,50)

# 1 = (a+1)*p3 + 4p => p=(1-(a+1)*p3)/4
p3 = (1-4*p)/(a+1)
p0 = a*p3 #= (A/(A+1) * (1-4*P))

A,P = np.meshgrid(a,p)

D = ((A/(A+1) * (1-4*P))+2*P)/(6*(1-((A/(A+1) * (1-4*P))+2*P)))
d = (p0+2*p)/(6*(1-(p0+2*p)))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x=np.linspace(0,0.25,10)
y=np.linspace(0,7.5,10)

# X,Y = np.meshgrid(y,x)
# z = np.full_like(Y, fill_value=0.25)
norm = mpl.colors.Normalize(vmin=0, vmax=1000)
# ax.plot_surface(X,Y,z)
ax.scatter(A,P,D,s=30, c=norm(abs(D-0.25)), cmap=cm.Greys)


plt.show()