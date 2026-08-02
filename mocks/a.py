import matplotlib.pyplot as plt
import numpy as np


dx = 4e-6  # meters
td = 0.01  # seconds, diffusion time step
tb = 0.5 * 3600 * 24  # 0.5 days, bacteria time step

Ks = 10  # mg/L
Xf = 40 / (0.1) ** 3  # 40mg/cm3 -> converted to mg/L
q = 8 / 3600 / 24  # 8 mg/mg day
b = 0.1 / 3600 / 24  # 0.1 1/day
Yg = 0.5
Sb = 15  # mg/L
Yca = Yg * Sb / 27 / Xf

t_ratio = tb // td

s = (dx * 100) ** 2 / (td / 3600 / 24)

sub_c = q * Xf * td

Sf = [Sb * px / 27 for px in range(28)]
pu = [q * (1 / (Ks + ssf)) * Xf * td for ssf in Sf]
pg = [Yca * ppu for ppu in pu]
print(pg)
pd = b * td

print(pg[0] * t_ratio)

# 1 = p0+p3 +4p p -> [0;0.25]
p = np.linspace(1e-6, 0.5, 100)

# assumption !!!
# p0 = a*p3

a = np.linspace(1e-6, 10, 100)

# 1 = (a+1)*p3 + 4p => p=(1-(a+1)*p3)/4
p3 = (1 - 4 * p) / (a + 1)
p0 = a * p3  # = (A/(A+1) * (1-4*P))

A, P = np.meshgrid(a, p)

D = (
    ((A / (A + 1) * (1 - 4 * P)) + 2 * P)
    / (6 * (1 - ((A / (A + 1) * (1 - 4 * P)) + 2 * P)))
    * s
)
d = (p0 + 2 * p) / (6 * (1 - (p0 + 2 * p)))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

z_plane = 0.8

ax.plot_surface(A, P, D, color="0.75", edgecolor="none", alpha=0.95)
ax.contour(
    A, P, D, levels=[z_plane], colors="black", linewidths=2, zdir="z", offset=z_plane
)

ax.set_xlabel("a")
ax.set_ylabel("p")
ax.set_zlabel("D*")

# ax.set_zlim(z_plane, np.nanmax(D))

ax.view_init(90, 0, 0)


plt.show()
