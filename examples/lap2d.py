import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from math import pi, cos, sin, exp


def force(xy):
    (x, y) = xy
    return -0.8e1 * pi ** 2 * cos(0.2e1 * pi * x) ** 2 * sin(0.2e1 * pi * y) ** 2 + \
        0.16e2 * sin(0.2e1 * pi * x) ** 2 * sin(0.2e1 * pi * y) ** 2 * pi ** 2 - \
        0.8e1 * sin(0.2e1 * pi * x) ** 2 * pi ** 2 * cos(0.2e1 * pi * y) ** 2


def uexact(xy):
    (x, y) = xy
    return sin(0.2e1 * pi * x) ** 2 * sin(0.2e1 * pi * y) ** 2


def fgrid(Lx, Nx, Ny, fn):
    dx = Lx/Nx
    dy = dx
    Ly = Ny*dx

    def pt(k):
        j = k//(Nx+1)
        i = k % (Nx+1)
        x = i*dx
        y = j*dx
        return (x, y)
    f = [fn(pt(k)) for k in range((Nx+1)*(Ny+1))]
    return f


def fplot(Lx, Nx, Ny, f):
    dx = Lx/Nx
    Ly = Ny*dx
    v2 = [[f[i+j*(Nx+1)] for i in range(Nx+1)] for j in range(Ny, -1, -1)]
    # , interpolation='nearest', cmap=cm.gist_rainbow)
    plt.imshow(v2, extent=(0, Lx, 0, Ly))
    plt.show()


def solve_lap(Lx, Nx, Ny, f):
    # liste des arêtes
    L = []
    # arêtes horizontales
    for i in range(Nx):
        for j in range(Ny+1):
            k = i+j*(Nx+1)
            l = (i+1)+j*(Nx+1)
            L.append((k, l))
    # arêtes horizontales
    for i in range(Nx+1):
        for j in range(Ny):
            k = i+j*(Nx+1)
            l = i+(j+1)*(Nx+1)
            L.append((k, l))
    # matrice au format coo
    val = []
    indl = []
    indk = []
    M = (Nx+1)*(Ny+1)
    dx = Lx/Nx
    for k in range(M):
        val.append(4/dx/dx)
        indl.append(k)
        indk.append(k)
    for (k, l) in L:
        val.append(-1/dx/dx)
        indl.append(l)
        indk.append(k)
        val.append(-1/dx/dx)
        indl.append(k)
        indk.append(l)
    # conditions aux limites
    bord = []
    for i in range(Nx+1):
        j = 0
        k = i+j*(Nx+1)
        bord.append(k)
        j = Ny
        k = i+j*(Nx+1)
        bord.append(k)
    for j in range(Ny+1):
        i = 0
        k = i+j*(Nx+1)
        bord.append(k)
        i = Nx
        k = i+j*(Nx+1)
        bord.append(k)
    for kb in bord:
        # print(kb)
        val[kb] = 1e20

    A = coo_matrix((val, (indl, indk)), shape=(M, M))
    A = A.tocsc()
    u = spsolve(A, f)
    print("min=", min(u), "max=", max(u))
    return u


Nx = 1200
Ny = 200
Lx = 1

f = fgrid(Lx, Nx, Ny, force)


u = solve_lap(Lx, Nx, Ny, f)


# fplot(Lx, Nx, Ny, u)

# uex = fgrid(Lx, Nx, Ny, uexact)
# print("minex=", min(uex), "maxex=", max(uex))
# fplot(Lx, Nx, Ny, uex)
