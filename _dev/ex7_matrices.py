#! /usr/bin/env python

from scipy.sparse import *
from numpy import *
from pylab import *
from scipy.linalg import *
import os

def spconvert(M):
    row=M[:,0].astype(int)-1
    col=M[:,1].astype(int)-1
    data=M[:,2]
    m=max(row)+1
    n=max(col)+1
    spmat=csr_matrix( (data,(row,col)), shape=(m,n) )
    spmat.eliminate_zeros()
    return spmat


# maps [-1,1] to [-1,1]
def my_cmap_transition(t):
#    return t
#    p=1./6.
#    return t+(copysign(abs(t)**p, t)-t)*(2./3)
#    return sign(t)
#    return t+(sign(t)-t)*(1./10)
#    d=100.
#    return t+(arctan(d*t)/math.atan(d)-t)*(1./2)
    return t+(2./3)*(where(t==0,t,copysign(
                1./(1.-log10(abs(where(t==0,1-t,t)))),t))-t)
#    return t+(1./3)*(copysign(
#        1+log10(abs(where(abs(t)<1e-16, 1e-17*ones_like(t), t)))/17, t)-t)



def my_colormap(n):
    t=linspace(0, 1, num=n+1)
#    t=(my_cmap_transition(2*t-1)+1)/2
    r=clip(2*t, 0., 1.)
    g=minimum(2*(1-t),2*t)
    b=clip(2*(1-t), 0., 1.)
    return transpose(array([r, g, b]))

cols=my_colormap(2**10)
mycm=matplotlib.colors.ListedColormap(cols, name='my_colormap')

def my_spy(M):
    if issparse(M):
        img=asarray(M.todense())
    else:
        img=asarray(M)
    maxv=amax(abs(img))
    img=my_cmap_transition(img/maxv)
    maxv=1
    imshow(img, vmin=-maxv, vmax=maxv, cmap=mycm, interpolation='none')
    # imshow(img, cmap=mycm, interpolation='none')
    # colorbar()

def min_dist(E):
    mind=Inf
    for i in range(len(E)):
        for j in range(i):
            mind = min(mind, abs(E[i]-E[j]))
    return mind

if 1:
    Mrho=spconvert(loadtxt('M0-matrix-rho.txt'))
    Arho=spconvert(loadtxt('A0-matrix-rho.txt'))
    if os.access('AJ0-matrix-rho.txt', os.F_OK):
        lump_jump = True
        AJrho=spconvert(loadtxt('AJ0-matrix-rho.txt'))
    else:
        lump_jump = False
    Mmom=spconvert(loadtxt('M0-matrix-mom.txt'))
    Amom=spconvert(loadtxt('A0-matrix-mom.txt'))
else:
    Mrho=spconvert(loadtxt('M1-matrix-rho.txt'))
    Arho=spconvert(loadtxt('A1-matrix-rho.txt'))
    if os.access('AJ1-matrix-rho.txt', os.F_OK):
        lump_jump = True
        AJrho=spconvert(loadtxt('AJ1-matrix-rho.txt'))
    else:
        lump_jump = False
    Mmom=spconvert(loadtxt('M1-matrix-mom.txt'))
    Amom=spconvert(loadtxt('A1-matrix-mom.txt'))

print 'Inverting Mrho ... shape =', Mrho.shape, ', nnz =', Mrho.nnz
iMrho=csr_matrix(inv(Mrho.todense()))
print 'Inverting Mmom ... shape =', Mmom.shape, ', nnz =', Mmom.nnz
iMmom=csr_matrix(inv(Mmom.todense()))

print 'Brho ...'
Brho=dot(iMrho, Arho)
if lump_jump:
    iMLrho=lil_matrix(Mrho.shape)
    iMLrho.setdiag(1./asarray(Mrho.sum(0))[0])
    Brho=Brho + dot(iMLrho, AJrho)

print 'Bmom ...'
Bmom=dot(Amom, iMmom)
print 'Erho ...'
Erho=eigvals(Brho.todense())
# Erho=eigvals(Arho.todense())
print Erho[real(Erho)>=0]
print 'Minimal eigenvalue distance =', min_dist(Erho)
print 'Emom ...'
Emom=eigvals(Bmom.todense())
# Emom=eigvals(Amom.todense())
print Emom[real(Emom)>=0]
print 'Minimal eigenvalue distance =', min_dist(Emom)
print 'done.'

figure(1)
plot(real(Erho), imag(Erho), 'bo', ms=5, label='Erho')
plot(real(Emom), imag(Emom), 'rD', ms=5, label='Emom')
legend()
grid('on')

figure(2)
# spy(iMrho, marker='.')
# spy(Brho, marker='.')

# my_spy(Mrho)
# my_spy(Mmom)
# my_spy(iMrho)
my_spy(Brho)
# my_spy(Bmom)
# my_spy(Arho)
# if lump_jump:
#     my_spy(AJrho)

# figure(3)
# t=linspace(-1,1,num=cols.shape[0])
# plot(t,my_cmap_transition(t),'.')
# axis('scaled')
# t=10**linspace(-16,0,num=17)
# print t
# print 1./my_cmap_transition(t)

# figure(4)
# C=eye(32)
# for i in range(len(C)):
#     for j in range(len(C[i])):
#         C[i,j] = 10**(-abs(i-j))
# my_spy(C)

show()
