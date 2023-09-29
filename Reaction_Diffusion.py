# %%#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:37:11 2019

@author: Shahriar Shadkhoo (KITP)
"""

import numpy as np , scipy as sp , random 
from scipy.integrate import dblquad , quad
from scipy.spatial import Voronoi 
import matplotlib.pyplot as plt
from math import atan2
from sympy.physics.quantum import TensorProduct
import os
import time

start_time = time.time()

owdFull = os.getcwd()

nu = 1          # The index of directionality of interactions

nx = 24         # int(input(print("Enter number of cells N_x:")))
ny = 24         # int(input(print("Enter number of cells N_y:")))

Emin = 1
Emax = 2
Edelta = 1

Elon = np.arange(Emin,Emax,Edelta)

D = 0.01         # Stochasticity strength

MG = 0#0.01        ## Morphogen gradient
T_mg = 1000 #100; 
tt = 1 

l0 = 1.0#/elon         # Initial length scale of the junctions
v_l  = 0.5 * l0    # Variance of lengths. See below for precise definition
v_lm = 0.5 * l0

Kf = 1 ; Kd = 0.1 ; alpha_eff = 5 ; beta_eff = 5 
at = 1 ; 

At = 1 
Amut = At

Bt = 1
Bmut = Bt
n_B = 20
Bmin = int(n_B * Bt)
Bmax = int(n_B * Bt) + 1
B = Bmin

xi = 0.5*l0 ; mutlen = 1 ; xi_uu = 1 * xi ; xi_uv = 1 * xi 

t_min = 0 ; t_max = 1000 ; dt = 0.1 ; s0 = 1000 # s0 is for image saving
t = np.arange(t_min, t_max , dt) * Kd ; Time = np.arange(t_min, t_max, dt)

Ab0 = 0.05
v_Ab_boundary = 0.5       # ratio of deviation to Ab0
v_Ab_bulk = 0.5
bc = 1           # boundary coefficient of initial distribution

c0 = 1 #1/100       ## concentration of cytoplasmic protein (for Mutant III)

mom = 1
lm = 0
coeff = 1
Nrun = 1

mutrange = 1#nx/8 # 1/4*nx*l0*np.sqrt(3)   
mutcoeff = 0
geocoeff = 0

### Self Integral of an edge with length l0 and diffusion length xi
def I_s(L, L_D):
    self_interaction = 2 * L_D * ( L - L_D * (1 - np.exp(-L/L_D)) )
    return self_interaction
    
############################# LOAD CENTROID COMMENTED #########################

mutant_points = []    
centroid = []       # creating the centroid list

################## Geometry mutant of higher disorder clone ###################

for i in range(4):
    for j in range(4):
        centroid.append([ np.sqrt(3) * l0 * (i + (1+(-1)**(j+1))/4 + np.exp( - mutcoeff * ( (i-nx/2)**2 + (j-ny/2)**2 ) / mutrange**2 ) * 0.7*v_l * ( random.random() -  random.random() )),\
        np.sqrt(3) * l0 * (j * np.sqrt(3)/2 + np.exp( - mutcoeff * ( (i-nx/2)**2 + (j-ny/2)**2 ) / mutrange**2 ) * 0.7*v_l * ( random.random() -  random.random() )  ) ]) 
    for j in range(4 , ny-4):
        centroid.append([ np.sqrt(3) * l0 * (i + (1+(-1)**(j+1))/4 + np.exp( - mutcoeff * ( (i-nx/2)**2 + (j-ny/2)**2 ) / mutrange**2 ) * 0.7*v_l * ( random.random() -  random.random() )),\
        np.sqrt(3) * l0 * (j * np.sqrt(3)/2 + np.exp( - mutcoeff * ( (i-nx/2)**2 + (j-ny/2)**2 ) / mutrange**2 ) * 0.7*v_l * ( random.random() -  random.random() )  ) ]) 
    centroid.append([centroid[i*ny + 0][0], centroid[i*ny + 0][1] + np.sqrt(3)/2 * np.sqrt(3) * l0 * (ny-4) ])
    centroid.append([centroid[i*ny + 1][0], centroid[i*ny + 1][1] + np.sqrt(3)/2 * np.sqrt(3) * l0 * (ny-4) ])
    centroid.append([centroid[i*ny + 2][0], centroid[i*ny + 2][1] + np.sqrt(3)/2 * np.sqrt(3) * l0 * (ny-4) ])
    centroid.append([centroid[i*ny + 3][0], centroid[i*ny + 3][1] + np.sqrt(3)/2 * np.sqrt(3) * l0 * (ny-4) ])

for i in range(4 , nx-4):
    for j in range(4):
        centroid.append([ np.sqrt(3) * l0 * (i + (1+(-1)**(j+1))/4 + np.exp( - mutcoeff * ( (i-nx/2)**2 + (j-ny/2)**2 ) / mutrange**2 ) * 0.7*v_l * ( random.random() -  random.random() )),\
        np.sqrt(3) * l0 * (j * np.sqrt(3)/2 + np.exp( - mutcoeff * ( (i-nx/2)**2 + (j-ny/2)**2 ) / mutrange**2 ) * 0.7*v_l * ( random.random() -  random.random() )  ) ]) 
    for j in range(4 , ny-4):
        if [i , j] not in mutant_points:
            centroid.append([ np.sqrt(3) * l0 * (i + (1+(-1)**(j+1))/4 + np.exp( - mutcoeff * ( (i-nx/2)**2 + (j-ny/2)**2 ) / mutrange**2 ) * v_l * ( random.random() -  random.random() )),\
            np.sqrt(3) * l0 * (j * np.sqrt(3)/2 + np.exp( - mutcoeff * ( (i-nx/2)**2 + (j-ny/2)**2 ) / mutrange**2 ) * v_l * ( random.random() -  random.random() )  ) ])
        else:  
            centroid.append([ np.sqrt(3) * l0 * (i + (1+(-1)**(j+1))/4 + np.exp( - mutcoeff * ( (i-nx/2)**2 + (j-ny/2)**2 ) / mutrange**2 ) * v_lm * ( random.random() -  random.random() )),\
            np.sqrt(3) * l0 * (j * np.sqrt(3)/2 + np.exp( - mutcoeff * ( (i-nx/2)**2 + (j-ny/2)**2 ) / mutrange**2 ) * v_lm * ( random.random() -  random.random() )  ) ])
    centroid.append([centroid[i*ny + 0][0], centroid[i*ny + 0][1] + np.sqrt(3)/2 * np.sqrt(3) * l0 * (ny-4) ])
    centroid.append([centroid[i*ny + 1][0], centroid[i*ny + 1][1] + np.sqrt(3)/2 * np.sqrt(3) * l0 * (ny-4) ])
    centroid.append([centroid[i*ny + 2][0], centroid[i*ny + 2][1] + np.sqrt(3)/2 * np.sqrt(3) * l0 * (ny-4) ])
    centroid.append([centroid[i*ny + 3][0], centroid[i*ny + 3][1] + np.sqrt(3)/2 * np.sqrt(3) * l0 * (ny-4) ])

for i in range(4*ny):
    centroid.append([centroid[i][0] + np.sqrt(3) * l0 * (nx-4), centroid[i][1]])
    
def Centroid_transf(r0 , mrange):    
    cen_trans = quad(  lambda r : ( geocoeff * np.exp( - r**2 /mrange**2 ) ) , 0 , r0 )
    return cen_trans[0]    
    
Lx = nx * l0 * np.sqrt(3) ; Ly = ny * l0 * 3/2

centrans = centroid
vor = Voronoi(centrans)

################################# EMPTY CORNERS ###############################            
borind_L = []         # Index of centroids of border regions (to be discarded)
borind_R = []

borind_U = []
borind_D = []

for i in range(2*ny):
    borind_L.append(i)
    
for i in range(nx):
    borind_D.append(i*(ny))
    borind_D.append(i*(ny)+1)
    
for i in range(nx):
    borind_U.append((i+1)*(ny)-2)
    borind_U.append((i+1)*(ny)-1)

for i in range((nx-2)*ny, (nx)*(ny)):
    borind_R.append(i) 
    
bor = borind_L + borind_D + borind_U + borind_R

#### Unnecessary introduction of non-overlapping borind instead of adding the L,R,U,D    
borind = []
    
for i in range(2*ny + 2):
    borind.append(i)
    
for i in range(3,nx-2):
    borind.append(i*(ny) - 2)
    borind.append(i*(ny) - 1)
    borind.append(i*(ny) + 0)    
    borind.append(i*(ny) + 1)
    
for i in range((nx-2)*(ny) - 2 , (nx)*(ny)):
    if i not in borind:
        borind.append(i)
        
###############################################################################

regc = []           # a copy of original regions to be edited (only bulk)
regb = []           # list of border regions
    
for reg in vor.regions:
    regc.append(reg)
    
for b in borind:
    regb.append(vor.regions[vor.point_region[b]])
    regc.remove(vor.regions[vor.point_region[b]])

regc.remove([])

######################### Relabeling the Bulk Vertices ########################

v = vor.vertices

vb = []

for ver in v:
    vb.append([])

for reg in regb:
    for ver in reg:
        if ver != -1:
            vb[ver].append(vor.regions.index(reg))
  
vbind = []          #### V Boundary Index (before relabeling)

for ver in vb:
    if len(ver) == 3: vbind.append(vb.index(ver))

vlist = []

for i in range(len(v)):
    vlist.append([i] + v.tolist()[i])

# removing the vertices in borders
for i in range(len(v)):
    if i in vbind:
        vlist.remove([i,v[i][0],v[i][1]])

# vind: only the index of new vertices not relabeled yet (map)
vind = []

for ver in vlist:           
    vind.append(ver[0])
    
for i in range(len(vlist)):
    vlist[i].remove(vlist[i][0])    

v = np.array(vlist)

########################## Reconstructing edge list ###########################

e = [edge for edge in vor.ridge_vertices if -1 not in edge \
and edge[0] not in vbind and edge[1] not in vbind ]

# Rebuilding the edge list from the new vertices
for edge in e:
    [edge[0],edge[1]] = [vind.index(edge[0]) , vind.index(edge[1])]

################## Relabeling the vertices in Bulk Regions ####################                
vor_regions = vor.regions

regions = []

for reg in regc:
    regions.append(reg)

for reg in regions:
    for ver in reg:
        regions[regions.index(reg)][reg.index(ver)] = vind.index(ver) 
        
###############################################################################    
    
N = len(v)               # number of vertices
M = len(e)               # voronoi adjacency = # edges
R = len(regions)

rx = v[:,0].reshape(N,1)                 # initializing and separating X, Y
ry = v[:,1].reshape(N,1)

############################ Sorting the vertices: ############################
####### The original order can be neither clockwise nor counterclockwise ######

def cenx(regions, rx, ry):
    Cenx = np.zeros((len(regions),1))
    for i in range(len(regions)):
        for vert in regions[i]:
            Cenx[i,0] += (rx[regions[i][regions[i].index(vert)]]) / len(regions[i])
    return Cenx
            
cenx = cenx(regions, rx, ry).reshape(len(regions),1) 

def ceny(regions, rx, ry):
    Ceny = np.zeros((len(regions),1))
    for i in range(len(regions)):
        for vert in regions[i]:
            Ceny[i,0] += (ry[regions[i][regions[i].index(vert)]]) / len(regions[i])
    return Ceny

ceny = ceny(regions, rx, ry).reshape(len(regions),1)


arctan = []

for i in range(len(regions)):  
    arctan.append([])    
    for vert in regions[i]:      
        arctan[i].append(  atan2(ry[regions[i][regions[i].index(vert)]]-ceny[i,0],\
        rx[regions[i][regions[i].index(vert)]]-cenx[i,0])  )    
        
ang_c = []     ### A copy of arctan for mapping the sorted angles   
        
for i in range(len(arctan)):
    ang_c.append([])
    for j in range(len(arctan[i])): ang_c[i].append(arctan[i][j])    
    
for reg in arctan:
    arctan[arctan.index(reg)] = sorted(arctan[arctan.index(reg)])  
        
reg_c = []
    
for i in range(len(regions)):
    reg_c.append([])

for i in range(len(arctan)):
    for ind in arctan[i]:
        reg_c[i].append(regions[i][ang_c[i].index(ind)]) 

regions = []
for i in range(len(reg_c)):
    regions.append([])
    for j in range(len(reg_c[i])):
        regions[i].append(reg_c[i][j])

########### Making all regions counterclockwise (Assuming convexity) ##########
############### CONVEXITY HOLDS TRUE IN THE INITIALIZED LATTICE ###############
for reg in regions:
    if reg != []:
        if np.cross(v[reg[1]]-v[reg[0]] , v[reg[2]]-v[reg[1]] ) < 0:
            regions[regions.index(reg)] = reg[::-1]

########################## Relabeling Point_Regions ###########################                        
pr = []                 # pr stands for point regions and is the same as original one
for num in vor.point_region.tolist():
    if vor_regions[num] in regc:
        pr.append(regc.index(vor_regions[num]))
    else: pr.append('NaN')
    
############################### Repeating edges ###############################

pairs1 = []
pairs2 = []
pairs = []              # Pairs of edges to be matched

cp1 = []
cp2 = []
cp = []                 # Pairs of cells

ripo = vor.ridge_points.tolist()

ripobu = vor.ridge_points.tolist()      ##### bulk ripo : in fact updated edges

for i in range(len(vor.ridge_points)):
    if (vor.ridge_points[i][0] in borind and vor.ridge_points[i][1] in borind):
        ripobu.remove(vor.ridge_points.tolist()[i])

for pp1 in ripo:
    if (bool(pp1[0] in borind_L) != bool(pp1[1] in borind_L) and bool((pp1[0] + (nx-4)*ny) in borind_R) != bool((pp1[1] + (nx-4)*ny) in borind_R) and -1 not in vor.ridge_vertices[ripo.index(pp1)] ):
        if [pp1[0] + (nx-4)*ny, pp1[1] + (nx-4)*ny] in ripo:
            pp2 = [pp1[0] + (nx-4)*ny, pp1[1] + (nx-4)*ny]
        else: pp2 = [pp1[1] + (nx-4)*ny, pp1[0] + (nx-4)*ny]
        pairs1.append([ripo.index(pp1) , ripo.index(pp2)])
        cp1.append([max(pp1),min(pp2)])
    
for pp1 in ripo:        
    if (bool(pp1[0] in borind_D) != bool(pp1[1] in borind_D) and bool((pp1[0] + ny-4) in borind_U) != bool((pp1[1] + ny-4) in borind_U) and -1 not in vor.ridge_vertices[ripo.index(pp1)]):
        if [pp1[0] + ny-4, pp1[1] + ny-4] in ripo:
            pp2 = [pp1[0] + ny-4, pp1[1] + ny-4]
        else: pp2 = [pp1[1] + ny-4, pp1[0] + ny-4]
        pairs2.append([ripo.index(pp1) , ripo.index(pp2)])
        cp2.append([max(pp1),min(pp2)])
    
# ee0 and cor0 correspond to Left and Down borders, ee1, cor1 to Right and Up        

# pairs1 and cp1 are for Left and Right.
# pairs2 and cp2 are for Down and Up.

pairs = pairs1 + pairs2
cp = cp1 + cp2

ee0 = []
for ee in pairs:
    if (ripo[ee[0]][0] in borind and ripo[ee[0]][1] in borind):
        ee0.append(ee)

ee1 = []
for ee in pairs:
    if (ripo[ee[1]][0] in borind and ripo[ee[1]][1] in borind):
        ee1.append(ee)

for ee in ee0:
    if ee in ee1:
        cp.remove(cp[pairs.index(ee)])
        pairs.remove(ee)

# CORNERS
cor0 = []               
cor1 = []

for ee in ee0: 
    if ee not in ee1:
        cor0.append(ee)
        cp.remove(cp[pairs.index(ee)])
        pairs.remove(ee)

for ee in ee1: 
    if ee not in ee0:
        cor1.append(ee)
        cp.remove(cp[pairs.index(ee)])
        pairs.remove(ee)
# These are the repeated edges in cor0, cor1.
# Instead of [0][0] or [0][1] choose repeated one in each cor0 or cor1

corner = [cor1[0][0] , cor0[0][1]]  
pairs.append(corner)

## for cp instead of the following, one can explicitely choose the one NOT in borind
cp.append( [ max(ripo[cor1[0][0]]) , min(ripo[cor0[0][1]]) ] )

#### At this point, pairs only includes edges that are supposed to appear ####

######################### Relabeling edges in PAIRS ###########################

for ee in pairs:
    [ee[0], ee[1]] = [ripobu.index(ripo[ee[0]]), ripobu.index(ripo[ee[1]])]
    
pairs_LR = []

for ee in pairs:
    if ( bool(ripobu[ee[0]][0] <= 2*ny) != bool(ripobu[ee[0]][1] <= 2*ny) ):
        pairs_LR.append(ee)
        
pairs_DU = []        

for ee in pairs:
    if ee not in pairs_LR: pairs_DU.append(ee)
    
###############################################################################
###############################################################################

# Introduction of necessary matrix variables
C = np.zeros((M,N))     # Connection matrix between edges and vertices
L = np.zeros((M,1)) 
Ab = np.zeros((2*M,1))         ### Bound A and B, for all JUNCTIONS
Bb = np.zeros((2*M , 1))

############################# STRUCTURAL FUNCTIONS ############################
def Connection(e,rx):
    C = np.zeros((len(e),len(rx)))
    for el in e:
        C[e.index(el),el[0]] = 1  ;  C[e.index(el),el[1]] = -1      # Connection matrix
    return C

C = Connection(e,rx)

CC = sp.sparse.coo_matrix(C)
    
############################# Connection Matrices #############################

ebor = []           ##### Edges of border, constructed based on updated PAIRS

for ee in pairs:
    if ee[0] not in ebor: ebor.append(ee[0])
    if ee[1] not in ebor: ebor.append(ee[1])

ebor_dual = []
for eorig in ebor:
    ebor_dual.append(eorig + len(e))

eeod = ebor + ebor_dual     # union of original border edges and their duals

ebulk = []

for el in e:
    if e.index(el) not in ebor:
        ebulk.append(e.index(el))

def rea(e, regions):
    REA = np.zeros((len(regions),2*len(e)))         ### Region-Edge connection network
    for ed in range(len(e)):
        if ed not in ebor:
            c1 = pr[ripobu[ed][0]] ; c2 = pr[ripobu[ed][1]]
            REA[c1][ed] = +1 ; REA[c2][ed + len(e)] = +1
    for ee in pairs:
        c1 = pr[cp[pairs.index(ee)][0]]
        c2 = pr[cp[pairs.index(ee)][1]]
        REA[c1][ee[0]] = +1/2  ; REA[c1][ee[1] + len(e)] = +1/2
        REA[c2][ee[1]] = +1/2  ; REA[c2][ee[0] + len(e)] = +1/2
        del c1, c2
    return REA

REA = rea(e, regions)

def reb(e, regions):
    REB = np.zeros((len(regions),2*len(e)))         ### Region-Edge connection network
    for ed in range(len(e)):
        if ed not in ebor:
            c1 = pr[ripobu[ed][0]] ; c2 = pr[ripobu[ed][1]]
            REB[c2][ed] = +1 ; REB[c1][ed + len(e)] = +1
    for ee in pairs:
        c1 = pr[cp[pairs.index(ee)][0]]
        c2 = pr[cp[pairs.index(ee)][1]]
        REB[c2][ee[0]] = +1/2  ; REB[c2][ee[1] + len(e)] = +1/2
        REB[c1][ee[1]] = +1/2  ; REB[c1][ee[0] + len(e)] = +1/2
        del c1, c2
    return REB

REB = reb(e, regions)

def reat(e, regions):
    REAT = np.zeros((2*len(e),len(regions)))         ### Region-Edge connection network
    for ed in range(len(e)):
        if ed not in ebor:
            c1 = pr[ripobu[ed][0]] ; c2 = pr[ripobu[ed][1]]
            REAT[ed][c1] = +1 ; REAT[ed + len(e)][c2] = +1
    for ee in pairs:
        c1 = pr[cp[pairs.index(ee)][0]]
        c2 = pr[cp[pairs.index(ee)][1]]
        REAT[ee[0]][c1] = +1  ; REAT[ee[1] + len(e)][c1] = +1
        REAT[ee[1]][c2] = +1  ; REAT[ee[0] + len(e)][c2] = +1
        del c1, c2
    return REAT

REAT = reat(e, regions)

def rebt(e, regions):
    REBT = np.zeros((2*len(e),len(regions)))         ### Region-Edge connection network
    for ed in range(len(e)):
        if ed not in ebor:
            c1 = pr[ripobu[ed][0]] ; c2 = pr[ripobu[ed][1]]
            REBT[ed][c2] = +1 ; REBT[ed + len(e)][c1] = +1
    for ee in pairs:
        c1 = pr[cp[pairs.index(ee)][0]]
        c2 = pr[cp[pairs.index(ee)][1]]
        REBT[ee[0]][c2] = +1  ; REBT[ee[1] + len(e)][c2] = +1
        REBT[ee[1]][c1] = +1  ; REBT[ee[0] + len(e)][c1] = +1
        del c1, c2
    return REBT

REBT = rebt(e, regions)

swap = TensorProduct(np.array([[0,1],[1,0]]) , np.identity(len(e)))
Cswap = sp.sparse.coo_matrix(swap)

L = np.sqrt( (CC.dot(rx))**2 + (CC.dot(ry))**2 )
LL = TensorProduct(np.ones((2,1)) , L)

REA = rea(e, regions)
REB = reb(e, regions)

CREA = sp.sparse.coo_matrix(REA) 
CREB = sp.sparse.coo_matrix(REB)

CREAT = sp.sparse.coo_matrix(REAT) 
CREBT = sp.sparse.coo_matrix(REBT) 

######################## Polarization Defined for EDGES #######################

def Ecenx(e , rx , ry):
    ecenx = np.zeros((len(e) , 1))
    for ed in e:
        ecenx[e.index(ed) , 0] = (1/2) * (rx[ed[0]] + rx[ed[1]])
    return ecenx
    
def Eceny(e , rx , ry):
    eceny = np.zeros((len(e) , 1))
    for ed in e:
        eceny[e.index(ed) , 0] = (1/2) * (ry[ed[0]] + ry[ed[1]])
    return eceny


def Edel(regions , e , rx , ry):
    PaxX = np.zeros((len(e) , 1)).reshape(len(e),1)
    PaxY = np.zeros((len(e) , 1)).reshape(len(e),1)
    
    for ed in range(len(e)):
        if ed not in ebor:
#            [cell1 , cell2] = [regions[pr[ripobu[ed][0]]], regions[pr[ripobu[ed][1]]]]
            cell1 = regions[pr[ripobu[ed][0]]]
            if cell1.index(e[ed][0]) == cell1.index(e[ed][1]) - 1:
                PaxX[ed,0] = -(ry[e[ed][0]] - ry[e[ed][1]])/L[ed]
                PaxY[ed,0] = +(rx[e[ed][0]] - rx[e[ed][1]])/L[ed]
            else:
                PaxX[ed,0] = -(ry[e[ed][1]] - ry[e[ed][0]])/L[ed]
                PaxY[ed,0] = +(rx[e[ed][1]] - rx[e[ed][0]])/L[ed]
    
    for ee in pairs:
        cell0 = regions[pr[cp[pairs.index(ee)][0]]]
        if cell0.index(e[ee[0]][0]) == cell0.index(e[ee[0]][1]) - 1:
            PaxX[ee[0],0] = -(ry[e[ee[0]][0]] - ry[e[ee[0]][1]])/L[ee[0]]
            PaxY[ee[0],0] = +(rx[e[ee[0]][0]] - rx[e[ee[0]][1]])/L[ee[0]]
        else:
            PaxX[ee[0],0] = -(ry[e[ee[0]][1]] - ry[e[ee[0]][0]])/L[ee[0]]
            PaxY[ee[0],0] = +(rx[e[ee[0]][1]] - rx[e[ee[0]][0]])/L[ee[0]]
        PaxX[ee[1],0] = - PaxX[ee[0],0]; PaxY[ee[1],0] = - PaxY[ee[0],0]
    return PaxX , PaxY

###############################################################################

def Per(regions, rx, ry):
    Per = np.zeros(len(regions))    
    for i in range(len(regions)):
        for j in range(len(regions[i])):
            Per[i] += np.sqrt((rx[regions[i][j]] - rx[regions[i][j-1]])**2 + (ry[regions[i][j]] - ry[regions[i][j-1]])**2)
    return Per

def Cenx(regions, rx, ry , per):
    cenx = np.zeros((len(regions),1))
    for i in range(len(regions)):
        for j in range(len(regions[i])):
            cenx[i,0] += (rx[regions[i][j]] + rx[regions[i][j-1]]) * (1/2) * np.sqrt((rx[regions[i][j]] - rx[regions[i][j-1]])**2 + (ry[regions[i][j]] - ry[regions[i][j-1]])**2)  / per[i]
    return cenx
 
def Ceny(regions, rx, ry , per):
    ceny = np.zeros((len(regions),1))
    for i in range(len(regions)):
        for j in range(len(regions[i])):
            ceny[i,0] += (ry[regions[i][j]] + ry[regions[i][j-1]]) * (1/2) * np.sqrt((rx[regions[i][j]] - rx[regions[i][j-1]])**2 + (ry[regions[i][j]] - ry[regions[i][j-1]])**2) / per[i]
    return ceny

def Delx(e, regions, rx, ry, cenx, ceny):
    delta_x = np.zeros((len(regions),2*len(e)))         ### Region-Edge connection network
    ddx = np.zeros((len(regions),2*len(e)))
    for ed in range(len(e)):
        if ed not in ebor:
            c1 = pr[ripobu[ed][0]] ; c2 = pr[ripobu[ed][1]]
            delta_x[c1][ed] = (( (rx[e[ed][0]] + rx[e[ed][1]])/2 - cenx[c1]  )) * (L[ed]**(lm)) / (( (rx[e[ed][0]] + rx[e[ed][1]])/2 - cenx[c1]  )**2 + (( (ry[e[ed][0]] + ry[e[ed][1]])/2 - ceny[c1]  )**2))**(0.5*(2-mom))
            delta_x[c2][ed + len(e)] = (( (rx[e[ed][0]] + rx[e[ed][1]])/2 - cenx[c2]  )) * (L[ed]**(lm)) / (( (rx[e[ed][0]] + rx[e[ed][1]])/2 - cenx[c2]  )**2 + (( (ry[e[ed][0]] + ry[e[ed][1]])/2 - ceny[c2]  )**2))**(0.5*(2-mom))
            ddx[c1][ed] = (( (rx[e[ed][0]] + rx[e[ed][1]])/2 - cenx[c1]  )) 
            ddx[c2][ed + len(e)] = (( (rx[e[ed][0]] + rx[e[ed][1]])/2 - cenx[c2]  )) 
    for ee in pairs:
        c1 = pr[cp[pairs.index(ee)][0]]
        c2 = pr[cp[pairs.index(ee)][1]]
        delta_x[c1][ee[0]] = 1/2 * (( (rx[e[ee[0]][0]] + rx[e[ee[0]][1]])/2 - cenx[c1]  )) * (L[ee[0]]**(lm)) / (( (rx[e[ee[0]][0]] + rx[e[ee[0]][1]])/2 - cenx[c1]  )**2 + (( (ry[e[ee[0]][0]] + ry[e[ee[0]][1]])/2 - ceny[c1]  )**2))**(0.5*(2-mom))           
        delta_x[c1][ee[1] + len(e)] = delta_x[c1][ee[0]] ## Factors of 1/2 is because of the
        ddx[c1][ee[0]] = (( (rx[e[ee[0]][0]] + rx[e[ee[0]][1]])/2 - cenx[c1]  ))
        delta_x[c2][ee[1]] = 1/2 * (( (rx[e[ee[1]][0]] + rx[e[ee[1]][1]])/2 - cenx[c2]  )) * (L[ee[1]]**(lm)) / (( (rx[e[ee[1]][0]] + rx[e[ee[1]][1]])/2 - cenx[c2]  )**2 + (( (ry[e[ee[1]][0]] + ry[e[ee[1]][1]])/2 - ceny[c2]  )**2))**(0.5*(2-mom))
        delta_x[c2][ee[0] + len(e)] = delta_x[c2][ee[1]] ## pairs at boundary condition
        ddx[c2][ee[1]] = (( (rx[e[ee[1]][0]] + rx[e[ee[1]][1]])/2 - cenx[c2]  ))
        del c1, c2
    return delta_x , ddx

def Dely(e, regions, rx, ry, cenx, ceny):
    delta_y = np.zeros((len(regions),2*len(e)))         ### Region-Edge connection network
    ddy = np.zeros((len(regions),2*len(e)))
    for ed in range(len(e)):
        if ed not in ebor:
            c1 = pr[ripobu[ed][0]] ; c2 = pr[ripobu[ed][1]]
            delta_y[c1][ed] = (( (ry[e[ed][0]] + ry[e[ed][1]])/2 - ceny[c1]  )) * (L[ed]**(lm)) / (( (rx[e[ed][0]] + rx[e[ed][1]])/2 - cenx[c1]  )**2 + (( (ry[e[ed][0]] + ry[e[ed][1]])/2 - ceny[c1]  )**2))**(0.5*(2-mom))
            delta_y[c2][ed + len(e)] = (( (ry[e[ed][0]] + ry[e[ed][1]])/2 - ceny[c2]  )) * (L[ed]**(lm)) / (( (rx[e[ed][0]] + rx[e[ed][1]])/2 - cenx[c2]  )**2 + (( (ry[e[ed][0]] + ry[e[ed][1]])/2 - ceny[c2]  )**2))**(0.5*(2-mom))
            ddy[c1][ed] = (( (ry[e[ed][0]] + ry[e[ed][1]])/2 - ceny[c1]  )) 
            ddy[c2][ed + len(e)] = (( (ry[e[ed][0]] + ry[e[ed][1]])/2 - ceny[c2]  )) 
    for ee in pairs:
        c1 = pr[cp[pairs.index(ee)][0]]
        c2 = pr[cp[pairs.index(ee)][1]]
        delta_y[c1][ee[0]] = 1/2 * (( (ry[e[ee[0]][0]] + ry[e[ee[0]][1]])/2 - ceny[c1]  )) * (L[ee[0]]**(lm)) / (( (rx[e[ee[0]][0]] + rx[e[ee[0]][1]])/2 - cenx[c1]  )**2 + (( (ry[e[ee[0]][0]] + ry[e[ee[0]][1]])/2 - ceny[c1]  )**2))**(0.5*(2-mom))
        delta_y[c1][ee[1] + len(e)] = delta_y[c1][ee[0]] 
        ddy[c1][ee[0]] = (( (ry[e[ee[0]][0]] + ry[e[ee[0]][1]])/2 - ceny[c1]  ))
        delta_y[c2][ee[1]] = 1/2 * (( (ry[e[ee[1]][0]] + ry[e[ee[1]][1]])/2 - ceny[c2]  )) * (L[ee[1]]**(lm)) / (( (rx[e[ee[1]][0]] + rx[e[ee[1]][1]])/2 - cenx[c2]  )**2 + (( (ry[e[ee[1]][0]] + ry[e[ee[1]][1]])/2 - ceny[c2]  )**2))**(0.5*(2-mom))
        delta_y[c2][ee[0] + len(e)] = delta_y[c2][ee[1]] 
        ddy[c2][ee[1]] = (( (ry[e[ee[1]][0]] + ry[e[ee[1]][1]])/2 - ceny[c2]  ))
        del c1, c2       
    return delta_y , ddy

def relx(e , regions , rx , ry , cenx , ceny):
    rel_x = np.zeros((2*len(e) , 1))
    for ed in range(len(e)):
        if ed not in ebor:
            c1 = pr[ripobu[ed][0]] ; c2 = pr[ripobu[ed][1]]
            rel_x[ed , 0] = (rx[e[ed][0]] + rx[e[ed][1]])/2 - cenx[c1]
            rel_x[ed + len(e) , 0] = (rx[e[ed][0]] + rx[e[ed][1]])/2 - cenx[c2]
    for ee in pairs:
        c1 = pr[cp[pairs.index(ee)][0]]
        c2 = pr[cp[pairs.index(ee)][1]]
        rel_x[ee[0] , 0] = (rx[e[ee[0]][0]] + rx[e[ee[0]][1]])/2 - cenx[c1]
        rel_x[ee[1] + len(e) , 0] = rel_x[ee[0]] 
        rel_x[ee[1] , 0] = (rx[e[ee[1]][0]] + rx[e[ee[1]][1]])/2 - cenx[c2]
        rel_x[ee[0] + len(e) , 0] = rel_x[ee[1]] 
        del c1, c2       
    return rel_x

def R_corr(Px , Py , Rij):

    Px = Cdelx.dot(Ab)
    Py = Cdely.dot(Ab)
    
    Pxt = np.transpose(Px)
    Pyt = np.transpose(Py)

    Z = sum(sum((Pxt * Px + Pyt * Py )/Rij))#/sum(sum(Pt * P))
        
    Q = sum(sum( ( Pxt * Px + Pyt * Py ) ))#/sum(sum(Pt * P))
    
    rc = Q/Z
    return rc

def P_corr(Px , Py , Rij):

    Px = Cdelx.dot(Ab)
    Py = Cdely.dot(Ab)

    P = np.sqrt(Px**2 + Py**2)        
    Pt = np.transpose(P)

    Z = sum(sum((Pt * P)/Rij))#/sum(sum(Pt * P))
        
    Q = sum(sum( ( Pt * P) ))#/sum(sum(Pt * P))
    
    Prc = Q/Z
    return Prc

def Ang_corr(Px , Py , Rij):

    Angx = np.transpose(Px/np.sqrt(Px**2 + Py**2))
    Angy = np.transpose(Py/np.sqrt(Px**2 + Py**2))
                
    Angxt = np.transpose(Angx)
    Angyt = np.transpose(Angy)
    
    ZAng = sum(sum( ( Angxt * Angx + Angyt * Angy ) / ( Rij ) ) )#/sum(sum(Pt * P))            
    QAng = sum(sum( ( Angxt * Angx + Angyt * Angy ) ) )#/sum(sum(Pt * P))
    
    rAngc = QAng/ZAng
    return rAngc

############################# NONOLOCAL MATRICES ##############################
############### SIMPLIFIED INTERACTIONS FOR ORDERED LATTICES ##################

## Interaction of nearest neighbor edges
def Int_NN(L1, L2, L_D, theta):
    int_NN = dblquad(  lambda r2, r1: np.exp( -(1/L_D) * np.sqrt( (np.cos(theta)*r2 + L1 - r1)**2 + ((np.sin(theta)) * r2)**2 ) ) , 0 , L2 , lambda r1: 0, lambda r1: L1  )
    return int_NN[0]

I_NN_uu = Int_NN(l0, l0, xi_uu, np.pi/3)
I_NN_uv = Int_NN(l0, l0, xi_uv, np.pi/3)

I_NN_mut_uu = Int_NN(l0, l0, xi_uu * mutlen, np.pi/3) * c0
I_NN_mut_uv = Int_NN(l0, l0, xi_uv * mutlen, np.pi/3) * c0


## Interaction of cross and parallel edges. Note the definition of L1, L2
def Int_CP(L1, L2, L_D, theta):      # L1 is the length of parallel edges and L2 is the short ones
    int_CP = dblquad(  lambda r2, r1: np.exp( -(1/L_D) * np.sqrt( (2*L2*np.sin(theta))**2 + (r1-r2)**2 ) ) , 0 , L1 , lambda r1: 0, lambda r1: L1  )
    return int_CP[0]

I_CP_uu = Int_CP(l0, l0, xi_uu, np.pi/3)
I_CP_uv = Int_CP(l0, l0, xi_uv, np.pi/3)

I_CP_mut_uu = Int_CP(l0, l0, xi_uu * mutlen, np.pi/3) * c0
I_CP_mut_uv = Int_CP(l0, l0, xi_uv * mutlen, np.pi/3) * c0

## Interaction of next to parallel edges
def Int_NP(L1, L2, L_D, theta): 
    int_NP = dblquad(  lambda r2, r1: np.exp( -(1/L_D) * np.sqrt( ((L2 + r2)*np.sin(theta))**2 + (L1-r1 + (L2-r2)*np.cos(theta))**2 ) ) , 0 , L2 , lambda r1: 0, lambda r1: L1  )
    return int_NP[0]
    
I_NP_uu = Int_NP(l0, l0, xi_uu, np.pi/3)
I_NP_uv = Int_NP(l0, l0, xi_uv, np.pi/3)

I_NP_mut_uu = Int_NP(l0, l0, xi_uu * mutlen, np.pi/3) * c0
I_NP_mut_uv = Int_NP(l0, l0, xi_uv * mutlen, np.pi/3) * c0

############################# Af , Bf Functions ###############################

def af(at, Ab, Ka, LL):
    Af = CREAT.dot( at - (CREA.dot((Ab + dt*Ka)*LL))/(CREA.dot(LL)) )
    return Af

def bf(bt, Ab, Ka, LL , nu):
    Bf = ( CREBT.dot( bt - (CREB.dot((Ab + dt*Ka)*LL))/(CREB.dot(LL)) ) )**nu
    return Bf

###############################################################################


#%% ##################### Dynamics and time integration  ######################

Ang_p = np.zeros((len(Time), len(Elon)))
epsilon_magnitude  = np.zeros(len(Elon))

for ratio in range(len(Elon)):

    owdE = os.getcwd()

    elon = Elon[ratio]

    elonname = 'Kf%02d' %(Kf/Kd) + '_xi%02d'%(100*xi) + '_vL%02d' %(100*v_l) + '_Bt%03d' %(100*Bt) # + '_Tmax%02d' %t_max + '_Elon%02d' %(10*elon)

    os.mkdir(elonname)        
    os.chdir(elonname)

    Ly = max(ry) - min(ry); # ry_mean = np.mean(ry)

    if ratio == 0 :
        ry = elon * ry
    else:
        ry = ry * elon/Elon[ratio - 1] 

    E = round(elon*10)

    L = np.sqrt( (CC.dot(rx))**2 + (CC.dot(ry))**2 )
    LL = TensorProduct(np.ones((2,1)) , L)
    
    per = Per(regions, rx, ry)
    
    cenx = Cenx(regions, rx, ry, per).reshape(len(regions),1) 
    ceny = Ceny(regions, rx, ry, per).reshape(len(regions),1)

    delx = Delx(e, regions, rx, ry, cenx, ceny)[0]
    dely = Dely(e, regions, rx, ry, cenx, ceny)[0]
        
    Cdelx = sp.sparse.coo_matrix(delx)
    Cdely = sp.sparse.coo_matrix(dely)
    
    Relx = relx(e , regions , rx , ry , cenx , ceny)

    dbl_e = 2 * e
    Ee = np.zeros((len(e) , 2*len(e)))
    
    for ed in range(len(e)):
        Ee[ed , ed] = +1
        Ee[ed , ed + len(e)] = -1
    
    EeSparse = sp.sparse.coo_matrix(Ee)

    ecenx = Ecenx(e , rx , ry)
    eceny = Eceny(e , rx , ry)
    
    ###############################################################################
    
    Alpha_NL = np.zeros((2*len(e) , 2*len(e)))
    
    Beta_NL = np.zeros((2*len(e) , 2*len(e)))
    
    ### alpha_0 and beta_0 are bare interactions
    alpha_0 = alpha_eff / I_s(l0, xi_uu) 
    beta_0 = beta_eff / I_s(l0, xi_uv)
    
    # We construct Alpha_NL and Beta_NL based on REA because that's the matrix that stores the edges
    # that belong to the same cell, since the basis of calculation is Ab, the concentration of A_bound.
    # REB relates to each edge, the other side of the edges in the adjacent cell
    
    dbl_e = 2 * e    
    mutant = []
    e_mut = []
    
    
    for edge in range(2*len(e)):
        if edge in ebor_dual:
            continue
        for i in range(len(regions)):
            if REA[i,edge] != 0:
                c1 = i
                break  #even this break is not necessar. there is only one element
        if regions[c1][regions[c1].index(dbl_e[edge][0]) - 1] == dbl_e[edge][1]: 
            vertex1 = dbl_e[edge][1] ; vertex2 = dbl_e[edge][0]
        else:
            vertex1 = dbl_e[edge][0] ; vertex2 = dbl_e[edge][1]
        for e_nghr in range(2*len(e)):
            if e_nghr in ebor_dual:
                continue
            if (e_nghr not in e_mut or edge not in e_mut):
                if (Alpha_NL[edge, e_nghr] == 0 or Beta_NL[edge, e_nghr] == 0):
                    if REA[c1,e_nghr] != 0: 
                        if regions[c1][regions[c1].index(dbl_e[e_nghr][0]) - 1] == dbl_e[e_nghr][1]: 
                            vn1 = dbl_e[e_nghr][1] ; vn2 = dbl_e[e_nghr][0]
                        else:
                            vn1 = dbl_e[e_nghr][0] ; vn2 = dbl_e[e_nghr][1]
                        if (vertex1 in dbl_e[e_nghr] or vertex2 in dbl_e[e_nghr] ) :
                            Alpha_NL[edge, e_nghr] = Alpha_NL[e_nghr, edge] = alpha_0 * I_NN_uu #Kernel_aa(edge , e_nghr)
                            Beta_NL[edge, e_nghr] = Beta_NL[e_nghr, edge] = beta_0 * I_NN_uv #Kernel_ab(edge , e_nghr)
                        elif (regions[c1][regions[c1].index(vertex2) - 2] == vn2 and regions[c1][regions[c1].index(vertex1) - 2] == vn1 ):
                            Alpha_NL[edge, e_nghr] = Alpha_NL[e_nghr, edge] = alpha_0 * I_NP_uu #Kernel_aa(edge , e_nghr)
                            Beta_NL[edge, e_nghr] = Beta_NL[e_nghr, edge] = beta_0 * I_NP_uv #Kernel_ab(edge , e_nghr)
                        else:
                            Alpha_NL[edge, e_nghr] = Alpha_NL[e_nghr, edge] = alpha_0 * I_CP_uu * np.sqrt(LL[edge] * LL[e_nghr]) #Kernel_aa(edge , e_nghr)
                            Beta_NL[edge, e_nghr] = Beta_NL[e_nghr, edge] = beta_0 * I_CP_uv * np.sqrt(LL[edge] * LL[e_nghr]) #Kernel_ab(edge , e_nghr)
            else: 
                if (Alpha_NL[edge, e_nghr] == 0 or Beta_NL[edge, e_nghr] == 0):
                    if REA[c1,e_nghr] != 0: 
                        if regions[c1][regions[c1].index(dbl_e[e_nghr][0]) - 1] == dbl_e[e_nghr][1]: 
                            vn1 = dbl_e[e_nghr][1] ; vn2 = dbl_e[e_nghr][0]
                        else:
                            vn1 = dbl_e[e_nghr][0] ; vn2 = dbl_e[e_nghr][1]
                        if (vertex1 in dbl_e[e_nghr] or vertex2 in dbl_e[e_nghr] ) :
                            Alpha_NL[edge, e_nghr] = Alpha_NL[e_nghr, edge] = alpha_0 * I_NN_mut_uu #Kernel_aa(edge , e_nghr)
                            Beta_NL[edge, e_nghr] = Beta_NL[e_nghr, edge] = beta_0 * I_NN_mut_uv #Kernel_ab(edge , e_nghr)
                        elif (regions[c1][regions[c1].index(vertex2) - 2] == vn2 and regions[c1][regions[c1].index(vertex1) - 2] == vn1 ):
                            Alpha_NL[edge, e_nghr] = Alpha_NL[e_nghr, edge] = alpha_0 * I_NP_mut_uu #Kernel_aa(edge , e_nghr)
                            Beta_NL[edge, e_nghr] = Beta_NL[e_nghr, edge] = beta_0 * I_NP_mut_uv #Kernel_ab(edge , e_nghr)
                        else:
                            Alpha_NL[edge, e_nghr] = Alpha_NL[e_nghr, edge] = alpha_0 * I_CP_mut_uu * np.sqrt(LL[edge] * LL[e_nghr]) #Kernel_aa(edge , e_nghr)
                            Beta_NL[edge, e_nghr] = Beta_NL[e_nghr, edge] = beta_0 * I_CP_mut_uv * np.sqrt(LL[edge] * LL[e_nghr]) #Kernel_ab(edge , e_nghr)
                
    for edd in ebor_dual: # With the next command this is unnecessary because eduals only have diagonal terms anyways
        Alpha_NL[edd,:] = 0 ; Alpha_NL[:,edd] = 0
        Beta_NL[edd,:] = 0 ; Beta_NL[:,edd] = 0
                
    for i in range(2*len(e)):
        Alpha_NL[i,i] = 0
        Beta_NL[i,i] = 0
    
    for ee in pairs:
        Alpha_NL[ee[0] + len(e), :] = Alpha_NL[ee[1], :]
        Alpha_NL[ee[1] + len(e), :] = Alpha_NL[ee[0], :]
        Beta_NL[ ee[0] + len(e), :] = Beta_NL[ ee[1], :]
        Beta_NL[ ee[1] + len(e), :] = Beta_NL[ ee[0], :]
    
    Alpha_NL *= coeff * Alpha_NL    
    Beta_NL  *= coeff * Beta_NL
    
    for i in range(2*len(e)):
        if (i not in e_mut):        
            Alpha_NL[i,i] = alpha_0 * I_s(LL[i] , xi_uu)
            Beta_NL[i,i] = beta_0 * I_s(LL[i] , xi_uv)
        else :
            Alpha_NL[i,i] = alpha_0 * c0 * I_s(LL[i] , xi_uu * mutlen)
            Beta_NL[i,i] = beta_0 * c0 * I_s(LL[i] , xi_uv * mutlen)
    
    Calpha = sp.sparse.coo_matrix(Alpha_NL)
    Cbeta = sp.sparse.coo_matrix(Beta_NL)    
    
    ############################# Elongation Part ############################

    L_0 = np.sqrt( (CC.dot(rx))**2 + (CC.dot(ry))**2 )
    
    LL_0 = TensorProduct(np.ones((2,1)) , L_0)

    delx_0 = Delx(e, regions, rx, ry, cenx, ceny)[1] 
    dely_0 = Dely(e, regions, rx, ry, cenx, ceny)[1]
    
    Dx = np.sqrt(delx_0**2)
    Dy = np.sqrt(dely_0**2)
    
    I_xx = np.dot(Dx**2, LL_0) 
    I_yy = np.dot(Dy**2, LL_0) 
    I_xy = np.dot(elon * delx_0 * dely_0, LL_0)
    
    for i in range(len(regions)):
        I_xx[i,0] = round(I_xx[i,0],7)
        I_yy[i,0] = round(I_yy[i,0],7)
        I_xy[i,0] = round(I_xy[i,0],7)
    
    I_tot = I_xx + I_yy ; I_dif = I_xx - I_yy ; I_2 = I_xx * I_yy
    
    lambda_1 = ( I_tot + np.sqrt(I_dif**2 + 4 * I_xy**2) )/2;    
    lambda_2 = ( I_tot - np.sqrt(I_dif**2 + 4 * I_xy**2) )/2;    
    epsilon = np.sqrt(lambda_1/lambda_2)     
    gamma = (lambda_1 - I_xx) / I_xy 
    elonx = np.sqrt(epsilon) - 1

    eps1 = (I_xx - I_yy)/(2 * per) ; eps2 = I_xy/per    
    eps_mag = np.sqrt((np.mean(eps1))**2 + (np.mean(eps2))**2)    
    epsilon_magnitude[ratio] = eps_mag
    
    theta1 = (1/2) * np.arccos(np.mean(eps1)/eps_mag) * 180/np.pi
    theta2 = (1/2) * np.arcsin(np.mean(eps2)/eps_mag) * 180/np.pi
    
    ############## Theta2 measured from y-axis ###############
    
    Theta_e = np.arctan( gamma ) * 180/np.pi    
    elonhat_x = 1/np.sqrt(1+gamma**2) ; elonhat_y = gamma/np.sqrt(1+gamma**2)    
    Ey = np.sqrt(np.mean(elonhat_y**2)) ; Ex = np.sqrt(np.mean(elonhat_x**2)) ; 
    
    for i in range(len(Theta_e)):
        if Theta_e[i] < 0:
            Theta_e [i] += 180
    
    Ang_elon = np.mean(Theta_e)
    
    ################################# Initialization ################################
    
    for run in range(Nrun):

        Ab_0 = np.zeros((2*M , 1))
        Bb_0 = np.zeros((2*M , 1))        
        
        V_A = np.zeros((2*M , 1))

        for i in range(2*M):
            V_A[i] = v_Ab_bulk * ( random.random() - random.random() )

        Ab_0 = Ab0 * ( 1 + V_A )

############ This part is only for fixed boundary condition signal ############
        
#        V_B = np.zeros((len(pairs_LR) , 2))
#        
#        for ee in range(len(pairs_LR)):
#            V_B[ee][0] = v_Ab_boundary * ( random.random() - random.random() )
#            V_B[ee][1] = v_Ab_boundary * ( random.random() - random.random() )
#            
#        for ee in range(len(pairs_LR)):
#            Ab_0[pairs_LR[ee][0]] = (1 * Ab0 * ( bc + V_B[ee][0] ) ) ; Ab_0[pairs_LR[ee][1] + len(e)] = Ab_0[pairs_LR[ee][0]]
##            Ab_0[pairs_LR[ee][1]] = (1 * Ab0 * ( bc + V_B[ee][1] ) ) ; Ab_0[pairs_LR[ee][0] + len(e)] = Ab_0[pairs_LR[ee][1]]
        
###############################################################################
        
        for ee in pairs:
            Ab_0[ee[0]] = (1 * Ab0 * (bc + v_Ab_boundary * ( random.random() - random.random() ) )); Ab_0[ee[1] + len(e)] = Ab_0[ee[0]]
            Ab_0[ee[1]] = (1 * Ab0 * (bc + v_Ab_boundary * ( random.random() - random.random() ) )); Ab_0[ee[0] + len(e)] = Ab_0[ee[1]]


        ##################### UNIFORM INITIAL POLARIZATION #####################
       
        Ab = Ab_0
        Bb_0[0:M  ,  0] = Ab_0[M:2*M , 0]
        Bb_0[M:2*M , 0] = Ab_0[0:M  ,  0]
        
        Bb = Bb_0
        
        at = At * np.ones((len(regions),1))
        bt = Bt * np.ones((len(regions),1))

        for mut in mutant:
            bt[mut] = Bmut
            at[mut] = Amut

        Af = af(at, Ab, np.zeros((2*len(e), 1)), LL)
        Bf = bf(bt, Ab, np.zeros((2*len(e), 1)), LL , nu) 
    
        afbf = Af * Bf
        
        P_avg = np.zeros((len(range(Bmax-Bmin)), 1))
        P_mag = np.zeros((len(range(Bmax-Bmin)), 1))
        
        P_avg_time = np.zeros((int((t_max-t_min)/dt), 1))
        P_mag_time = np.zeros((int((t_max-t_min)/dt), 1))
            
        I_avg_time = np.zeros((int((t_max-t_min)/dt), 1))
        I_mag_time = np.zeros((int((t_max-t_min)/dt), 1))

        Px = Cdelx.dot(Ab - Bb)/2
        Py = Cdely.dot(Ab - Bb)/2
                
        PxA = Cdelx.dot(Ab)
        PyA = Cdely.dot(Ab)

        PxB = Cdelx.dot(Bb)
        PyB = Cdely.dot(Bb)

        P = np.sqrt(Px**2 + Py**2) 
        P_init = P
        
        PA = np.sqrt(PxA**2 + PyA**2)    
        PB = np.sqrt(PxB**2 + PyB**2)

        Pxt = np.transpose(Px)
        Pyt = np.transpose(Py)
        
        Pt = np.transpose(P)
        
        for i in range(len(regions)):
            Px[i,0] = round(Px[i,0],10)
            Py[i,0] = round(Py[i,0],10)
            PxA[i,0] = round(PxA[i,0],7)
            PyA[i,0] = round(PyA[i,0],7)
            PxB[i,0] = round(PxB[i,0],7)
            PyB[i,0] = round(PyB[i,0],7)
            
            
        P0 = np.sqrt(np.mean(Px)**2 + np.mean(Py)**2)
        
            
        print(P0)
        print(np.mean(np.sqrt(Px**2 + Py**2)))
        
        Ang_avg = np.zeros((int((t_max-t_min)/dt), 1))        
        Ang_std = np.zeros((int((t_max-t_min)/dt), 1))    
        deltaAng = np.zeros((R , 1))        

        PE_avg = np.zeros((int((t_max-t_min)/dt), 1))    
        P2av_E2av = np.zeros((int((t_max-t_min)/dt), 1))
        Pav_Eav = np.zeros((int((t_max-t_min)/dt), 1))

        DD = np.zeros((int((t_max-t_min)/dt), 1))    
        SS = np.zeros((int((t_max-t_min)/dt), 1))
        QQ = np.zeros((int((t_max-t_min)/dt), 1))

        Rc = np.zeros((int((t_max-t_min)/dt), 1))          # correlation length vs time
        PRc = np.zeros((int((t_max-t_min)/dt), 1))          # correlation length vs time
        RAngc = np.zeros((int((t_max-t_min)/dt), 1))          # correlation length vs time

        Qmean = np.zeros((int((t_max-t_min)/dt), 1))
        Qvar = np.zeros((int((t_max-t_min)/dt), 1))
    
        Afmean = np.zeros((int((t_max-t_min)/dt), 1))
        AfBfvar = np.zeros((int((t_max-t_min)/dt), 1))
        
        Rij = np.zeros((len(regions) , len(regions)))

        for i in range(len(regions)):
            for j in range(len(regions)):
                Rij[i,j] = np.sqrt( ( min(elon*(nx - 4)*l0*np.sqrt(3)/2 - abs( cenx[i,0] - cenx[j,0] ) + 1 , abs( cenx[i,0] - cenx[j,0] ) + 1) )**2 + ( min((ny  - 4)*l0*3/2 - abs( ceny[i,0] - ceny[j,0] ) + 1 , abs( ceny[i,0] - ceny[j,0] )  + 1 ))**2)
        
        Rc[0,0] = R_corr(Px , Py , Rij)
        PRc[0,0] = P_corr(Px , Py , Rij)
        RAngc[0,0] = Ang_corr(Px , Py , Rij)
    
        Qmean[0,0] = np.mean(P)
        Qvar[0,0] = np.sqrt(np.var(P))

        Afmean[0,0] = np.mean(afbf)
        AfBfvar[0,0] = np.sqrt(np.var(afbf))

        afbf_init = afbf

        y_av = np.mean(Py/P) ; x_av = np.mean(Px/P)
        Ang_avg[0,0] = np.arctan2(y_av , x_av) * 180/np.pi    
        Ang_std[0,0] = np.sqrt( np.mean( deltaAng**2 ) )
        Ang = np.arctan2(Py,Px) * 180 /np.pi                
        for c in range(R):    
            deltaAng[c,0] = min((360 - abs(Ang[c , 0] - Ang_avg[0 , 0])) , abs(Ang[c , 0] - Ang_avg[0,0]))            

        perpcorr0 = np.sqrt((elonhat_x * Py - elonhat_y * Px)**2)#/np.mean(P)
        perpang0 = np.sqrt((elonhat_x * Py - elonhat_y * Px)**2)#/P
        PE_avg[0] = np.mean(perpang0)
        P2av_E2av[0] = abs(np.sqrt(np.mean(elonhat_x**2)) * np.sqrt(np.mean(Py**2)) - np.sqrt(np.mean(elonhat_y**2)) * np.sqrt(np.mean(Px**2)) )#/ ( np.sqrt(np.mean(Px**2 + Py**2)) )#* np.sqrt(np.mean(elonhat_x**2) + np.mean(elonhat_y**2)))
        Pav_Eav[0] = abs((np.mean(elonhat_x)) * (np.mean(Py)) - (np.mean(elonhat_y)) * (np.mean(Px)) )#/ ( np.sqrt(np.mean(Px**2 + Py**2)) )#* np.sqrt(np.mean(elonhat_x**2) + np.mean(elonhat_y**2)))
        Ang_p[0,ratio] = abs(min(180 - abs(Ang_avg[0] - Ang_elon)  , abs(Ang_avg[0] - Ang_elon))) #min(abs(Ang_avg[0] - Ang_elon) , abs( 180 - Ang_avg[0] + Ang_elon) )
            
        px_mut = np.zeros(( len(mutant), int((t_max-t_min)/dt) ))
        py_mut = np.zeros(( len(mutant), int((t_max-t_min)/dt) ))
        
        mut = np.zeros((len(mutant) , len(regions)))
        
        for i in range(len(mutant)):
            mut[i, mutant[i]] = 1
        
        Cmut = sp.sparse.coo_matrix(mut)
            
        px_mut[: , 0] = Cmut.dot(Px).reshape(len(mutant)) 
        py_mut[: , 0] = Cmut.dot(Py).reshape(len(mutant))
        
        p_avg_mut = np.zeros((int((t_max-t_min)/dt) , 1))
        p_mag_mut = np.zeros((int((t_max-t_min)/dt) , 1))
        
        p_avg_wt = np.zeros((int((t_max-t_min)/dt) , 1))
        p_mag_wt = np.zeros((int((t_max-t_min)/dt) , 1))
        
        N_m = len(mutant)
        N_wt = len(regions) - N_m
        
        p_avg_mut[0 , 0] = np.sqrt( (np.mean(px_mut[:,0]))**2 + (np.mean(py_mut[:,0]))**2 )
        p_mag_mut[0 , 0] = np.mean( np.sqrt( (px_mut[:,0]**2) + (py_mut[:,0]**2) ) )
        
        p_avg_wt[0 , 0] = np.sqrt( ((R * np.mean(Px) - N_m * np.mean(px_mut[:,0]))/N_wt)**2 + ((R * np.mean(Py) - N_m * np.mean(py_mut[:,0]))/N_wt)**2 )
        p_mag_wt[0 , 0] = ( R * np.mean( np.sqrt( (Px[:,0]**2) + (Py[:,0]**2) ) ) - N_m * np.mean( np.sqrt( (px_mut[:,0]**2) + (py_mut[:,0]**2) ) ) ) / N_wt
    
        P_mag_time[0 , 0] = np.mean(P)
        P_avg_time[0 , 0] = np.sqrt(np.mean(Px)**2 + np.mean(Py)**2)
        
        DDx = PxA - PxB ; SSx = PxA + PxB
        DDy = PyA - PyB ; SSy = PyA + PyB
        
        DD_init = np.sqrt(DDx**2 + DDy**2)
        SS_init = np.sqrt(SSx**2 + SSy**2)
        QQ_init = np.mean(DD_init)
        
        DD[0 , 0] = np.mean(DD_init)
        SS[0 , 0] = np.mean(SS_init)
        QQ[0 , 0] = QQ_init
            
        fig = plt.figure()
        fig.set_figheight(3*elon*l0 + max(ry) + 3*elon*l0 + min(ry))
        fig.set_figwidth(3*l0 + max(rx) + 3*l0 + min(rx) )
        for el in e:
            lines = plt.plot([rx[el[0]],rx[el[1]]],[ry[el[0]],ry[el[1]]]) 
            plt.setp(lines, color='k', linewidth=5)                

        ax = fig.add_subplot(111)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(100)
            item.set_fontname("Times New Roman")
            fname = 'InitEd_e%02d'%elon + '_r%02d.svg' %run  
        plt.savefig(fname)        
        plt.close()
        
        fig = plt.figure()
        fig.set_figheight(3*elon*l0 + max(ry) + 3*elon*l0 + min(ry))
        fig.set_figwidth(3*l0 + max(rx) + 3*l0 + min(rx) )
        for el in e:
            lines = plt.plot([rx[el[0]],rx[el[1]]],[ry[el[0]],ry[el[1]]]) 
            plt.setp(lines, color='k', linewidth=5)                
        plt.quiver(cenx, ceny, Px/1, Py/1, pivot='mid', alpha = 0.5)
        plt.quiver(cenx, ceny, Px/1, Py/1, pivot='mid', edgecolor='steelblue', facecolor='steelblue', linewidth=elon**2)     
        
        ax = fig.add_subplot(111)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(100)
            item.set_fontname("Times New Roman")
            fname = 'P_Cell_t00'#_e%02d'%elon + '_r%02d.svg' %run  
        plt.savefig(fname)        
        plt.close()

        fig = plt.figure()
        fig.set_figheight(3*elon*l0 + max(ry) + 3*elon*l0 + min(ry))
        fig.set_figwidth(3*l0 + max(rx) + 3*l0 + min(rx) )
        for el in e:
            lines = plt.plot([rx[el[0]],rx[el[1]]],[ry[el[0]],ry[el[1]]]) 
            plt.setp(lines, color='k', linewidth=5)                

        ax = fig.add_subplot(111)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(100)
            item.set_fontname("Times New Roman")
            fname = 'I_Cell_t00'#_e%02d'%elon + '_r%02d.svg' %run  
        plt.savefig(fname)        
        plt.close()
            
        fig = plt.figure()        
        edd = np.arange(2*M)
        plt.plot(edd, Ab_0 + Bb_0, 'r', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'AbPlusBb_Init'
        plt.savefig(fname)
        plt.close()
        
        fig = plt.figure()        
        edd = np.arange(2*M)
        plt.plot(edd, Ab_0 - Bb_0, 'b', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'AbMinusBb_Init'
        plt.savefig(fname)
        plt.close()
        

        fig = plt.figure()        
        cell = np.arange(len(regions))
        plt.plot(cell, DD_init/QQ_init, 'm', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'DD_Init'
        plt.savefig(fname)
        plt.close()
        
        fig = plt.figure()        
        cell = np.arange(len(regions))
        plt.plot(cell, SS_init/QQ_init, 'c', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'SS_Init'
        plt.savefig(fname)
        plt.close()

    ############################# INITIAL ROSE PLOTS #############################

        N_rose = 24             
        theta = np.linspace(0.0, 2 * np.pi, N_rose, endpoint=False)        
        width = (2 * np.pi / N_rose)#(360 / N_rose) # * np.ones(N_rose)        
        
        Ang_hist_Ed = np.mod(Ang_hist_Ed , 2*np.pi)
        Ang_range_Ed = np.floor(Ang_hist_Ed / width)
        radii0_Ed = np.zeros(N_rose)
        rr = np.reshape(radii , (24 , 1))
        
        for i in range(len(Ang_range_Ed)):
            radii0_Ed[ int(Ang_range_Ed[i,0]) ] += 1/len(e)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        bars_Ed = ax.bar(theta, radii0_Ed, width=width, bottom=0.0)

        radii_Ed = radii0_Ed.tolist()
        theta_label_Ed = np.mod( radii_Ed.index(np.max(radii_Ed)) + N_rose/2 , N_rose ) * width * 180/np.pi
        pmax_Ed = max(radii_Ed)
        ax.set_rmax(pmax_Ed)        
        ax.set_rticks([ round(0.25 * pmax_Ed , 2) , round(0.5 * pmax_Ed , 2) , round(0.75 * pmax_Ed , 2) , round(1.00 * pmax_Ed , 2) ] )
        ax.set_rlabel_position(theta_label_Ed)  # get radial labels away from plotted line
        ax.grid(True)
        
        # Use custom colors and opacity
        for r_Ed, bar_Ed in zip(radii_Ed, bars_Ed):
            bar_Ed.set_facecolor(plt.cm.GnBu( ( r_Ed ) / max(radii_Ed) ))
            bar_Ed.set_alpha(1)    
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")                                
        fname_Ed = 'RosePol_Initial_Edge_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname_Ed)
        plt.close()      


        N_rose = 24             
        theta = np.linspace(0.0, 2 * np.pi, N_rose, endpoint=False)        
        width = (2 * np.pi / N_rose)#(360 / N_rose) # * np.ones(N_rose)        
        
        Ang_hist = np.arctan2(Py , Px) # * 180/np.pi
        Ang_hist = np.mod(Ang_hist , 2*np.pi)
        Ang_range = np.floor(Ang_hist / width)
        radii0 = np.zeros(N_rose)
        
        for i in range(len(Ang_range)):
            radii0[ int(Ang_range[i,0]) ] += 1/len(regions)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        bars = ax.bar(theta, radii0, width=width, bottom=0.0)

        radii = radii0.tolist()
        theta_label = np.mod( radii.index(np.max(radii)) + N_rose/2 , N_rose ) * width * 180/np.pi
        pmax = max(radii)
        ax.set_rmax(pmax)
#        ax.set_rticks([25%, 50%, 75%, 100%])  # less radial ticks
        ax.set_rticks([ round(0.25 * pmax , 2) , round(0.5 * pmax , 2) , round(0.75 * pmax , 2) , round(1.00 * pmax , 2) ] )
        ax.set_rlabel_position(theta_label)  # get radial labels away from plotted line
        ax.grid(True)
        
        # Use custom colors and opacity
        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.GnBu( ( r ) / max(radii) ))
            bar.set_alpha(1)    
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")                                
        fname = 'RosePol_Initial_Cell_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()      



#        Ab = Ab_0
    
        #################### TIME EVOLUTION LOOP ####################
        
        for s in range(1,int((t_max-t_min)/dt)):      # CHECK THIS +1
        
        #################### RUNGE-KUTTA MATRICES ####################
            noise = D * ( np.random.rand(2*len(e),1) - np.random.rand(2*len(e),1) ) 
            
            RKa = np.zeros((2*len(e),1))
        
            Ka1 = np.zeros((2*len(e),1))
            Ka2 = np.zeros((2*len(e),1))
            Ka3 = np.zeros((2*len(e),1))
            Ka4 = np.zeros((2*len(e),1))
    
        ############################### Runge-Kutta 4 #############################
    
            Ka1 = Kf * af(at, Ab, np.zeros((2*len(e), 1)), LL) * bf(bt, Ab, np.zeros((2*len(e), 1)), LL , nu) * (1 + Calpha.dot(Ab)) - Kd * (Ab) * (1 + Cbeta.dot(Cswap.dot(Ab)) ) + (1) * noise * LL**(1/2) + MG * Relx * np.exp( - tt * s*dt/T_mg)
    
        ###########################################################################        
        
            Ka2 = Kf * af(at, Ab, Ka1/2, LL) * bf(bt, Ab, Ka1/2, LL , nu) * (1 + Calpha.dot(Ab + dt*Ka1/2) ) - Kd * (Ab + dt*Ka1/2) * (1 + Cbeta.dot(Cswap.dot((Ab + dt*Ka1/2))) ) + (1) * noise * LL**(1/2) + MG * Relx * np.exp( - tt * s*dt/T_mg)   
            
        ###########################################################################
        
            Ka3 = Kf * af(at, Ab, Ka2/2, LL) * bf(bt, Ab, Ka2/2, LL , nu) * (1 + Calpha.dot(Ab + dt*Ka2/2) ) - Kd * (Ab + dt*Ka2/2) * (1 + Cbeta.dot(Cswap.dot((Ab + dt*Ka2/2))) ) + (1) * noise * LL**(1/2) + MG * Relx * np.exp( - tt * s*dt/T_mg)     
            
        ###########################################################################
        
            Ka4 = Kf * af(at, Ab, Ka3, LL) * bf(bt, Ab, Ka3, LL , nu) * (1 + Calpha.dot(Ab + dt*Ka3)) - Kd * (Ab + dt*Ka3) * (1 + Cbeta.dot(Cswap.dot((Ab + dt*Ka3))) ) + (1) * noise * LL**(1/2) + MG * Relx * np.exp( - tt * s*dt/T_mg)
            
        ###########################################################################
            
            RKa = (Ka1 + 2 * Ka2 + 2 * Ka3 + Ka4)
        
            Ab = Ab + (dt/6) * RKa

############# This part is only for fixed boundary condition signal ############
#
#            for ee in range(len(pairs_LR)):
#                Ab[pairs_LR[ee][0]] = (1 * Ab0 * ( bc + V_B[ee][0] ) ) ; Ab_0[pairs_LR[ee][1] + len(e)] = Ab_0[pairs_LR[ee][0]]
##                Ab[pairs_LR[ee][1]] = (1 * Ab0 * ( bc + V_B[ee][1] ) ) ; Ab_0[pairs_LR[ee][0] + len(e)] = Ab_0[pairs_LR[ee][1]]
#
################################################################################

            Bb[0:M , 0] = Ab[M:2*M , 0]
            Bb[M:2*M , 0] = Ab[0:M , 0]        

            for ee in range(len(pairs)):
                Bb[pairs[ee][1] , 0] = Ab[pairs[ee][0] , 0]
                Bb[pairs[ee][1] + len(e) , 0] = Ab[pairs[ee][0] + len(e) , 0]

            Px = Cdelx.dot(Ab - Bb)/2
            Py = Cdely.dot(Ab - Bb)/2

            PxA = Cdelx.dot(Ab)
            PyA = Cdely.dot(Ab)
    
            PxB = Cdelx.dot(Bb)
            PyB = Cdely.dot(Bb)

            for i in range(len(regions)):
                Px[i,0] = round(Px[i,0],10)
                Py[i,0] = round(Py[i,0],10)
                PxA[i,0] = round(PxA[i,0],7)
                PyA[i,0] = round(PyA[i,0],7)
                PxB[i,0] = round(PxB[i,0],7)
                PyB[i,0] = round(PyB[i,0],7)
                    
            px_mut[:, s] = Cmut.dot(Px).reshape(len(mutant))
            py_mut[:, s] = Cmut.dot(Py).reshape(len(mutant)) 
            p_avg_mut[s] = np.sqrt( (np.mean( px_mut[:,s] ) )**2 + (np.mean(py_mut[:,s]))**2 )
            p_mag_mut[s] = np.mean( np.sqrt( (px_mut[:,s]**2) + (py_mut[:,s]**2) ) )
            
            P = np.sqrt(Px**2 + Py**2)
            PA = np.sqrt(PxA**2 + PyA**2)
            PB = np.sqrt(PxB**2 + PyB**2)
            for i in range(len(regions)):
                P[i,0] = round(P[i,0],7)
                PA[i,0] = round(P[i,0],7)
                PB[i,0] = round(P[i,0],7)
                
            P_mag_time[s , 0] = np.mean(P)
            P_avg_time[s , 0] = np.sqrt(np.mean(Px)**2 + np.mean(Py)**2)
            
            DDx = PxA - PxB ; SSx = PxA + PxB
            DDy = PyA - PyB ; SSy = PyA + PyB
            
            DD[s , 0] = np.mean(np.sqrt(DDx**2 + DDy**2))
            SS[s , 0] = np.mean(np.sqrt(SSx**2 + SSy**2))
            QQ[s , 0] = DD[s , 0]
                
            p_avg_wt[s , 0] = np.sqrt( ((R * np.mean(Px) - N_m * np.mean(px_mut[:,s]))/N_wt)**2 + ((R * np.mean(Py) - N_m * np.mean(py_mut[:,s]))/N_wt)**2 )
            p_mag_wt[s , 0] = ( R * np.mean( np.sqrt( (Px**2) + (Py**2) ) ) - N_m * np.mean( np.sqrt( (px_mut[:,s]**2) + (py_mut[:,s]**2) ) ) ) / N_wt
    
            Qmean[s,0] = np.mean(P)
            Qvar[s,0] = np.sqrt(np.var(P))

            Rc[s , 0] = R_corr(Px , Py , Rij)
            PRc[s , 0] = P_corr(Px, Py , Rij)
            RAngc[s , 0] = Ang_corr(Px , Py , Rij)

            Af = af(at, Ab, np.zeros((2*len(e), 1)), LL)
            Bf = bf(bt, Ab, np.zeros((2*len(e), 1)), LL , nu)
            afbf = Af * Bf
            Afmean[s,0] = np.mean(afbf)
            AfBfvar[s,0] = np.sqrt(np.var(afbf))

            y_av = np.mean(Py/P);   x_av = np.mean(Px/P)                
            Ang_avg[s , 0] = np.arctan2(y_av , x_av) * 180 /np.pi    
            Ang = np.arctan2(Py,Px) * 180 /np.pi               
            deltaAng = np.zeros((R , 1))
            for c in range(R):
                deltaAng[c , 0] = min((360 - abs(Ang[c , 0] - Ang_avg[s,0])) , abs(Ang[c , 0] - Ang_avg[s,0]))
            Ang_std[s , 0] = np.sqrt( np.mean( deltaAng**2 ) )
            
            perpcorr = abs(elonhat_x * Py - elonhat_y * Px)#/P_mag_time[s]    
            perpang = abs(elonhat_x * Py - elonhat_y * Px)#/P    
            PE_avg[s] = np.mean(perpang)
            P2av_E2av[s] = abs(np.sqrt(np.mean(elonhat_x**2)) * np.sqrt(np.mean(Py**2)) - np.sqrt(np.mean(elonhat_y**2)) * np.sqrt(np.mean(Px**2)) )#/ ( np.sqrt(np.mean(Px**2 + Py**2)) )#* np.sqrt(np.mean(elonhat_x**2) + np.mean(elonhat_y**2)))
            Pav_Eav[s] = abs((np.mean(elonhat_x)) * (np.mean(Py)) - (np.mean(elonhat_y)) * (np.mean(Px)) )
            Ang_p[s, ratio] = np.minimum( abs( abs(np.mean(Theta_e)) - abs(Ang_avg[s,0]) )  , 180 - abs( abs(np.mean(Theta_e) - abs(Ang_avg[s,0])) ) )  #abs(min(180 - abs(Ang_avg[s,0] - Ang_elon)  , abs(Ang_avg[s,0] - Ang_elon))) 

            if any(al <= 0 for al in bf(bt, Ab, np.zeros((2*len(e), 1)), LL , nu) ) : print("FUCK! bf")   #(CREB.dot(Ab*LL))/(CREB.dot(LL)) ): print("FUCK!")
            if any(al <= 0 for al in af(at, Ab, np.zeros((2*len(e), 1)), LL) ) : print("FUCK! af")   #(CREB.dot(Ab*LL))/(CREB.dot(LL)) ): print("FUCK!")
    
            if s % s0 == 0:           
                                        
                fig = plt.figure()
                fig.set_figheight(3*elon*l0 + max(ry) + 3*elon*l0 + min(ry))
                fig.set_figwidth(3*l0 + max(rx) + 3*l0 + min(rx) )
                for el in e:
                    lines = plt.plot([rx[el[0]],rx[el[1]]],[ry[el[0]],ry[el[1]]]) 
                    plt.setp(lines, color='k', linewidth=5)                                        

                ax = fig.add_subplot(111)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(100)
                    item.set_fontname("Times New Roman")
                fname = 'PolEdge_t%02d'%s + '_e%02d'%elon + '_r%02d.svg' %run  
                plt.savefig(fname)                        
                plt.close()

                fig = plt.figure()
                fig.set_figheight(3*elon*l0 + max(ry) + 3*elon*l0 + min(ry))
                fig.set_figwidth(3*l0 + max(rx) + 3*l0 + min(rx) )
                for el in e:
                    lines = plt.plot([rx[el[0]],rx[el[1]]],[ry[el[0]],ry[el[1]]]) 
                    plt.setp(lines, color='k', linewidth=5)                                        
                plt.quiver(cenx, ceny, Px/1, Py/1, pivot='mid', alpha = 0.5)
                plt.quiver(cenx, ceny, Px/1, Py/1, pivot='mid', edgecolor='steelblue', facecolor='steelblue', linewidth=elon**2)                                
                ax = fig.add_subplot(111)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(100)
                    item.set_fontname("Times New Roman")
                fname = 'P_Cell_t%03d'%(s*dt*Kd) # + '_e%02d'%elon + '_r%02d.svg' %run   
                plt.savefig(fname)                        
                plt.close()

                fig = plt.figure()
                fig.set_figheight(3*elon*l0 + max(ry) + 3*elon*l0 + min(ry))
                fig.set_figwidth(3*l0 + max(rx) + 3*l0 + min(rx) )
                for el in e:
                    lines = plt.plot([rx[el[0]],rx[el[1]]],[ry[el[0]],ry[el[1]]]) 
                    plt.setp(lines, color='k', linewidth=5)                                        

                ax = fig.add_subplot(111)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(100)
                    item.set_fontname("Times New Roman")
                fname = 'I_Cell_t%02d'%(s*dt*Kd) # + '_e%02d'%elon + '_r%02d.svg' %run   
                plt.savefig(fname)                        
                plt.close()

                print(s)


        fig = plt.figure()
        fig.set_figheight(3*elon*l0 + max(ry) + 3*elon*l0 + min(ry))
        fig.set_figwidth(3*l0 + max(rx) + 3*l0 + min(rx) )
        for el in e:
            lines = plt.plot([rx[el[0]],rx[el[1]]],[ry[el[0]],ry[el[1]]]) 
            plt.setp(lines, color='k', linewidth=5)   

        ax = fig.add_subplot(111)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(100)
            item.set_fontname("Times New Roman")
        fname = 'finalEdge_e%02d'%elon + '_r%02d.svg'%run   
        plt.savefig(fname)        
        plt.close()

        fig = plt.figure()
        fig.set_figheight(3*elon*l0 + max(ry) + 3*elon*l0 + min(ry))
        fig.set_figwidth(3*l0 + max(rx) + 3*l0 + min(rx) )
        for el in e:
            lines = plt.plot([rx[el[0]],rx[el[1]]],[ry[el[0]],ry[el[1]]]) 
            plt.setp(lines, color='k', linewidth=5)   
        plt.quiver(cenx, ceny, Px/1, Py/1, pivot='mid', alpha = 0.5)
        plt.quiver(cenx, ceny, Px/1, Py/1, pivot='mid', edgecolor='steelblue', facecolor='steelblue', linewidth=elon**2)                
        ax = fig.add_subplot(111)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(100)
            item.set_fontname("Times New Roman")
        fname = 'Ptot_final_Cell.svg'  
        plt.savefig(fname)        
        plt.close()

        fig = plt.figure()
        fig.set_figheight(3*elon*l0 + max(ry) + 3*elon*l0 + min(ry))
        fig.set_figwidth(3*l0 + max(rx) + 3*l0 + min(rx) )
        for el in e:
            lines = plt.plot([rx[el[0]],rx[el[1]]],[ry[el[0]],ry[el[1]]]) 
            plt.setp(lines, color='k', linewidth=5)   
        plt.quiver(cenx, ceny, PxA/1, PyA/1, pivot='mid', alpha = 0.5)
        plt.quiver(cenx, ceny, PxA/1, PyA/1, pivot='mid', edgecolor='steelblue', facecolor='steelblue', linewidth=elon**2)                
        ax = fig.add_subplot(111)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(100)
            item.set_fontname("Times New Roman")
        fname = 'PA_final_Cell.svg'
        plt.savefig(fname)        
        plt.close()

        fig = plt.figure()
        fig.set_figheight(3*elon*l0 + max(ry) + 3*elon*l0 + min(ry))
        fig.set_figwidth(3*l0 + max(rx) + 3*l0 + min(rx) )
        for el in e:
            lines = plt.plot([rx[el[0]],rx[el[1]]],[ry[el[0]],ry[el[1]]]) 
            plt.setp(lines, color='k', linewidth=5)   
        plt.quiver(cenx, ceny, PxB/1, PyB/1, pivot='mid', alpha = 0.5)
        plt.quiver(cenx, ceny, PxB/1, PyB/1, pivot='mid', edgecolor='steelblue', facecolor='steelblue', linewidth=elon**2)                
        ax = fig.add_subplot(111)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(100)
            item.set_fontname("Times New Roman")
        fname = 'PB_final_Cell.svg'
        plt.savefig(fname)        
        plt.close()

        for i in range(len(regions)):
            Px[i,0] = round(Px[i,0],10)
            Py[i,0] = round(Py[i,0],10)
            PxA[i,0] = round(PxA[i,0],7)
            PyA[i,0] = round(PyA[i,0],7)
            PxB[i,0] = round(PxB[i,0],7)
            PyB[i,0] = round(PyB[i,0],7)

        P_avg[B - Bmin ,0] = np.sqrt(np.mean(Px)**2 + np.mean(Py)**2)
        P_mag[B - Bmin ,0] = np.mean(np.sqrt(Px**2 + Py**2))
        print(Bt, P_avg[B - Bmin ,0], P_mag[B - Bmin ,0])

        fig = plt.figure()
        plt.plot(t, P_avg_time, 'b', lw=2)    
        plt.plot(t, P_mag_time, 'g', lw=2)
        plt.plot(t, P_avg_time/P_mag_time, 'c', lw=2)    
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
        fname = 'PsvsT_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()    
        
        fig = plt.figure()
        plt.plot(t, Ang_avg, 'b', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
        fname = 'AngvsT_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        plt.plot(t, Ang_std, 'c', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
        fname = 'AngstdT_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()    
        plt.plot(t, Rc, 'b', lw=2)
        plt.plot(t, PRc, 'c', lw=2)
        plt.plot(t, RAngc, 'r', lw=2)    
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
        fname = 'Rc_RAngc_vsT_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()

        Af = af(at, Ab, np.zeros((2*len(e), 1)), LL) ; Bf = bf(at, Ab, np.zeros((2*len(e), 1)), LL , nu)

        fig = plt.figure()
        afbf = Af * Bf
        edd = np.arange(2*len(e))
        plt.plot(edd, afbf_init/np.mean(afbf_init), 'r', lw=2)
        plt.plot(edd, afbf/np.mean(afbf), 'b', lw=2)
        plt.ylim(0, 1.2 * max( max(afbf)/np.mean(afbf) , max(afbf_init)/np.mean(afbf_init) ))
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
        fname = 'AfBf_Init_Final_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()        
        P_final = np.sqrt(Px**2 + Py**2)        
        cell = np.arange(len(regions))
        plt.plot(cell, P_init/np.mean(P_init), 'r', lw=2)
        plt.plot(cell, P_final/np.mean(P_final), 'b', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
        fname = 'Q_Init_Final_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(t, Qvar, 'c', lw=2)    
        plt.plot(t, Qvar/Qmean, 'r', lw=2)    
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
        fname = 'QvarvsT_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()    

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(t, AfBfvar, 'b', lw=2)    
        plt.plot(t, AfBfvar/Afmean, 'r', lw=2)    
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
        fname = 'AfBfvarvsT_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()      

        fig = plt.figure()
        plt.plot(t, P2av_E2av, 'c', lw=2)                
        plt.plot(t, Pav_Eav, 'b', lw=2)    
        plt.plot(t, PE_avg, 'g', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
        fname = 'P_E_corr_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()    

        fig = plt.figure()
        plt.plot(t, Ang_p[:,ratio], 'k', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
            fname = 'Ang_p_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        plt.plot(t, SS/DD, 'r', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
            fname = 'SS_DD_vsT.svg'
        plt.savefig(fname)
        plt.close()
        
        
        fig = plt.figure()
        plt.plot(t, SS, 'r', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
            fname = 'SS_vsT.svg'
        plt.savefig(fname)
        plt.close()
        
        
        fig = plt.figure()
        plt.plot(t, DD, 'b', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")    
            fname = 'DD_vsT.svg'
        plt.savefig(fname)
        plt.close()
        
        
        N_rose = 24             
        theta = np.linspace(0.0, 2 * np.pi, N_rose, endpoint=False)        
        width = (2 * np.pi / N_rose)#(360 / N_rose) # * np.ones(N_rose)        
        
        Ang_hist = np.arctan2(Py , Px) # * 180/np.pi
        Ang_hist = np.mod(Ang_hist , 2*np.pi)
        Ang_range = np.floor(Ang_hist / width)
        radii0 = np.zeros(N_rose)
        
        for i in range(len(Ang_range)):
            radii0[ int(Ang_range[i,0]) ] += 1/len(regions)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        bars = ax.bar(theta, radii0, width=width, bottom=0.0)
        radii = radii0.tolist()
        theta_label = np.mod( radii.index(np.max(radii)) + N_rose/2 , N_rose ) * width * 180/np.pi
        pmax = max(radii)
        ax.set_rmax(pmax)
        ax.set_rticks([ round(0.25 * pmax , 2) , round(0.5 * pmax , 2) , round(0.75 * pmax , 2) , round(1.00 * pmax , 2) ] )
        ax.set_rlabel_position(theta_label)  # get radial labels away from plotted line
        ax.grid(True)
        
        # Use custom colors and opacity
        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.GnBu( ( r ) / max(radii) ))
            bar.set_alpha(1)    
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")                                
        fname = 'RosePol_Cell_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close()      
        
        
        xmin, xmax = plt.xlim()  # return the current ylim
        plt.xlim(xmin, xmax)     # set the ylim to ymin, ymax            fig = plt.figure()            
        fig = plt.figure()            
        ax = fig.add_subplot(111)
        fig.set_figheight(3)
        fig.set_figwidth(5)
        plt.xlim(xmax = 1) ; plt.xlim(xmin = 0)
        plt.scatter(elonx, P, c="teal", alpha=1, marker=r'o')
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")                                
        fname = 'cell-by-cell-PE_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close
        

        xmin, xmax = plt.xlim()  # return the current ylim
        plt.xlim(xmin, xmax)     # set the ylim to ymin, ymax            fig = plt.figure()            
        ymin, ymax = plt.ylim()  # return the current ylim
        plt.ylim(ymin, ymax)     # set the ylim to ymin, ymax
        Ang_pE = abs(90 - abs(Ang))#np.minimum( abs(AngP0 - 90) , abs(AngP0))             
        fig = plt.figure()            
        ax = fig.add_subplot(111)
        fig.set_figheight(3)
        fig.set_figwidth(5)
        plt.scatter(Theta_e, Ang_pE, c="royalblue", alpha=1, marker=r'*')
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")                                
        fname = 'cell-by-cell-Ang_e%02d'%elon + '_r%02d.svg'%run
        plt.savefig(fname)
        plt.close
        
        
        fig = plt.figure()        
        edd = np.arange(2*M)
        plt.plot(edd, (Ab_0 + Bb_0)/np.mean((Ab_0 + Bb_0)), 'r', lw=2)
        plt.plot(edd, (Ab + Bb)/np.mean((Ab + Bb)), 'b', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'AbPlusBb_final.svg'
        plt.savefig(fname)
        plt.close()
        
        fig = plt.figure()        
        edd = np.arange(2*M)
        plt.plot(edd, (Ab_0 - Bb_0)/np.mean((Ab_0 + Bb_0)/2), 'r', lw=2)
        plt.plot(edd, (Ab - Bb)/np.mean((Ab + Bb)/2), 'b', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'AbMinusBb_final.svg'
        plt.savefig(fname)
        plt.close()
        
        DDx = PxA - PxB ; SSx = PxA + PxB;
        DDy = PyA - PyB ; SSy = PyA + PyB
        
        DD_final = np.sqrt(DDx**2 + DDy**2)
        SS_final = np.sqrt(SSx**2 + SSy**2)
        QQ_final = np.mean(DD_final)

        fig = plt.figure()        
        cell = np.arange(len(regions))
        plt.plot(cell, DD_init/QQ_init, 'm', lw=2)
        plt.plot(cell, DD_final/QQ_final, 'c', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'DD_final.svg'
        plt.savefig(fname)
        plt.close()
        
        fig = plt.figure()        
        cell = np.arange(len(regions))
        plt.plot(cell, SS_init/QQ_init, 'r', lw=2)
        plt.plot(cell, SS_final/QQ_final, 'b', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'SS_final.svg'
        plt.savefig(fname)
        plt.close()
        

        fig = plt.figure()        
        cell = np.arange(len(regions))
        plt.plot(cell, SS_init/DD_init, 'r', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'SSi_over_DDi.svg'
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()        
        cell = np.arange(len(regions))
        plt.plot(cell, SS_final/DD_final, 'b', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'SSf_over_DDf.svg'
        plt.savefig(fname)
        plt.close()
        
        
        fig = plt.figure()        
        cell = np.arange(len(regions))
        plt.plot(cell, DD_init/DD_final, 'b', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'DDi_over_DDf.svg'
        plt.savefig(fname)
        plt.close()        

        fig = plt.figure()        
        cell = np.arange(len(regions))
        plt.plot(cell, SS_init/SS_final, 'r', lw=2)
        ax = fig.add_subplot(111)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        fname = 'SSi_over_SSf.svg'
        plt.savefig(fname)
        plt.close()        

        

#        os.chdir(owdE)
                
    fig = plt.figure()
    for ratio in range(len(Elon)):
        plt.plot(t, Ang_p[:, ratio], lw=2)    
    ax = fig.add_subplot(111)
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")    
    fname = 'Ang_p.svg'
    plt.savefig(fname)
    plt.close()    

El = np.sqrt(Elon) - 1

fig = plt.figure()
plt.plot(El, Ang_p[-1,:], 'k', lw=2)
ax = fig.add_subplot(111)
for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")    
    fname = 'AngVsElon'
plt.savefig(fname)
plt.close()

os.chdir(owdFull)
        
print("--- %s seconds ---" % (time.time() - start_time))

duration = 1  # second
freq = 440  # Hz
os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))




