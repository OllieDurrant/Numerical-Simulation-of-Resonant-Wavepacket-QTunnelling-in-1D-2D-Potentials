# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 17:15:43 2025

@author: ollie
"""

#FIRST STEP
#V = 0 
#in natural units, hbar = m = 1
'''
d/dt * phi(t) = i/2 * (M.phi(t))

M is the differential operator, being a tridiagonal matrix

'''
#%% import and define functions
import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from scipy.sparse import kron, identity 
import matplotlib.colors as colors


def make_potential(x, kind, **kwargs):
    """
    Returns a diagonal sparse potential matrix V(x)
    for different potential types.

    Parameters: 
    x : array of spatial grid points.
    kind : str of type of potential ('barrier', 'well', 'harmonic', 'double_well', etc).
    kwargs : Extra parameters dependent on type of potential (height, width, position, etc.)

    Returns:
    Vx : 1D array of otential values at each grid point (for plotting).
    V_diag : scipy.sparse.csr_matrix diagonal sparse potential operator for use in the Hamiltonian.
    """
    
    Vx = np.zeros_like(x)

    if kind == "barrier":
        # rectangular barrier, defaults
        height = kwargs.get("height", np.pi)
        width = kwargs.get("width",1)
        x1 = kwargs.get("x1", 0)
        x2 = kwargs.get("x2", x1+width)
        Vx[(x > x1) & (x < x2)] = height

    elif kind == "well":
        # square well (negative potential)
        width = kwargs.get("width",1)        
        x1 = kwargs.get("x1", 0)
        x2 = kwargs.get("x2", x1+width)
        depth = kwargs.get("depth", -5)
        Vx[(x > x1) & (x < x2)] = depth
    
    elif kind == "double_barrier":
        #2 rectangular barriers
        width = kwargs.get('width',1) 
        x1 = kwargs.get("x1", -1.5)
        x2 = kwargs.get("x2", x1+width)
        gap = kwargs.get("gap", 1)
        x3 = kwargs.get("x3", x2 + gap)
        x4 = kwargs.get("x4", x3 + width) 
        height = kwargs.get("height", 5)
        Vx[(x > x1) & (x < x2)] = height
        Vx[(x > x3) & (x < x4)] = height    
    
    elif kind == 'triple_barrier':
        #3 rectangular barriers
        width = kwargs.get('width',1) 
        x1 = kwargs.get("x1", -1.5)
        x2 = kwargs.get("x2", x1+width)
        gap = kwargs.get("gap", 1)
        x3 = kwargs.get("x3", x2 + gap)
        x4 = kwargs.get("x4", x3 + width) 
        x5 = kwargs.get("x5", x4 + gap)
        x6 = kwargs.get("x6", x5 + width)
        height = kwargs.get("height", 5)
        Vx[(x > x1) & (x < x2)] = height
        Vx[(x > x3) & (x < x4)] = height    
        Vx[(x > x5) & (x < x6)] = height    
        
        
    elif kind == "harmonic": #have to zoom in or not plot all of Vx to see what happens
        k = kwargs.get("k", k0)  # spring constant
        Vx = 0.5 * k * x**2

    elif kind == "double_well":
        A= kwargs.get("a", a)
        Vx = 0.02 * (x**2 - A**2)**2  # symmetric double well

    elif kind == "none":
        Vx[:] = 0

    else:
        raise ValueError(f"Unknown potential kind: {kind}")

    # convert to sparse diagonal matrix
    V = diags(Vx, 0, format="csr")
    return Vx, V

def choose_initial(x,y,kind,a_x, a_y, sigma,k0,dx):
    ''''a function to choose which initial 2d psi to use, either directional or radially symmetric
        radial is momentum with 0 direction, directions are for moving each direction. Returns normalised wavefunction'''
    G = 1/((2*np.pi)**0.25 * np.sqrt(sigma))
    X, Y = np.meshgrid(x, y, indexing='xy')
    R = np.sqrt(X**2 + Y**2)
    if kind == 'radial':
        psi = G * np.exp(-(R**2)/(4*sigma**2)) * (np.exp(1j*k0*R) + np.exp(-1j*k0*R))
        psi_init_2D = psi
    elif kind == 'cartesian':
        psi_x = G * np.exp(-(x-a)**2 /(4*sigma**2)) * (np.exp(1j*k0*x) + np.exp(-1j*k0*x))
        psi_y = G * np.exp(-(y-a)**2 /(4*sigma**2)) * (np.exp(1j*k0*y) + np.exp(-1j*k0*y))
        psi_init_2D = psi_x[:, None] * psi_y[None, :]  
    
    elif kind == 'east':
        psi_x = G * np.exp(-(x-a_x)**2 /(4*sigma**2)) 
        psi_y = G * np.exp(-(y-a_y)**2 /(4*sigma**2)) * (np.exp(1j*k0*y)) #my x and y axes must be switched somewhere
        psi_init_2D = psi_x[:, None] * psi_y[None, :] 

    elif kind == 'northeast':
        psi_x = G * np.exp(-(x-a_x)**2 /(4*sigma**2)) * np.exp(1j*k0*x)
        psi_y = G * np.exp(-(y-a_y)**2 /(4*sigma**2)) * np.exp(1j*k0*y)
        psi_init_2D = psi_x[:, None] * psi_y[None, :] 
    
    elif kind == 'southwest':
        psi_x = G * np.sqrt(sigma) * np.exp(-(x-a_x)**2 /(4*sigma**2)) * np.exp(1j*-k0*x)
        psi_y = G * np.sqrt(sigma) * np.exp(-(y-a_y)**2 /(4*sigma**2)) * np.exp(1j*-k0*y)
        psi_init_2D = psi_x[:, None] * psi_y[None, :] 
    
    else:
        raise ValueError("Unknown potential type")
    
    #normalisation
    prob_density = np.abs(psi_init_2D)**2
    norm = np.sqrt(np.sum(prob_density) * dx**2)    
    psi_init_2D /= norm
    #turn into vector 
    psi_init_flat = psi_init_2D.ravel()            # shape (N*N = N**2)
    return psi_init_flat


def make_potential2D(x, y, kind="none", **kwargs):
    '''same as the function for 1d but now takes x and y inputs'''
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    Vxy = np.zeros_like(X)   # 2D potential grid
    R2 = X**2 + Y**2
    if kind == "none":
        pass
    
    elif kind == 'square_barrier':
        width = kwargs.get('width', 1)
        d = kwargs.get('d', 10)
        height = kwargs.get('height', 5)
        
        d2 = width + d
        
        inner = (np.abs(X) < d) & (np.abs(Y) < d)
        outer = (np.abs(X) < d2) & (np.abs(Y) < d2)

        mask = outer & (~inner)
    
        Vxy[mask] = height
        
        
    elif kind == "circular_barrier":
        r1 = kwargs.get("r1",  4)
        r2 = kwargs.get("r2",  5)
        height = kwargs.get("height", 5)
            
        mask = (R2 <= r2**2) & (R2 >= r1**2)
        Vxy[mask] = height

    elif kind == "well":
        x1 = kwargs.get("x1", -1)
        x2 = kwargs.get("x2",  1)
        depth = kwargs.get("depth", -5)

        mask = (X > x1) & (X < x2)
        Vxy[mask] = depth

    elif kind == 'double_circular_barrier':
        r1 = kwargs.get('r1',5)
        width = kwargs.get('width',1)
        gap = kwargs.get('gap',1)
        height = kwargs.get('height', 5)
        
        r2 = r1 + width 
        r3 = r2 + gap
        r4 = r3 + width 

        mask = ((R2 <= r2**2) & (R2 >= r1**2)) | ((R2 <= r4**2) & (R2 >= r3**2))
        Vxy[mask] = height
    
    elif kind == 'double_square_barrier':
        d1 = kwargs.get('d', 7)
        width = kwargs.get('width', 1)
        gap = kwargs.get('gap', 1)
        height = kwargs.get('height', 5)
        
        d2 = d1 + width
        d3 = d1 + width + gap
        d4 = d1 + gap + width*2

        inner_1 = (np.abs(X) < d1) & (np.abs(Y) < d1)
        outer_1 = (np.abs(X) < d2) & (np.abs(Y) < d2)

            
        inner_2 = (np.abs(X) < d3) & (np.abs(Y) < d3)
        outer_2 = (np.abs(X) < d4) & (np.abs(Y) < d4)

        mask_1 = outer_1 & (~inner_1)
        mask_2 = outer_2 & (~inner_2)
    
        mask = mask_1 | mask_2
        Vxy[mask] = height
        
    elif kind == "harmonic":
        k = kwargs.get("k", 0.05)
        Vxy = 0.5 * k * (X**2 + Y**2)

    else:
        raise ValueError("Unknown potential type")

    # flatten and build operator
    V2D = diags(Vxy.ravel(), 0, format="csr")
    return Vxy, V2D

def build_absorber(x, w, power):
    '''Builds a 1d absorber profile for 0 at interior and 1 at exterior, w is the width and power is the strength'''
    # distance to nearest x-boundary
    left = x - x[0]
    right = x[-1] - x
    d = np.minimum(left, right)

    # fraction s goes from 0 (interior) to 1 (at boundary)
    s = np.clip((w-d)/w,0,1)
    A = s**power
    return A

def Tcoeff(E,V0, a):
    '''Analytically finds the transmission constant for a 1D barrier of height V0 and width a'''
   
    if E == V0: #avoid division by 0
        E -= 1e-10

    if E < V0: 
        k2 = np.sqrt(2*(V0-E))
        pref = (V0**2) / (4 * E * (V0 - E))
        T = 1/(1 + pref * np.sinh(k2*a)**2)
    elif E > V0:
        k2 = np.sqrt(2*(E-V0))
        pref = (V0**2) / (4 * E * (E - V0))
        T = 1/(1 + pref * np.sin(k2*a)**2)    
    return T
  
def T_double_coeff(E,V0, a, b):
    '''Analytically finds the transmission constant for a 1D double barrier, where E is energy of particle
    V0 is height of the barriers, a is the width of the barriers and b is the gap between them. FOR E<<V0'''

    if E == V0:
        E -= 1e-10
        
    k = np.sqrt(2*E)
    k2 = np.sqrt(2*(V0-E) + 0j) #0j allows complex values
  
    A_E = (V0*np.sinh(k2*a)**2)/(4*E*(V0-E))
    B_E = np.sin(k*b)**2 / np.sinh(k2*a)**2
  
    T = 1/(1 + A_E * (1 + B_E))
    return np.real_if_close(T)

def TDSE(t,phi):
    return -1j * (H @ phi) #need to have t as solve_ivp requires 2 arguments

#%% 1a) initial


#define constants 
N = 1001 
x = np.linspace(-25,25,N)
dx = x[1] - x[0]
a = -10
sigma = 2
k0 = 2

#define psi at t=0 
psi_x = 1/((2*np.pi)**(0.25) * np.sqrt(sigma)) * np.exp(-(x-a)**2 /(4*sigma**2)) * np.exp(1j*k0*x)


# Define diagonals
main_diag = -2 * np.ones(N)
off_diag  = 1 * np.ones(N-1)

# Build sparse tridiagonal matrix, might as well start using sparse now 
M = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr") / dx**2
H = -1/2 * M + 0


#now create initial conditions and integration conditions 
t_span = (0,10) 
N_t = 3001
t_eval = np.linspace(t_span[0],t_span[1], N_t)

#create solution using solve_ivp
sol= solve_ivp(TDSE, t_span, psi_x, t_eval = t_eval, rtol = 1e-8, atol = 1e-8)

#plot the final waveform
psi_final = sol.y[:, -1]
plt.plot(x, np.abs(psi_final)**2, color='tab:red',lw = 2.5, label = 'numerical wave packet @ 10s')
plt.plot(x,np.abs(psi_x)**2, color = 'tab:blue', lw=2.5, label = 'initial numerical wave packet')

#analytic solution 
v = k0
delta_t = t_span[1]/(2*sigma**2)
psi_ati = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-a)**2/(2*sigma**2)) #initial analytical psi
psi_atf = 1/(sigma*np.sqrt(2*np.pi)*np.sqrt(1+delta_t**2)) * np.exp(-(x-a-v*t_span[1])**2/(2*sigma**2*(1+delta_t**2))) #final analytic psi

plt.plot(x,psi_ati,'-.' ,c = 'black', lw=1.5, label = 'initial analytical wavepacket')
plt.plot(x,psi_atf,'--',c = 'black', lw=1.5, label = 'analytical wavepacket @ 10s')
plt.xlabel('$x$')
plt.ylabel('Probability density $|\phi(x,t)|^2$')
plt.legend()
plt.show()

#%% 1b) little video plot showing how this wavepacket moves through time for the above
plt.ion()   # interactive mode on
fig, ax = plt.subplots()

line1, = ax.plot(x, np.abs(psi_x)**2, 'b', label='initial')
line2, = ax.plot(x, np.abs(sol.y[:, 0])**2, 'r', label='evolving')
ax.legend()
ax.set_xlabel('$x$')
ax.set_ylabel('$|\phi(x,t)|^2$')


for i in range(0,int(N_t/3)):   #not for all of N_t to speed it up a little  
    line2.set_ydata(np.abs(sol.y[:, i*3])**2)
    ax.set_title(f't = {sol.t[i*3]:.2f}s')
    plt.pause(0.01)

plt.ioff()
plt.show()

#%% 2a) time to introduce a potential 

#define constants 
N = 1001 
x = np.linspace(-25,25,N)
dx = x[1] - x[0]
a = -10
sigma = 2
k0 = 2

#define psi at t=0 
psi_x = 1/((2*np.pi)**(0.25) * np.sqrt(sigma)) * np.exp(-(x-a)**2 /(4*sigma**2)) * np.exp(1j*k0*x)


# Define diagonals
main_diag = -2 * np.ones(N)
off_diag  = 1 * np.ones(N-1)

# Build sparse tridiagonal matrix
M = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr") / dx**2

#now choose potential parameters 
start = 0
width = 1
gap = 2
end = start + width
height = 2
kind = 'barrier'

Vx, V = make_potential(x, kind, height = height, x1 = start, x2 = end)
#Vx, V = make_potential(x, "double_barrier", height = height, x1 = start, width = width, gap = gap)

#now build absorber 
cap_width = 5 #width from edge
cap_strength = k0 
power = 3 #order magnitude of absorber
Ax = build_absorber(x, cap_width, power)

# CAP function (negative imaginary potential)
CAP = -1j * cap_strength * Ax
CAPop = diags(CAP, 0, format='csr')

#and create H: 
H = -1/2 * M  + V + CAPop

#now integration conditions 
t_span = (0,12) 
N_t = 3001
t_eval = np.linspace(t_span[0],t_span[1], N_t)

#create solution using solve_ivp
sol= solve_ivp(TDSE, t_span, psi_x, t_eval = t_eval, rtol = 1e-8, atol = 1e-8)
psi_final = sol.y[:, -1]

plt.ion()   # interactive mode on
fig, ax = plt.subplots()

#line1, = ax.plot(x, np.abs(psi_x)**2, 'b', label='initial') #only used if want to compare to original 
line2, = ax.plot(x, np.abs(sol.y[:, 0])**2, 'r', label='evolving')
ax.plot(x,Vx/4) #potential can be scaled to fit nicer onto graph, or not plotted (in harmonic case) 
ax.legend()
ax.set_xlabel('$x$')
ax.set_ylabel('|\phi(x,t)|^2$')


for i in range(0,int(N_t/5)):    
    line2.set_ydata(np.abs(sol.y[:, i*5])**2)
    ax.set_title(f't = {sol.t[i*5]:.2f}s')
    plt.pause(0.01)

plt.ioff()
plt.show()

#transmission and reflection and test
prob_all = np.trapz(np.abs(psi_final)**2,x)
mask_trans = x > end       # region beyond the barrier
mask_refl = x < start 
mask_betw = (x > (start + width)) & (x < (end-width)) #only useful in double barrier cases
T = np.trapz(np.abs(psi_final[mask_trans])**2, x[mask_trans])
R = np.trapz(np.abs(psi_final[mask_refl])**2, x[mask_refl])
Betw = np.trapz(np.abs(psi_final[mask_betw])**2, x[mask_betw])

print(f'final probability is: {prob_all:.3f}') 
print(f"Transmission probability: {T:.3f}")
print(f"Reflection probability: {R:.3f}")
print(f"Between probability: {Betw:.3f}")

#analytical value ONLY FOR SINGLE BARRIER OTHERWISE WILL BE WRONG
E = k0**2/2 + 1/(8*sigma**2)
T_anal = Tcoeff(E,height,width)
print(f'Analytical transmission coefficient is: {T_anal:.4f}')

#%%
frames = [0, int(N_t/2), N_t-1]

for idx, f in enumerate(frames):
    psi2 = np.abs(sol.y[:, f])**2
    np.savetxt(f"frame_{idx+1}.txt", np.column_stack((x, psi2, Vx)))
#%% 2b) analyse of transmission probability vs height 

import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
'''
analytical equation for transmission coeffecicent T for a barrier of height V_0 and width a 

T = (1 + 1/4 * (V_0^2/E(E(V_0-E)))^2  * sinh^2(k_2*a))^(-1)

where k_2 = sqrt(2mE/hbar^2)

'''
#define constants 
N = 501
x = np.linspace(-30,30,N)
dx = x[1] - x[0]
a = -10
sigma = 2
k0 = 2 

E = k0**2/2 + 1/(8*sigma**2) #small offset due to the contribution of the spread of momentas


#define psi at t=0 
psi_x = 1/((2*np.pi)**(0.25) * np.sqrt(sigma)) * np.exp(-(x-a)**2 /(4*sigma**2)) * np.exp(1j*k0*x)

# Define diagonals
main_diag = -2 * np.ones(N)
off_diag  = 1 * np.ones(N-1)

# Build sparse tridiagonal matrix, m 
M = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr") / dx**2

#now build absorber 
cap_width = 5 #width from edge
cap_strength = k0 #eta parameter
power = 3 #strength of absorber
Ax = build_absorber(x, cap_width, power)

# CAP function (negative imaginary potential)
CAP = -1j * cap_strength * Ax
CAPop = diags(CAP, 0, format='csr')


N_t = 3001
t_span = (0,15) 
t_eval = np.linspace(t_span[0],t_span[1], N_t)

start = 0
end = 1 #start and end of barrier
width = end - start

T_height = np.array([])
anal_height = np.array([])
P_tot = np.array([])

Vx = np.zeros_like(x)
V_sparse = diags([np.ones_like(x)], [0], format="csr")  # ensures non-empty .data

anal_height = np.array([])

range_height = np.linspace(2,10,50)
#first for height
for i in range_height:
    # update only the numerical values of Vx
    Vx[:] = 0
    Vx[(x > start) & (x < end)] = i
    V_sparse.data[:] = Vx  # update the diagonal entries, same sparse pattern
    
    # construct Hamiltonian using the updated potential
    H = -0.5 * M + V_sparse + CAPop

    #solve for each height
    sol = solve_ivp(TDSE, t_span, psi_x, t_eval = t_eval, rtol = 1e-8, atol = 1e-8)
    psi_final = sol.y[:, -1]
    
    #calculate transmission and reflection and append them
    mask_trans = x > end       # region beyond the barrier
    mask_refl = x < start 
    P = np.trapz(np.abs(psi_final)**2,x)
    T = np.trapz(np.abs(psi_final[mask_trans])**2, x[mask_trans])
    
    P_tot = np.append(P_tot, P)
    T_height = np.append(T_height, T)
    
    
    T_anal = Tcoeff(E, i, width)
    anal_height = np.append(anal_height, T_anal)
    print('height', i)


ratio = E/(range_height)

#plt.plot(ratio, P_tot, label = 'Total prob')
plt.plot(ratio, T_height/P_tot,'-r', label = 'changing height')
plt.plot(ratio, anal_height, '--k',markersize = 3, label = 'analytical height for plane wave')
plt.show()
plt.xlabel('E/V0')
plt.ylabel('Probability of Transmission of particle')
plt.legend()
#%%
data = np.vstack([ratio, T_height/P_tot, anal_height]).T
np.savetxt("changing_height.csv", data,
           delimiter=",")
#%% 2c) same thing, ran straight after so keeping variables


#then for width
height = 3
range_width = np.arange(0, 4, dx+0.001) #difference in values must be larger than dx 
deltaW = range_width[1] - range_width[0]


T_width = np.array([])
anal_width = np.array([])

for i in range_width:
    # update only the numerical values of Vx
    Vx[:] = 0
    Vx[(x > 0) & (x < i)] = height
    V_sparse.data[:] = Vx  # update the diagonal entries, same sparse pattern
    
    # construct Hamiltonian using the updated potential
    H = -0.5 * M + V_sparse

    #solve for each height
    sol = solve_ivp(TDSE, t_span, psi_x, t_eval = t_eval, rtol = 1e-8, atol = 1e-8)
    psi_final = sol.y[:, -1]
    
    #calculate transmission and reflection and append them
    mask_trans = x > i       # region beyond the barrier
    T = np.trapz(np.abs(psi_final[mask_trans])**2, x[mask_trans])
    T_width = np.append(T_width, T)
        
    T_anal = Tcoeff(E, height, i)
    anal_width = np.append(anal_width, T_anal)
    print('Height', i)




plt.plot(range_width, T_width, '-r', label = 'changing width')
plt.plot(range_width, anal_width, '--k',markersize = 3, label = 'analytical width')

plt.show()
plt.xlabel('Width of barriers')
plt.ylabel('Probability of Transmission of particle')
plt.legend()
#%%
data = np.vstack([range_width, T_width, anal_width]).T
np.savetxt("changing_width.csv", data,
           delimiter=",")

#%% 3a) Now a look at resonant tunnelling, first for the 1 barrier case 

#define constants 
N = 2001
x = np.linspace(-60,130,N)
dx = x[1] - x[0]
a = -20
sigma = 10
 
# Define diagonals
main_diag = -2 * np.ones(N)
off_diag  = 1 * np.ones(N-1)

# Build sparse tridiagonal matrix, might as well start using sparse now 
M = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr") / dx**2


#now choose potential 
start = 30
width = 3
end = start + width
height = 20

Vx, V = make_potential(x, "barrier", height = height, x1 = start, width = width)

#now build absorber     
cap_width = 5 #width from edge
cap_strength = 3 #eta parameter, similar to k 
power = 3 #strength of absorber
Ax = build_absorber(x, cap_width, power)

# CAP function (negative imaginary potential)
CAP = -1j * cap_strength * Ax
CAPop = diags(CAP, 0, format='csr')


H = -1/2 * M  + V + CAPop


prob_trans = np.array([])
prob_total = np.array([])
prob_refl = np.array([])
prob_betw = np.array([])

mask_trans = x > end       # region beyond the barrier
mask_refl = x < start
mask_betw = (x > start) & (x < end)

E_range = np.linspace(height, height*3, 20)
k_range = np.sqrt(2*E_range - 1/(4*sigma**2)) #should make it linear in E not k, as E is proportional to k**2
anal_T = []

for i, k in enumerate(k_range):
    #change t boundaries 
    t_span = (0,130/k)
    t_eval = np.linspace(*t_span, 3000)

    #solve for each k value 
    psi_x = 1/((2*np.pi)**(0.25) * np.sqrt(sigma)) * np.exp(-(x-a)**2 /(4*sigma**2)) * np.exp(1j*k*x)
    sol = solve_ivp(TDSE, t_span, psi_x, t_eval = t_eval, rtol = 1e-8, atol = 1e-8)
    psi_final = sol.y[:, -1]
    
    #calculate transmission and reflection and append them
    prob_all = np.trapz(np.abs(psi_final)**2,x)
    prob_T = np.trapz(np.abs(psi_final[mask_trans])**2, x[mask_trans])
    prob_F = np.trapz(np.abs(psi_final[mask_refl])**2, x[mask_refl])
    prob_B = np.trapz(np.abs(psi_final[mask_betw])**2, x[mask_betw])
    
    
    prob_trans = np.append(prob_trans, prob_T)
    prob_refl = np.append(prob_refl,prob_F)
    prob_betw = np.append(prob_betw, prob_B)
    prob_total = np.append(prob_total,prob_all)
    
    #calc analytical 
    T_ana = Tcoeff(E_range[i], height, width)
    anal_T = np.append(anal_T, T_ana)
    print(i)




ratio = E_range/height
plt.figure(figsize = (12,10))
plt.plot(ratio, anal_T, 'k*',label = 'Analytical transmission')
plt.plot(ratio, prob_trans,label = 'transmission')
plt.plot(ratio, prob_refl,label = 'Reflection')
plt.plot(ratio, prob_betw, label = 'wavefunction stuck inbetween')
plt.plot(ratio, prob_total,label = 'total')
plt.legend()
plt.xlabel('E/V_0 value')
plt.ylabel('Probability values')
plt.show()
#%%
data = np.vstack([ratio, prob_trans, prob_betw, prob_total, anal_T]).T
np.savetxt("1D_1barrier_resonant_data_with_anal1-3.csv", data,
           delimiter=",")
#%% 3b) second for the double barrier case  (triple actually wooo)
 
#define constants 
N = 2001
x = np.linspace(-30,90,N)
dx = x[1] - x[0]
a = -15
sigma = 5
 
# Define diagonals
main_diag = -2 * np.ones(N)
off_diag  = 1 * np.ones(N-1)

# Build sparse tridiagonal matrix, might as well start using sparse now 
M = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr") / dx**2


#now choose potential 
start = 15
width = 0.2
gap = 5
end = start + gap*2 + width*3 #for triple or double whatever
height = 50

Vx, V = make_potential(x, "double_barrier", height = height, x1 = start, width = width, gap = gap)

#now build absorber     
cap_width = 5 #width from edge
cap_strength = 3 #eta parameter, similar to k 
power = 3 #strength of absorber
Ax = build_absorber(x, cap_width, power)

# CAP function (negative imaginary potential)
CAP = -1j * cap_strength * Ax
CAPop = diags(CAP, 0, format='csr')


H = -1/2 * M  + V + CAPop


prob_trans = np.array([])
prob_total = np.array([])
prob_refl = np.array([])
prob_betw = np.array([])

mask_trans = x > end       # region beyond the barrier
mask_refl = x < start
mask_betw = (x > start) & (x < end)

E_range = np.linspace(1,10, 10)
k_range = np.sqrt(2*E_range - 1/(4*sigma**2)) #should make it linear in k not E, as E is proportional to k**2
anal_T = []

for i, k in enumerate(k_range):
    #change t boundaries 
    t_span = (0,60/k)
    t_eval = np.linspace(*t_span, 3000)

    #solve for each k value 
    psi_x = 1/((2*np.pi)**(0.25) * np.sqrt(sigma)) * np.exp(-(x-a)**2 /(4*sigma**2)) * np.exp(1j*k*x)
    sol = solve_ivp(TDSE, t_span, psi_x, t_eval = t_eval, rtol = 1e-8, atol = 1e-8)
    psi_final = sol.y[:, -1]
    
    #calculate transmission and reflection and append them
    prob_all = np.trapz(np.abs(psi_final)**2,x)
    prob_T = np.trapz(np.abs(psi_final[mask_trans])**2, x[mask_trans])
    prob_F = np.trapz(np.abs(psi_final[mask_refl])**2, x[mask_refl])
    prob_B = np.trapz(np.abs(psi_final[mask_betw])**2, x[mask_betw])
    
    prob_trans = np.append(prob_trans, prob_T)
    prob_refl = np.append(prob_refl,prob_F)
    prob_betw = np.append(prob_betw,prob_B)
    prob_total = np.append(prob_total,prob_all)
    
    print(i)


#%%

ratio = E_range/height
plt.figure(figsize = (12,10))
plt.plot(ratio, prob_trans,label = 'transmission')
plt.plot(ratio, prob_refl,label = 'Reflection')
plt.plot(ratio, prob_betw, label = 'Between')
plt.plot(ratio, prob_total,label = 'total')
plt.legend()
plt.xlabel('E/V_0 value')
plt.ylabel('Probability values')
plt.show()


#%%

data = np.vstack([ratio, prob_trans, prob_betw, prob_total]).T
np.savetxt("1D_2barrier_resonant_data1-10.csv", data,
           delimiter=",")




#%% 4a) now for 2d  

#need a 2d potential and hamilitonian 
#and initial psi, that is easy tho 
#define constants 
N = 201
x = np.linspace(-25,25,N)
y = np.linspace(-25,25,N)
X, Y = np.meshgrid(x, y, indexing='xy')
R = X**2 + Y**2
dx = x[1] - x[0] #do not need to a define a dy as it is the same
a_x = 8
a_y = 8
sigma = 2
k0 = 2

t_span = (0,30) 
N_t = 2001
t_eval = np.linspace(t_span[0],t_span[1], N_t)


psi_initial = choose_initial(x,y,'east',a_x, a_y, sigma,k0,dx)

#now for M, as Mx and My are the same so only need to define one
'''
M = M_x ⊗ I_y + I_x ⊗ M_y

they switch due to using row-major ordering in discretised grid 
'''

# Hamiltonian construction
main_diag = -2 * np.ones(N)
off_diag = np.ones(N-1)
Mx = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr") / dx**2
I_x = identity(N, dtype=float).tocsr()
#only defined for x, as they are the same for y 

M = kron(Mx,I_x) + kron(I_x,Mx)

#now build absorber 
cap_width = 5 #width from edge
cap_strength = k0 #eta parameter
power = 3 #strength of absorber
Ax = build_absorber(x, cap_width, power)
Ay = build_absorber(y, cap_width, power)
Absorb_profile = np.maximum.outer(Ax, Ay)   # 0 interior, 1 near box edges

# CAP function (negative imaginary potential)
CAP = -1j * cap_strength * Absorb_profile
CAPop = diags(CAP.ravel(), 0, format='csr')

#now for V, first the parameters used. I have included all the possible additional parameters below so easier to just switch the label
d = 5
r1 = 12
r2 = 13
height = 5
width = 0.5
gap = 3
k = 0.05
kind = 'harmonic'

Vxy, V2D = make_potential2D(x, y, kind, r1 = r1,r2=r2, width = width, gap = gap, height = height, d = d, k=k)

H = -0.5*M + V2D + CAPop #adding the absorber makes it non-hermitian

#use solver
sol = solve_ivp(TDSE, t_span, psi_initial,t_eval=t_eval, rtol=1e-8, atol=1e-8)



plt.ion()
fig, ax = plt.subplots()

# Initial frame
psi0_2D = sol.y[:, 0].reshape(N,N)
im = ax.imshow(np.abs(psi0_2D)**2,
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin="lower",
               cmap="inferno",
               #norm=colors.LogNorm(vmin=1e-8,vmax=np.max(np.abs(psi0_2D)**2)) #toggle comment if dont want log scale
               )

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("t = 0")

#add on potential 
# circle_1 = patches.Circle((0,0), r1,fill = False, edgecolor='white', linewidth = 2)
# ax.add_patch(circle_1)



cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Probability density $|\psi(x,y)|^2$')

ax.contour(X, Y, Vxy, levels=10, colors="white", alpha=0.4, linewidths=0.7)

plt.show()

# Classical trajectory
omega = np.sqrt(k)

x_class = a_x * np.cos(omega * t_eval) + k0/omega * np.sin(omega * t_eval)
y_class = a_x * np.cos(omega * t_eval) + 0/omega * np.sin(omega * t_eval)
# Classical point marker
(classical_line,) = ax.plot([], [], 'rx', markersize=4)

# Animation
for i in range(0, int(N_t/5)):
    psi_frame = sol.y[:, i*5].reshape(N, N)
    im.set_data(np.abs(psi_frame)**2)
    # update classical trajectory position
    classical_line.set_data([x_class[i*5]], [y_class[i*5]])
    ax.set_title(f"t = {sol.t[i*5]:.3f}")
    plt.pause(0.01)

plt.ioff()
plt.show() 


psi_final_2D = (sol.y[:,-1]).reshape(N, N)
prob_all = np.trapz(np.trapz(np.abs(psi_final_2D)**2, x, axis=0), y) #integrate over both x and y
print(f"Total probability: {prob_all:.4f}")

#%% following is only for saving frames of animations, not needed to be ran in evaluation of project
frames = [0, int(N_t/6), int(N_t-1)]

for idx, f in enumerate(frames):
    plt.ion()
    fig, ax = plt.subplots()

    circle_1 = patches.Circle((0,0), r1,fill = False, edgecolor='white', linewidth = 2)
    ax.add_patch(circle_1)

    psi = sol.y[:,f].reshape(N, N)
    im = ax.imshow(np.abs(psi)**2,
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin="lower",
               cmap="inferno",
               norm=colors.LogNorm(vmin=1e-8,vmax=np.max(np.abs(psi0_2D)**2))
               )
    

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability density $|\psi(x,y)|^2$')
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


    #Save figure
    plt.savefig(
        f"2Dcircular_{idx+1}.png",
        dpi=300,           # looks good on slides
        bbox_inches='tight'
    )
#%% 4b) for looking at the prob. stats of the single 2d case 
#now for transmission
if kind == 'circular_barrier':
    mask_refl = R < r2**2
    mask_between = ((R > r1**2) & (R < r2**2)).astype(float)


elif kind == 'double_circular_barrier':
    mask_refl = R < (r1)**2
    r2 = r1+width*2 + gap
    mask_between = ((R > r1**2) & (R < r2**2)).astype(float)

    
elif kind == 'double_square_barrier':
    d2 = d + width 
    d3 = d2 + gap
    d4 = d3 + width
    mask_refl = (np.abs(X) < d) & (np.abs(Y) < d)
    mask_between = (np.maximum(np.abs(X), np.abs(Y)) > d2) & (np.maximum(np.abs(X), np.abs(Y)) < d3)

elif kind == 'harmonic': #not made to look at stats for this case, not a helpful graph 
    mask_refl = True
    mask_between = True


plt.imshow(Absorb_profile, origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar(label='Absorb profile (0..1)')
plt.title('Absorber profile')
plt.show()



plt.figure()
plt.imshow(Vxy, origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar(label='Potential (0-x.max())')
plt.title('Potential used')
plt.show()

prob_in = np.zeros(sol.y.shape[1])
prob_total = np.zeros(sol.y.shape[1])
prob_between = np.zeros(sol.y.shape[1])

for i in range(sol.y.shape[1]):
    psi_frame = sol.y[:, i].reshape((y.size, x.size), order='C')
    abs2 = np.abs(psi_frame)**2
    prob_total[i] = np.sum(abs2) * dx**2 #dx = dy
    prob_in[i] = np.sum(abs2 * mask_refl) * dx**2
    prob_between[i] = np.sum(abs2 * mask_between) * dx**2

print("P_total initial, mid, final:", prob_total[0], prob_total[len(prob_total)//2], prob_total[-1])



#Plot results
plt.figure(figsize=(7,5))
plt.plot(sol.t, prob_in, label="Probability inside")
plt.plot(sol.t, prob_total, label="Total probability in box")
plt.plot(sol.t, prob_between, label='Probability between the barriers')
plt.xlabel("time")
plt.ylabel("Probability")
plt.legend()
plt.title("Probability evolution inside region & whole box")
plt.show()


#%% 4c) for looping over energies to have a look at resonant tunnelling

#initial values
a_x = 0
a_y = 0
r1 = 5
height = 20
gap = 3
d = 5
kind = 'double_square_barrier'
sigma = 2
N = 101
x = np.linspace(-10,10,N)
y = np.linspace(-10,10,N)
X, Y = np.meshgrid(x, y, indexing='xy')
R = X**2 + Y**2
dx = x[1] - x[0] #do not need to a define a dy as it is the same
width = dx*5 #scaled with dx 

# Hamiltonian construction
main_diag = -2 * np.ones(N)
off_diag = np.ones(N-1)
Mx = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr") / dx**2
I_x = identity(N, dtype=float).tocsr()
#only defined for x, as they are the same for y 

M = kron(Mx,I_x) + kron(I_x,Mx)

#now build absorber 
cap_width = 0.5 #width from edge
cap_strength = 5 #eta parameter
power = 3 #strength of absorber
Ax = build_absorber(x, cap_width, power)
Ay = build_absorber(y, cap_width, power)
Absorb_profile = np.maximum.outer(Ax, Ay)   # 0 interior, 1 near box edges

# CAP function (negative imaginary potential)

CAP = -1j * cap_strength * Absorb_profile
CAPop = diags(CAP.ravel(), 0, format='csr')

Vxy, V2D = make_potential2D(x, y, kind, r1 = r1,d=d, width = width, gap = gap, height = height)

H = -0.5*M + V2D + CAPop #adding the absorber makes it non-hermitian



if kind == 'circular_barrier':
    mask_refl = R < r2**2
    mask_between = ((R > r1**2) & (R < r2**2)).astype(float)


elif kind == 'double_circular_barrier':
    mask_refl = R < (r1)**2
    r2 = r1+width*2 + gap
    mask_between = ((R > r1**2) & (R < r2**2)).astype(float)

    
elif kind == 'double_square_barrier':
    d2 = d + width 
    d3 = d2 + gap
    d4 = d3 + width
    mask_refl = (np.abs(X) < d) & (np.abs(Y) < d)
    mask_between = (np.maximum(np.abs(X), np.abs(Y)) > d2) & (np.maximum(np.abs(X), np.abs(Y)) < d3)
   

E_range = np.linspace(1, 10, 20)
k_range = np.sqrt(2*E_range - 1/(4*sigma**2))

prob_in = np.zeros(k_range.size)
prob_total = np.zeros(k_range.size)
prob_between = np.zeros(k_range.size)


for i, k0 in enumerate(k_range):
    #first set up initial psi 
    psi_initial = choose_initial(x,y,'east',a_x, a_y, sigma,k0,dx)

    #adjust k values 
    t_span = (0,110/k0) #since width of box is 20 should allow for a ~5 oscillations     
    N_t = 2001
    t_eval = np.linspace(t_span[0],t_span[1], N_t)
    
    #use solver  
    sol = solve_ivp(TDSE, t_span, psi_initial,t_eval=t_eval, rtol=1e-8, atol=1e-8)
    
    psi_final = sol.y[:, -1].reshape((y.size, x.size), order='C')
    abs2 = np.abs(psi_final)**2
    prob_total[i] = np.sum(abs2) * dx**2 #dx = dy
    prob_in[i] = np.sum(abs2 * mask_refl) * dx**2
    prob_between[i] = np.sum(abs2 * mask_between) * dx**2

    print(i)



ratio = E_range/height
plt.figure(figsize=(10,8))
#plt.plot(ratio, prob_in,'-r', label = 'amplitude still in inner well')
plt.plot(ratio, prob_between,'-b' ,label = 'Amplitude left between barriers')
#plt.plot(ratio, prob_total,'-g', label = 'total probability')
plt.xlabel('E/V0')
plt.ylabel('Probability')
plt.legend()
plt.show()

#%%

data = np.vstack([k_range, ratio, prob_in, prob_between, prob_total]).T
np.savetxt("2Dcircular_east_resonant_data1-10,200height20.csv", data,
           delimiter=",")







