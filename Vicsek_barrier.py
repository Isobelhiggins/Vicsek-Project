import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
 
# parameters
L = 32.0 # size of box
rho = 3.0 # density
N = int(rho*L**2) # number of particles
print("N", N) 
 
r0 = 1.0 # interaction radius
deltat = 1.0 # time step
factor = 0.5
v0 = r0/deltat*factor # velocity
iterations = 200 # animation frames
eta = 0.15 # noise
 
# initialise positions and angles
positions = np.random.uniform(0,L,size=(N,2))
angles = np.random.uniform(-np.pi, np.pi,size=N)

# define barrier position and size
barrier_x_start, barrier_x_end = 10, 20
barrier_y_start, barrier_y_end = 10, 20

# ensure particles are not generated inside the barrier
positions = np.zeros((N, 2))
for i in range(N):
    position = np.random.uniform(0, L, size = 2)
    while barrier_x_start <= position[0] <= barrier_x_end and barrier_y_start <= position[1] <= barrier_y_end:
        position = np.random.uniform(0, L, size = 2)
    positions[i] = position

def barrier_collision(position, angle):
    # calculate next position based on current position and angle
    next_position = position + (v0 * np.column_stack((np.cos(angle), np.sin(angle))) * deltat)
    
    # check if next position is at the barrier
    hit_barrier_x = (barrier_x_start <= next_position[:,0]) & (next_position[:,0] <= barrier_x_end)
    hit_barrier_y = (barrier_y_start <= next_position[:,1]) & (next_position[:,1] <= barrier_y_end)
    hit_barrier = hit_barrier_x & hit_barrier_y
    
    # return current position if hit barrier, otherwise continue updating position
    next_position[hit_barrier] = position[hit_barrier]
    
    return next_position
 
def animate(i):
    print(i)
 
    global positions, angles
    tree = cKDTree(positions,boxsize=[L,L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0,output_type='coo_matrix')
 
    #important 3 lines: we evaluate a quantity for every column j
    data = np.exp(angles[dist.col]*1j)
    # construct  a new sparse marix with entries in the same places ij of the dist matrix
    neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())
    # and sum along the columns (sum over j)
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
     
     
    angles = np.angle(S)+eta*np.random.uniform(-np.pi, np.pi, size=N)
    
    # update position with barrier collision check
    positions = barrier_collision(positions, angles)
 
    # boundary conditions of box
    positions[positions > L] -= L
    positions[positions < 0] += L
 
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(angles), np.sin(angles), angles)
    return qv,

fig, ax= plt.subplots(figsize=(6,6))
 
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), clim=[-np.pi, np.pi], cmap="hsv")
ax.add_patch(plt.Rectangle((barrier_x_start, barrier_y_start), barrier_x_end - barrier_x_start, barrier_y_end - barrier_y_start, color = "grey", alpha = 0.5)) 
ax.set_title(f"Vicsek model for {N} particles with an attrative barrier, using cKDTree")
anim = FuncAnimation(fig,animate,np.arange(1, iterations),interval=5, blit=True)
writer = FFMpegWriter(fps = 10, metadata = dict(artist = "Isobel"), bitrate = 1800)
anim.save("Vicsek_barrier.mp4", writer = writer, dpi = 100)
plt.show()