import numba
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# parameters
L = 10.0 # size of box
rho = 1.0 # density
N = int(rho * L**2) # number of particles
print("N", N)

r0 = 1.0 # interaction radius
deltat = 1.0 # time step
factor = 0.5
v0 = r0 / deltat * factor # velocity
iterations = 500 # animation frames
eta = 0.15 # noise/randomness
max_neighbours = N # maximum number of neighbours a particle might have

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) # from 0 to 2pi rad

# empty array for average angles of particles every 10 time steps
average_angles = np.empty(iterations // 10)

# histogram for average particle density in different areas of the box
bins = int(L / (r0/2))
hist, xedges, yedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, density = False)

# define barrier position and size
barrier_x_start, barrier_x_end = 3, 7
barrier_y_start, barrier_y_end = 3, 7

# ensure particles are not generated inside the barrier
positions = np.zeros((N, 2))
for i in range(N):
    position = np.random.uniform(0, L, size = 2)
    while barrier_x_start <= position[0] <= barrier_x_end and barrier_y_start <= position[1] <= barrier_y_end:
        position = np.random.uniform(0, L, size = 2)
    positions[i] = position

@numba.njit()
def barrier_collision(position, angle):
    
    while True:
        # calculate next position based on current position and angle
        next_position = position + v0 * np.array([np.cos(angle), np.sin(angle)]) * deltat
    
        # check if next position is at the barrier
        # if particle is at barrier, make it bounce backwards a bit
        if barrier_x_start <= next_position[0] <= barrier_x_end and barrier_y_start <= next_position[1] <= barrier_y_end:
            position = position - (5.0 * v0 * np.array([np.cos(angle), np.sin(angle)]) * deltat)
        else:
            return next_position # continue updating position if not at barrier

@numba.njit(parallel = True)
def update(positions, angles):
    
    # empty arrays to hold updated positions and angles
    new_positions = np.empty_like(positions)
    new_angles = np.empty_like(angles)
    neighbour_angles = np.empty(max_neighbours)
    
    # loop over all particles
    for i in numba.prange(N):
        count_neighbour = 0
        # distance to other particles
        for j in range(N):
            distance = np.linalg.norm(positions[i] - positions[j])
            # if within interaction radius add angle to list
            if (distance < r0) & (distance != 0):
                neighbour_angles[count_neighbour] = angles[j]
                count_neighbour += 1
         
        # if there are neighbours, calculate average angle and noise/randomness       
        if count_neighbour > 0:
            average_angle = np.angle(np.sum(np.exp(neighbour_angles[:count_neighbour] * 1j)))
            noise = eta * np.random.uniform(-np.pi, np.pi)
            new_angles[i] = average_angle + noise # updated angle with noise
        else:
            # if no neighbours, keep current angle
            new_angles[i] = angles[i]
        
        # update position with barrier collision check
        new_positions[i] = barrier_collision(positions[i], new_angles[i])
        # boundary conditions of box
        new_positions[i] %= L
        
    return new_positions, new_angles
    
def animate(frames):
    print(frames)
    
    global positions, angles, hist
        
    new_positions, new_angles = update(positions, angles)
    
    # save positions and angles to compressed npz files
    np.savez_compressed(f"pos_ang_arrays/positions_angles{frames}.npz", positions = new_positions, angles = new_angles)
    
    data = np.load(f"pos_ang_arrays/positions_angles{frames}.npz")
    positions = data["positions"]
    angles = data["angles"]
    
    # update the empty array with average angle
    average_angles[frames // 10] = np.angle(np.mean(np.exp(angles * 1j)))
    
    # add particle positions to the histogram
    hist += np.histogram2d(positions[:,0], positions[:,1], bins = [xedges, yedges], density = False)[0]
    
    # plotting
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(angles), np.sin(angles), angles)
    return qv,

fig, ax = plt.subplots(figsize = (6, 6))  
 
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
ax.add_patch(plt.Rectangle((barrier_x_start, barrier_y_start), barrier_x_end - barrier_x_start, barrier_y_end - barrier_y_start, color = "grey", alpha = 0.5))
ax.set_title(f"Vicsek model for {N} particles with an attractive barrier")
anim = FuncAnimation(fig, animate, frames = range(iterations), interval = 5, blit = True)
writer = FFMpegWriter(fps = 10, metadata = dict(artist = "Isobel"), bitrate = 1800)
#anim.save("Vicsek_loops_barrier.mp4", writer = writer, dpi = 100)
plt.show()

fig, ax2 = plt.subplots()

ax2.plot(range(0, iterations, 10), average_angles)
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Average Angle (rad)")
ax2.set_title("Alignment of Particles over Time")
#plt.savefig("barrier_alignment.png")
plt.show()

# normalise the histogram to cartesian coordinates for plotting
hist_normalised = hist.T / sum(hist)

fig, ax3 = plt.subplots()

cax = ax3.imshow(hist_normalised, extent = [0, L, 0, L], origin = "lower", cmap = "hot", aspect = "auto")
ax3.set_xlabel("X Position")
ax3.set_ylabel("Y Position")
ax3.set_title("Normalised 2D Histogram of Particle Density")
fig.colorbar(cax, ax = ax3, label = "Density")
#plt.savefig("barrier_densitymap.png")
plt.show()