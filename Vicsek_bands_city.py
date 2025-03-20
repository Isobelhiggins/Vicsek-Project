import numba
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = "11"
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

# parameters
L = 50.0 # size of box
rho = 2.0 # density
N = int(rho * L**2) # number of particles
print("N:", N)
r0 = 1.0 # interaction radius
deltat = 1.0 # time step
factor = 1.0
v0 = 0.5 #r0 / deltat * factor # velocity
iterations = 5000 # animation frames
eta = 0.2 # noise/randomness
max_neighbours = N // 2 #  guess a good value, max is N

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N)

# cell list
cell_size = 1.0 * r0
lateral_num_cells = int(L / cell_size)
total_num_cells = lateral_num_cells ** 2
max_particles_per_cell = int(rho * cell_size ** 2 * 10)

# average angles and order parameter
time_step = 10 # interval to record data
angle_time_step = np.empty(time_step)
angle_t = 0
average_angles = [] # empty array for average angles
alignment_data = {} # dictionary for aligment data 
order_time_step = np.empty(time_step)
order_t = 0
order_parameters = [] # empty array for order parameters
order_data = {} # dictionary for order parameter data
block_size = 20 # for block averaging
std_threshold = 0.05 # standard deviation threshold for steady state to be reached
steady_blocks = 5 # consecutive blocks that meet std_threshold criteria

# histogram for average particle density
bins = int(L / r0)
hist, xedges, yedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, density = False)

# clustering
cluster_threshold = r0 # distance to be within same cluster
num_clusters_list = [] # empty array for number of clusters
cluster_particles_list = [] # empty array for average num of particles per cluster
num_clusters_data = {} # dictionary for cluster data
cluster_particles_data = {}

# velocity flux
tot_vx_all = np.zeros((bins, bins)) # velocity x components for all particles
tot_vy_all = np.zeros((bins, bins))
counts_all = np.zeros((bins, bins)) # number of particles in cell
vxedges = np.linspace(0, 1, bins + 1) # bin edges for meshgrid
vyedges = np.linspace(0, 1, bins + 1)

# define barrier positions and sizes
barriers = []
num_barriers = 5
min_size = 5
max_size = 15
gap = 5 # gap between barriers

# barrier collision avoidance
turnfactor = np.pi/2
boundary = r0
turned = np.zeros(N, dtype = bool)
    
def random_barriers(barriers):
    for i in range(num_barriers):
        while True:
            # randomly generate barrier coordinates
            width = np.random.randint(min_size, max_size + 1)
            height = np.random.randint(min_size, max_size + 1)
            x_start = np.random.randint(0 + gap, L - width - gap) # make sure there is a gap between barriers and box
            y_start = np.random.randint(0 + gap, L - height - gap)
            x_end = x_start + width
            y_end = y_start + height
            
            # make sure there is a gap between barriers and that they do not overlap
            for b_x_start, b_x_end, b_y_start, b_y_end in barriers:
                if ((x_start < b_x_end + gap) and (x_end > b_x_start - gap) and
                    (y_start < b_y_end + gap) and (y_end > b_y_start - gap)):
                    break
            else:
                barriers.append((x_start, x_end, y_start, y_end))
                break
        
    return barriers

# generate random barriers
barriers = random_barriers(barriers)

# save barrier coordinates for plotting
np.save("plotting_data/bands_city/barriers.npy", barriers)

# ensure particles are not generated inside the barrier
positions = np.zeros((N, 2))
for i in range(N):
    position = np.random.uniform(0, L, size = 2)
    # continue generating the particle position until it's not inside a barrier
    while any ((barrier_x_start - boundary) <= position[0] <= (barrier_x_end + boundary) and
                (barrier_y_start - boundary) <= position[1] <= (barrier_y_end + boundary)
                for barrier_x_start, barrier_x_end, barrier_y_start, barrier_y_end in barriers):
        position = np.random.uniform(0, L, size = 2)
    positions[i] = position
    
@numba.njit()
def barrier_collision(position, angle, turned, barriers):
    # calculate next position based on current position and angle
    next_position = (position + v0 * np.array([np.cos(angle), np.sin(angle)]) * deltat)
    next_angle = angle
    
    # if particle has not turned
    if turned == False:
        for barrier_x_start, barrier_x_end, barrier_y_start, barrier_y_end in barriers:
        # if particle's next position is within the boundary, turn
            if (((barrier_x_start - boundary) <= next_position[0] <= (barrier_x_end + boundary)) and
                ((barrier_y_start - boundary) <= next_position[1] <= (barrier_y_end + boundary))):
                next_angle += turnfactor
                turned = True
                
                next_position = (position + v0 * np.array([np.cos(next_angle), np.sin(next_angle)]) * deltat)
                break
    
    for barrier_x_start, barrier_x_end, barrier_y_start, barrier_y_end in barriers:   
        # if particle's next position is within barrier, do not update position for 1 time step, but turn
        if ((barrier_x_start <= next_position[0] <= barrier_x_end) and
            (barrier_y_start <= next_position[1] <= barrier_y_end)):
            next_angle += turnfactor
            turned = True
            return position, next_angle, turned
            
    return next_position, next_angle, turned

@numba.njit()
def get_cell_index(pos, cell_size, num_cells):
    return int(pos[0] // cell_size) % num_cells, int(pos[1] // cell_size) % num_cells

@numba.njit(parallel=True)
def initialize_cells(positions, cell_size, num_cells, max_particles_per_cell):
    
    # create cell arrays
    cells = np.full((num_cells, num_cells, max_particles_per_cell), -1, dtype = np.int32)  # -1 means empty
    cell_counts = np.zeros((num_cells, num_cells), dtype = np.int32)
    
    # populate cells with particle indices
    for i in  numba.prange(positions.shape[0]):
        cell_x, cell_y = get_cell_index(positions[i], cell_size, num_cells)
        idx = cell_counts[cell_x, cell_y]
        if idx < max_particles_per_cell:
            cells[cell_x, cell_y, idx] = i  # add particle index to cell
            cell_counts[cell_x, cell_y] += 1  # update particle count in this cell
    return cells, cell_counts

@numba.njit
def average_angle(new_angles):
    # average angle of particles
    return np.angle(np.sum(np.exp(new_angles * 1.0j)))

average_angles = [average_angle(positions)]

@numba.njit
def order_parameter(angles):
    avg_velocity = np.array([np.mean(np.cos(angles)), np.mean(np.sin(angles))])
    order_param = np.linalg.norm(avg_velocity) # norm of avg velocity
    return order_param

def steady_state(order_parameters):
    steady_reached = False
    steady_time = 0
    steady_blocks_count = 0
    
    # iterate over order parameters in blocks
    for i in range(len(order_parameters) - block_size + 1):
        block = order_parameters[i:(i + block_size)] # separate one block from the list (from i to i+block_size)
        running_average = np.mean(block) # average order parameter for that block
        running_std = np.std(block) # standard deviation for that block
                
        # check block srandard deviation with threshold
        if running_std < std_threshold:
            steady_blocks_count += 1 # increment consecutive blocks that meet std threshold
            
            # check if number of consecutive steady blocks meets criteria
            if steady_blocks_count >= steady_blocks:
                steady_reached = True
                steady_time = i + block_size # time of steady state reached
                break
        
        # if not enough consecutive blocks, reset count to 0    
        else:
            steady_blocks_count = 0
                
    return steady_reached, steady_time

def clusters(positions, L, threshold):
    # taking into account periodic boundary conditions
    total = 0
    for d in range(positions.shape[1]):
        pd = pdist(positions[:, d].reshape(positions.shape[0],1))
        pd[pd > L * 0.5] -= L
        total += pd ** 2
    total = np.sqrt(total)
    square = squareform(total)
    
    # clustering
    clustering = DBSCAN(eps = threshold, metric = "precomputed").fit(square)
    labels = clustering.labels_ # assign cluster labels to each point (points labelled -1 are noise)
    unique_labels = set(labels) # unique clustering labels
    
    # exclude noise in number of clusters calculation
    if -1 in labels:
        num_clusters = len(unique_labels) - 1
    else:
        num_clusters = len(unique_labels)
        
    # average number of particles per cluster
    if num_clusters > 0:
        cluster_particles = len(positions) / num_clusters
    else:
        cluster_particles = 0
        
    return labels, num_clusters, cluster_particles 

@numba.njit
def velocity_flux(positions, angles, v0):
    
    tot_vx = np.zeros(positions.shape[0]) # empty array for total velocity x components
    tot_vy = np.zeros(positions.shape[0]) # empty array for total velocity y components
    
    for i in numba.prange(positions.shape[0]):
        tot_vx[i] = v0 * np.cos(angles[i])
        tot_vy[i] = v0 * np.sin(angles[i])
    
    return tot_vx, tot_vy

@numba.njit(parallel=True)
def update(positions, angles, turned, barriers, cell_size, num_cells, max_particles_per_cell):
    
    # empty arrays to hold updated positions and angles
    N = positions.shape[0]
    new_positions = np.empty_like(positions)
    new_angles = np.empty_like(angles)
    new_turned = np.empty_like(turned)
      
    # initialize cell lists
    cells, cell_counts = initialize_cells(positions, cell_size, num_cells, max_particles_per_cell)

    # loop over all particles
    for i in numba.prange(N):  # parallelize outer loop
        neighbour_angles = np.empty(max_neighbours)
        count_neighbour = 0

        # get particle's cell, ensuring indices are integers
        cell_x, cell_y = get_cell_index(positions[i], cell_size, num_cells)

        # check neighboring cells (3x3 neighborhood)
        for cell_dx in (-1, 0, 1):
            for cell_dy in (-1, 0, 1):
                # ensure neighbor_x and neighbor_y are integers
                neighbour_x = int((cell_x + cell_dx) % num_cells)
                neighbour_y = int((cell_y + cell_dy) % num_cells)

                # check each particle in the neighboring cell
                for idx in range(cell_counts[neighbour_x, neighbour_y]):
                    j = cells[neighbour_x, neighbour_y, idx]
                    if i != j:  # avoid self-comparison
                        # calculate squared distance for efficiency
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        distance_sq = dx * dx + dy * dy
                        # compare with squared radius
                        if distance_sq < r0 * r0:
                            if count_neighbour < max_neighbours:
                                neighbour_angles[count_neighbour] = angles[j]
                                count_neighbour += 1

        # apply noise using Numba-compatible randomness
        noise = eta * (np.random.random() * 2 * np.pi - np.pi)

        # if neighbours, calculate average angle
        if count_neighbour > 0:
            average_angle = np.angle(np.sum(np.exp(neighbour_angles[:count_neighbour] * 1j)))
            new_angles[i] = average_angle + noise
        else:
            # if no neighbours, keep current angle
            new_angles[i] = angles[i] + noise
        
        # update position with barrier collision check
        new_positions[i], new_angles[i], new_turned[i] = barrier_collision(positions[i], new_angles[i], turned[i], barriers)
        # boundary conditions of box
        new_positions[i] %= L

    return new_positions, new_angles, new_turned

def animate(frames):
    print(frames)
    global positions, angles, turned, angle_time_step, angle_t, order_time_step, order_t, hist, tot_vx_all, tot_vy_all, counts_all, vxedges, vyedges
    
    new_positions, new_angles, new_turned = update(positions, angles, turned, barriers, cell_size, lateral_num_cells, max_particles_per_cell)
        
    # update global variables
    delta_pos = new_positions - positions # change in positions
    positions = new_positions
    angles = new_angles
    turned = new_turned
    
    # update the empty array with average angle
    angle_time_step[angle_t] = average_angle(new_angles)
    if angle_t == time_step - 1:  # check if array filled
        average_angles.append(average_angle(angle_time_step))
        angle_t = 0  # reset t
        angle_time_step = np.empty(time_step)  # reinitialise array
    else:
        angle_t += 1  # increment t
    
    # update the empty array with order parameter    
    order_time_step[order_t] = order_parameter(new_angles)
    if order_t == time_step - 1:
        order_parameters.append(np.mean(order_time_step))
        order_t = 0
        order_time_step = np.empty(time_step)
    else:
        order_t += 1
    
    # add particle positions to the histogram once reached steady state
    if frames >= 3000:
        hist += np.histogram2d(positions[:,0], positions[:,1], bins = [xedges, yedges], density = False)[0]
    
    # update cluster arrays
    if frames % time_step == 0:
        labels, num_clusters, cluster_particles = clusters(positions, L, cluster_threshold)
        num_clusters_list.append(num_clusters) # number clusters
        cluster_particles_list.append(cluster_particles) # number particles per cluster
    
    # calculate velocity fluxes once reached steady state
    if frames >= 3000:
        tot_vx, tot_vy = velocity_flux(positions, angles, v0)
        
        # histograms for the x and y velocity components
        H_vx, vxedges, vyedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, weights = delta_pos[:,0])
        H_vy, vxedges, vyedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, weights = delta_pos[:,1])
        counts, vxedges, vyedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins)
        
        tot_vx_all += H_vx
        tot_vy_all += H_vy
        counts_all += counts # hist of number of particles
    
    # can be commented out if not running the animation and just saving data
    # Update the quiver plot
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
    
    # uncomment to save positions and angles to npz files for plotting in Vicsek_plotting.ipynb
    # np.savez_compressed(f"plotting_data/bands_city/pos_ang{frames}.npz", positions = np.array(positions, dtype = np.float16), angles = np.array(angles, dtype = np.float16))
    
    # can be commented out if not running the animation and just saving data
    return qv,

# comment out to hide the animation
# Vicsek Model for N Particles Animation
fig, ax = plt.subplots(figsize = (3.5, 3.5)) 
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
for barrier_x_start, barrier_x_end, barrier_y_start, barrier_y_end in barriers:
    ax.add_patch(plt.Rectangle((barrier_x_start, barrier_y_start), barrier_x_end - barrier_x_start, barrier_y_end - barrier_y_start, color = "grey", alpha = 0.5))
    ax.add_patch(plt.Rectangle((barrier_x_start - boundary, barrier_y_start - boundary), (barrier_x_end + boundary) - (barrier_x_start - boundary), (barrier_y_end + boundary) - (barrier_y_start - boundary), edgecolor = "grey", fill = False))
anim = FuncAnimation(fig, animate, frames = range(0, iterations), interval = 5, blit = True)
# writer = FFMpegWriter(fps = 10, metadata = dict(artist = "Isobel"), bitrate = 1800)
# anim.save("simulation_videos/Vicsek_bands_city.mp4", writer = writer, dpi = 300)
plt.show()
 
# uncomment to save analysis data to npz files for plotting in Vicsek_plotting.ipynb

# intialise arrays   
# average_angles = []
# order_parameters = []
# hist = np.empty((len(xedges) - 1, len(yedges) - 1))
# num_clusters_list = []
# cluster_particles_list = []

# run animation and update arrays
# for frame in range(0, iterations + 1):
#     animate(frame)
          
#     alignment_data = average_angles 
#     order_data = order_parameters

# steady_reached, steady_time = steady_state(order_parameters)
# print(f"Steady state reached in {steady_time * 10} frames")

# num_clusters_data = num_clusters_list
# cluster_particles_data = cluster_particles_list

# save data to npz files for plotting
# np.savez_compressed(f"plotting_data/bands_city/avg_ang.npz", angles = np.array(average_angles, dtype = np.float16))
# np.savez_compressed(f"plotting_data/bands_city/hist.npz", hist = np.array(hist, dtype = np.float64))
# np.savez_compressed(f"plotting_data/bands_city/flow.npz", vx = np.array(tot_vx_all, dtype = np.float32), vy = np.array(tot_vy_all, dtype = np.float32), counts = np.array(counts_all, dtype = np.float32), vxedges = np.array(vxedges, dtype = np.float32), vyedges = np.array(vyedges, dtype = np.float32))
# np.savez_compressed(f"plotting_data/bands_city/order.npz", order = np.array(order_parameters, dtype = np.float16), steady_reached = np.array(steady_reached, dtype = np.bool_), steady_time = np.array(steady_time, dtype = np.float16))
# np.savez_compressed(f"plotting_data/bands_city/clusters.npz", num_clust = np.array(num_clusters_list, dtype = np.float16), particle_clust = np.array(cluster_particles_list, dtype = np.float16))