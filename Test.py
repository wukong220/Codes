import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, gaussian_kde, multivariate_normal
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import matplotlib.transforms as mtransforms
import seaborn as sns
import warnings
from itertools import combinations, product, permutations, islice
warnings.filterwarnings('ignore')

# Function to calculate mean and variance for scatter plots
def statistics(data):
    unique_coords, indices, counts = np.unique(data[:, :2], axis=0, return_inverse=True, return_counts=True)
    sum_values = np.bincount(indices, weights=data[:, 2])
    mean_values = sum_values / counts
    sum_values_squared = np.bincount(indices, weights=data[:, 2] ** 2)
    var_values = (sum_values_squared / counts) - (mean_values ** 2)
    return unique_coords, mean_values, var_values, counts

class InitialConfig:
    def __init__(self, dimensions=3, Lbox=100, Wid=0.5, Phi=0.03351, Sigma=1.0, N=100, chain_distance=0.945):
        self.dimensions = dimensions
        self.Lbox = Lbox
        self.Wid = Wid
        self.Phi = Phi
        self.Sigma = Sigma
        self.N = N
        self.chain_distance = chain_distance
        self.obstacle_positions = []
        self.chain_positions = []
        self.hash_grid = defaultdict(list)
        self.grid_size = 2 * Wid
        self.num_grid = int(Lbox // self.grid_size)

    def place_obstacles_with_hash(self):
        volume_box = self.Lbox ** self.dimensions
        volume_obstacle = (4 / 3 * np.pi * self.Wid ** 3) if self.dimensions == 3 else (np.pi * self.Wid ** 2)
        num_obstacles = int(self.Phi * volume_box // volume_obstacle)

        for _ in range(num_obstacles):
            while True:
                position = np.random.rand(self.dimensions) * self.Lbox
                hash_key = self.tuple((position // self.grid_size).astype(int))

                overlap = False
                for neighbor_key in self.neighboring_keys(hash_key):
                    for neighbor_pos in self.hash_grid[neighbor_key]:
                        if np.linalg.norm(position - neighbor_pos) < 2 * self.Wid:
                            overlap = True
                            break

                if not overlap:
                    self.obstacle_positions.append(position)
                    self.hash_grid[hash_key].append(position)
                    break
        print(f"{self.obstacle_positions[:10]}")

    def neighboring_keys(self, key):
        self.delta_keys = [tuple(d) for d in np.ndindex(*([3] * self.dimensions))]
        return [tuple(map(lambda x, dx: x + dx, key, dkey)) for dkey in self.delta_keys()]

    def place_chain_with_kdtree(self):
        obstacles_tree = KDTree(self.obstacle_positions)
        start_position = np.random.rand(self.dimensions) * self.Lbox
        self.chain_positions.append(start_position)

        direction = np.random.randn(self.dimensions)
        direction /= np.linalg.norm(direction)

        for i in range(1, self.N):
            next_position = self.chain_positions[-1] + direction * self.chain_distance
            while (next_position.min() < 0 or next_position.max() > self.Lbox or
                   obstacles_tree.query(next_position)[0] < 1.05):
                direction += 0.1 * np.random.randn(self.dimensions)
                direction /= np.linalg.norm(direction)
                next_position = self.chain_positions[-1] + direction * self.chain_distance

            self.chain_positions.append(next_position)
        print(f"{self.chain_positions[:10]}")

    def save_to_lammps(self, filename):
        with open(filename, 'w') as f:
            f.write("LAMMPS data file for initial config\n")
            total_atoms = len(self.obstacle_positions) + len(self.chain_positions)
            f.write(f"{total_atoms} atoms\n")
            f.write("2 atom types\n")

            for dim, label in zip(range(self.dimensions), ['x', 'y', 'z']):
                f.write(f"0 {self.Lbox} {label}lo {label}hi\n")

            f.write("\nAtoms\n")
            atom_id = 1

            for pos in self.obstacle_positions:
                f.write(f"{atom_id} 1 {' '.join(map(str, pos))}\n")
                atom_id += 1

            for pos in self.chain_positions:
                f.write(f"{atom_id} 2 {' '.join(map(str, pos))}\n")
                atom_id += 1

# generate both 2D and 3D toroidal structures
class ParticleStructure:
    """
    Generate the coordinates of points for toroidal structure in 2D and 3D
    
    Parameters:
    - dimension (int): 2 for 2D and 3 for 3D
    - inner_radius (float): inner radius of the toroidal structure
    - outer_radius (float): outer radius of the toroidal structure
    - num_points (int): number of points to generate
    
    Returns:
    - points (numpy.ndarray): N x dimension array of points
    """
    
    def __init__(self, particle_density=1, Rin=15, Wid = 10, is_3D=True):
        self.particle_density = particle_density
        self.Rin = Rin
        self.Rout = Rin + Wid
        self.R_ring = Rin + Wid/2
        self.r_torus = Wid / 2
        self.is_3D = is_3D
        self.particles = None
        
    def circle(self, radius):
        N_points = int(self.particle_density * 2 * np.pi * radius)
        theta = np.linspace(0, 2 * np.pi, N_points)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        return np.vstack([x, y]).T
    
    def generate_coordinates(self):
        if self.is_3D:
            # For 3D Torus
            N_points = int(self.particle_density * 2 * np.pi * self.R_ring)
            theta = np.linspace(0, 2 * np.pi, N_points)
            phi = np.linspace(0, 2 * np.pi, N_points)
            theta, phi = np.meshgrid(theta, phi)
            
            x = (self.R_ring + self.r_torus * np.cos(theta)) * np.cos(phi)
            y = (self.R_ring + self.r_torus * np.cos(theta)) * np.sin(phi)
            z = self.r_torus * np.sin(theta)
            self.particles = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        else:
            # For 2D Ring
            outer_ring = self.circle(self.Rout)
            inner_ring = self.circle(self.Rin)
            self.particles = np.vstack([outer_ring, inner_ring])
        print(f"{self.particles[:10]}")
        return self.particles
    
    def write_to_data_file(self, file_name):
        with open(file_name, "w") as f:
            f.write("LAMMPS Data File for 2D/3D Structure\n")
            f.write(f"{len(self.particles)} atoms\n")
            f.write("1 atom types\n")
            f.write("\n")
            f.write("Atoms\n")
            f.write("\n")
            for i, coord in enumerate(self.particles):
                f.write(f"{i+1} 1 {' '.join(map(str, coord))}\n")
                
    def show_coordinates(self):
        if self.is_3D:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.particles[:, 0], self.particles[:, 1], self.particles[:, 2], c='r', marker='o')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_zlim(-self.R_ring, self.R_ring)
            plt.title('3D Torus Structure')
            plt.show()
        else:
            plt.scatter(self.particles[:, 0], self.particles[:, 1], c='r', marker='o')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('2D Ring Structure')
            plt.axis('equal')
            plt.show()

class RDFPlotter:
    def __init__(self, natom=100, alatt=10, dr=0.1, rmax=10):
        self.natom = natom
        self.alatt = alatt
        self.dr = dr
        self.rmax = rmax
        self.layer_number = int(self.rmax / self.dr)
        self.atoms = self.build_crystral()

    class Atom:
        def __init__(self):
            self.x_cor = None
            self.y_cor = None

    def build_crystral(self):
        atoms = []
        for ii in range(0, self.natom):
            this_atom = self.Atom()
            this_atom.x_cor = ii % self.alatt
            this_atom.y_cor = int(ii / self.alatt)
            atoms.append(this_atom)
        return atoms

    def calculate_distance(self, atomii, atomjj):
        dr = np.sqrt((atomii.x_cor - atomjj.x_cor)**2 + (atomii.y_cor - atomjj.y_cor)**2)
        return dr

    def to_density(self, gr):
        y = []
        for ii in range(len(gr)):
            r = ii * self.dr
            s = 2 * np.pi * r * self.dr
            rho = 0 if s == 0 else gr[ii] / s / self.natom
            y.append(rho / (self.natom / (self.alatt ** 2)))
        return y

    def plot_rdf(self):
        # Single point RDF
        atom_flag = self.atoms[20]
        gr_single = np.zeros(self.layer_number, dtype='int')
        for atom in self.atoms:
            drij = self.calculate_distance(atom_flag, atom)
            if (drij < self.rmax):
                layer = int(drij / self.dr)
                gr_single[layer] += 1

        # Multiple points RDF
        gr_multi = np.zeros(self.layer_number, dtype='int')
        for atom_flag in self.atoms:
            gr = np.zeros(self.layer_number, dtype='int')
            for atom in self.atoms:
                drij = self.calculate_distance(atom_flag, atom)
                if (drij < self.rmax):
                    layer = int(drij / self.dr)
                    gr[layer] += 1
            gr_multi += gr

        gr_multi = gr_multi / len(self.atoms)

        y_single = self.to_density(gr_single)
        y_multi = self.to_density(gr_multi)
        x = [ii * self.dr for ii in range(len(gr_single))]

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # Single point RDF
        ax_a.plot(x, y_single)
        ax_a.set_title("Single Point Radial Distribution Function")
        ax_a.set_xlabel("Distance")
        ax_a.set_ylabel("g(r)")

        # Multiple points RDF
        ax_c.plot(x, y_multi)
        ax_c.set_title("Averaged Radial Distribution Function")
        ax_c.set_xlabel("Distance")
        ax_c.set_ylabel("g(r)")

        plt.show()

class Plot:
    def __init__(self, bins = 20):
        self.num_bins = bins
        self.num_pdf = bins*100

    def original(self):
        # 生成模拟数据
        frames = 200  # 时间帧数
        atoms = 100  # 粒子数
        times = np.arange(frames)
        ids = np.arange(atoms)
        data = np.random.rand(frames, atoms)
        data_dict = {
            "x": np.repeat(times, atoms),  # 时间帧（t坐标）
            "y": np.tile(ids, frames),  # 粒子ID（s坐标）
            "z": data.flatten(),  # 幅度（r坐标）
        }

        # ----------------------------> prepare data <----------------------------#
        keys, values = list(data_dict.keys()), list(data_dict.values())
        x_label, y_label, z_label = keys[0], keys[1], keys[2]
        x, y, z = values[0], values[1], values[2]
        simp_x, simp_y, simp_z = x, y, z

        # Calculate bin size and mid-bin values
        bin_size_z = (z.max() - z.min()) / 50
        bin_size_y = (y.max() - y.min()) / 50
        bin_size_x = (x.max() - x.min()) / 50
        mid_z = (np.floor(z.min() / bin_size_z) + np.ceil(z.max() / bin_size_z)) / 2 * bin_size_z
        mid_y = (np.floor(y.min() / bin_size_y) + np.ceil(y.max() / bin_size_y)) / 2 * bin_size_y
        mid_x = (np.floor(x.min() / bin_size_x) + np.ceil(x.max() / bin_size_x)) / 2 * bin_size_x
        data_e = np.column_stack([x, y, z])[(z >= mid_z - bin_size_z / 2) & (z <= mid_z + bin_size_z / 2)]
        data_f = np.column_stack([x, z, y])[(y >= mid_y - bin_size_y / 2) & (y <= mid_y + bin_size_y / 2)]
        data_g = np.column_stack([y, z, x])[(x >= mid_x - bin_size_x / 2) & (x <= mid_x + bin_size_x / 2)]

        unique_coords_e, _, _, counts_e = statistics(data_e)
        unique_coords_f, _, _, counts_f = statistics(data_f)
        unique_coords_g, _, _, counts_g = statistics(data_g)

        plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='Times New Roman')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15

        # ----------------------------> plot figure<----------------------------#
        # Prepare figure and subplots
        fig = plt.figure(figsize=(20, 9))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.13, top=0.9, wspace=0.5, hspace=0.5)
        gs = GridSpec(2, 5, figure=fig)
        ax_a = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        ax_b = fig.add_subplot(gs[0, 2])
        ax_c = fig.add_subplot(gs[0, 3])
        ax_d = fig.add_subplot(gs[0, 4])
        ax_e = fig.add_subplot(gs[1, 2], sharex=ax_b, sharey=ax_b)
        ax_f = fig.add_subplot(gs[1, 3], sharex=ax_c, sharey=ax_c)
        ax_g = fig.add_subplot(gs[1, 4], sharex=ax_d, sharey=ax_d)

        #plot figure#
        # ----------------------------> ax_a <----------------------------#
        sc_a = ax_a.scatter(simp_x, simp_y, simp_z, c=simp_z, cmap='rainbow') #, vmin=df_grp['mean'].min(), vmax=df_grp['mean'].max())
        #ax_a.axhline(y=mid_y, linestyle='--', lw=1.5, color='black')  # Selected Particle ID
        #ax_a.axvline(x=mid_x, linestyle='--', lw=1.5, color='black')  # Selected Time frame

        # axis settings
        ax_a.set_title(f'({z_label}, {x_label}, {y_label}) in 3D Space', fontsize=20)
        ax_a.set_xlabel(x_label, fontsize=20)
        ax_a.set_xlim(min(simp_x), max(simp_x))
        ax_a.set_ylabel(y_label, fontsize=20)
        ax_a.set_ylim(min(simp_y), max(simp_y))
        ax_a.set_zlabel(z_label, fontsize=20)
        ax_a.set_zlim(min(simp_z), max(simp_z))
        # colorbar
        axpos = ax_a.get_position()
        caxpos = mtransforms.Bbox.from_extents(axpos.x0 - 0.05, axpos.y0, axpos.x0 - 0.03, axpos.y1)
        cax = fig.add_axes(caxpos)
        cbar = plt.colorbar(sc_a, ax=ax_a, cax=cax)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.set_xlabel(z_label, fontsize=20)

        # linewidth and note
        ax_a.annotate("(a)", (-0.2, 0.9), textcoords="axes fraction", xycoords="axes fraction", va="center",
                    ha="center", fontsize=20)
        ax_a.tick_params(axis='both', which="major", width=2, labelsize=15, pad=7.0)
        ax_a.tick_params(axis='both', which="minor", width=2, labelsize=15, pad=4.0)
        # axes lines
        ax_a.spines['bottom'].set_linewidth(2)
        ax_a.spines['left'].set_linewidth(2)
        ax_a.spines['right'].set_linewidth(2)
        ax_a.spines['top'].set_linewidth(2)

        ## ----------------------------> ax_bcd <----------------------------#
        for ax, data, axis_labels, note in zip([ax_b, ax_c, ax_d],
                                         [np.column_stack([simp_x, simp_y, simp_z]), np.column_stack([simp_x, simp_z, simp_y]), np.column_stack([simp_y, simp_z, simp_x])],
                                         [(x_label, y_label, z_label), (x_label, z_label, y_label), (y_label, z_label, x_label)],
                                         ['(b)', '(c)', '(d)']):
            unique_coords, mean_values, var_values = statistics(data)[:3]
            sc = ax.scatter(unique_coords[:, 0], unique_coords[:, 1], c=mean_values, s=(var_values + 1) * 10, cmap='rainbow', alpha=0.7)

            # axis settings
            ax.set_title(fr'$\langle\ {labels[2]}\ \rangle$ in {labels[0]}-{labels[1]} Space', loc='right', fontsize=20)
            ax.set_xlabel(labels[0], fontsize=20)
            ax.set_ylabel(labels[1], fontsize=20)
            ax.set_xlim(min(unique_coords[:, 0]), max(unique_coords[:, 0]))
            ax.set_ylim(min(unique_coords[:, 1]), max(unique_coords[:, 1]))

            # colorbar
            axpos = ax.get_position()
            caxpos = mtransforms.Bbox.from_extents(axpos.x1 + 0.005, axpos.y0, axpos.x1 + 0.015, axpos.y1)
            cax = fig.add_axes(caxpos)
            cbar = plt.colorbar(sc, ax=ax, cax=cax)
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.ax.set_xlabel(labels[2], fontsize=20)

            # linewidth and note
            ax.annotate(note, (-0.2, 0.9), textcoords="axes fraction", xycoords="axes fraction", va="center", ha="center", fontsize=20)
            ax.tick_params(axis='both', which="major", width=2, labelsize=15, pad=7.0)
            ax.tick_params(axis='both', which="minor", width=2, labelsize=15, pad=4.0)
            #axes lines
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)

        # ----------------------------> ax_efg <----------------------------#
        for ax, data, bin, counts, axis_labels, note in zip([ax_e, ax_f, ax_g],
                                                      [unique_coords_e, unique_coords_f, unique_coords_g],
                                                      [({mid_z - bin_size_z / 2}, {mid_z + bin_size_z / 2}), ({mid_y - bin_size_y / 2}, {mid_y + bin_size_y / 2}),({mid_x - bin_size_x / 2}, {mid_x + bin_size_x / 2})],
                                                      [counts_e, counts_f, counts_g],
                                                      [(x_label, y_label, z_label), (x_label, z_label, y_label), (y_label, z_label, x_label)],
                                                      ['(e)', '(f)', '(g)']):

            sc = ax.scatter(data[:, 0], data[:, 1], s=counts*50, color="blue", alpha=0.6)
            # axis settings
            ax.set_title(fr'${labels[2]}_0\ \in$ [{bin[0]}, {bin[1]}]', loc='right', fontsize=20)
            ax.set_xlabel(labels[0], fontsize=20)
            ax.set_ylabel(labels[1], fontsize=20)

            # linewidth and note
            ax.annotate(note, (-0.2, 0.9), textcoords="axes fraction", xycoords="axes fraction", va="center",
                        ha="center", fontsize=20)
            ax.tick_params(axis='both', which="major", width=2, labelsize=15, pad=7.0)
            ax.tick_params(axis='both', which="minor", width=2, labelsize=15, pad=4.0)
            #axes lines
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
        plt.show()

    def distribution(self):
        """
        Compute the normalized distribution of the data on the s-t plane.

        Parameters:
        - data (np.array): A 2D array representing the data at each (s, t) point.
        - bins (tuple): The number of bins along each dimension (s, t).

        Returns:
        - normalized_distribution (np.array): A 2D array representing the normalized distribution.
        """

        # 生成模拟数据
        frames = 100  # 时间帧数
        atoms = 50  # 粒子数
        times = np.arange(frames)
        ids = np.arange(atoms)
        data = np.random.rand(frames, atoms)

        # 创建一个字典来存储数据
        data_dict = {
            "t": np.repeat(times, atoms),  # 时间帧（t坐标）
            "s": np.tile(ids, frames),  # 粒子ID（s坐标）
            "r": data.flatten(),  # 幅度（r坐标）
        }

        keys, values = list(data_dict.keys()), list(data_dict.values())
        x_label, y_label, z_label = keys[0], keys[1], keys[2]
        x, y, z = values[0], values[1], values[2]

        # Create a 2D histogram and bin centers
        bin_id, pdf_id = self.num_bins//2, self.num_pdf//2
        hist_x, x_bins = np.histogram(x, bins=self.num_bins, density=True)
        hist_y, y_bins = np.histogram(y, bins=self.num_bins, density=True)
        x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2
        x_range = np.linspace(min(x), max(x), self.num_pdf)
        y_range = np.linspace(min(y), max(y), self.num_pdf)
        pdf_x = gaussian_kde(x).evaluate(x_range)
        pdf_y = gaussian_kde(y).evaluate(y_range)
        hist_2D, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], density=True)

        #specific y or x
        hist_x_at_y = hist_2D[bin_id, :]/np.sum(hist_2D[bin_id, :])
        hist_y_at_x = hist_2D[:, bin_id]/np.sum(hist_2D[:, bin_id])

        # Plotting
        fig = plt.figure(figsize=(25, 15))
        gs = GridSpec(2, 4, figure=fig)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        ax_a = fig.add_subplot(gs[0:2, 0:2])
        ax_b = fig.add_subplot(gs[0, 2], sharex=ax_a)
        ax_c = fig.add_subplot(gs[0, 3], sharex=ax_a)
        ax_d = fig.add_subplot(gs[1, 2], sharey=ax_a)
        ax_e = fig.add_subplot(gs[1, 3], sharey=ax_a)

        # Plot fz(x,y)
        im = ax_a.imshow(hist_2D, interpolation='nearest', origin='lower',
                    extent=[x_bin_centers[0], x_bin_centers[-1], y_bin_centers[0], y_bin_centers[-1]])
        ax_a.set_title(f"$f^{z_label}({x_label},{y_label})$")
        ax_a.set_xlabel(f"{x_label}")
        ax_a.set_ylabel(f"{y_label}")
        ax_a.set_xlim(x_bin_centers[0], x_bin_centers[-1])
        ax_a.set_ylim(y_bin_centers[0], y_bin_centers[-1])
        plt.colorbar(im, ax=ax_a, orientation='vertical')

        # Plot Fzy(x)
        ax_b.bar(x_bin_centers, hist_x_at_y, width=(x_bins[1] - x_bins[0]), alpha = 0.7, label="histogram")
        ax_b.set_title(f"$f^{z_label}({x_label}; {y_label}_0)$"+f" with {y_label}0 = {y_bins[bin_id]:.2f}")
        ax_b.set_xlabel(f"{x_label}")
        ax_b.set_ylabel("Frequency")
        ax_b.set_ylim(min(hist_x_at_y), max(hist_x_at_y)*1.1)

        ax_c.bar(x_bin_centers, hist_x, width=(x_bins[1] - x_bins[0]), alpha = 0.7, label="histogram")
        ax_c.plot(x_range, pdf_x, 'r', label='PDF')
        ax_c.set_title(f"$f^{z_label}_{y_label}({x_label})$")
        ax_c.set_xlabel(f"{x_label}")
        ax_c.set_ylabel("Frequency")
        ax_c.set_ylim(min(hist_x), max(hist_x) *1.1)

        # Plot Fzx(y)
        ax_d.barh(y_bin_centers, hist_y_at_x, height=(y_bins[1] - y_bins[0]), alpha = 0.7, label="histogram")
        ax_d.set_title(f"$f^{z_label}({y_label}; {x_label}_0)$"+f" with {x_label}0 = {x_bins[bin_id]:.2f}")
        ax_d.set_xlabel("Frequency")
        ax_d.set_ylabel(f"{y_label}")
        ax_d.set_xlim(min(hist_y_at_x), max(hist_y_at_x)*1.1)

        # Plot Fzx(y)
        ax_e.barh(y_bin_centers, hist_y, height=(y_bins[1] - y_bins[0]), alpha = 0.7, label="histogram")
        ax_e.plot(pdf_y, y_range, 'r', label='PDF')
        ax_e.set_title(f"$f^{z_label}_{x_label}({y_label})$")
        ax_e.set_xlabel("Frequency")
        ax_e.set_xlim(min(hist_y), max(hist_y)*1.1)
        ax_e.set_ylabel(f"{y_label}")

        plt.rc('font', family='Times New Roman')
        plt.tight_layout()
        plt.show()

    def xyz():
        # Generate synthetic data for demonstration
        np.random.seed(0)
        x = np.random.choice(np.linspace(0, 10, 11), size=500)
        y = np.random.choice(np.linspace(0, 10, 11), size=500)
        z = 2 * np.sin(x) + 3 * np.cos(y) + np.random.normal(0, 1, 500)  # some function of x and y plus noise

        # Process data: Calculate mean and standard deviation of z for each (x, y) pair
        data = np.column_stack((x, y, z))
        unique_coords, indices, counts = np.unique(data[:, :2], axis=0, return_inverse=True, return_counts=True)
        sum_values = np.bincount(indices, weights=data[:, 2])
        mean_values = sum_values / counts
        sum_values_squared = np.bincount(indices, weights=data[:, 2] ** 2)
        std_values = np.sqrt((sum_values_squared / counts) - (mean_values ** 2))

        # Create the plots
        plt.figure(figsize=(18, 6))

        # 1. Scatter plot with error bars and color-coded mean
        plt.subplot(1, 3, 1)
        sc = plt.scatter(unique_coords[:, 0], unique_coords[:, 1], c=mean_values, cmap='viridis', edgecolors='k')
        plt.errorbar(unique_coords[:, 0], unique_coords[:, 1], yerr=std_values, fmt='none', ecolor='red', alpha=0.5,
                     label='Std Dev')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Scatter Plot with Error Bars & Color-coded Mean')
        plt.colorbar(label='Mean z')
        plt.legend()

        # 2. Contour plot for mean z values
        plt.subplot(1, 3, 2)
        grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        grid_z = np.empty_like(grid_x)
        grid_z.fill(np.nan)
        for i in range(len(unique_coords)):
            x_idx = np.where(grid_x[0, :] == unique_coords[i, 0])[0][0]
            y_idx = np.where(grid_y[:, 0] == unique_coords[i, 1])[0][0]
            grid_z[y_idx, x_idx] = mean_values[i]
        plt.contourf(grid_x, grid_y, grid_z, cmap='viridis')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Contour Plot for Mean z values')

        # 3. KDE for the density of z values
        plt.subplot(1, 3, 3)
        sns.kdeplot(x=x, y=y, weights=counts[indices], cmap='viridis', fill=True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('KDE for Density of z values')
        plt.colorbar(label='Density')

        plt.tight_layout()
        plt.show()

def var2str(variable):
    # transform to latex
    if variable.lower() == 'msd':
        return r'\mathrm{MSD}'
    latex_label = variable[0].upper()
    subscript, superscript = "", ""
    for char in variable[1:]:
        if char.isnumeric():  # If the character is a number, it will be a superscript
            superscript += char
        else:  # Otherwise, it will be part of the subscript
            subscript += char
    if subscript:
        latex_label += r'_{\mathrm{' + subscript + '}}'
    if superscript:
        latex_label += '^{' + superscript + '}'

    # transform to abbreviation
    abbreviation = variable[0].upper()
    trailing_number = ''.join(filter(str.isdigit, variable))
    base_variable = variable.rstrip(trailing_number)
    if len(base_variable) > 1 and not all(char.isnumeric() for char in base_variable[1:]):
        abbreviation += base_variable[1].lower()
    if trailing_number:
        abbreviation += trailing_number
    return fr"${latex_label}$", abbreviation

if __name__ == "__main__":
    # Example usage
    rdf_plotter = RDFPlotter()
    plot = Plot()
    #rdf_plotter.plot_rdf()

    # Generate initial config for 3D
    config_3D = InitialConfig()
    #config_3D.place_obstacles_with_hash()
    #config_3D.place_chain_with_kdtree()

    # Initialize and test the class
    particle_density=2
    particle_structure_3D = ParticleStructure(particle_density, is_3D=True)
    particle_structure_2D = ParticleStructure(particle_density, is_3D=False)
    #coords_3D = particle_structure_3D.generate_coordinates()
    #coords_2D = particle_structure_2D.generate_coordinates()
    #particle_structure_3D.write_to_data_file("/mnt/data/torus_data.lammps")
    #particle_structure_2D.show_coordinates()

    #plot.original()
    #plot.distribution(data_dict)
    #plot.xyz()

####################################################################
    # data_r[Pe][N][file][frame][atom]
    # len_Pe, len_N, len_file, len_frame, len_atom = len(Pes), len(Ns), len(files), len(frames), len(atoms)
    # Pe_arr = np.repeat(Pes, len_N * len_file * len_frame * len_atom)
    # N_arr = np.tile(np.repeat(Ns, len_file * len_frame * len_atom), len_Pe)
    # file_arr = np.tile(np.repeat(files, len_frame * len_atom), len_Pe * len_N)
    # frame_arr = np.tile(np.repeat(frames, len_atom), len_Pe * len_N * len_file)
    # atom_arr = np.tile(atoms, len_Pe * len_N * len_file * len_frame)

    # It seems that there was an error. Let's try again.
    # We need to make sure that the dimensions match for pcolormesh.

    # Create a sample DataFrame
    mean_z = {'x': [1, 1, 5, 3, 3, 7], 'y': [1, 2, 1, 4, 1, 2], 'z': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    std_z = {'z': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]}
    count_z = {'z': [1, 2, 1, 2, 1, 2]}

    df = pd.DataFrame({
        'x': mean_z['x'],
        'y': mean_z['y'],
        'mean_z': mean_z['z'],
        'std_z': std_z['z'],
        'count_z': count_z['z']
    })

params = {
    'num_chains': 1,
    'Dimend': [2.0,3.0],

}
data_shape = (5, 2001, 100, 3)
data = np.random.rand(*data_shape)
Lx = 10.0
frames = data.shape[1]

class BasePlot:
    def __init__(self, df, jump=True):
        self.df = df
        self.jump = jump
    def set_style(self):
        '''plotting'''
        plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
    def colorbar(self, fig, ax, sc, label, is_3D=False, loc="right"):
        '''colorbar'''
        axpos = ax.get_position()
        if is_3D:
            caxpos = mtransforms.Bbox.from_extents(axpos.x0 - 0.05, axpos.y0, axpos.x0 - 0.03, axpos.y1)
        else:
            if loc == "right":
                caxpos = mtransforms.Bbox.from_extents(axpos.x1 + 0.005, axpos.y0, axpos.x1 + 0.015, axpos.y1)
            elif loc == "left":
                caxpos = mtransforms.Bbox.from_extents(axpos.x0 - 0.07, axpos.y0, axpos.x0 - 0.05, axpos.y1)
        cax = fig.add_axes(caxpos)
        cbar = plt.colorbar(sc, ax=ax, cax=cax)
        cbar.ax.yaxis.set_ticks_position(loc)
        cbar.ax.set_xlabel(label, fontsize=20, labelpad=10)
    def set_axes(self, ax, data, labels, title, is_3D=False, rotation=-60, loc="right"):
        x, y, z = data[:3]
        xlabel, ylabel, zlabel = labels[:3]
        # axis settings
        if is_3D:
            rotation = 0
            loc = 'center'
            ax.set_zlabel(zlabel, fontsize=20, labelpad=10)
            ax.set_zlim(min(z), max(z))
        ax.tick_params(axis='x', rotation=rotation)
        ax.set_title(title, loc=loc, fontsize=20)
        ax.set_xlabel(xlabel, fontsize=20, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=20, labelpad=10)
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
    def adding(self, ax, note, is_3D=False):
        # linewidth and note
        ax.annotate(note, (-0.2, 0.9), textcoords="axes fraction", xycoords="axes fraction", va="center", ha="center", fontsize=20)
        if is_3D:
            ax.tick_params(axis='x', which="both", width=2, labelsize=15, pad=-3.0)
            ax.tick_params(axis='y', which="both", width=2, labelsize=15, pad=1.0)
            ax.tick_params(axis='z', which="both", width=2, labelsize=15, pad=2.0)
        else:
            ax.tick_params(axis='both', which="both", width=2, labelsize=15, pad=5.0)

        # axes lines
        for spine in ["bottom", "left", "right", "top"]:
            ax.spines[spine].set_linewidth(2)
    def save_fig(self, fig, path):
        pdf = PdfPages(path)
        pdf.savefig(fig, dpi=500, transparent=True)
        pdf.close()
class Plotter(BasePlot):
    def scatter(self, fig, ax, data, labels, note, is_3D=False):
        x, y, z, w = data
        xlabel, ylabel, zlabel, wlabel = labels
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H']
        title = f"{xlabel}-{ylabel}-{zlabel}-{wlabel} Space" if is_3D else f"({zlabel}, {wlabel}) in {xlabel}-{ylabel} Space"
        if is_3D:
            sc = ax.scatter(x, y, z, c=w, cmap="rainbow", vmin=w.min(), vmax=w.max())
            self.colorbar(fig, ax, sc, wlabel, is_3D)
        else:
            for idx, uw in enumerate(np.unique(w)):
                mask = (w == uw)
                marker = markers[idx % len(markers)]
                sc = ax.scatter(x[mask], y[mask], c=z[mask], cmap="rainbow", s=80, marker=marker, vmin=z.min(), vmax=z.max())
            self.colorbar(fig, ax, sc, labels[2], is_3D)

            for val, marker in w_marker.items():
                mask = w == val
            sc = ax.scatter(x[mask], y[mask], c=z[mask], cmap="rainbow", s=w[mask] * 10, marker=marker,  vmin=z.min(), vmax=z.max())
            self.colorbar(fig, ax, sc, zlabel, is_3D)

        self.set_axes(ax, data, labels, title, is_3D)
        self.adding(ax, note, is_3D)
    def Rg2(self, fig_save, variable="Rg2"):
        timer = Timer(variable)
        timer.start()

        #----------------------------> figure settings <----------------------------#
        if os.path.exists(f"{fig_save}.pdf") and self.jump:
            print(f"==>{fig_save}.pdf is already!")
            logging.info(f"==>{fig_save}.pdf is already!")
            return True
        else:
            print(f"{fig_save}.pdf")
            logging.info(f"{fig_save}.pdf")

        # ----------------------------> preparing<----------------------------#
        data_set = [tuple(self.df[label].values for label in label_set) for label_set in [("Pe", "N", "W", variable), ("Pe", "N", variable, "W"),
                                                                                                                            ("Pe", "W", variable, "N"), ("N", "W", variable, "Pe")]]
        labels_set = [("Pe", "N", "W", var2str(variable)[0]), ("Pe", "N", var2str(variable)[0], "W"),
                            ("Pe", "W", var2str(variable)[0], "N"), ("N", "W", var2str(variable)[0], "Pe")]
        notes = (["(A)"], ["(B)","(a)", "(b)"], ["(C)", "(c)", "(d)"], ["(D)", "(e)", "(f)"])

        # ----------------------------> plot figures<----------------------------#
        self.set_style()
        #Prepare figure and subplots
        fig = plt.figure(figsize=(18, 25))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95, wspace=0.6, hspace=0.5)
        gs = GridSpec(6, 4, figure=fig)

        axes_3D = [fig.add_subplot(gs[0:3, 0:2], projection='3d')] + [fig.add_subplot(gs[i:i+2, j:j+2], projection='3d') for i, j in [(0, 2), (3, 0), (3, 2)]]
        axes_2D = [fig.add_subplot(gs[i, j]) for i, j in [(2, 2), (2, 3), (5, 0), (5, 1), (5, 2), (5, 3)]]

        # ----------------------------> plotting <----------------------------#
        for i, (data, labels, note) in enumerate(zip(data_set, labels_set, notes)):
            x, y, z, w = data
            xlabel, ylabel, zlabel, wlabel = labels
            # ----------------------------> plot3D<----------------------------#
            self.scatter(fig, axes_3D[i], data, labels, note[0], True)
            if i == 0:
                continue
            # ----------------------------> plot2D<----------------------------#
            self.scatter(fig, axes_2D[2*(i-1)], data, labels, note[1])
            self.scatter(fig, axes_2D[2*(i-1) + 1], (w, z, x, y), (wlabel, zlabel, xlabel, ylabel), note[2])

        # ----------------------------> save fig <----------------------------#
        fig = plt.gcf()
        self.save_fig(fig, f"{fig_save}.pdf")

        # ax.legend(loc='upper left', frameon=False, ncol=int(np.ceil(len(Arg1) / 5.)), columnspacing = 0.1, labelspacing = 0.1, bbox_to_anchor=[0.0, 0.955], fontsize=10)
        #fig0.savefig(f"{fig_save}.png", format="png", dpi=1000, transparent=True)
        timer.count("saving figure")
        plt.show()
        plt.close()
        timer.stop()
        # -------------------------------Done!----------------------------------------#
        return False

class plotGraph:
    def __init__(self, df):
        self.df = df

    def set_style(self):
        '''plotting'''
        plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15

    def colorbar(self, ax, sc, label, is_3D=False):
        '''colorbar'''
        axpos = ax.get_position()
        caxpos = mtransforms.Bbox.from_extents(axpos.x0 - 0.05, axpos.y0, axpos.x0 - 0.03, axpos.y1) if is_3D else \
                 mtransforms.Bbox.from_extents(axpos.x1 + 0.005, axpos.y0, axpos.x1 + 0.015, axpos.y1)
        cax = self.fig.add_axes(caxpos)
        cbar = plt.colorbar(sc, ax=ax, cax=cax)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.set_xlabel(label, fontsize=20)

    def set_axes(self, ax, data, labels, is_3D=False):
        x, y, z, w = data
        xlabel, ylabel, zlabel, wlabel = labels
        title = f"{xlabel}-{ylabel}-{zlabel}-{wlabel} Space" if is_3D else f"({zlabel}, {wlabel}) in {xlabel}-{ylabel} Space"
        ax.set_title(title, loc='center', fontsize=20)
        # axis settings
        ax.tick_params(axis='x', rotation=-45)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        if is_3D:
            ax.set_zlabel(zlabel, fontsize=20)
            ax.set_zlim(min(z), max(z))

    def adding(self, ax, note, is_3D=False):
        # linewidth and note
        ax.annotate(note, (-0.2, 0.9), textcoords="axes fraction", xycoords="axes fraction", va="center", ha="center", fontsize=20)
        if is_3D:
            ax.tick_params(axis='x', which="both", width=2, labelsize=15, pad=-3.0)
            ax.tick_params(axis='y', which="both", width=2, labelsize=15, pad=1.0)
            ax.tick_params(axis='z', which="both", width=2, labelsize=15, pad=2.0)
        else:
            ax.tick_params(axis='both', which="both", width=2, labelsize=15, pad=5.0)

        # axes lines
        for spine in ["bottom", "left", "right", "top"]:
            ax.spines[spine].set_linewidth(2)

    def plot_scatter(self, ax, data, labels, note, is_3D=False):
        x, y, z, w = data
        xlabel, ylabel, zlabel, wlabel = labels
        if is_3D:
            sc = ax.scatter(x, y, z, c=w, cmap="rainbow", vmin=w.min(), vmax=w.max())
            self.colorbar(ax, sc, wlabel, is_3D)
        else:
            sc = ax.scatter(x, y, c=z, cmap="rainbow", s=w * 10, vmin=z.min(), vmax=z.max())
            self.colorbar(ax, sc, zlabel, is_3D)

        self.set_axes(ax, data, labels, is_3D)
        self.adding(ax, note, is_3D)

    def plot_graphs(self, variable="Rg2"):
        # ----------------------------> plot figures<----------------------------#
        self.set_style()
        #Prepare figure and subplots
        self.fig = plt.figure(figsize=(16, 24))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.13, top=0.9, wspace=0.6, hspace=0.5)
        gs = GridSpec(6, 4, figure=self.fig)

        axes_3D = [self.fig.add_subplot(gs[0:3, 0:2], projection='3d')] + [self.fig.add_subplot(gs[i:i+2, j:j+2], projection='3d') for i, j in [(0, 2), (3, 0), (3, 2)]]
        axes_2D = [self.fig.add_subplot(gs[i, j]) for i, j in [(2, 2), (2, 3), (5, 0), (5, 1), (5, 2), (5, 3)]]

        # ----------------------------> preparing<----------------------------#
        data_set = [tuple(self.df[label].values for label in label_set) for label_set in [("Pe", "N", "W", variable), ("Pe", "N", variable, "W"),
                                                                                                                            ("Pe", "W", variable, "N"), ("N", "W", variable, "Pe")]]
        labels_set = [("Pe", "N", "W", var2str(variable)[0]), ("Pe", "N", var2str(variable)[0], "W"),
                            ("Pe", "W", var2str(variable)[0], "N"), ("N", "W", var2str(variable)[0], "Pe")]
        notes = (["(A)"], ["(B)","(a)", "(b)"], ["(C)", "(c)", "(d)"], ["(D)", "(e)", "(f)"])

        # ----------------------------> plotting <----------------------------#
        for i, (data, labels, note) in enumerate(zip(data_set, labels_set, notes)):
            x, y, z, w = data
            xlabel, ylabel, zlabel, wlabel = labels
            # ----------------------------> plot3D<----------------------------#
            self.plot_scatter(axes_3D[i], data, labels, note[0], True)
            if i == 0:
                continue
            # ----------------------------> plot2D<----------------------------#
            self.plot_scatter(axes_2D[2*(i-1)], data, labels, note[1])
            self.plot_scatter(axes_2D[2*(i-1) + 1], (w, z, x, y), (wlabel, zlabel, xlabel, ylabel), note[2])
        plt.show()

# Generate some example DataFrame; replace this with your actual data

data = pd.DataFrame({
    'Pe': np.random.rand(100),
    'N': np.random.rand(100),
    'W': np.random.rand(100),
    'nu': np.random.rand(100)
})

df = pd.DataFrame(data)

# 定义变量
nu = np.array(df['nu'])
N = np.array(df['N'])
W = np.array(df['W'])

# 创建网格数据
xi = np.linspace(N.min(), N.max(), 100)
yi = np.linspace(W.min(), W.max(), 100)
X, Y = np.meshgrid(xi, yi)
Z = griddata((N, W), nu, (X, Y), method='cubic')

# 创建图形
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# ax_c Pcolormesh热图
c = axs[0, 1].pcolormesh(X, Y, Z, shading='gouraud', cmap='viridis')
fig.colorbar(c, ax=axs[0, 1], extend='both')
axs[0, 1].set_xlabel('N')
axs[0, 1].set_ylabel('W')
axs[0, 1].set_title('Pcolormesh Heatmap of nu')

# 等高线图
CS = axs[1, 0].contour(X, Y, Z, levels=14, linewidths=0.5, colors='k')
axs[1, 0].contourf(X, Y, Z, levels=14, cmap="rainbow")
axs[1, 0].clabel(CS, inline=True, fontsize=8)
axs[1, 0].set_xlabel('N')
axs[1, 0].set_ylabel('W')
axs[1, 0].set_title('Contour plot of nu')

# 密度图（Hexbin）
hb = axs[1, 1].hexbin(N, W, C=nu, gridsize=50, cmap='inferno', bins='log')
fig.colorbar(hb, ax=axs[1, 1], extend='both')
axs[1, 1].set_xlabel('N')
axs[1, 1].set_ylabel('W')
axs[1, 1].set_title('Hexbin plot of nu')

# 调整子图间距
plt.tight_layout()
plt.show()

# 如果您有其他的可视化需求，例如3D曲面图、散点图矩阵等，请补充您的需求，我可以继续为您提供相应的代码和解释。

