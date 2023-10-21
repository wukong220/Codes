import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from scipy.stats import gaussian_kde
import matplotlib.transforms as mtransforms
import warnings
warnings.filterwarnings('ignore')

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
        frames = 2000  # 时间帧数
        atoms = 100  # 粒子数
        times = np.arange(frames)
        ids = np.arange(atoms)
        data = np.random.rand(frames, atoms)
        data[0][1] = data[0][0]
        data_dict = {
            "x": np.repeat(times, atoms),  # 时间帧（t坐标）
            "y": np.tile(ids, frames),  # 粒子ID（s坐标）
            "z": data.flatten(),  # 幅度（r坐标）
        }

        # ----------------------------> prepare data <----------------------------#
        keys, values = list(data_dict.keys()), list(data_dict.values())
        x_label, y_label, z_label = keys[0], keys[1], keys[2]
        x, y, z = values[0], values[1], values[2]

        # Convert the data to a DataFrame for easier manipulation
        df_org = pd.DataFrame(data_dict)
        #df_org['x'][0] = df_org['x'][10]
        #df_org['y'][0] = df_org['y'][10]
        grouped = df_org.groupby(['x', 'y'])
        mean_z, std_z, count_z = grouped['z'].mean().reset_index(), grouped['z'].std().fillna(0).reset_index(), grouped['z'].count().reset_index()
        df = pd.DataFrame({
            'x': mean_z['x'],
            'y': mean_z['y'],
            'mean_z': mean_z['z'],
            'std_z': std_z['z'],
            'count_z': count_z['z']
        })
        mid_x, mid_y = df_org.loc[(df_org['x'] - (df_org['x'].max() + df_org['x'].min()) / 2).abs().idxmin()]['x'], df_org.loc[(df_org['y'] - (df_org['y'].max() + df_org['y'].min()) / 2).abs().idxmin()]['y']
        df_org_slicex, df_org_slicey = df_org[df_org['x'] == mid_x][['y', 'z']], df_org[df_org['y'] == mid_y][['x', 'z']]
        df_slicex, df_slicey = df[df['x'] == mid_x].sort_values(by='y'), df[df['y'] == mid_y].sort_values(by='x')

        # Find unique x and y values to create a grid
        unique_x = np.sort(df['x'].unique())
        unique_y = np.sort(df['y'].unique())
        grid_z = np.empty((len(unique_y), len(unique_x)))
        grid_z.fill(np.nan)
        for index, row in df.iterrows():
            x_idx = np.where(unique_x == row['x'])[0][0]
            y_idx = np.where(unique_y == row['y'])[0][0]
            grid_z[y_idx, x_idx] = row['mean_z']

        # ----------------------------> plot figure<----------------------------#
        # plt.subplots_adjust(left=0.15, right=0.85, bottom=0.13, top=0.9, wspace=0.2, hspace=0.2)
        # Create the layout
        fig = plt.figure(figsize=(18, 9))
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        gs = GridSpec(2, 4, figure=fig)
        ax_a = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        ax_b = fig.add_subplot(gs[0, 2])
        ax_c = fig.add_subplot(gs[0, 3], sharey=ax_b)
        ax_d = fig.add_subplot(gs[1, 2], sharex=ax_b)
        ax_e = fig.add_subplot(gs[1, 3], sharex=ax_b, sharey=ax_b)

        # ----------------------------> plot figure<----------------------------#
        #plt.subplots_adjust(left=0.15, right=0.85, bottom=0.13, top=0.9, wspace=0.2, hspace=0.2)
        sc_a = ax_a.scatter(x, y, z, c=z, cmap='rainbow', vmin=z.min(), vmax=z.max())
        sc_b = ax_b.scatter(df['x'], df['y'], c=df['mean_z'], s=(df['std_z'] + 1) * 10, cmap='rainbow', alpha=0.7, vmin=z.min(), vmax=z.max())

        sc_c = ax_c.scatter(df_org_slicex['z'], df_org_slicex['y'], color='k')
        ax_c.errorbar(df_slicex['mean_z'], df_slicex['y'], yerr=df_slicex['std_z'], fmt='^-', color='r', ecolor='r', capsize=5, alpha=0.6)

        sc_d = ax_d.scatter(df_org_slicey['x'], df_org_slicey['z'], color='k')
        ax_d.errorbar(df_slicey['x'], df_slicey['mean_z'], xerr=df_slicey['std_z'], fmt='^-', color='r', ecolor='r', capsize=5, alpha=0.6)

        cmap = ax_e.pcolormesh(unique_x, unique_y, grid_z, shading='auto', cmap='rainbow', vmin=z.min(), vmax=z.max())

        # ----------------------------> adding <----------------------------#
        ax_b.axhline(y=mid_y, linestyle='--', lw=1.5, color='black')  # Selected Particle ID
        ax_b.axvline(x=mid_x, linestyle='--', lw=1.5, color='black')  # Selected Time frame
        ax_e.axhline(y=mid_y, linestyle='--', lw=1.5, color='black')  # Selected Particle ID
        ax_e.axvline(x=mid_x, linestyle='--', lw=1.5, color='black')  # Selected Time frame

        axpos = ax_a.get_position()
        caxpos = mtransforms.Bbox.from_extents(axpos.x0 - 0.07, axpos.y0, axpos.x0 - 0.05, axpos.y1)
        cax = fig.add_axes(caxpos)
        cbar = plt.colorbar(sc_a, ax=ax_a, cax=cax)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.set_xlabel(f'{z_label}', fontsize=20)
        for i, txt in enumerate(df['count_z']):
            if txt > 1:
                ax_b.annotate(str(txt), (df['x'].iloc[i], df['y'].iloc[i]))

        # ----------------------------> axis settings <----------------------------#
        ax_a.set_title(f'({z_label}, {x_label}, {y_label}) in 3D Space', fontsize=20)
        ax_a.set_xlabel(f'{x_label}', fontsize=20)
        ax_a.set_xlim(min(x), max(x))
        ax_a.set_ylabel(f'{y_label}', fontsize=20)
        ax_a.set_ylim(min(y), max(y))
        ax_a.set_zlabel(f'{z_label}', fontsize=20)
        ax_a.set_zlim(min(z), max(z))

        ax_b.set_title(f'<{z_label}> in {x_label}-{y_label} Space', loc='right', fontsize=20)
        ax_b.set_xlabel(f'{x_label}', fontsize=20)
        ax_b.set_xlim(min(x), max(x))
        ax_b.set_ylabel(f'{y_label}', fontsize=20)
        ax_b.set_ylim(min(y), max(y))

        ax_c.set_title(f'${x_label}_0$={mid_x}', loc='right', fontsize=20)
        ax_c.set_xlabel(f'{z_label}({y_label}, ${x_label}_0$)', fontsize=20)
        ax_c.set_xlim(min(z), max(z))
        ax_c.set_ylabel(f'{y_label}', fontsize=20)

        ax_d.set_title(f'${y_label}_0$={mid_y}', loc='right', fontsize=20)
        ax_d.set_xlabel(f'{x_label}', fontsize=20)
        ax_d.set_ylabel(f'{z_label}({x_label}, ${y_label}_0$)', fontsize=20)
        ax_d.set_ylim(min(z), max(z))

        ax_e.set_title(f'<{z_label}>', loc='right', fontsize=20)
        ax_e.set_xlabel(f'{x_label}', fontsize=20)
        ax_e.set_ylabel(f'${y_label}$', fontsize=20)
        # ----------------------------> linewidth <----------------------------#
        for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e], ['(a)', '(b)', '(c)', '(d)', '(e)' ]):
            ax.annotate(label, (-0.3, 0.9), textcoords="axes fraction", xycoords="axes fraction", va="center",
                        ha="center", fontsize=20)
            ax.tick_params(axis='both', which="major", width=2, labelsize=15, pad=7.0)
            ax.tick_params(axis='both', which="minor", width=2, labelsize=15, pad=4.0)
            # ----------------------------> axes lines <----------------------------#
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)

        # ----------------------------> save fig <----------------------------#
        plt.show()
        plt.close()
        # -------------------------------Done!----------------------------------------#

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

    plot.original()
    #plot.distribution(data_dict)

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