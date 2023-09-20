import numpy as np
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import defaultdict

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
    
    def __init__(self, particle_density=1, R_ring=20, r_torus=5, is_3D=True):
        self.particle_density = particle_density
        self.R_ring = R_ring
        self.Rin = R_ring - r_torus
        self.Rout = R_ring + r_torus
        self.r_torus = r_torus
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
            outer_ring = self.circle(self.R_ring + self.r_torus)
            inner_ring = self.circle(self.R_ring - self.r_torus)
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
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_zlim(-self.R_ring, self.R_ring)
            plt.title('3D Torus Structure')
            plt.show()
        else:
            plt.scatter(self.particles[:, 0], self.particles[:, 1], c='r', marker='o')
            plt.xlabel('X')
            plt.ylabel('Y')
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
        axs[0].plot(x, y_single)
        axs[0].set_title("Single Point Radial Distribution Function")
        axs[0].set_xlabel("Distance")
        axs[0].set_ylabel("g(r)")

        # Multiple points RDF
        axs[1].plot(x, y_multi)
        axs[1].set_title("Averaged Radial Distribution Function")
        axs[1].set_xlabel("Distance")
        axs[1].set_ylabel("g(r)")

        plt.show()

class Plot:
    def __init__(self, bins = 10):
        self.num_bins = bins

    def distribution(self):
        # Generate synthetic data for 2D example
        np.random.seed(0)
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)

        # Create bins

        x_bins = np.linspace(min(x), max(x), self.num_bins)
        y_bins = np.linspace(min(y), max(y), self.num_bins)

        # Count frequency of points in each bin
        hist_x, _ = np.histogram(x, bins=x_bins)
        hist_y, _ = np.histogram(y, bins=y_bins)

        # Normalize the histograms
        hist_x = hist_x / np.sum(hist_x)
        hist_y = hist_y / np.sum(hist_y)

        # Create bin centers
        x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2

        # Create a 2D histogram (fy(x))
        hist_2D, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        hist_2D = hist_2D / np.sum(hist_2D)

        # Create bin centers for 2D histogram
        x_bin_centers_2D = (x_edges[:-1] + x_edges[1:]) / 2
        y_bin_centers_2D = (y_edges[:-1] + y_edges[1:]) / 2

        # Re-create the code for 2D example with corrected imshow parameters
        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot fy(x)
        axs[0].imshow(hist_2D, interpolation='nearest', origin='lower',
                    extent=[x_bin_centers_2D[0], x_bin_centers_2D[-1], y_bin_centers_2D[0], y_bin_centers_2D[-1]])
        axs[0].set_title("fy(x)")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")

        # Plot Fy(x)
        axs[1].bar(x_bin_centers, hist_y, width=(x_bins[1] - x_bins[0]))
        axs[1].set_title("Fy(x)")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("Frequency")

        # Plot Fx(y)
        axs[2].barh(y_bin_centers, hist_x, height=(y_bins[1] - y_bins[0]))
        axs[2].set_title("Fx(y)")
        axs[2].set_xlabel("Frequency")
        axs[2].set_ylabel("y")

        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    # Example usage
    rdf_plotter = RDFPlotter()
    #rdf_plotter.plot_rdf()
    #distribution()

    # Generate initial config for 3D
    config_3D = InitialConfig(dimensions_3D, Lbox, Wid, Phi, Sigma, N)
    #config_3D.place_obstacles_with_hash()
    #config_3D.place_chain_with_kdtree()

    # Initialize and test the class
    particle_density=2
    particle_structure_3D = ParticleStructure(particle_density, is_3D=True)
    particle_structure_2D = ParticleStructure(particle_density, is_3D=False)
    coords_3D = particle_structure_3D.generate_coordinates()
    coords_2D = particle_structure_2D.generate_coordinates()
    #particle_structure_3D.write_to_data_file("/mnt/data/torus_data.lammps")
    particle_structure_3D.show_coordinates()
    #particle_structure_2D.show_coordinates()