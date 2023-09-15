# Adding support for both 2D and 3D systems
from collections import defaultdict
import numpy as np
from scipy.spatial import KDTree

class InitialConfig:
    def __init__(self, dimensions, Lbox, Wid, Phi, Sigma, N, chain_distance=0.945):
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

    def hash_function(self, position):
        return tuple((position // self.grid_size).astype(int))

    def place_obstacles_with_hash(self):
        volume_box = self.Lbox ** self.dimensions
        volume_obstacle = (4 / 3 * np.pi * self.Wid ** 3) if self.dimensions == 3 else (np.pi * self.Wid ** 2)
        num_obstacles = int(self.Phi * volume_box // volume_obstacle)

        for _ in range(num_obstacles):
            while True:
                position = np.random.rand(self.dimensions) * self.Lbox
                hash_key = self.hash_function(position)

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

    def neighboring_keys(self, key):
        return [tuple(map(lambda x, dx: x + dx, key, dkey)) for dkey in self.delta_keys()]

    def delta_keys(self):
        return [tuple(d) for d in np.ndindex(*([3] * self.dimensions))]

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

# Parameters for 3D
dimensions_3D = 3
Lbox = 100
Wid = 0.5
Phi = 0.03351
Sigma = 1.0
N = 100

# Generate initial config for 3D
config_3D = InitialConfig(dimensions_3D, Lbox, Wid, Phi, Sigma, N)
config_3D.place_obstacles_with_hash()
print(f"{config_3D.obstacle_positions[:10]}")
config_3D.place_chain_with_kdtree()
print(f"{config_3D.chain_positions[:10]}")
