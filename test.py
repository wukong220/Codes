import math
import matplotlib.pyplot as plt
import numpy as np

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
        dr = math.sqrt((atomii.x_cor - atomjj.x_cor)**2 + (atomii.y_cor - atomjj.y_cor)**2)
        return dr

    def to_density(self, gr):
        y = []
        for ii in range(len(gr)):
            r = ii * self.dr
            s = 2 * math.pi * r * self.dr
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

# Example usage
        #rdf_plotter = RDFPlotter()
        #rdf_plotter.plot_rdf()
        
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Generate synthetic data for 2D example
        np.random.seed(0)
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)
        
        # Create bins
        num_bins = 10
        x_bins = np.linspace(min(x), max(x), num_bins)
        y_bins = np.linspace(min(y), max(y), num_bins)
        
        # Count frequency of points in each bin
        hist_x, _ = np.histogram(x, bins=x_bins)
        hist_y, _ = np.histogram(y, bins=y_bins)
        
        # Normalize the histograms
        hist_x = hist_x / np.sum(hist_x)
        hist_y = hist_y / np.sum(hist_y)
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for 2D example
np.random.seed(0)
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)

# Create bins
num_bins = 10
x_bins = np.linspace(min(x), max(x), num_bins)
y_bins = np.linspace(min(y), max(y), num_bins)

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

