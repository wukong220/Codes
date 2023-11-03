import re
import os
import sys
import time
import glob
import shutil
import logging
import platform
import subprocess
from datetime import datetime, timedelta
from itertools import combinations, product, permutations, islice

import pandas as pd
import numpy as np
from numba import vectorize, jit
from scipy.interpolate import splrep, splev
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from scipy.spatial import KDTree
from collections import defaultdict

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages as PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(filename='Run.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------Const-------------------------------------------
_BACT = "Bacteria"
HOST = platform.system()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
input_file = sys.argv[1] if len(sys.argv) > 1 else None
usage = "Run.py infile or bsub < infile.lsf"

#-----------------------------------Parameters-------------------------------------------
types = ["Chain", _BACT, "Ring"]
envs = ["Anlus", "Rand", "Slit"]
task, check, jump = ["Simus", "Anas", "Plots"][2], True, False
#-----------------------------------Dictionary-------------------------------------------
#参数字典
params = {
    'labels': {'Types': types[0:1], 'Envs': envs[2:3]},
    'marks': {'labels': [], 'config': []},
    'restart': [False, "equ"],
    'Queues': {'7k83!': 1.0, '9654!': 1.0},
    # 动力学方程的重要参数
    'Temp': 1.0,
    'Gamma': 100,
    'Trun': 5,
    'Dimend': 3,
    #'Dimend': [2,3],
    'num_chains': 1,
}

class _config:
    def __init__(self, Dimend, Type, Env, Params = params):
        self.config = {
            "Linux": {
                _BACT: {'N_monos': [3], 'Xi': 1000, 'Fa': [1.0],}, # 'Fa': [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0],},
                "Chain": {'N_monos': [20, 40, 80, 100, 150, 200, 250, 300], 'Xi': 0.0, 'Fa': [20.0, 100.0], #'Fa': [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 100.0],
                          #'Temp': [1.0, 0.2, 0.1, 0.05, 0.01],
                          # 'Gamma': [0.1, 1, 10, 100]
                          },
                "Ring": {'N_monos': [20, 40, 80, 100, 150, 200, 250, 300], 'Xi': 0.0, 'Fa': [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 100.0],
                         'Gamma': [0.1, 1, 10, 100],
                         # 'Temp': [1.0, 0.2, 0.1, 0.05, 0.01],
                         },

                "Anlus": {2: {'Rin': [0.0], 'Wid': [0.0]},
                              #3: {'Rin': [0.0], 'Wid': [0.0]},
                              #2: {'Rin': [5.0, 10.0, 15.0, 20.0, 30.0], 'Wid': [5.0, 10.0, 15.0, 20.0, 30.0]},
                              3: {'Rin': [5.0, 10.0, 15.0, 20.0, 30.0], 'Wid': [5.0, 10.0, 15.0, 20.0, 30.0]},
                            },
                "Rand":{2: {'Rin': [0.1256, 0.314, 0.4], 'Wid': [1.5, 2.0, 2.5]},
                            #2: {'Rin': [0.1256], 'Wid': [0.5, 1.0]},
                            #2: {'Rin': [0.314], 'Wid': [1.0]},
                            #2: {'Rin': [0.0628], 'Wid': [1.0, 1.5, 2.0, 2.5]},
                            3: {'Rin': [0.0314, 0.0628, 0.1256], 'Wid': [1.0, 1.5, 2.0, 2.5]},
                            },
                "Slit":{2: {"Rin":[0.0],"Wid":[5.0, 10.0, 15.0, 20.0]},
                        3: {"Rin":[0.0],"Wid":[3.0, 5.0, 10.0, 15.0, 20.0]},
                        },
                },

            "Darwin": {
                _BACT: {'N_monos': 3, 'Xi': 1000, 'Fa': 1.0},
                "Chain": {'Xi': 0.0,
                              #'N_monos': [20, 40, 80, 100, 150, 200, 250, 300],
                              'N_monos': [40, 100, 200],
                              'Fa': [0.0, 1.0], #'Fa': [0.0, 1.0, 10.0],
                              #'Temp': [0.2],
                          },
                "Ring": {'N_monos': [100], 'Xi': 0.0, 'Fa': [1.0], 'Gamma': [1.0]},

                "Anlus":{2: {'Rin': [0.0], 'Wid': [0.0]},
                            3: {'Rin': [0.0], 'Wid': [0.0]},
                            #3: {'Rin': [5.0], 'Wid': [5.0]},
                            },
                "Rand": {2: {'Rin': 0.4,  'Wid': 2.0},
                             3: {'Rin': 0.0314, 'Wid': 2.5},
                            },
                "Slit": {2: {"Rin": [0.0], "Wid": [5.0]},
                         3: {"Rin": [0.0], "Wid": [3.0, 5.0]}, #10.0, 15.0]},
                         },
            },
        }
        self.Params = Params
        self.labels = [t+e for t in self.Params['labels']['Types'] for e in self.Params['labels']['Envs']]
        self.Params['marks']['labels'] = self.labels
        self.Dimend = Dimend
        self.Type = Type
        self.Env = Env
        self.Label = self.Type+self.Env
        self.Params["marks"]["config"] = self.Label
        if self.Type in self.config[HOST]:
            self.Params.update(self.config[HOST][self.Type])
        if self.Env in self.config[HOST]:
            self.Params.update(self.config[HOST][self.Env][self.Dimend])
    def set_dump(self, Run):
        """计算 Tdump 的值: xu yu"""
        # 定义 dimension 和 dump 的映射
        dim_to_dump = {2: "xu yu vx vy", 3: "xu yu zu vx vy vz"}
        try:
            Run.Dump = dim_to_dump[self.Dimend]
        except KeyError:
            logging.error(f"Error: Wrong Dimend to run => dimension != {self.Dimend}")
            raise ValueError(f"Invalid dimension: {self.Dimend}")

        Run.Tdump = 2 * 10 ** Run.eSteps // Run.Frames
        Run.Tdump_ref = Run.Tdump // 100
        if HOST == "Darwin" and task == "Simus":
            Run.Tdump = Run.Tdump_ref

        Run.Tinit = Run.Frames * Run.Tdump_ref // 5
        Run.TSteps = Run.Frames * Run.Tdump
        Run.Tequ = Run.TSteps
        Run.Tref = Run.Frames * Run.Tdump_ref
        Run.Params["Total Run Steps"] = Run.TSteps

        if self.Type == _BACT:
            Run.Tdump //= 10
            Run.Tequ //= 100
##########################################END!###############################################################
class _run:
    def __init__(self, Dimend, Gamma, Temp, Trun, Params = params, Frames = 2000):
        self.Params = Params
        self.Queue = "7k83!"
        self.set_queue()
        self.Gamma = Gamma
        self.Trun = Trun
        self.Dimend = Dimend
        self.Frames = Frames
        self.Temp = Temp
        self.SkipRows = 9

        self.dt = 0.001
        self.Seed = self.set_seed()
        self.eSteps = 9 if self.Gamma == 1000 else 8
        self.Damp = 1.0 / self.Gamma
    def set_seed(self):
        return np.random.randint(700000000, 800000001)
    def set_queue(self):
        queues = {
            "7k83!": {"Usage": 1.0,  "hosts": ['g009', 'g008', 'a016']},
            "9654!": {"Usage": 1.0, "hosts": ['a017']}
        }
        if HOST != "Darwin":
            try:
                bqueues = subprocess.check_output(['bqueues']).decode('utf-8') # Decode the output here
                bhosts = subprocess.check_output(['bhosts']).decode('utf-8')
            except subprocess.CalledProcessError as e:
                logging.error(f"Error: {e}")
                raise ValueError(f"Error: {e}")

            myques = list(queues.keys())
            queue_info = {key: {"NJOBS": 0, "PEND": 0,"RUN": 0, "runs": 0, "cores": 0, "occupy": 0, "Avail": 0, "Usage": 0} for key in myques}
            myhosts = {key: value["hosts"] for key, value in queues.items()}
            for line in bqueues.strip().split('\n')[1:]:  # Skip the header line
                columns = line.split()
                if columns[0] in myques:
                    queue_info[columns[0]]["NJOBS"] = int(columns[7])
                    queue_info[columns[0]]["PEND"] = int(columns[8])
                    queue_info[columns[0]]["RUN"] = int(columns[9])

            for line in bhosts.strip().split('\n')[1:]:
                columns = line.split()
                for iqueue in myques:
                    if columns[0] in myhosts[iqueue]:
                        queue_info[iqueue]["runs"] += int(columns[5])
                        queue_info[iqueue]["cores"] +=  int(columns[3])

            for iqueue in myques:
                try:
                    bjobs = subprocess.check_output(['bjobs', '-u', 'all', '-q',  f'{iqueue}']).decode('utf-8')
                except subprocess.CalledProcessError as e:
                    logging.error(f"Error: {e}")
                    raise ValueError(f"Error: {e}")
                for line in bjobs.strip().split('\n')[1:]:
                    columns = line.split()
                    start_time = datetime.strptime(f"{columns[-4]} {columns[-3]} {datetime.now().year} {columns[-2]}", "%b %d %Y %H:%M")
                    if datetime.now() - start_time > timedelta(hours=24) and columns[2] == "RUN":
                        cores = int(columns[-1].split('*')[0]) if '*' in columns[-1] else 1
                        queue_info[iqueue]["occupy"] += cores
                queue_info[iqueue]["Avail"] = queue_info[iqueue]["cores"] - queue_info[iqueue]["occupy"] + 1
                queue_info[iqueue]["Usage"] = np.around( (queue_info[iqueue]["PEND"] + queue_info[iqueue]["RUN"] - queue_info[iqueue]["occupy"] ) / queue_info[iqueue]["Avail"], 3)
                self.Params["Queues"][iqueue] = queue_info[iqueue]["Usage"]
                if queue_info[iqueue]["PEND"] == 0:
                    self.Queue = max(myques, key=lambda x: queue_info[x]['cores'] - queue_info[x]['RUN'])
                elif queue_info[iqueue]["PEND"] > 0:
                    self.Queue = min(myques, key=lambda x: queue_info[x]['Usage']) #print(f"queue = {self.Queue}, queue_info: {queue_info}")
        return self.Queue
    def sub_file(self, Path, infiles):
        logging.info(f">>> Preparing sub file: {Path.simus}")
        for infile in infiles:
            print(">>> Preparing sub file......")
            dir_file = os.path.join(f"{Path.simus}", infile)
            bsub=[
                f'#!/bin/bash',
                f'',
                f'#BSUB -J {infile}_{Path.Jobname}',
                f'#BSUB -Jd "active chain"',
                f'#BSUB -r',
                f'#BSUB -q {self.Queue}',
                f'#BSUB -n 1',
                f'#BSUB -oo {dir_file}.out',
                f'#BSUB -eo {dir_file}.err',
                f'export RUN_ON_CLUSTER=true',
                f'cd {Path.simus}',
                f'echo "python3 Run.py {infile}"',
                f'python3 Run.py {infile}',
                #f'echo "mpirun -np 1 lmp_wk -i {infile}.in"',
                #f'mpirun -np 1 lmp_wk -i {infile}.in',
            ]

            with open(f"{dir_file}.lsf", "w") as file:
                for command in bsub:
                    file.write(command + "\n")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
class _init:
    def __init__(self, Config, Trun, Rin, Wid, N_monos, num_chains = params["num_chains"]):
        self.sigma_equ, self.mass, self.sigma = 0.94, 1.0, 1.0
        self.Ks = 300.0
        self.Config, self.Trun = Config, Trun
        self.Rin, self.Wid = Rin, Wid
        self.particle_density = 2
        self.N_monos, self.num_chains = int(N_monos), num_chains
        self.num_monos = self.N_monos * self.num_chains
        self.Env = "Free" if (self.Rin < 1e-6 and self.Wid < 1e-6) else self.Config.Env
        self.Rring = self.N_circles()
        self.jump = False
        self.set_box()   #set box

        if (self.Config.Type == "Ring" and self.Env == "Anlus") and (self.Env == "Slit" and self.Rin > 1e-6):
            self.jump = True
            print(f"I'm sorry => '{self.Config.Label}' is not ready! when Dimend = {Params['Dimend']}")
            logging.warning(f"I'm sorry => '{self.Label}' is not ready!")

        if self.Env == "Anlus":
            self.Rout = self.Rin + self.Wid  # outer_radius
            self.R_ring = self.Rin + self.Wid / 2
            self.R_torus = self.Wid / 2

            self.Nobs_Rin, self.Nobs_Rout = self.N_ring(self.particle_density, self.Rin, self.sigma), self.N_ring(self.particle_density, self.Rout, self.sigma)
            self.Nobs_ring, self.Nobs_torus = self.N_ring(self.particle_density, self.R_ring, self.sigma), self.N_ring(self.particle_density, self.R_torus, self.sigma)
            self.num_obs = int(self.Nobs_ring) * int(self.Nobs_torus) if self.Config.Dimend == 3 else int(self.Nobs_Rin + self.Nobs_Rout)
            theta = [np.linspace(0, 2 * np.pi, int(2 * np.pi * R / self.sigma_equ + 1))[:-1] for R in self.Rring]
            if self.Config.Dimend == 2 and (self.num_monos > sum(itheta.size for itheta in theta)):
                print("N_monos is too Long!")
                logging.warning("N_monos is too Long!")
                self.jump = True

        elif self.Env == "Rand":
            self.num_obs = int(np.ceil(self.Rin * self.v_box / self.v_obs))
        else:
            self.num_obs = 0

        #include Free chain
        self.sigma12 = self.sigma + 2 * self.Wid if (self.Env == "Rand") else 2 * self.sigma
        self.total_particles = self.num_obs + self.num_monos
        if self.Config.Type == "Ring":
            self.bonds = self.num_monos - self.num_chains*0
            self.angles = self.num_monos - self.num_chains*1
        else:
            self.bonds = self.num_monos - self.num_chains*1
            self.angles = self.num_monos - self.num_chains*2
        if (self.num_chains != 1):
            logging.error(f"ERROR => num_chains = {self.num_chains} is not prepared!\nnum_chains must be 1")
            raise ValueError(f"ERROR => num_chains = {self.num_chains} is not prepared!\nnum_chains must be 1")
    def set_box(self):
        """计算盒子大小"""
        if self.Env == "Anlus":
            self.Lbox = self.Rin + self.Wid + 1
        else:
            if self.Config.Type == "Chain":
                self.Lbox = self.N_monos/4 + 10
                if self.Env == "Slit":
                    self.Lbox = self.N_monos/2 + 5
            elif self.Config.Type == "Ring":
                self.Lbox = self.N_monos / 4 + 5
            elif self.Config.Type == _BACT:
                self.Lbox = self.N_monos * 10
            else:
                print(f"ERROR: Wrong model type! => Config.Type = {self.Config.Type}")
                logging.warning(f"ERROR: Wrong model type! => Config.Type = {self.Config.Type}")
        self.v_box = (2 * self.Lbox) ** self.Config.Dimend

        if self.Config.Dimend == 2:
            self.zlo = -self.sigma/2
            self.zhi = self.sigma/2
            self.v_obs = np.pi * self.Wid ** 2
        elif self.Config.Dimend == 3:
            self.zlo = - self.Lbox
            self.zhi = self.Lbox
            self.v_obs = 4 / 3 * np.pi * self.Wid ** 3
        else:
            logging.error(f"Error: Invalid Dimend  => dimension != {Config.Dimend}")
            raise ValueError(f"Error: Invalid Dimend  => dimension != {Config.Dimend}")
    def N_ring(self, density, Radius, sigma):
        return np.ceil(density * 2 * np.pi * Radius / sigma)
    def N_circles(self):
        inter = self.sigma_equ + 0.2
        start = self.Rin + inter + 0.5 if self.Env == "Anlus" else self.N_monos * self.sigma_equ/(2 * np.pi)
        stop = self.Rin + self.Wid - inter
        circles = int((stop - start) / inter) + 1 if self.Env == "Anlus" else 0
        return np.linspace(start, start + inter * circles, circles + 1)
    def set_torus(self, R_torus, Nobs_ring, Nobs_torus=0):
        theta = np.linspace(0, 2 * np.pi, int(Nobs_ring+1))[:-1]
        phi = np.linspace(0, 2 * np.pi, int(Nobs_torus+1))[:-1] if self.Config.Dimend == 3 else 0
        theta, phi = np.meshgrid(theta, phi)

        x =  np.around((self.R_ring + R_torus * np.cos(phi)) * np.cos(theta) * self.sigma, 5)
        y =  np.around((self.R_ring + R_torus * np.cos(phi)) * np.sin(theta) * self.sigma, 5)
        z =  np.around(R_torus * np.sin(phi) * self.sigma, 5)

        return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

    def write_header(self, file):
        # 写入文件头部信息
        file.write("{} LAMMPS data file for initial configuration:\n\n".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
        
        # 写入原子数目和边界信息
        file.write(f"{int(self.total_particles)} atoms\n\n")
        file.write(f"{self.bonds} bonds\n\n")
        file.write(f"{self.angles} angles\n\n")
        file.write("5 atom types\n\n")
        file.write("1 bond types\n\n")
        file.write("3 angle types\n\n")
        file.write(f"-{self.Lbox} {self.Lbox} xlo xhi\n")
        file.write(f"-{self.Lbox} {self.Lbox} ylo yhi\n")
        file.write(f"{self.zlo} {self.zhi} zlo zhi\n\n")
        file.write("Masses\n\n")
        file.write(f"1 {self.mass}\n")
        file.write(f"2 {self.mass}\n")
        file.write(f"3 {self.mass}\n")
        file.write(f"4 {self.mass}\n")
        file.write(f"5 {self.mass}\n\n")
    def write_chain(self, file):
        """
        Writes chain data into the file.

        Parameters:
        - file: file object to write into
        - system_dim: dimension of the system, either 2 or 3
        - chain_type: type of the chain, either 'polymer' or 'bacteria'
        - N_monos: number of monomers in the chain
        - sigma_equ: equilibrium bond length
        - Rchain: initial radius for polymer chain
        - theta0: initial angle
        - dtheta_chain: change in angle per monomer
        """
        #inlude Free chain
        file.write("Atoms\n\n")
        if self.Config.Dimend == 2:
            chain_coords = []
            theta0 = 0
            for iRring in self.Rring:
                theta = np.linspace(theta0, 2 * np.pi+theta0, int(2 * np.pi * iRring / self.sigma_equ + 1))[:-1]
                x = np.around(iRring * np.cos(theta) * self.sigma_equ, 5)
                y = np.around(iRring * np.sin(theta) * self.sigma_equ, 5)
                z = np.zeros_like(x)
                chain_coords.append(np.column_stack([x, y, z]))
                theta0 += theta[0] - theta[1]
            chain = np.vstack(chain_coords)[:self.N_monos]
            for i, coord in enumerate(chain):
                atom_type = 2  # Default to "middle" of the chain
                if i == 0:
                    atom_type = 1  # Head
                elif i == self.N_monos - 1:
                    atom_type = 3  # Tail
                file.write(f"{i + 1} 1 {atom_type} {' '.join(map(str, coord))}\n")

        elif self.Config.Dimend == 3:
            inter = self.sigma_equ + 0.2
            stop = (self.Wid- self.sigma_equ - 0.2) / 2  if self.Env == "Anlus" else self.N_monos * self.sigma_equ/(2 * np.pi)
            circles = int(stop / inter) + 1 if self.Env == "Anlus" else 0

            chain_coords = []
            theta0, phi0 = 0, 0
            for iRchain in np.linspace(0.5, inter * circles, circles+1)[:-1][::-1]:
                Nchain = self.N_ring(1, iRchain, self.sigma_equ)
                phi = np.linspace(phi0, 2 * np.pi + phi0, int(Nchain + 1))[:-1] if self.Config.Dimend == 3 else [0]
                for iphi in phi:
                    Nring = self.N_ring(1, self.R_ring + iRchain * np.cos(iphi), self.sigma_equ)
                    theta = np.linspace(theta0, 2 * np.pi+theta0, int(Nring + 1))[:-1]
                    x = np.round((self.R_ring + self.sigma + iRchain * np.cos(iphi)) * np.cos(theta) * self.sigma_equ, 5)
                    y = np.round((self.R_ring + self.sigma + iRchain * np.cos(iphi)) * np.sin(theta) * self.sigma_equ, 5)
                    z = np.full_like(theta, np.around(iRchain * np.sin(iphi) * self.sigma_equ, 5))
                    chain_coords.append(np.column_stack([x, y, z]))
                    theta0 += theta[0] - theta[1]
                phi0 += phi[0] - phi[1]

            # Prepare the chain for writing to file
            chain = np.vstack(chain_coords)[:self.N_monos]
            for i, coord in enumerate(chain):
                atom_type = 2  # Default to "middle" of the chain
                if i == 0:
                    atom_type = 1  # Head
                elif i == self.N_monos - 1:
                    atom_type = 3  # Tail
                file.write(f"{i + 1} 1 {atom_type} {' '.join(map(str, coord))}\n")
    def write_anlus(self, file):
        self.particles = None
        if self.Config.Dimend == 2:
            outer_ring = self.set_torus(self.R_torus, self.Nobs_Rout)
            inner_ring = self.set_torus(-self.R_torus, self.Nobs_Rin)
            for i, coord in enumerate(inner_ring):
                file.write(f"{self.N_monos+i+1} 1 4 {' '.join(map(str, coord))}\n")
            for i, coord in enumerate(outer_ring):
                file.write(f"{int(self.N_monos+self.Nobs_Rin+i+1)} 1 5 {' '.join(map(str, coord))}\n")
        elif self.Config.Dimend == 3:
            torus = self.set_torus(self.R_torus, self.Nobs_ring, self.Nobs_torus)
            for i, coord in enumerate(torus):
                file.write(f"{self.N_monos+i+1} 1 4 {' '.join(map(str, coord))}\n")
    def periodic_distance(self, pos1, pos2):
        delta = np.abs(pos1 - pos2)
        delta = np.where(delta > 0.5 * self.Lbox, delta - self.Lbox, delta)
        return np.sqrt((delta ** 2).sum())
    def neighbor_keys(self, hash_key):
        """Generate neighbor hash keys for a given hash_key."""
        dim = len(hash_key)
        for d in range(-1, 2):
            for i in range(dim):
                neighbor_key = list(hash_key)
                neighbor_key[i] += d
                yield tuple(neighbor_key)
    def write_rand(self, file):
        # obstacles: harsh grid and size
        self.obs_positions = []
        self.bound = self.Lbox
        #self.bound = self.Lbox - self.Wid - 0.56 * self.sigma
        self.hash_grid = defaultdict(list)
        self.grid_size = 2 * self.Wid
        for i in range(self.num_obs):
            while True:
                position = -self.bound + 2 * np.random.rand(self.Config.Dimend) * self.bound
                hash_key = tuple((position // self.grid_size).astype(int))
                # Check for overlaps using hash grid
                overlap = False
                for neighbor_key in self.neighbor_keys(hash_key):
                    for neighbor_pos in self.hash_grid[neighbor_key]:
                        if self.periodic_distance(position, neighbor_pos) < self.grid_size:
                            overlap = True
                            break
                if not overlap:
                    self.obs_positions.append(position)
                    self.hash_grid[hash_key].append(position)
                    pos = np.append(position, 0.0) if self.Config.Dimend == 2 else position
                    file.write(f"{int(self.N_monos + i + 1)} 1 4 {' '.join(map(str, pos))}\n")
                    break
        return self.obs_positions
    def write_potential(self, file):
        #写入bonds and angles
        file.write("\nBonds\n\n")
        for i in range(self.bonds):
            i1 = (i + 1) if (i + 1)%self.N_monos == 0 else (i+1) % self.N_monos
            i2 = (i + 2) if (i + 2)%self.N_monos == 0 else (i+2) % self.N_monos
            file.write(f"{i+1} 1 {i1} {i2}\n")

        file.write("\nAngles\n\n")
        for i in range(self.angles):
            i1 = (i + 1) if (i + 1)%self.N_monos == 0 else (i+1) % self.N_monos
            i2 = (i + 2) if (i + 2)%self.N_monos == 0 else (i+2) % self.N_monos
            i3 = (i + 3) if (i + 3)%self.N_monos == 0 else (i+3) % self.N_monos
            if i == 0:
                file.write(f"{i + 1} 1 {i1} {i2} {i3}\n")
            elif i == self.angles - 1:
                file.write(f"{i + 1} 3 {i1} {i2} {i3}\n")
            else:
                file.write(f"{i + 1} 2 {i1} {i2} {i3}\n")

    def data_file(self, Path):
        # 初始构型的原子信息: theta, x, y, z
        logging.info("==> Preparing initial data file......")
        # 打开data文件以进行写入
        for infile in [f"{i:03}" for i in range(1, self.Trun + 1)]:
            data_file = os.path.join(f"{Path.simus}", f'{infile}.{self.Config.Type[0].upper()}{self.Env[0].upper()}.data')
            print(f"==> Preparing initial data {infile}......")
            with open(f"{data_file}", "w") as file:
                self.write_header(file)
                self.write_chain(file)
                if self.Env == "Anlus":
                   self.write_anlus(file)
                elif self.Env == "Rand":
                    self.write_rand(file)
                self.write_potential(file)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#############################################################################################################
class _model:
    def __init__(self, Init, Run, Fa, Xi):
        self.Init = Init
        self.Config = Init.Config
        self.Run = Run
        self.type = Init.Config.Type.lower()
        self.Fa = Fa
        self.Xi = Xi
        self.Kb = self.Xi * Init.N_monos/4
        #for directory
        self.Pe = self.Fa / self.Run.Temp

        self.dump = {"init": "INIT", "equ": "EQU", "data": "DATA", "refine": "REFINE"}
        self.dump_read = "x y" if self.Config.Dimend == 2 else "x y z"

        # setup
        self.fix_cmd = {}
        dimension = self.Config.Dimend
        env = self.Init.Env
        wid = self.Init.Wid+1
        axis = "y" if dimension == 2 else "z"
        unfix = '\n'.join([
            '',
            'unfix           LANG',
            'unfix           NVE',
            'unfix           FREEZE',
        ])
        fix_wall_init = '\n'.join([
                f'fix             WALL1 {self.type} wall/lj126 {axis}lo v_lo 1.0 1.0 1.12246',
                f'fix             WALL2 {self.type} wall/lj126 {axis}hi v_hi 1.0 1.0 1.12246',
            ])
        fix_wall = '\n'.join([
                f'fix             WALL1 {self.type} wall/lj126 {axis}lo {-wid / 2} 1.0 1.0 1.12246',
                f'fix             WALL2 {self.type} wall/lj126 {axis}hi {wid / 2} 1.0 1.0 1.12246',
            ])
        unfix_wall = '\n'.join([
                'unfix           WALL1',
                'unfix           WALL2',
            ])
        fix2D = '' if dimension == 3 else '\nfix             2D all enforce2d'
        unfix2D = '' if dimension == 3 else '\nunfix           2D'
        if env == "Slit":
            self.bound = f'boundary        p {"f" if dimension == 2 else "p"} {"f" if dimension == 3 else "p"}'
            self.lh = '\n'.join([
                '',
                f'variable        lo equal ramp({-self.Init.Lbox/2},{-wid / 2})',
                f'variable        hi equal ramp({self.Init.Lbox/2},{wid / 2})',
                '',
            ])
            self.fix_cmd.update({"init": fix_wall_init + "\n" + fix2D, "data": fix_wall + "\n" + fix2D})
            self.fix_cmd["unfix"] = '\n'.join([unfix, unfix_wall, unfix2D])
        else:
            self.bound = f'boundary        p p p'
            self.lh = ''
            self.fix_cmd.update({"init": fix2D, "data": fix2D})
            self.fix_cmd["unfix"]  = unfix + "\n" + unfix2D

        # pairs, bonds, angles
        self.pair = {
        "SOFT": '\n'.join([
                '# pair potential and  soft potential',
                f'pair_style      hybrid/overlay lj4422/cut 1.03201 soft {1.03201 * (self.Init.Wid * 2 + self.Init.sigma)}',
                'pair_coeff      *3 *3 lj4422/cut 1 1.0',
                'pair_modify     shift yes',
                f'pair_coeff      *3 4*5 soft 1 {1.03201 * self.Init.sigma12 / 2}',
                f'pair_coeff      4*5 4*5 soft 1',
         ]),
        "INIT": '\n'.join([
                '# pair potential and  soft potential',
                f'pair_style      lj/cut 1.12246',
                'pair_coeff      *3 *3 1 1.0',
                'pair_modify     shift yes',
                f'pair_coeff      *3 4*5 1 {self.Init.sigma12 / 2} {1.12246 * self.Init.sigma12 / 2}',
                f'pair_coeff      4*5 4*5 1 {self.Init.Wid * 2 + self.Init.sigma} {1.12246 * (self.Init.Wid * 2 + self.Init.sigma)}',
         ]),
        "INIT4422": '\n'.join([
                '# pair potential and  soft potential',
                f'pair_style      lj4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          