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
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(filename='Run.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#-----------------------------------Const-------------------------------------------
_BACT = "Bacteria"
HOST = platform.system()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
input_file = sys.argv[1] if len(sys.argv) > 1 else None
#-----------------------------------Parameters-------------------------------------------
types = ["Chain", _BACT, "Ring"]
envs = ["Anlus", "Rand", "Slit"]
tasks = ["Simus", "Anas"]
#-----------------------------------Dictionary-------------------------------------------
#参数字典
params = {
    'labels': {'Types': types[0:1], 'Envs': envs[0:1]},
    'marks': {'labels': [], 'config': []},
    'task': tasks[1],
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
                "Chain": {'N_monos': [20, 40, 80, 100, 150, 200, 250, 300], 'Xi': 0.0,
                              'Fa': [1.0],
                              'Temp': [0.2],
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
                         3: {"Rin": [0.0], "Wid": [3.0]},
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
        if HOST == "Darwin" and params['task'] == "Simus":
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
##########################################END!###############################################################

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
                f'pair_style      lj4422/cut 1.03201',
                'pair_coeff      *3 *3 1 1.0',
                'pair_modify     shift yes',
                f'pair_coeff      *3 4*5 1 {self.Init.sigma12 / 2} {1.03201 * self.Init.sigma12 / 2}',
                f'pair_coeff      4*5 4*5 1 {self.Init.Wid * 2 + self.Init.sigma} {1.03201 * (self.Init.Wid * 2 + self.Init.sigma)}',
         ]),
        "LJ": '\n'.join([
            '#pair potential',
            f'pair_style      lj/cut 1.12246',
            'pair_modify	    shift yes',
            'pair_coeff      *3 *3 1 1.0',
            f'pair_coeff      *3 4*5 1 {self.Init.sigma12/2} {1.12246 * self.Init.sigma12/2}',
            'pair_coeff      4*5 4*5  1 1.0 0.0',
        ]),
        "LJ4422": '\n'.join([
            '#pair potential',
            f'pair_style      lj4422/cut 1.03201',
            'pair_modify	    shift yes',
            'pair_coeff      *3 *3 1 1.0',
            f'pair_coeff      *3 4*5 1 {self.Init.sigma12/2} {1.03201 * self.Init.sigma12/2}',
            'pair_coeff      4*5 4*5  1 1.0 0.0',
        ])}

        self.bond = {
        "harmonic": '\n'.join([
            '# Bond potential',
            'bond_style      harmonic',
            'bond_coeff      1 4000 1.0',
            'special_bonds   lj/coul 1.0 1.0 1.0',
        ]),
        "fene4422": '\n'.join([
            '# Bond potential',
            'bond_style      fene4422',
            f'bond_coeff      1 {self.Init.Ks} 1.05 1.0 1.0',
            'special_bonds   lj/coul 0.0 1.0 1.0',
        ])}

        self.angle = {
        "harmonic": '\n'.join([
            '# angle potential',
            'angle_style     harmonic',
            f'angle_coeff     * {self.Kb} 180',
        ]),
        "hybrid": '\n'.join([
            '# Angle potential',
            'angle_style     hybrid actharmonic_h2 actharmonic actharmonic_t',
            f'angle_coeff     1 actharmonic_h2 {self.Kb} 180 {self.Fa} {self.Fa}',
            f'angle_coeff     2 actharmonic   {self.Kb} 180 {self.Fa}',
            f'angle_coeff     3 actharmonic_t {self.Kb} 180 {self.Fa}',
        ]),
        "actharmonic": '\n'.join([
            '# Angle potential',
            'angle_style      actharmonic',
            f'angle_coeff     * {self.Kb} 180 {self.Fa}',
        ])}

    def write_section(self, file, cmds):
        """向文件中写入一个命令区块"""
        for command in cmds:
            file.write(command + "\n")

    def setup(self, dimend: int, dir_file: str, read: str) -> list:
        """Setup the basic environment for LAMMPS simulation."""
        return [
            '# Setup',
            'echo		        screen',
            'units           lj',
            f'dimension       {dimend}',
            self.bound,
            'atom_style      angle',
            self.lh,
            f'variable        Pre_soft equal ramp(0.0,10000.0)',
            f'variable        dir_file string {dir_file}',
            '',
            read,
            '',
            '#groups',
            'group           head type 1',
            'group           end type 3',
            f'group			      {self.type} type 1 2 3',
            'group			      obs type 4 5',
            '',
            'velocity		all set 0.0 0.0 0.0',
            '',
        ]

    def iofile(self, file, title):
        lmp_rest = '' if title == self.dump["data"] else f".{title.lower()}"
        lmp_trj = '' if title == self.dump["data"] else f".{self.type}_{title.lower()}"
        if file == "restart":
            return f'${{dir_file}}{lmp_rest}.restart'
        elif file == "dump":
            return f'${{dir_file}}{lmp_trj}.lammpstrj'

    def configure(self, prompt, temp, damp, run):
        return [
            '# for communication',
            'comm_style      brick',
            'comm_modify     mode single cutoff 3.0 vel yes',
            '',
            'neighbor	      1.5 bin',
            f'neigh_modify	  every 1 delay 0 check yes exclude group {self.type} obs',
            '',
            prompt,
            '##################################################################',
            f"#fix 	          SOFT all adapt 1 pair soft a *4 4 v_Pre_soft",
            f'fix      	      LANG obs langevin {temp} {temp} {damp} {run.set_seed()}',
            f'fix             NVE obs nve/limit 0.01',
            '',
            'reset_timestep    0',
            'timestep        0.001',
            'run             100000',
            '',
            'unfix           LANG',
            'unfix           NVE',
            '',
        ]

    def potential(self, prompt: str, pair: str, bond: str, angle: str, exclude="exclude none") -> list:
        """Define the potential parameters for LAMMPS simulation."""
        return [
            prompt,
            '##################################################################',
            pair,
            '',
            bond,
            '',
            angle,
            '##################################################################',
        ]

    def fix(self, prompt: str, temp: float, damp: float, run) -> list:
        """Define the fix parameters for LAMMPS simulation."""
        self.fix_cmd["langevin"] = "\n".join([
            f'fix      	      LANG {self.type} langevin {temp} {temp} {damp} {run.set_seed()}',
            f'fix             NVE {self.type} nve/limit 0.01' if 'init' in prompt else f'fix             NVE {self.type} nve',
            f'fix             FREEZE obs setforce 0.0 0.0 0.0',
        ])
        fix_cmd = self.fix_cmd["init"] if "init" in prompt else self.fix_cmd["data"]
        return [
            '# for communication',
            'comm_style      brick',
            'comm_modify     mode single cutoff 3.0 vel yes',
            '',
            'neighbor	      1.5 bin',
            'neigh_modify	  every 1 delay 0 check yes exclude none',
            '',
            f'{prompt}',
            '##################################################################',
            self.fix_cmd["langevin"],
            fix_cmd,
            '',
        ]

    def run(self, title: str, timestep: int, tdump: int, run) -> list:
        """Define the run parameters for LAMMPS simulation."""
       #log, unfix, dump
        log_cmd ='log	            ${dir_file}.log' if title == self.dump["equ"] else ''
        unfix_cmd = '' if title == self.dump["data"] else self.fix_cmd["unfix"]
        type, dt = ('all', "0.001") if title == self.dump["init"] else (self.type, run.dt)

        return [
            f'dump	        	{title} {type} custom {tdump} {self.iofile("dump", title)} id type {run.Dump}',
            f'dump_modify     {title} sort id',
            '',
            'reset_timestep	0',
            f'timestep        {dt}',
            f'thermo		      {timestep // 200}',
            log_cmd,
            f'restart         {tdump//10}  ${{dir_file}}.a.restart  ${{dir_file}}.b.restart ',
            f'run	            {timestep}',
            f'write_restart   {self.iofile("restart", title)}',
            unfix_cmd,
            f'undump          {title}',
            '',
        ]

    def in_file(self, Path):
        """Main function to write the in-file for LAMMPS simulation."""
        #from LJ + harmonic ==> LJ4422 + FENE4422
        Config, Init, Run = Path.Config, Path.Init, Path.Run
#        Run.Trun = 1000
        logging.info(f"==> Writing infile ==> {Path.simus}")

        for infile in [f"{i:03}" for i in range(1, Run.Trun + 1)]:
            print(f"==> Writing infile: {infile}......")
            try:
                #setup
                dir_file = os.path.join(f"{Path.simus}", infile)
                if Run.Params["restart"][0]:
                    read = f'read_restart       {self.iofile("restart", Run.Params["restart"][1])}'
                else:
                    read = f'read_data       {dir_file}.{Config.Type[0].upper()}{Init.Env[0].upper()}.data'

                # potential
                detach_potential = self.potential('# for configuration', self.pair["INIT4422"], self.bond["harmonic"], self.angle["harmonic"])
                if Config.Type == "Ring":
                    initial_potential = self.potential(f'# {Config.Type} for initialization', self.pair["LJ4422"], self.bond["fene4422"], self.angle["harmonic"])
                    run_potential = self.potential('# for equalibrium', self.pair["LJ4422"], self.bond["fene4422"], self.angle["actharmonic"])
                else:
                    initial_potential = self.potential('# for initialization', self.pair["LJ"], self.bond["harmonic"], self.angle["harmonic"])
                    run_potential = self.potential('# for equalibrium', self.pair["LJ"], self.bond["harmonic"], self.angle["hybrid"])

                #f'# fix		 	        BOX all deform 1 x final 0.0 {Init.Lbox} y final 0.0 {Init.Lbox} units box remap x',
                detach_configure = self.configure('# for configuration', 0.1, 0.01, Run)
                initial_fix = self.fix('# for initialization',  1.0, 1.0, Run)
                equal_fix = self.fix('# for equalibrium', Run.Temp, Run.Damp, Run)
                data_fix = self.fix('# for data', Run.Temp, Run.Damp, Run)
                refine_read = [
                    '# for refine',
                    '##################################################################',
                    f'read_dump        {self.iofile("dump", self.dump["equ"])} {Run.Tequ} {self.dump_read} wrapped no format native',
                    '',
                ]
                # run
                v_init, v_equ, v_data, v_refine = self.dump["init"], self.dump["equ"], self.dump["data"], self.dump["refine"]
                initial_run = self.run(v_init, Run.Tinit, Run.Tinit//20, Run)
                equal_run = self.run(v_equ, Run.Tequ, Run.Tequ//200, Run)
                data_run = self.run(v_data, Run.TSteps, Run.TSteps//Run.Frames, Run)
                refine_run = self.run(v_refine, Run.Tref, Run.Tref//Run.Frames, Run)

                # Define LAMMPS 参数
                with open(f"{dir_file}.in", "w") as file:
                    self.write_section(file, self.setup(Config.Dimend, dir_file, read))
                    if not Run.Params["restart"][0]:
                        if Init.Env == "Rand":
                            self.write_section(file, detach_potential)
                            self.write_section(file, detach_configure)
                        self.write_section(file, initial_potential)
                        self.write_section(file, initial_fix)
                        self.write_section(file, initial_run)
                    self.write_section(file, run_potential)
                    self.write_section(file, equal_fix)
                    self.write_section(file, equal_run)
                    self.write_section(file, data_fix)
                    self.write_section(file, data_run)
                    self.write_section(file, refine_read)
                    self.write_section(file, refine_run)
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                raise ValueError(f"An error occurred: {e}")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#############################################################################################################

class _path:
    def __init__(self, Model):
        #host = os.getcwd()
        self.host = os.path.abspath(os.path.dirname(os.getcwd()))
        self.mydirs = ["Codes", "Simus", "Anas", "Figs"]
        self.Model = Model
        self.Init = Model.Init
        self.Config = Model.Init.Config
        self.Run = Model.Run
        self.jump = self.build_paths()
        self.filename = os.path.basename(os.path.abspath(__file__))

    def build_paths(self):
        self.simus = os.path.join(self.host, self.mydirs[1])
        for dir in self.mydirs:
            self.host_dir = os.path.join(self.host, dir)
            os.makedirs(self.host_dir, exist_ok=True)
        #2D_100G_1.0Pe_Chain
        self.dir1= f"{self.Config.Dimend}D_{self.Run.Gamma:.1f}G_{self.Model.Pe}Pe_{self.Config.Type}"
        #5.0R5.0_100N1_Anulus
        if self.Init.Env == "Free":
            self.dir2 = f"{self.Init.N_monos}N{self.Init.num_chains}_{self.Init.Env}"
        else:
            self.dir2 = f"{self.Init.Rin}R{self.Init.Wid}_{self.Init.N_monos}N{self.Init.num_chains}_{self.Init.Env}"

        #1.0T_0.0Xi_8T5
        self.dir3 = f"{self.Run.Temp}T_{self.Model.Xi}Xi_{self.Run.eSteps}T{self.Run.Trun}"
        if self.Config.Type == _BACT:
            self.Jobname = f"{self.Model.Pe}Pe_{self.Config.Dimend}{self.Config.Type[0].upper()}{self.Init.Env[0].upper()}"
        else:
            self.Jobname = f"{self.Init.N_monos}N_{self.Config.Dimend}{self.Config.Type[0].upper()}{self.Init.Env[0].upper()}"
        #/Users/wukong/Data/Simus/2D_100G_1.0Pe_Chain/5.0R5.0_100N1_Anulus/1.0T_0.0Xi_8T5
        self.simus = os.path.join(self.simus, self.dir1, f'{self.dir2}', self.dir3)
        #/Users/wukong/Data/Simus/2D_100G_1.0Pe_Chain/5.0R5.0_100N1_Anulus/1.0T_0.0Xi_8T5/5.0R5.0_100N1_CA.data

        #Anas and Figures
        self.fig0 = self.simus.replace(self.mydirs[1], self.mydirs[3])
        self.lmp_trj = os.path.join(self.simus, f"{self.Run.Trun:03}.lammpstrj")
        return os.path.exists(self.lmp_trj)

    def show(self):
        print(f"host: {self.host}\nmydirs: {self.mydirs}\n"
              f"Path.dir1: {self.dir1}\nPath.dir2: {self.dir2}\nPath.dir3: {self.dir3}\n"
              f"Path.simus: {self.simus}\nPath.fig0: {self.fig0}")
#############################################################################################################

class _anas:
    def __init__(self, Path, Pe):
        self.Path, self.Init, self.Run, self.Config = Path, Path.Init, Path.Run, Path.Config
        self.Pe = Pe
        self.chunk = 9
        self.simp = 10
        self.jump = True

    def set_dump(self):
        is_bacteria = (self.Config.Type == _BACT)
        dump = {
            2: "xu yu" + (" vx vy" if is_bacteria else ""),
            3: "xu yu zu" + (" vx vy vz" if is_bacteria else "")
        }
        try:
            return dump[self.Config.Dimend].split(" ")
        except KeyError:
            logging.error(f"Error: Wrong Dimension to run => dimension != {self.Config.Dimend}")
            raise ValueError(f"Invalid dimension: {self.Config.Dimend}")

    def unwrap_x(self, Lx):
        frames = self.data.shape[1]
        for i in range(1, frames):
            while True:
                dx = self.data[:, i, :, 0] - self.data[:, i - 1, :, 0]
                crossed = np.abs(dx) > Lx / 2
                if not np.any(crossed):
                    break
                for file_idx in range(self.data.shape[0]):
                    for atom_idx in range(self.data.shape[2]):
                        if crossed[file_idx, atom_idx]:
                            self.data[file_idx, i:, atom_idx, 0] -= (np.sign(dx) * Lx * crossed)[file_idx, atom_idx]
        return self.data

    def read_data(self):
        timer = Timer("Read")
        timer.start()
        dump = self.set_dump()
        lox, hix = -self.Init.Lbox * 1025, self.Init.Lbox * 1023

        self.data = np.zeros((self.Run.Trun, self.Run.Frames+1, self.Init.num_monos, len(dump)))
        # read the lammpstrj files with 2001 frames
        print(f"{self.Path.simus}")
        for index, ifile in enumerate([f"{i:03}" for i in range(1, self.Run.Trun + 1)]):
            dir_file = os.path.join(f"{self.Path.simus}", f"{ifile}.lammpstrj")
            logging.info(f"==> Reading {ifile}.lammpstrj file: ......")
            print(f"==> Reading {ifile}.lammpstrj file: ......")
            # extract natoms, time steps, and check the last time step
            names = list(pd.read_csv(dir_file, skiprows=7, nrows=0, delim_whitespace=True, header=1).columns[2:])
            natoms = pd.read_csv(dir_file, skiprows=3, nrows=1, delim_whitespace=True, header=None).iloc[0][0]
            dstep = pd.read_csv(dir_file, skiprows=self.Init.num_monos + self.chunk + 1, nrows=1, delim_whitespace=True, header=None).iloc[0][0]
            lastStep = pd.read_csv(dir_file, skiprows=self.Run.Frames * (self.Init.num_monos + self.chunk) + 1, nrows=1, delim_whitespace=True, header=None).iloc[0][0]
            if natoms != self.Init.num_monos:
                logging.error(f"ERROR: Wrong atoms => {natoms} != {self.Init.num_monos}")
                raise ValueError(f"ERROR: Wrong atoms => {natoms} != {self.Init.num_monos}")
            elif lastStep != self.Run.Frames * self.Run.Tdump:
                logging.error(f"ERROR: Wrong timesteps => {lastStep} != {self.Run.Frames} * {self.Run.Tdump}")
                raise ValueError(f"ERROR: Wrong timesteps => {lastStep} != {self.Run.Frames} * {self.Run.Tdump}")
            skiprows = np.array(list(map(lambda x: np.arange(self.chunk) + (self.Init.num_monos + self.chunk) * x, np.arange(self.Run.Frames+1)))).ravel()
            try:
                df = pd.read_csv(dir_file, skiprows=skiprows, delim_whitespace=True, header=None, names=names, usecols=dump)
            except Exception as e:
                logging.error(f"ERROR: Could not read the file due to {e}")
                raise ValueError(f"ERROR: Could not read the file due to {e}")

            # data[ifile][iframe][iatom][id, xu, yu]
            self.data[index] = df.to_numpy().reshape((self.Run.Frames+1, self.Init.num_monos, len(dump)))
            #np.save(self.Path.simus, data)

        # unwrap data
        Lx = hix - lox
        if self.Pe > 50:
            self.unwrap_x(Lx)  # get real coordinates
        # data = np.copy(self.data)
        # for i in range(1, frames):
        # dx = data[:, i, :, 0] - data[:, i - 1, :, 0]
        # data[:, i, :, 0] -= np.sign(dx) * Lx * (np.abs(dx) > Lx / 2)
        timer.count("Read and unwrap data")
        timer.stop()
        return self.data

    def set_data(self, str="Com", flag=False):
        self.describe = str
        dict = {}
        files, frames, atoms, _ = self.data.shape
        # data[ifile][iframe][iatom][xu, yu]
        data_com = self.data - np.expand_dims(np.mean(self.data, axis=2), axis=2)
        data = np.mean(data_com, axis=0)

        # debug
        #describe(self.data, "self.data", flag=False)
        # describe(data_com, "data_com", flag=False)
        #describe(data, "data")
        # center of mass and average of files

        if atoms != self.Init.num_monos or frames != (self.Run.Frames+1):
            message = f"Wrong Atoms or Frames: atoms != num_monos => {atoms} != {self.Init.num_monos}; frames != Frames => {frames} != {self.Run.Frames+1}"
            print(message)
            logging.info(message)
        times, ids, modules = np.arange(frames)*self.Run.dt*self.Run.Tdump, np.arange(1,atoms+1,1), np.linalg.norm(data[ ..., :], axis = -1)

        if flag:
            data = data[::self.simp, ...]
            frames = data.shape[0]
            times, modules = np.arange(frames)*self.Run.dt*self.Run.Tdump * self.simp, np.linalg.norm(data[ ..., :], axis = -1)

        dict = {
            #"x": [np.random.normal(0, 1, frames), np.random.normal(0, 1, frames * atoms), ],  # Frame numbers (t coordinate)
            #"y": [np.random.random(atoms), np.random.random(frames * atoms),],  # Particle IDs (s coordinate)
            #"z": [modules, modules.flatten(),],  # Magnitude (r coordinate)
            "t": [times, np.repeat(times, atoms),],  # Frame numbers (t coordinate)
            "s": [ids, np.tile(ids, frames),],  # Particle IDs (s coordinate)
            "r": [modules, modules.flatten(),],  # Magnitude (r coordinate)
        }
        return dict

    def save_data(self):
        # data[ifile][iframe][iatom][xu, yu]
        self.read_data()
        #print(f"data.shape: {self.data.shape}")
        data_com = self.data - np.expand_dims(np.mean(self.data, axis=2), axis=2)
        data_Rcom = np.linalg.norm(data_com[ ..., :], axis = -1)

        # Rg2_time[iframe]
        Rg2_time = np.mean(np.mean(data_Rcom ** 2, axis=-1), axis=0) #average over atoms and files
        Rg2 = np.mean(Rg2_time) # average over time
        message = f"saving Rg2_time: {self.Path.fig0}"
        print(message)
        logging.info(message)
        np.save(os.path.join(self.Path.fig0, "Rg2_time.npy"), Rg2_time)

        #MSD

class _plot:
    def __init__(self, Path, bins = 20):
        self.Path, self.Init, self.Run, self.Config = Path, Path.Init, Path.Run, Path.Config
        self.runfile = "dana_org"
        self.num_bins = bins
        self.num_pdf = bins*10
        self.jump = True

    def original(self):
        timer = Timer("Original")
        timer.start()
        # ----------------------------> prepare data <----------------------------#
        keys, values = list(self.data_dict.keys()), list(self.data_dict.values())
        x_label, y_label, z_label = keys[0], keys[1], keys[2]
        x, y, z = self.data_dict[keys[0]][1], self.data_dict[keys[1]][1], self.data_dict[keys[2]][1]
        simp_x, simp_y, simp_z = self.simp_dict[keys[0]][1], self.simp_dict[keys[1]][1], self.simp_dict[keys[2]][1]

        # Calculate bin size and mid-bin values
        bin_size_z = (z.max() - z.min()) / 200
        bin_size_y = (y.max() - y.min()) / 200
        bin_size_x = (x.max() - x.min()) / 200
        sorted_x, sorted_y, sorted_z = np.sort(x), np.sort(y), np.sort(z)
        mid_x = sorted_x[np.argmin(np.abs(sorted_x - (x.max() - x.min())/2))]
        mid_y = sorted_y[np.argmin(np.abs(sorted_y - (y.max() - y.min())/2))]
        mid_z = sorted_z[np.argmin(np.abs(sorted_z - (z.max() - z.min())/2))]
        data_e = np.column_stack([x, y, z])[(z >= mid_z - bin_size_z / 2) & (z <= mid_z + bin_size_z / 2)]
        data_f = np.column_stack([x, z, y])[(y >= mid_y - bin_size_y / 2) & (y <= mid_y + bin_size_y / 2)]
        data_g = np.column_stack([y, z, x])[(x >= mid_x - bin_size_x / 2) & (x <= mid_x + bin_size_x / 2)]

        unique_coords_e, _, _, counts_e = statistics(data_e)
        unique_coords_f, _, _, counts_f = statistics(data_f)
        unique_coords_g, _, _, counts_g = statistics(data_g)

        #----------------------------> figure settings <----------------------------#
        fig_save = os.path.join(f"{self.Path.fig0}", f"{self.describe}.Org.({z_label},{x_label},{y_label})")

        if os.path.exists(f"{fig_save}.pdf") and self.jump:
            print(f"==>{fig_save}.pdf is already!")
            logging.info(f"==>{fig_save}.pdf is already!")
            return True
        else:
            print(f"{fig_save}.pdf")
            logging.info(f"{fig_save}.pdf")
        pdf = PdfPages(f"{fig_save}.pdf")
        #plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
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
        ax_a.set_xlabel(x_label, fontsize=20, labelpad=10)
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
            unique_coords, mean_values, std_values = statistics(data)[:3]
            sc = ax.scatter(unique_coords[:, 0], unique_coords[:, 1], c=mean_values, s=(std_values + 1) * 10, cmap='rainbow', alpha=0.7)

            # axis settings
            ax.set_title(fr'$\langle\ {axis_labels[2]}\ \rangle$ in {axis_labels[0]}-{axis_labels[1]} Space', loc='right', fontsize=20)
            ax.tick_params(axis='x', rotation=-45)
            ax.set_xlabel(axis_labels[0], fontsize=20)
            ax.set_ylabel(axis_labels[1], fontsize=20)
            ax.set_xlim(min(unique_coords[:, 0]), max(unique_coords[:, 0]))
            ax.set_ylim(min(unique_coords[:, 1]), max(unique_coords[:, 1]))

            # colorbar
            axpos = ax.get_position()
            caxpos = mtransforms.Bbox.from_extents(axpos.x1 + 0.005, axpos.y0, axpos.x1 + 0.015, axpos.y1)
            cax = fig.add_axes(caxpos)
            cbar = plt.colorbar(sc, ax=ax, cax=cax)
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.ax.set_xlabel(axis_labels[2], fontsize=20)

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
                                                      [({np.around(mid_z - bin_size_z / 2, 2)}, {np.around(mid_z + bin_size_z / 2, 2)}),
                                                       ({np.around(mid_y - bin_size_y / 2, 2)}, {np.around(mid_y + bin_size_y / 2, 2)}),
                                                       ({np.around(mid_x - bin_size_x / 2, 2)}, {np.around(mid_x + bin_size_x / 2, 2)})],
                                                      [counts_e, counts_f, counts_g],
                                                      [(x_label, y_label, z_label), (x_label, z_label, y_label), (y_label, z_label, x_label)],
                                                      ['(e)', '(f)', '(g)']):

            sc = ax.scatter(data[:, 0], data[:, 1], s=counts*50, color="blue", alpha=0.6)
            # axis settings
            ax.set_title(fr'${axis_labels[2]}_0\ \in$ [{bin[0]}, {bin[1]}]', loc='right', fontsize=20)
            ax.set_xlabel(axis_labels[0], fontsize=20)
            ax.set_ylabel(axis_labels[1], fontsize=20)

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

        # ----------------------------> save fig <----------------------------#
        fig0 = plt.gcf()
        pdf.savefig(fig0, dpi=300, transparent=True)
        pdf.close()

        # ax.legend(loc='upper left', frameon=False, ncol=int(np.ceil(len(Arg1) / 5.)), columnspacing = 0.1, labelspacing = 0.1, bbox_to_anchor=[0.0, 0.955], fontsize=10)
        #fig0.savefig(f"{fig_save}.png", format="png", dpi=1000, transparent=True)
        timer.count("saving figure")
        plt.show()
        plt.close()
        timer.stop()
        # -------------------------------Done!----------------------------------------#
        return False

    def distribution(self):
        timer = Timer("Distribution")
        timer.start()
        # ----------------------------> prepare data <----------------------------#
        bin_id, pdf_id = self.num_bins // 2, self.num_pdf // 2
        keys, values = list(self.data_dict.keys()), list(self.data_dict.values())
        for i, j, k in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
            x_label, y_label, z_label = keys[i], keys[j], keys[k]
            x, y, z = values[i][1], values[j][1], values[k][1]
            # Create a 2D histogram and bin centers
            hist_x, x_bins = np.histogram(x, bins=self.num_bins, density=True)
            hist_y, y_bins = np.histogram(y, bins=self.num_bins, density=True)
            x_bin_centers, y_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2, (y_bins[:-1] + y_bins[1:]) / 2
            x_range, y_range = np.linspace(min(x), max(x), self.num_pdf), np.linspace(min(y), max(y), self.num_pdf)
            pdf_x, pdf_y = gaussian_kde(x).evaluate(x_range), gaussian_kde(y).evaluate(y_range)
            hist_2D = np.histogram2d(x, y, bins=[x_bins, y_bins], density=True)[0]
            #specific y or x
            hist_x_at_y, hist_y_at_x = hist_2D[:, bin_id]/np.sum(hist_2D[:, bin_id]), hist_2D[bin_id, :]/np.sum(hist_2D[bin_id, :])
            #hist_x_at_y, hist_y_at_x = np.nan_to_num(hist_x_at_y), np.nan_to_num(hist_y_at_x)

            #----------------------------> figure settings <----------------------------#
            fig_save = os.path.join(f"{self.Path.fig0}", f"{self.describe}.Dist.f^{z_label}({x_label},{y_label})")
            if os.path.exists(f"{fig_save}.pdf") and self.jump:
                print(f"==>{fig_save}.pdf is already!")
                logging.info(f"==>{fig_save}.pdf is already!")
                return True
            else:
                print(f"{fig_save}.pdf")
                logging.info(f"{fig_save}.pdf")
            pdf = PdfPages(f"{fig_save}.pdf")

            #plt.clf()
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.labelsize'] = 15
            plt.rcParams['ytick.labelsize'] = 15

            # ----------------------------> plot figure<----------------------------#
            #plt.subplots_adjust(left=0.15, right=0.85, bottom=0.13, top=0.9, wspace=0.2, hspace=0.2)
            # Create the layout
            fig = plt.figure(figsize=(18, 9))
            fig.subplots_adjust(wspace=0.5, hspace=0.5)
            gs = GridSpec(2, 4, figure=fig)
            ax_a = fig.add_subplot(gs[0:2, 0:2])
            ax_b = fig.add_subplot(gs[0, 2], sharex=ax_a)
            ax_c = fig.add_subplot(gs[0, 3], sharex=ax_a)
            ax_d = fig.add_subplot(gs[1, 2], sharey=ax_a)
            ax_e = fig.add_subplot(gs[1, 3], sharey=ax_a)

            # Plot fz(x,y)
            cmap = ax_a.pcolormesh(x_bins, y_bins, hist_2D.T, shading='auto', cmap='rainbow')
            # Plot Fz(x;y0)
            ax_b.bar(x_bin_centers, hist_x_at_y, width=(x_bins[1] - x_bins[0]), alpha = 0.7, label="histogram")

            # Plot Fzy(x)
            ax_c.bar(x_bin_centers, hist_x, width=(x_bins[1] - x_bins[0]), alpha = 0.7, label="histogram")
            ax_c.plot(x_range, pdf_x, 'r', label='PDF')
            # Plot Fz(y;x0)
            ax_d.barh(y_bin_centers, hist_y_at_x, height=(y_bins[1] - y_bins[0]), alpha = 0.7, label="histogram")
            # Plot Fzx(y)
            ax_e.barh(y_bin_centers, hist_y, height=(y_bins[1] - y_bins[0]), alpha = 0.7, label="histogram")
            ax_e.plot(pdf_y, y_range, 'r', label='PDF')

            # ----------------------------> adding <----------------------------#
            ax_a.axhline(y=y_bin_centers[bin_id], linestyle='--', lw = 1.5, color='black')  # Selected Particle ID
            ax_a.axvline(x=x_bin_centers[bin_id], linestyle='--', lw = 1.5, color='black')  # Selected Time frame

            axpos = ax_a.get_position()
            caxpos = mtransforms.Bbox.from_extents(axpos.x0 - 0.07, axpos.y0, axpos.x0 - 0.05, axpos.y1)
            cax = fig.add_axes(caxpos)
            cbar = plt.colorbar(cmap, cax=cax)
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.set_xlabel(fr"$f^{z_label}({x_label},{y_label})$", fontsize=20)

            # ----------------------------> axis settings <----------------------------#
            ax_a.set_title(fr"$f^{z_label}({x_label},{y_label})$", fontsize=20)
            ax_a.set_xlabel(f"{x_label}", fontsize=20)
            ax_a.set_ylabel(f"{y_label}", fontsize=20)
            ax_a.set_xlim(x_bin_centers[0], x_bin_centers[-1])
            ax_a.set_ylim(y_bin_centers[0], y_bin_centers[-1])

            ax_b.set_title(fr"${y_label}_0$ = {y_bin_centers[bin_id]:.2f}", loc='right', fontsize=20)
            ax_b.set_xlabel(f"{x_label}", fontsize=20)
            ax_b.tick_params(axis='x', rotation=45)
            ax_b.set_ylabel(fr"$f^{z_label}({x_label}; {y_label}_0)$", fontsize=20)
            ax_b.set_ylim(0, max(hist_x_at_y) * 1.1)
            ax_c.set_title("Distribution", loc='right', fontsize=20)
            ax_c.set_xlabel(f"{x_label}", fontsize=20)
            ax_c.tick_params(axis='x', rotation=45)
            ax_c.set_ylabel(fr"$f^{z_label}_{y_label}({x_label})$", fontsize=20)
            ax_c.set_ylim(0, max(hist_x) *1.1)

            ax_d.set_title(fr"${x_label}_0$ = {x_bin_centers[bin_id]:.2f}", loc='right', fontsize=20)
            ax_d.set_xlabel(fr"$f^{z_label}({y_label}; {x_label}_0)$", fontsize=20)
            ax_d.set_ylabel(f"{y_label}", fontsize=20)
            ax_d.set_xlim(0, max(hist_y_at_x)*1.1)

            ax_e.set_title('Distribution', loc='right', fontsize=20)
            ax_e.set_xlabel(fr"$f^{z_label}_{x_label}({y_label})$", fontsize=20)
            ax_e.set_ylabel(f"{y_label}", fontsize=20)
            ax_e.set_xlim(0, max(hist_y)*1.1)

            # ----------------------------> linewidth <----------------------------#
            for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e], ['(a)', '(b)', '(c)', '(d)', '(e)',]):
                ax.annotate(label, (-0.3, 0.9), textcoords="axes fraction", xycoords="axes fraction", va="center", ha="center", fontsize=20)
                ax.tick_params(axis='both', which="major", width=2, labelsize=15, pad=7.0)
                ax.tick_params(axis='both', which="minor", width=2, labelsize=15, pad=4.0)
                # ----------------------------> axes lines <----------------------------#
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)
                ax.spines['top'].set_linewidth(2)

            timer.count("saving figure")
            # ----------------------------> save fig <----------------------------#
            #plt.tight_layout()
            fig0 = plt.gcf()
            pdf.savefig(fig0, dpi=1000, transparent=True)
            pdf.close()

            #print("saving png......")
            # ax.legend(loc='upper left', frameon=False, ncol=int(np.ceil(len(Arg1) / 5.)), columnspacing = 0.1, labelspacing = 0.1, bbox_to_anchor=[0.0, 0.955], fontsize=10)
            #fig0.savefig(f"{fig_save}.png", format="png", dpi=1000, transparent=True)
            plt.show()
            plt.close()
            # -------------------------------Done!----------------------------------------#
        timer.stop()
        return False

    def distics(self):
        print("-----------------------------------Done!--------------------------------------------")
    ##################################################################
    def MSD(self):
        print("-----------------------------------Done!--------------------------------------------")
    def Rg(self):
        Rg2_time = np.load(os.path.join(self.Path.fig0, "Rg2_time.npy")) #average over atoms and files


        # Plotting the Rg2 for the first file across all frames
        plt.plot(Rg2_time[:])
        plt.xlabel('time')
        plt.ylabel(fr'$R_g^2$')
        plt.title(fr'$R_g^2$ average over files')
        plt.show()

        print("-----------------------------------Done!--------------------------------------------")
    def Cee(self):
        print("-----------------------------------Done!--------------------------------------------")
    ##################################################################
    def plot(self):
        timer = Timer("Plot")
        timer.start()
        if HOST == "Darwin": #MacOS
            self.jump = False
        self.read_data()
        self.data_dict, self.simp_dict = self.set_data("Com"), self.set_data("Com", flag=True)
        #plotting
        self.original()
        self.distribution()

        timer.count("plot")
        timer.stop()

    def sub_file(self):
        Path = self.Path
        print(f">>> Preparing sub file......")
        logging.info(f">>> Preparing sub file......")
        dir_file = os.path.join(f"{Path.fig0}", f"{self.runfile}")
        bsub = [
            f'#!/bin/bash',
            f'',
            f'#BSUB -J {Path.Jobname}_Anas',
            f'#BSUB -Jd "Analysis: original data"',
            f'#BSUB -r',
            f'#BSUB -q {self.Run.Queue}',
            f'#BSUB -n 1',
            f'#BSUB -oo {dir_file}.out',
            f'#BSUB -eo {dir_file}.err',
            f'source ~/.bashrc',
            f'cd {Path.fig0}',
            f'echo "python3 {dir_file}.py"',
            f'python3 {dir_file}.py',
        ]
        with open(f"{dir_file}.lsf", "w") as file:
            for command in bsub:
                file.write(command + "\n")
        print("-----------------------------------Done!--------------------------------------------")
#############################################################################################################

class Timer:
    def __init__(self, tip = "start", func=time.perf_counter):
            self.tip = tip
            self.elapsed = 0.0
            self._func = func
            self._start = None
            self.time_dict = {}

    def start(self):
            self.elapsed = 0.0
            if self._start is not None:
                    raise RuntimeError('Already started')
            self._start = self._func()
            print(f"-------------------------------{self.tip}----------------------------------------")

    def stop(self):
            if self._start is None:
                    raise RuntimeError('Not started')
            end = self._func()
            self.elapsed += end - self._start
            print(f"-------------------------------{self.tip}: Done!----------------------------------------")
            #logging.info(str, ":", self.elapsed)
            self._start = None

    def count(self, str="Time"):
            end = self._func()
            self.elapsed += end - self._start
            self.time_dict[str] = self.elapsed
            print(f"{str}: {self.elapsed}")
            self.elapsed = 0.0
            self._start = self._func()

    def reset(self):
            self.elapsed = 0.0
            
    @property
    def running(self):
            return self._start is not None    
    def __enter__(self):
            self.start()
            return self
    
    def __exit__(self, *args):
            self.stop()

#############################################################################################################
def convert2array(x):
    # 如果x已经是一个列表，numpy数组，直接转换为numpy数组
    if isinstance(x, (list, np.ndarray, tuple)):
        return np.array(x)
    # 如果x是一个单一的数值，转换为包含该数值的numpy数组
    elif isinstance(x, (int, float, str)):
        return np.array([x])
    else:
        raise ValueError("Unsupported type!")

def statistics(data):
    '''for plot original'''
    unique_coords, indices, counts = np.unique(data[:, :2], axis=0, return_inverse=True, return_counts=True)
    sum_values = np.bincount(indices, weights=data[:, 2])
    mean_values = sum_values / counts
    sum_values_squared = np.bincount(indices, weights=data[:, 2] ** 2)
    std_values = np.sqrt( (sum_values_squared / counts) - (mean_values ** 2))
    return unique_coords, mean_values, std_values, counts

def describe(dataset, str="data", flag = True):
    data = dataset.flatten()
    print(f"==>{str}\n"
          f"shape:{dataset.shape}\n"
          f"max: {np.max(data)}\n"
          f"min: {np.min(data)}\n"
          f"mean: {np.mean(data)}\n"
          f"var: {np.var(data)}\n"
          f"first: {data[:10]}\n"
          f"middle: {data[len(data) // 2 - 5: len(data) // 2 + 5]}\n"
          f"last: {data[-10:]}\n"
          "------------------------------------------Done!----------------------------------------")
    if flag:
        sys.exit()

__all__ = [
    "params",
    "_config",
    "_run",
    "_init",
    "_model",
    "_path",
    "_plot",
    "Timer",
    "convert2array",
    "check_params",
    "describe",
]

# -----------------------------------bsub-------------------------------------------#
def exe_simus(task, path, infile):
    dir_file = os.path.join(path, infile)
    if task == "Run":
        print(f">>> Running jobs: {infile}......")
        logging.info(f">>> Running jobs: {infile}......")

        print(f"mpirun -np 1 lmp_wk -i {dir_file}.in")
        subprocess.run(f"mpirun -np 1 lmp_wk -i {dir_file}.in", shell=True)
        print(f"{dir_file}.in ==> Done! \n")

    elif task == "Submit":
        print(f">>> Running jobs : {infile}......")
        logging.info(f">>> Running jobs : {infile}......")

        print(f"bsub < {dir_file}.lsf")
        subprocess.run(f"bsub < {dir_file}.lsf", shell=True)
        print(f"Submitted: {dir_file}")
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    else:
        message = f"ERROR: Wrong task => {task} != Run or Submit"
        print(message)
        logging.error(message)
        raise ValueError(message)

def sub_job(Path):
    # Initialize directories and files
    Init, Model, Run = Path.Init, Path.Model, Path.Run
    run_on_cluster = os.environ.get("RUN_ON_CLUSTER", "false")
    if Path.jump:  # jump for repeat: check lammpstrj
        return

    # Create simulation directory and copy Run.py
    message = f"dir_simus => {Path.simus}"
    print(message)
    logging.info(message)
    os.makedirs(Path.simus, exist_ok=True)
    shutil.copy2(
        os.path.join(Path.host, Path.mydirs[0], Path.filename),
        os.path.join(Path.simus, Path.filename))

    # prepare files
    infiles = [f"{i:03}" for i in range(1, Run.Trun + 1)]
    Init.data_file(Path)
    Model.in_file(Path)  # return
    Run.sub_file(Path, infiles)
    # excute jobs
    if input_file:
        # running files
        if HOST == "Darwin":  # macOS
            exe_simus("Run", Path.simus, input_file)
        elif HOST == "Linux" and run_on_cluster == "false":  # 登陆节点
            exe_simus("Submit", Path.simus, input_file)
    else:
        for infile in infiles:
            # running files
            if HOST == "Darwin":  # macOS
                exe_simus("Run", Path.simus, infile)
            # submitting files
            elif HOST == "Linux" and run_on_cluster == "false":  # 登陆节点
                exe_simus("Submit", Path.simus, infile)

# -----------------------------------Act-----------------v--------------------------#
def check_params(params):
    print("===> Caveats: Please confirm the following parameters(with Dimend, Type, Env fixed):")
    for key, value in islice(params.items(), 1, None):
        user_input = input(f"{key} = {value}    (y/n)?: ")
        if user_input.lower() == "y":
            break
        elif user_input.lower() == '':
            continue
        else:
            new_value = input(f"Please enter the new value for {key}: ")
            params[key] = type(value)(new_value)  # 更新值，并尝试保持原来的数据类型
    return params

def simus(params):
    check = True
    for iDimend in convert2array(params['Dimend']):
        for iType in convert2array(params['labels']['Types']):
            for iEnv in convert2array(params['labels']['Envs']):
                Config = _config(iDimend, iType, iEnv, params)
                if platform.system() == "Linux":  # params['task'] == "Simus" and
                    mpl.use("agg")
                    if check:
                        params = check_params(params)
                        check = False

                for iGamma in convert2array(params['Gamma']):
                    for iTemp in convert2array(params['Temp']):
                        # paras for run: Gamma, Temp, Queue, Frames, Trun, Dimend, Temp, Dump
                        Run = _run(Config.Dimend, iGamma, iTemp, params['Trun'])  # print(Run.__dict__)
                        Config.set_dump(Run)  # print(f"{params['marks']['config']}, {Run.Dump}, {Run.Tdump}")
                        for iRin in convert2array(params['Rin']):
                            for iWid in convert2array(params['Wid']):
                                for iN in convert2array(params['N_monos']):
                                    # paras for init config: Rin, Wid, N_monos, L_box
                                    Init = _init(Config, Run.Trun, iRin, iWid, iN)
                                    if Init.jump:  # chains are too long or invalid label
                                        continue
                                    queue = Run.set_queue()  # print(f"{queue}\n", params["Queues"])
                                    for iFa in convert2array(params['Fa']):
                                        for iXi in convert2array(params['Xi']):
                                            # paras for model: Pe(Fa), Xi(Kb)
                                            Model = _model(Init, Run, iFa, iXi)
                                            Path = _path(Model)  # for directory
                                            sub_job(Path)

if __name__ == "__main__":
    task = params["task"]
    usage = "Run.py infile or bsub < infile.lsf"
    print(usage)
    # simulation or analysis
    ###################################################################
    # Simulations
    if task == "Simus":
        if "Codes" in CURRENT_DIR:
            simus(params)
        elif "Simus" in CURRENT_DIR: # 计算节点: "Run.py infile" == "bsub < infile.lsf"
            try:
                if input_file:
                    exe_simus("Run", CURRENT_DIR, input_file)
                else:
                    for infile in [f"{i:03}" for i in range(1, params['Trun'] + 1)]:
                        exe_simus("Run", CURRENT_DIR, infile)
            except Exception as e:
                print(usage)
                logging.error(f"An error occurred: {e}")
                raise ValueError(f"An error occurred: {e}")

    # Analysis
    elif task == "Anas":
        if "Codes" in CURRENT_DIR:
            check = True
            for iDimend in convert2array(params['Dimend']):
                for iType in convert2array(params['labels']['Types']):
                    for iEnv in convert2array(params['labels']['Envs']):
                        Config = _config(iDimend, iType, iEnv, params)
                        if platform.system() == "Linux":  # params['task'] == "Simus" and
                            mpl.use("agg")
                            if check:
                                check_params(params)
                                check = False

                        for iGamma in convert2array(params['Gamma']):
                            for iTemp in convert2array(params['Temp']):
                                # paras for run: Gamma, Temp, Queue, Frames, Trun, Dimend, Temp, Dump
                                Run = _run(Config.Dimend, iGamma, iTemp, params['Trun'])  # print(Run.__dict__)
                                Config.set_dump(Run)  # print(f"{params['marks']['config']}, {Run.Dump}, {Run.Tdump}")
                                for iRin in convert2array(params['Rin']):
                                    for iWid in convert2array(params['Wid']):
                                        for iN in convert2array(params['N_monos']):
                                            # paras for init config: Rin, Wid, N_monos, L_box
                                            Init = _init(Config, Run.Trun, iRin, iWid, iN)
                                            if Init.jump:  # chains are too long or invalid label
                                                continue
                                            queue = Run.set_queue()  # print(f"{queue}\n", params["Queues"])
                                            for iFa in convert2array(params['Fa']):
                                                for iXi in convert2array(params['Xi']):
                                                    # paras for model: Pe(Fa), Xi(Kb)
                                                    Model = _model(Init, Run, iFa, iXi)
                                                    Path = _path(Model)  # for directory
                                                    ###################################################################
                                                    task = params["task"]
                                                    Init, Model, Run = Path.Init, Path.Model, Path.Run
                                                    run_on_cluster = os.environ.get("RUN_ON_CLUSTER", "false")

                                                    if Path.jump:  # according to lammpstrj files
                                                        Anas = _anas(Path, Model.Pe)
                                                        Plot = _plot(Anas)

                                                        # copy Run.py
                                                        message = f"dir_figs => {Path.fig0}"
                                                        os.makedirs(Path.fig0, exist_ok=True)
                                                        shutil.copy2(os.path.join(Path.host, Path.mydirs[0], Path.filename),
                                                                     os.path.join(Path.fig0, f"{Plot.runfile}.py"))
                                                        print(message)
                                                        logging.info(message)

                                                        # prepare files
                                                        #Plot.sub_file()

                                                        # running files
                                                        dir_file = os.path.join(f"{Path.fig0}", f"{Plot.runfile}")
                                                        if HOST == "Darwin":  # macOS
                                                            print(">>> Plotting tests......")
                                                            logging.info(">>> Plotting tests......")
                                                            #Plot.plot()
                                                            Anas.save_data()
                                                            print(f"==> Done! \n ==> Please check the results and submit the plots!")

                                                        # submitting files
                                                        elif HOST == "Linux" and run_on_cluster == "false":  # 登陆节点
                                                            print(">>> Submitting plots......")
                                                            logging.info(">>> Submitting plots......")
                                                            print(f"bsub < {dir_file}.lsf")
                                                            subprocess.run(f"bsub < {dir_file}.lsf", shell=True)
                                                            print(f"Submitted: {dir_file}.py")
                                                    else:
                                                        message = f"File doesn't exist in data: {Path.lmp_trj}"
                                                        print(message)
                                                        logging.info(message)
        elif "Anas" in CURRENT_DIR:
            print(">>> Plotting tests......")
            logging.info(">>> Plotting tests......")

        elif "Figs" in CURRENT_DIR:
            print(">>> Plotting tests......")
            logging.info(">>> Plotting tests......")
            Plot.plot()
            print(f"==> Done! \n ==> Please check the results and submit the plots!")