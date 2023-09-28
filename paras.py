import os
import re
import sys
import subprocess
import shutil
import platform
import time
import logging
from datetime import datetime, timedelta
from itertools import combinations, product

import pandas as pd
import numpy as np
from numba import vectorize, float64
from scipy.interpolate import splrep, splev
from scipy.stats import norm
from scipy.spatial import KDTree
from collections import defaultdict

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages as PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms
logging.basicConfig(filename='paras.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#-----------------------------------Const-------------------------------------------
_BACT = "Bacteria"
HOST = platform.system()
#-----------------------------------Parameters-------------------------------------------
types = ["Chain", _BACT, "Ring"]
envs = ["Anlus", "Rand", "Slit"]
tasks = ["Simus", "Anas"]
#-----------------------------------Dictionary-------------------------------------------
#参数字典
params = {
    'labels': {'Types': types[0:1], 'Envs': envs[0:1]},
    'marks': {'labels': [], 'config': []},
    'task': tasks[0],
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
                _BACT: {'N_monos': [3], 'Xi': 1000, 'Fa': [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0],},
                "Chain": {'N_monos': [20, 40, 80, 100, 150, 200, 250, 300], 'Xi': 0.0, 'Fa': [1.0], #'Fa': [0.0, 0.1, 1.0, 5.0, 10.0],
                          'Temp': [0.2, 0.1, 0.05, 0.01], #'Temp': [1.0],
                          # 'Gamma': [0.1, 1, 10, 100]
                          },
                "Ring": {'N_monos': [20, 40, 80, 100, 150, 200, 250, 300], 'Xi': 0.0, 'Fa': [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 100.0],
                         'Gamma': [0.1, 1, 10, 100],
                         # 'Temp': [1.0, 0.2, 0.1, 0.05, 0.01],
                         },

                "Anlus": {2: {'Rin': [0.0], 'Wid': [0.0]},
                              3: {'Rin': [0.0], 'Wid': [0.0]},
                              #2: {'Rin': [5.0, 10.0, 15.0, 20.0, 30.0], 'Wid': [5.0, 10.0, 15.0, 20.0, 30.0]},
                              #3: {'Rin': [5.0, 10.0, 15.0, 20.0, 30.0], 'Wid': [5.0, 10.0, 15.0, 20.0, 30.0]},
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
                "Chain": {'N_monos': [100], 'Xi': 0.0,
                              'Fa': [1.0]},
                "Ring": {'N_monos': [100], 'Xi': 0.0, 'Fa': [1.0], 'Gamma': [1.0]},

                "Anlus":{2: {'Rin': [10.0], 'Wid': [3.0]},
                            3: {'Rin': [10.0], 'Wid': [10.0]},
                            },
                "Rand": {2: {'Rin': 0.314,  'Wid': 1.0},
                             3: {'Rin': 0.0314, 'Wid': 2.5},
                            },
                "Slit": {2: {"Rin": [0.0], "Wid": [3.0]},
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
        if HOST == "Darwin":
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
    
    def bsubs(self, Path):
        Run = Path.Run
        logging.info(f">>> Preparing sub file: {Path.dir_data}")
        for infile in [f"{i:03}" for i in range(1, Run.Trun + 1)]:
            print(">>> Preparing sub file......")
            dir_file = os.path.join(f"{Path.dir_data}", infile)
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
                f'cd {Path.dir_data}',
                f'echo "mpirun -np 1 lmp_wk -i {infile}.in"',
                f'mpirun -np 1 lmp_wk -i {infile}.in',
            ]
            
            with open(f"{dir_file}.lsf", "w") as file:
                for command in bsub:
                    file.write(command + "\n")
            if HOST == "Darwin":
                print(">>> for test......")
                print(f"mpirun -np 4 lmp_wk -i {dir_file}.in")
                subprocess.run(f"mpirun -np 4 lmp_wk -i {dir_file}.in", shell=True)
                print(f"{dir_file}.in ==> Done! \n ==> Please check the results and submit the jobs!")
            elif HOST == "Linux":
                print(">>> Submitting jobs......")
                print(f"bsub < {dir_file}.lsf")
                subprocess.run(f"bsub < {dir_file}.lsf", shell=True)
                print(f"Submitted: {dir_file}")

    def run(self, Path, Plot):
        Init, Model = Path.Init, Path.Model
        task = self.Params["task"]
        if task == "Simus":
            try:
                # prepare files and submit
                Init.data_file(Path)
                Model.in_file(Path)
                self.bsubs(Path)
                #Run.bsubs(Path, 1)
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                raise ValueError(f"An error occurred: {e}")
        elif task == "Anas":
            try:
                print("Plot")
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                raise ValueError(f"An error occurred: {e}")
##########################################END!###############################################################

class _init:
    def __init__(self, Config, Trun, Rin, Wid, N_monos, num_chains = params["num_chains"]):
        self.sigma_equ, self.mass, self.sigma = 0.94, 1.0, 1.0
        self.Ks = 300.0
        self.Config, self.Trun = Config, Trun
        self.Rin, self.Wid = Rin, Wid
        self.particle_density = 2 if self.Config.Dimend == 3 else 2
        self.N_monos, self.num_chains = int(N_monos), num_chains
        self.num_monos = self.N_monos * self.num_chains

        self.Env = "Free" if (self.Rin < 1e-6 and self.Wid < 1e-6) else self.Config.Env
        self.jump = self.set_box()   #set box
        if (self.Config.Type == "Ring" and self.Env == "Anlus") and (self.Env == "Slit" and self.Rin > 1e-6):
            self.jump = True
            print(f"I'm sorry => '{self.Config.Label}' is not ready! when Dimend = {Params['Dimend']}")
            logging.warning(f"I'm sorry => '{self.Label}' is not ready!")

        if self.Config.Env == "Anlus":
            self.Rout = self.Rin + self.Wid  # outer_radius
            self.R_ring = self.Rin + self.Wid / 2
            self.R_torus = self.Wid / 2

            self.num_Rin = np.ceil(self.particle_density * 2 * np.pi * self.Rin / self.sigma)
            self.num_Rout = np.ceil(self.particle_density * 2 * np.pi * self.Rout / self.sigma)
            self.num_ring = np.ceil(self.particle_density * 2 * np.pi * self.R_ring / self.sigma)
            self.num_torus = np.ceil(self.particle_density * 2 * np.pi * self.R_torus / self.sigma)
            self.num_obs = int(self.num_ring) * int(self.num_torus) if self.Config.Dimend == 3 else int(self.num_Rin + self.num_Rout)

        elif self.Config.Env == "Rand":
            self.num_obs = int(np.ceil(self.Rin * self.v_box / self.v_obs))
        elif self.Config.Env == "Slit":
            self.num_obs = 0
        else:
            logging.error(f"Error: wrong environment! => self.Env = {self.Env}")
            raise ValueError(f"Error: wrong environment! => self.Env = {self.Env}")

        self.sigma12 = self.sigma + 2 * self.Wid if (self.Env == "Rand") else 2 * self.sigma
        self.Rchain = self.Rin + self.sigma + 0.5 if self.Env == "Anlus" else (self.Rin + self.sigma_equ + self.N_monos * self.sigma_equ/(2 * np.pi))
        self.dtheta_chain = self.set_dtheta(self.Rchain) if self.Env == "Anlus" else self.set_dtheta(self.Rchain, self.N_monos)
        self.theta0 = - 2 * self.dtheta_chain if self.Env == "Anlus" else - 4 * self.dtheta_chain
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
            self. Lbox = self.Rin + self.Wid + 1
            if (self.num_monos > np.pi * (self.Wid * self.Wid + self.Wid * ( 2 * self.Rin - 1) - 2 * self.Rin) ):
                print("N_monos is too Long!")
                logging.warning("N_monos is too Long!")
                return True
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

        return False
    
    def set_dtheta(self, R, num_R=None):
        """计算角度间隔"""
        if num_R is None:
            return 2 * np.pi / np.floor(2 * np.pi * R / self.sigma_equ)
        elif num_R == 0:
            return 0
        elif num_R < 10:
            return self.sigma_equ/R
        else:
            return 2 * np.pi / num_R

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

        file.write("Atoms\n\n")
        Rchain = self.Rchain
        dtheta_chain = self.dtheta_chain
        # 写入链的原子信息
        circle = 1
        theta = self.theta0
        phi = 0 # Only used in 3D

        for i in range(self.N_monos):
            x, y, z = 0.0, 0.0, 0.0
            if theta >= 2 * np.pi - 5 * circle * dtheta_chain:
                circle += 1
                Rchain += self.sigma_equ + 0.2
                dtheta_chain = 2 * np.pi / np.floor(2* np.pi * Rchain/self.sigma_equ)
                theta = theta - 2 * np.pi - dtheta_chain

            if self.Config.Dimend == 2:
                x = np.around(Rchain * np.cos(theta) * self.sigma_equ, 5)
                y = np.around(Rchain * np.sin(theta) * self.sigma_equ, 5)
            elif self.Config.Dimend == 3:
                x = np.around(Rchain * np.cos(phi) * np.cos(theta) * self.sigma_equ, 5)
                y = np.around(Rchain * np.cos(phi) * np.sin(theta) * self.sigma_equ, 5)
                z = np.around(Rchain * np.sin(phi) * self.sigma_equ, 5)
            theta += dtheta_chain

            # Write atom information
            atom_type = 2  # Default to "middle" of the chain
            if i == 0:
                atom_type = 1  # Head
            elif i == self.N_monos - 1:
                atom_type = 3  # Tail
            file.write(f"{i + 1} 1 {atom_type} {x} {y} {z}\n")

    def set_torus(self, R_torus, N_ring, N_torus=0):
        theta = np.linspace(0, 2 * np.pi, int(N_ring))
        phi = np.linspace(0, 2 * np.pi, int(N_torus)) if self.Config.Dimend == 3 else 0
        theta, phi = np.meshgrid(theta, phi)

        x =  np.around((self.R_ring + R_torus * np.cos(phi)) * np.cos(theta), 5)
        y =  np.around((self.R_ring + R_torus * np.cos(phi)) * np.sin(theta), 5)
        z =  np.around(R_torus * np.sin(phi), 5)

        return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

    def write_anlus(self, file):
        self.particles = None
        if self.Config.Dimend == 2:
            outer_ring = self.set_torus(self.R_torus, self.num_Rout)
            inner_ring = self.set_torus(-self.R_torus, self.num_Rin)
            for i, coord in enumerate(inner_ring):
                file.write(f"{self.N_monos+i+1} 1 4 {' '.join(map(str, coord))}\n")
            for i, coord in enumerate(outer_ring):
                file.write(f"{int(self.N_monos+self.num_Rin+i+1)} 1 5 {' '.join(map(str, coord))}\n")
        elif self.Config.Dimend == 3:
            torus = self.set_torus(self.R_torus, self.num_ring, self.num_torus)
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
        self.bound = self.Lbox - self.Wid - 0.56 * self.sigma
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
            data_file = os.path.join(f"{Path.dir_data}", f'{infile}.{self.Config.Type[0].upper()}{self.Env[0].upper()}.data')
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
        logging.info(f"==> Writing infile ==> {Path.dir_data}")

        for infile in [f"{i:03}" for i in range(1, Run.Trun + 1)]:
            print(f"==> Writing infile: {infile}......")
            try:
                #setup
                dir_file = os.path.join(f"{Path.dir_data}", infile)
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
        
    def build_paths(self):
        self.simus = os.path.join(self.host, self.mydirs[1])
        for dir in self.mydirs:
            self.host_dir = os.path.join(self.host, dir)
            subprocess.run(f"mkdir -p {self.host_dir}", shell=True)
        #2D_100G_1.0Pe_Chain
        self.dir1= f"{self.Config.Dimend}D_{self.Run.Gamma:.1f}G_{self.Model.Pe}Pe_{self.Config.Type}"
        #5.0R5.0_100N1_Anulus
        if self.Init.Env == "Free":
            self.dir2 = f"{self.Init.N_monos}N{self.Init.num_chains}"
        else:
            self.dir2 = f"{self.Init.Rin}R{self.Init.Wid}_{self.Init.N_monos}N{self.Init.num_chains}"

        #1.0T_0.0Xi_8T5
        self.dir3 = f"{self.Run.Temp}T_{self.Model.Xi}Xi_{self.Run.eSteps}T{self.Run.Trun}"
        if self.Config.Type == _BACT:
            self.Jobname = f"{self.Model.Pe}Pe_{self.Config.Dimend}{self.Config.Type[0].upper()}{self.Init.Env[0].upper()}"
        else:
            self.Jobname = f"{self.Init.N_monos}N_{self.Config.Dimend}{self.Config.Type[0].upper()}{self.Init.Env[0].upper()}"
        #/Users/wukong/Data/Simus/2D_100G_1.0Pe_Chain/5.0R5.0_100N1_Anulus/1.0T_0.0Xi_8T5
        self.dir_data = os.path.join(self.simus, self.dir1, f'{self.dir2}_{self.Init.Env}', self.dir3)
        #/Users/wukong/Data/Simus/2D_100G_1.0Pe_Chain/5.0R5.0_100N1_Anulus/1.0T_0.0Xi_8T5/5.0R5.0_100N1_CA.data
        subprocess.run(f"mkdir -p {self.dir_data}", shell=True)
        shutil.copy2(os.path.join(self.host, self.mydirs[0], "paras.py"), os.path.join(self.dir_data, "paras.py"))
        print(f"dir_data => {self.dir_data}")
        logging.info(f"dir_data => {self.dir_data}")
        #Figures
        self.fig1 = os.path.join(self.host, self.mydirs[3], self.dir1, f'{self.dir2}_{self.Init.Env}', self.dir3)
        subprocess.run(f"mkdir -p {self.fig1}", shell=True)

        if os.path.exists(os.path.join(self.dir_data, f"{self.Run.Trun:03}.lammpstrj")):
            return True
        else:
            return False
#############################################################################################################

class _plot:
    def __init__(self, Path):
        self.Path, self.Init, self.Run, self.Config = Path, Path.Init, Path.Run, Path.Config
        self.chunk = 9

    def set_dump(self):
        is_bacteria = (self.Config.Type == _BACT)
        dump = {
            2: "xu yu" + (" vx vy" if is_bacteria else ""),
            3: "xu yu zu" + (" vx vy vz" if is_bacteria else "")
        }
        try:
            return ["id"] + dump[self.Config.Dimend].split(" ")
        except KeyError:
            logging.error(f"Error: Wrong Dimension to run => dimension != {self.Config.Dimend}")
            raise ValueError(f"Invalid dimension: {self.Config.Dimend}")

    def read_data(self):
        dump = self.set_dump()
        self.data = np.zeros((self.Run.Trun, self.Run.Frames+1, self.Init.num_monos, len(dump)))
        logging.info(f"==> Reading {ifile}.lammpstrj file: ......")
        # read the lammpstrj files with 2001 frames
        for index, ifile in enumerate([f"{i:03}" for i in range(1, self.Run.Trun + 1)]):
            dir_file = os.path.join(f"{self.Path.dir_data}", f"{ifile}.lammpstrj")
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
                logging.error(f"ERROR: Wrong timesteps => {lastStep} != {self.Run.Fradfmes} * {self.Run.Tdump}")
                raise ValueError(f"ERROR: Wrong timesteps => {lastStep} != {self.Run.Fradfmes} * {self.Run.Tdump}")
            skiprows = np.array(list(map(lambda x: np.arange(self.chunk) + (self.Init.num_monos + self.chunk) * x, np.arange(self.Run.Frames+1)))).ravel()
            try:
                df = pd.read_csv(dir_file, skiprows=skiprows, delim_whitespace=True, header=None, names=names, usecols=dump)
            except Exception as e:
                logging.error(f"ERROR: Could not read the file due to {e}")
                raise ValueError(f"ERROR: Could not read the file due to {e}")

            # data[ifile][iframe][iatom][id, xu, yu]
            self.data[index] = df.to_numpy().reshape((self.Run.Frames+1, self.Init.num_monos, len(dump)))
            #np.save(self.Path.dir_data, data)
        return self.data

    def plot1(self):
        fig_save = os.path.join(f"{self.Path.fig1}", "r(S,T)Dist")
        pdf = PdfPages(f"{fig_save}.pdf")
        print(f"{fig_save}.pdf")
        logging.info(f"{fig_save}.pdf")
        # ----------------------------> read data <----------------------------#
        # 1. Plot particle ID vs time, color-coded by the magnitude of the particle coordinates
        self.data = self.read_data()
        ids = self.data[0, 0, :, 0]
        times = np.arange((self.Run.Frames+1))*self.Run.dt*self.Run.Tdump
        data = np.linalg.norm(self.data[0, ..., 1:3], axis = -1)
        at_id = self.Init.num_monos//2
        at_time = self.Run.Frames//2

        #----------------------------> figure settings <----------------------------#
        plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='Times New Roman')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15

        #plt.subplots_adjust(left=0.15, right=0.85, bottom=0.13, top=0.9, wspace=0.2, hspace=0.2)
        # Create the layout
        fig = plt.figure(figsize=(15, 15))
        gs = GridSpec(4, 4, figure=fig)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        ax_a = fig.add_subplot(gs[0:2, 0:2])
        ax_b = fig.add_subplot(gs[0:2, 2], sharey=ax_a)
        ax_c = fig.add_subplot(gs[0:2, 3], sharey=ax_a)
        ax_d = fig.add_subplot(gs[2, 0:2], sharex=ax_a)
        ax_e = fig.add_subplot(gs[3, 0:2], sharex=ax_a)
        ax_f = fig.add_subplot(gs[2:, 2:])

        # ----------------------------> plot figure<----------------------------#
        cmap = ax_a.pcolormesh(times[:self.Run.Frames+1], np.arange(1, self.Init.num_monos+1), data.T, shading='auto', cmap='rainbow')
        # ax_b: Distribution of magnitudes for a specific time frame
        ax_b.plot(data[at_time, :], ids, 'k')
        # ax_c: Distribution of magnitudes across all time frames for a specific atom
        #ax_c.hist(data[:, at_id], bins=20, density=True, orientation='horizontal')
        pdf_atoms = norm.pdf(np.arange(self.Init.num_monos), 25, 5)
        ax_c.plot(pdf_atoms, np.arange(1, self.Init.num_monos + 1), 'r')
        # ax_d: Magnitude vs time for a specific atom (e.g., atom 50)
        ax_d.plot(times, data[:, at_id], 'b')
        # ax_e: Distribution of magnitudes across all atoms for a specific time frame (e.g., 50th frame)
        pdf_time = norm.pdf(times, 5, 1)
        ax_e.plot(times, pdf_time, 'r')
        # ax_f: Overall distribution of magnitudes
        ax_f.plot(np.sort(data.ravel()), np.linspace(0, 1, (self.Run.Frames+1) * self.Init.num_monos), 'k')
        ax_f2 = ax_f.twinx()
        pdf_f_times = np.histogram(data.flatten(), bins=50, density=True)[0]
        ax_f2.bar(range(len(pdf_f_times)), pdf_f_times, color='c', alpha=0.5)

        # ----------------------------> adding <----------------------------#
        ax_a.axhline(y=at_id, linestyle='--', lw = 1.5, color='black')  # Selected Particle ID
        ax_a.axvline(x=at_time*self.Run.dt*self.Run.Tdump, linestyle='--', lw = 1.5, color='black')  # Selected Time frame

        axpos = ax_a.get_position()
        caxpos = mtransforms.Bbox.from_extents(axpos.x0 - 0.07, axpos.y0, axpos.x0 - 0.05, axpos.y1)
        cax = fig.add_axes(caxpos)
        cbar = plt.colorbar(cmap, cax=cax)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.set_xlabel(r'$r$', fontsize=20)
        # ----------------------------> axis settings <----------------------------#
        ax_a.set_xlabel(r"$t$", fontsize=20)
        ax_a.set_ylabel(r"$s$", fontsize=20)
        ax_b.set_xlabel(r'$r_t(s)$', fontsize=20)
        ax_b.set_ylabel(r'$s$', fontsize=20)
        ax_b.set_title(f'Time = {int(at_time*self.Run.dt*self.Run.Tdump)}', loc='right')
        ax_c.set_xlabel(r'$p_t(s)$', fontsize=20)
        ax_c.set_ylabel(r'$s$', fontsize=20)
        ax_c.set_title(f'probability', loc='right')
        ax_d.set_xlabel(r"$t$", fontsize=20)
        ax_d.set_ylabel(r'$r_s(t)$', fontsize=20)
        ax_d.set_title(f'Atom = {at_id}', loc='right')
        ax_e.set_xlabel(r"$t$", fontsize=20)
        ax_e.set_ylabel(r'$p_s(t)$', fontsize=20)
        ax_e.set_title('probability', loc='right')
        ax_f.set_xlabel(r'$r$', fontsize=20)
        ax_f.set_ylabel(r'$p_s(r)$', fontsize=20)
        ax_f2.set_ylabel(r'$p_t(r)$', fontsize=20)
        ax_f.set_title(r'PDF across modules (Time)', loc='right')

        # ----------------------------> linewidth <----------------------------#
        for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
            ax.annotate(label, (-0.1, 0.9), textcoords="axes fraction", xycoords="axes fraction", va="center", ha="center", fontsize=20)
            ax.tick_params(axis='both', which="major", width=2, labelsize=15, pad=7.0)
            ax.tick_params(axis='both', which="minor", width=2, labelsize=15, pad=4.0)
            # ----------------------------> axes lines <----------------------------#
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)

        print("saving pdf......")
        # ----------------------------> save fig <----------------------------#
        #plt.tight_layout()
        fig1 = plt.gcf()
        #pdf.savefig(fig1, dpi=1000, transparent=True)
        #pdf.close()

        #print("saving png......")
        # ax.legend(loc='upper left', frameon=False, ncol=int(np.ceil(len(Arg1) / 5.)), columnspacing = 0.1, labelspacing = 0.1, bbox_to_anchor=[0.0, 0.955], fontsize=10)
        #fig1.savefig(f"{fig_save}.png", format="png", dpi=1000, transparent=True)
        plt.show()
        plt.close()
        # -------------------------------Done!----------------------------------------#

    ##################################################################
    def MSD():
        print("-----------------------------------Done!--------------------------------------------")
    def Rg():
        print("-----------------------------------Done!--------------------------------------------")
#############################################################################################################

class Timer:
    def __init__(self, func=time.perf_counter):
            self.elapsed = 0.0
            self._func = func
            self._start = None
        
    def start(self):
            self.elapsed = 0.0
            if self._start is not None:
                    raise RuntimeError('Already started')
            self._start = self._func()
        
    def stop(self, str="Time"):
            if self._start is None:
                    raise RuntimeError('Not started')
            end = self._func()
            self.elapsed += end - self._start
            print(str, ":", self.elapsed)
            logging.info(str, ":", self.elapsed)
            self._start = None
        
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
timer = Timer()
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
]
