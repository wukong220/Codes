import re
import os
import sys
import time
import shutil
import logging
import platform
import subprocess
from datetime import datetime, timedelta
from itertools import islice

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev, griddata, UnivariateSpline
from scipy.stats import norm, linregress, gaussian_kde, multivariate_normal

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
# -----------------------------------Const-------------------------------------------
HOST, CURRENT_DIR = platform.system(), os.path.dirname(os.path.abspath(__file__))
usage = "bsub < infile.lsf"
usage = "var: exe_analysis  => var(Pe, N, W): plot_job => Exp(var): exe_plot"
#-----------------------------------Variable-------------------------------------------
class Property:
    def __init__(self, name, path, scale="\\nu", dtime=True, paras=['Pe', 'N', 'W']):
        self.name = name
        self.path = path
        self.dtime = dtime
        self.scale = scale
        self.paras = paras
    def show(self):
        print(f"name={self.name}\npath={self.path}\ndtime={self.dtime}\nscale={self.scale}\nparas={self.paras}")
class Echo:
    def __init__(self):
        logging.basicConfig(filename='Run.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    def info(self, message):
        logging.info(message)
        print(message)
    def error(self, message):
        logging.error(message)
        raise ValueError(message)
    def warn(self, message):
        logging.warning(message)
        print(message)
echo, Rcom  = Echo(), Property("Rcom", "Rcom")
Rg, Rgp = Property("Rg", "Rg2_time", dtime=False), Property("Rgp", "Rg2p", dtime=False),
Re, Rep = Property("Re", "Re2_time", dtime=False), Property("Rep", "Re2p", dtime=False),
MSD, MSDp = Property("MSD", "MSDt", "D"), Property("MSDp", "MSDpt", "Dp")
Cee = Property("Cee", "Ceet", "\\tau_R")
#-----------------------------------Parameters-------------------------------------------
task, JOBS = ["Simus", "Anas", "Plots"][2], [Rg, MSD] #Rg,
check  = (task != "Plots")
if task == "Simus":
    BSUB, jump = True, True
else:
    BSUB, jump = False, False
#-----------------------------------Dictionary-------------------------------------------
params = {
    'labels': {'Types': ["Chain", "Bacteria", "Ring"][0:1],
                'Envs': ["Anlus", "Rand", "Slit"][2:3]},
    'marks': {'labels': [], 'config': []},
    'restart': (False, "equ"),
    'BSUB, jump': (BSUB, jump),
    'Queues': {'7k83!': 1.0, '9654!': 1.0},
    # 动力学方程的重要参数
    'Temp': 1.0,
    'Gamma': 10,
    'Trun': [1, 20],
    'Dimend': 3,
    #'Dimend': [2,3],
    'Frames': 2000,
    'num_chains': 1,
}
#---------------------------------------------------------------------------------------
class _config:
    def __init__(self, Dimend, Type, Env, Params = params):
        self.config = {
            "Linux": {
                "Bacteria": {'N_monos': [3], 'Xi': 1000, 'Fa': [1.0],}, # 'Fa': [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0],},
                "Chain": {'N_monos': [20, 40, 80, 100, 150, 200, 250, 300], 'Xi': 0.0,
                          'Fa': [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 100.0],
                          # 'Gamma': [0.1, 1, 10, 100], #'Temp': [1.0, 0.2, 0.1, 0.05, 0.01],
                          },
                "Slit": {2: {"Rin": [0.0], "Wid": [5.0, 10.0, 15.0, 20.0]},
                          3: {"Rin": [0.0], "Wid": [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]},
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
                },

            "Darwin": {
                "Bacteria": {'N_monos': 3, 'Xi': 1000, 'Fa': 1.0},
                "Chain": {'N_monos': [20, 40, 80, 100, 150, 200, 250, 300], 'Xi': 0.0,
                              #'Gamma': 100.0, 'Fa': [1.0, 5.0, 10.0], #, 20.0, 100.0
                              #'N_monos': [20, 40, 80, 100, 150, 200, 250, 300], 'Xi': 0.0, 'Fa': [1.0, 5.0, 10.0, 20.0, 100.0], 'Gamma': 100,
                              #'N_monos': [10], 'Fa': [1.0, 10.0], "Xi": 0.0, "Trun": [1, 2], "Frames": 200, 'Gamma': 100,
                              'N_monos': [80, 100], 'Fa': [1.0], 'Gamma': 100
                          },
                "Slit": {2: {"Rin": [0.0], "Wid": [5.0]},
                         3: {"Rin": [0.0], "Wid": [3.0]}, #1.0, 3.0, 5.0, 10.0, 15.0, 20.0]}, #0.0, 2.0
                         },

                "Ring": {'N_monos': [100], 'Xi': 0.0, 'Fa': [1.0], 'Gamma': [1.0]},
                "Anlus":{2: {'Rin': [0.0], 'Wid': [0.0]},
                            3: {'Rin': [0.0], 'Wid': [0.0]},
                            #3: {'Rin': [5.0], 'Wid': [5.0]},
                            },
                "Rand": {2: {'Rin': 0.4,  'Wid': 2.0},
                             3: {'Rin': 0.0314, 'Wid': 2.5},
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
            echo.error(f"Error: Invalid Dimension to run => Config.Dimend != {self.Dimend}")

        Run.Tdump = 2 * 10 ** Run.eSteps // Run.Frames
        Run.Tdump_ref = Run.Tdump // 100
        if HOST == "Darwin" and task == "Simus":
            Run.Tdump = Run.Tdump_ref

        Run.Tinit = Run.Frames * Run.Tdump_ref // 5
        Run.TSteps = Run.Frames * Run.Tdump
        Run.Tequ = Run.TSteps
        Run.Tref = Run.Frames * Run.Tdump_ref
        Run.Params["Total Run Steps"] = Run.TSteps

        if self.Type == "Bacteria":
            Run.Tdump //= 10
            Run.Tequ //= 100
#############################################################################################################
class _run:
    def __init__(self, Dimend, Gamma, Temp, Trun, Params = params, Frames = params["Frames"]):
        self.Params = Params
        self.Queue = "7k83!"
        self.set_queue()
        self.Gamma = Gamma
        self.Trun = Trun
        if task != "Simus":
            #if  HOST == "Linux" or task == "Plots":
              #  self.Trun[0] = 6  # supplement
            if HOST == "Darwin":
                if task == "Anas":
                    self.Trun[1] = 5
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
                echo.error(f"Error: {e}")

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
                    echo.error(f"Error: {e}")
                for line in bjobs.strip().split('\n')[1:]:
                    columns = line.split()
                    start_time = datetime.strptime(f"{columns[-4]} {columns[-3]} {datetime.now().year} {columns[-2]}", "%b %d %Y %H:%M")
                    if datetime.now() - start_time > timedelta(hours=24) and columns[2] == "RUN":
                        cores = int(columns[-1].split('*')[0]) if '*' in columns[-1] else 1
                        queue_info[iqueue]["occupy"] += cores
                queue_info[iqueue]["Avail"] = queue_info[iqueue]["cores"] - queue_info[iqueue]["occupy"] + 1
                queue_info[iqueue]["Usage"] = np.around( (queue_info[iqueue]["PEND"] + queue_info[iqueue]["RUN"] - queue_info[iqueue]["occupy"] ) / queue_info[iqueue]["Avail"], 3)
                self.Params["Queues"][iqueue] = queue_info[iqueue]["Usage"]
                #if queue_info[iqueue]["PEND"] == 0:
                    #print(queue_info[iqueue]['Usage'])
                    #self.Queue = iqueue
                #elif queue_info[iqueue]["PEND"] > 0:
                self.Queue = min(myques, key=lambda x: queue_info[x]['Usage']) #print(f"queue = {self.Queue}, queue_info: {queue_info}")
        return self.Queue
    def sub_file(self, Path, infiles):
        echo.info(f">>> Preparing sub file: {Path.simus}")
        for infile in infiles:
            #print(">>> Preparing sub file......")
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
                f'source ~/.bashrc',
                f'export RUN_ON_CLUSTER=true',
                f'cd {Path.simus}',
                #f'echo "python3 Run.py {infile}\n$(date)" ',
                #f'python3 Run.py {infile}',
                f'echo -e "mpirun -np 1 lmp_wk -i {infile}.in\n$(date)"',
                f'mpirun -np 1 lmp_wk -i {infile}.in',
            ]

            with open(f"{dir_file}.lsf", "w") as file:
                for command in bsub:
                    file.write(command + "\n")
        echo.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
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
        self.R_ring = 0
        self.jump = False
        self.set_box()   #set box
        if (self.Config.Type == "Ring" and self.Env == "Anlus") and (self.Env == "Slit" and self.Rin > 1e-6):
            self.jump = True
            echo.warning(f"I'm sorry => '{self.Label}' is not ready! when Dimend = {Params['Dimend']}")

        if self.Env == "Anlus":
            self.Rout = self.Rin + self.Wid  # outer_radius
            self.R_ring = self.Rin + self.Wid / 2
            self.R_torus = self.Wid / 2

            self.Nobs_Rin, self.Nobs_Rout = self.N_ring(self.particle_density, self.Rin, self.sigma), self.N_ring(self.particle_density, self.Rout, self.sigma)
            self.Nobs_ring, self.Nobs_torus = self.N_ring(self.particle_density, self.R_ring, self.sigma), self.N_ring(self.particle_density, self.R_torus, self.sigma)
            self.num_obs = int(self.Nobs_ring) * int(self.Nobs_torus) if self.Config.Dimend == 3 else int(self.Nobs_Rin + self.Nobs_Rout)
            theta = [np.linspace(0, 2 * np.pi, int(2 * np.pi * R / self.sigma_equ + 1))[:-1] for R in self.Rring]
            if self.Config.Dimend == 2 and (self.num_monos > sum(itheta.size for itheta in theta)):
                echo.warning("N_monos is too Long!")
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
            echo.error(f"ERROR => num_chains = {self.num_chains} is not prepared!\nnum_chains must be 1")
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
            elif self.Config.Type == "Bacteria":
                self.Lbox = self.N_monos * 10
            else:
                echo.warning(f"ERROR: Wrong model type! => Config.Type = {self.Config.Type}")
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
            echo.error(f"Error: Invalid Dimend  => dimension != {Config.Dimend}")
    def N_ring(self, density, Radius, sigma):
        return np.ceil(density * 2 * np.pi * Radius / sigma)
    def N_circles(self):
        inter = self.sigma_equ + 0.2
        start = self.Rin + inter + 0.5 if self.Env == "Anlus" else (self.N_monos+0.5) * self.sigma_equ/(2 * np.pi)
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
        atomTypes = 5
        file.write(f"{int(self.total_particles)} atoms\n\n")
        file.write(f"{self.bonds} bonds\n\n")
        file.write(f"{self.angles} angles\n\n")
        file.write(f"{atomTypes} atom types\n\n")
        file.write("1 bond types\n\n")
        file.write("3 angle types\n\n")
        file.write(f"-{self.Lbox} {self.Lbox} xlo xhi\n")
        file.write(f"-{self.Lbox} {self.Lbox} ylo yhi\n")
        file.write(f"{self.zlo} {self.zhi} zlo zhi\n\n")
        file.write("Masses\n\n")
        for i in range(atomTypes):
            file.write(f"{i+1} {self.mass}\n")
        file.write("\n")
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
            stop = (self.Wid - self.sigma_equ - 0.2) / 2  if self.Env == "Anlus" else self.N_monos * self.sigma_equ/(2 * np.pi)
            (circles, flag) = (int(stop / inter) + 1, 1) if self.Env == "Anlus" else (0, 0)
            chain_coords = []
            theta0, phi0 = 0, 0
            for iRchain in [stop] if circles == 0 else np.linspace(0.5, inter * circles, circles+1)[:-1][::-1]:
                Nchain = self.N_ring(1, iRchain, self.sigma_equ)
                phi = np.linspace(phi0, 2 * np.pi + phi0, int(Nchain + 1))[:-1] if self.Config.Dimend == 3 and self.Env == "Anlus" else [0, 0]
                for iphi in phi:
                    Nring = self.N_ring(1, self.R_ring + iRchain * np.cos(iphi), self.sigma_equ)
                    theta = np.linspace(theta0, 2 * np.pi+theta0, int(Nring + 1))[:-1]
                    x = np.round((self.R_ring + self.sigma * flag + iRchain * np.cos(iphi)) * np.cos(theta) * self.sigma_equ, 5)
                    y = np.round((self.R_ring + self.sigma * flag + iRchain * np.cos(iphi)) * np.sin(theta) * self.sigma_equ, 5)
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
        echo.info("==> Preparing initial data file......")
        # 打开data文件以进行写入
        for infile in [f"{i:03}" for i in range(self.Trun[0], self.Trun[1] + 2)]:
            data_file = os.path.join(f"{Path.simus}", f'{infile}.{self.Config.Type[0].upper()}{self.Env[0].upper()}.data')
            #print(f"==> Preparing initial data {infile}......")
            with open(f"{data_file}", "w") as file:
                self.write_header(file)
                self.write_chain(file)
                if self.Env == "Anlus":
                   self.write_anlus(file)
                elif self.Env == "Rand":
                    self.write_rand(file)
                self.write_potential(file)
        echo.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
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
                f'change_box             all {axis} final {-wid / 2} {wid / 2}',
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
        echo.info(f"==> Writing infile ==> {Path.simus}")

        for infile in [f"{i:03}" for i in range(Run.Trun[0], Run.Trun[1] + 2)]:
            #print(f"==> Writing infile: {infile}......")
            dir_file = os.path.join(f"{Path.simus}", infile)
            if Init.Env == "Test":
                file_content = f"""
# Setup
echo                screen
units           lj
dimension       3
boundary        p p p
atom_style      angle

variable        dir_file string {dir_file}
read_data       {dir_file}.CF.data

#groups
group                  chain type 1 2 3

# for initialization
##################################################################
#pair potential
pair_style      lj/cut 1.12246
pair_modify     shift yes
pair_coeff      * * 1 1.0

# Bond potential
bond_style      harmonic
bond_coeff      1 4000 1.0
special_bonds   lj/coul 1.0 1.0 1.0

# angle potential
angle_style     harmonic
angle_coeff     * 0.0 180
##################################################################
# for communication
comm_style      brick
comm_modify     mode single cutoff 3.0 vel yes

neighbor        1.5 bin
neigh_modify    every 1 delay 0 check yes 

# for equalibrium
##################################################################
fix             LANG chain langevin 1.0 1.0 {Run.Damp} 743050779
fix             NVE chain nve

dump            EQU chain custom 1000000 ${{dir_file}}.chain_equ.lammpstrj id type xu yu zu
dump_modify     EQU sort id

reset_timestep  0
timestep        0.001
thermo          1000000
log             ${{dir_file}}.log
restart         1000000  ${{dir_file}}.a.restart  ${{dir_file}}.b.restart 
run             200000000
write_restart   ${{dir_file}}.equ.restart

undump          EQU

dump            DATA chain custom 100000 ${{dir_file}}.lammpstrj id type xu yu zu vx vy vz
dump_modify     DATA sort id

reset_timestep  0
run             200000000
write_restart   ${{dir_file}}.restart
                """
                with open(f"{dir_file}.in", 'w') as file:  # Replace with your file path
                    file.write(file_content.strip())
            else:
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
                    echo.error(f"An error occurred: {e}")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
class _path:
    def __init__(self, Model):
        #host = os.getcwd()
        #self.host = os.path.abspath(os.path.dirname(os.getcwd()))
        self.host = re.match(r"(.*?/Data/)", os.getcwd()).group(1)
        self.mydirs = ["Codes", "Simus", "Figs"]
        self.Model = Model
        self.Init = Model.Init
        self.Config = Model.Init.Config
        self.Run = Model.Run
        self.jump = self.build_paths()
        self.run_py = "Run.py"
    def build_paths(self):
        self.base = os.path.join(self.host, self.mydirs[1])
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
        self.dir3 = f"{self.Run.Temp}T_{self.Model.Xi}Xi"
        if self.Config.Type == "Bacteria":
            self.Jobname = f"{self.Model.Pe}Pe_{self.Config.Dimend}{self.Config.Type[0].upper()}{self.Init.Env[0].upper()}"
        else:
            self.Jobname = f"{self.Init.N_monos}N_{self.Config.Dimend}{self.Config.Type[0].upper()}{self.Init.Env[0].upper()}"
        #/Users/wukong/Data/Simus/2D_100G_1.0Pe_Chain/5.0R5.0_100N1_Anulus/1.0T_0.0Xi_8T5
        self.simus0 = os.path.join(self.base, self.dir1, f'{self.dir2}', f"{self.dir3}_{self.Run.eSteps}T{self.Run.Trun[0]}-{self.Run.Trun[1]}")
        self.simus1 = os.path.join(self.base, self.dir1, f'{self.dir2}', f"{self.dir3}_{self.Run.eSteps}T{self.Run.Trun[0]-1}")
        self.simus2 = os.path.join(self.base, self.dir1, f'{self.dir2}', f"{self.dir3}_{self.Run.eSteps}T{self.Run.Trun[1]}")
        self.simus = self.simus2 if Trun[0] == 1 else self.simus0
        #/Users/wukong/Data/Simus/2D_100G_1.0Pe_Chain/5.0R5.0_100N1_Anulus/1.0T_0.0Xi_8T5/5.0R5.0_100N1_CA.data

        #Anas and Figures
        self.fig = self.simus.replace(self.mydirs[1], self.mydirs[2])
        self.fig0 = self.simus0.replace(self.mydirs[1], self.mydirs[2])
        self.fig1 = self.simus2.replace(self.mydirs[1], self.mydirs[2])
        self.lmp_trj = os.path.join(self.simus, f"{self.Run.Trun[1]:03}.lammpstrj")
        for i, path in enumerate([self.simus, self.simus1, self.simus2, self.simus0]):
            lmp_trj = os.path.join(path, f"{self.Run.Trun[1]:03}.lammpstrj")
            if os.path.exists(lmp_trj):
                self.lmp_trj = lmp_trj
                return True
            else:
                if i == 3:
                    if HOST != "Darwin":
                        echo.info(f"File doesn't exist in Path: {lmp_trj}")
                    return False
    def show(self):
        print(f"host: {self.host}\nmydirs: {self.mydirs}\n"
              f"Path.dir1: {self.dir1}\nPath.dir2: {self.dir2}\nPath.dir3: {self.dir3}\n"
              f"Path.simus: {self.simus}\nPath.fig: {self.fig}")
#############################################################################################################
class _anas:
    def __init__(self, Path, Pe):
        self.Path, self.Init, self.Run, self.Config = Path, Path.Init, Path.Run, Path.Config
        self.Pe = Pe
        self.chunk = 9
        self.jump = jump
    def set_dump(self):
        is_bacteria = (self.Config.Type == "Bacteria")
        dump = {
            2: "xu yu" + (" vx vy" if is_bacteria else ""),
            3: "xu yu zu" + (" vx vy vz" if is_bacteria else "")
        }
        try:
            return dump[self.Config.Dimend].split(" ")
        except KeyError:
            echo.error(f"Error: Invalid dimension to run => Config.Dimend != {self.Config.Dimend}")
    def read_data(self):
        timer = Timer("Read")
        timer.start()
        dump = self.set_dump()
        lox, hix = -self.Init.Lbox * 1025, self.Init.Lbox * 1023

        self.data = np.zeros((self.Run.Trun[1], self.Run.Frames+1, self.Init.num_monos, len(dump)))
        # read the lammpstrj files with 2001 frames
        for index, ifile in enumerate([f"{i:03}" for i in range(1, self.Run.Trun[1] + 1)]):
            for i, path in enumerate([self.Path.simus, self.Path.simus1, self.Path.simus2, self.Path.simus0]):
                dir_file = os.path.join(f"{path}", f"{ifile}.lammpstrj")
                if os.path.exists(dir_file):
                    break
                elif i == 3:
                    echo.error(f"ERROR: Wrong Trun => ifile = {dir_file} while Trun = {self.Run.Trun}")
            echo.info(f"==> Reading {dir_file} file: ......")
            # extract natoms, time steps, and check the last time step
            names = list(pd.read_csv(dir_file, skiprows=7, nrows=0, delim_whitespace=True, header=1).columns[2:])
            natoms = pd.read_csv(dir_file, skiprows=3, nrows=1, delim_whitespace=True, header=None).iloc[0][0]
            dstep = pd.read_csv(dir_file, skiprows=self.Init.num_monos + self.chunk + 1, nrows=1, delim_whitespace=True, header=None).iloc[0][0]
            lastStep = pd.read_csv(dir_file, skiprows=self.Run.Frames * (self.Init.num_monos + self.chunk) + 1, nrows=1, delim_whitespace=True, header=None).iloc[0][0]
            if natoms != self.Init.num_monos:
                echo.error(f"ERROR: Wrong atoms => {natoms} != {self.Init.num_monos}")
            elif lastStep != self.Run.Frames * self.Run.Tdump:
                echo.error(f"ERROR: Wrong timesteps => {lastStep} != {self.Run.Frames} * {self.Run.Tdump}")
            skiprows = np.concatenate([np.arange(self.chunk) + (self.Init.num_monos + self.chunk) * x for x in range(self.Run.Frames + 1)])
            try:
                df = pd.read_csv(dir_file, skiprows=skiprows, delim_whitespace=True, header=None, names=names, usecols=dump)
            except Exception as e:
                echo.error(f"ERROR: Could not read the file due to {e}")

            # data[ifile][iframe][iatom][id, xu, yu]
            self.data[index] = df.to_numpy().reshape((self.Run.Frames+1, self.Init.num_monos, len(dump)))
            #np.save(self.Path.simus, data)

        # unwrap data
        Lx = hix - lox
        if self.Pe > 50:
            self.data = unwrap_x(self.data, Lx)  # get real coordinates
        # data = np.copy(self.data)
        # for i in range(1, frames):
        # dx = data[:, i, :, 0] - data[:, i - 1, :, 0]
        # data[:, i, :, 0] -= np.sign(dx) * Lx * (np.abs(dx) > Lx / 2)
        timer.count("Read and unwrap data")
        timer.stop()
        return self.data
class BasePlot:
    def __init__(self, variable, fig_save):
        self.variable = variable
        self.df = variable.df
        self.fig_save = fig_save
        self.jump = jump
#---------------------------------------------------------------------------#
    def fig_path(self, mark=None, fig_save=None):
        fig_file = f"{fig_save if fig_save else self.fig_save}{('.' + mark) if mark else ''}"
        if os.path.exists(f"{fig_file}.pdf") and self.jump:
            echo.info(f"==>{fig_file}.pdf is already!")
            return True
        else:
            echo.info(f"{fig_file}.pdf")
        return fig_file
    def set_style(self, mag, size=(6.4, 5.4), inter=(0.2, 0.5), edge=(1.0, 1.5), title=""):
        '''plotting:  axes = [ax for row in axes for ax in row]'''
        plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        self.notes = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)"]
        self.markers = ['o', '^', 's', '<', 'p', '*', 'D', 'v', 'H', '>']
        # ----------------------------> setup: canvas<----------------------------#
        (cols, rows), (width, heigh), (wspace, hspace), ledge = mag, size, inter, edge[0]
        redge = edge[1] if rows > 2 else 1.0
        left, bot = ledge / (width * cols), redge / (heigh * rows)
        right, top = 1-left, 1-bot
        fig = plt.figure(figsize=(width * cols, heigh * rows))
        plt.subplots_adjust(left=left, right=right, bottom=bot, top=top, wspace=wspace, hspace=hspace)
        gs = GridSpec(rows, cols, figure=fig)
        fig.suptitle(title, fontsize=25, y=1.0)
        return fig, gs
    def colorbar(self, ax, data, label, is_3D=False):
        loc, pad = ("left", 0.05) if is_3D else ("right", 0.05)
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=Normalize(vmin=data.min(), vmax=data.max()))
        sm.set_array([])
        if HOST == "Linux":
            cbar = plt.colorbar(sm, ax=ax, pad=pad)
        elif HOST == "Darwin":
            cbar = plt.colorbar(sm, ax=ax, location=loc, pad=pad)
        cbar.ax.yaxis.set_ticks_position(loc)
        #cbar.ax.set_xlabel(label, fontsize=20, labelpad=10)
        cbar.ax.set_title(label, fontsize=20, pad=10)
    def set_axes(self, ax, data, labels, title, is_3D=False, fill=True, rotation=0, loc="right", log=False):
        # label settings
        ax.set_title(title, loc=loc, fontsize=20)
        axis_labels = {
            "x": labels[0],
            "y": labels[1],
            "z": labels[2] if is_3D else None
        }
        if log:
            lo, hi = 0.5, 3.0
            ax.set_xscale('log')
            ax.set_yscale('log')
            if is_3D:
                ax.set_zscale('log')
        else:
            lo, hi = 0.1, 1.2
            ax.grid(True)

        # Set axis limits and rotation
        for i, (axis_data, axis_name) in enumerate(zip(data[:2 + is_3D], "xyz"[:2 + is_3D])):
            getattr(ax, f'set_{axis_name}label')(axis_labels[axis_name], fontsize=20, labelpad=5)
            min_val, max_val = min(axis_data), max(axis_data)
            if fill:
                getattr(ax, f'set_{axis_name}lim')(min_val, max_val)
            else:
                getattr(ax, f'set_{axis_name}lim')((min_val - 1) * lo, max_val * hi)
            if axis_name == 'x':
                ax.tick_params(axis='x', rotation=rotation)
    def adding(self, ax, note, pos=-0.25, is_3D=False):
        # linewidth and note
        ax.annotate(note, (pos, 0.9), textcoords="axes fraction", xycoords="axes fraction", va="center", ha="center", fontsize=20)
        if is_3D:
            ax.tick_params(axis='x', which="both", width=2, labelsize=15, pad=-3.0)
            ax.tick_params(axis='y', which="both", width=2, labelsize=15, pad=1.0)
            ax.tick_params(axis='z', which="both", width=2, labelsize=15, pad=2.0)
        else:
            ax.tick_params(axis='both', which="both", width=2, labelsize=15, pad=5.0)

        # axes lines
        for spine in ["bottom", "left", "right", "top"]:
            ax.spines[spine].set_linewidth(2)
class _plot(BasePlot):
    def __init__(self, Path, bins=20):
        self.Path = Path
        self.Init, self.Run = Path.Init, Path.Run
        self.simp = 10
        self.num_bins = bins
        self.num_pdf = bins*10
        self.jump = jump
    def data2dict(self, flag=False):
        dict = {}
        _, frames, atoms = self.data.shape
        # data[ifile][iframe][iatom]
        data = np.mean(self.data, axis=0)

        if atoms != self.Init.num_monos or frames != (self.Run.Frames+1):
            echo.info(f"Wrong Atoms or Frames: atoms != num_monos => {atoms} != {self.Init.num_monos}; "
                         f"frames != Frames => {frames} != {self.Run.Frames+1}")
        times, ids = np.arange(frames)*self.Run.dt*self.Run.Tdump, np.arange(1,atoms+1,1)

        if flag:
            data = data[::self.simp, ...]
            frames = data.shape[0]
            times = np.arange(frames)*self.Run.dt*self.Run.Tdump * self.simp
        dict = {
            #"x": [np.random.normal(0, 1, frames), np.random.normal(0, 1, frames * atoms), ],  # Frame numbers (t coordinate)
            #"y": [np.random.random(atoms), np.random.random(frames * atoms),],  # Particle IDs (s coordinate)
            #"z": [data, data.flatten(),],  # Magnitude (r coordinate)
            "t": [times, np.repeat(times, atoms), "t"],  # Frame numbers (t coordinate)
            "s": [ids, np.tile(ids, frames), "s"],  # Particle IDs (s coordinate)
            var2str(self.variable.name)[0]: [data, data.flatten(), var2str(self.variable.name)[1]],  # Magnitude (r coordinate)
        }
        return dict
    def dist_data(self, x):
        bin_id, pdf_id = self.num_bins // 2, self.num_pdf // 2
        # Create a 2D histogram and bin centers
        hist_x, x_bins = np.histogram(x, bins=self.num_bins, density=True)
        x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        x_range = np.linspace(min(x), max(x), self.num_pdf)
        pdf_x = gaussian_kde(x).evaluate(x_range)
        return hist_x, x_bins, x_bin_centers, x_range, pdf_x
    def adding(self, ax, note, loc=(-0.2, 0.9)):
        # linewidth and note
        ax.annotate(note, loc, textcoords="axes fraction", xycoords="axes fraction", va="center",
                    ha="center", fontsize=20)
        ax.tick_params(axis='both', which="both", width=2, labelsize=15, pad=7.0)
        # ax.tick_params(axis='both', which="minor", width=2, labelsize=15, pad=4.0)
        # axes lines
        for spine in ["bottom", "left", "right", "top"]:
            ax.spines[spine].set_linewidth(2)
    ##################################################################
    def original(self):
        timer = Timer("Original")
        timer.start()
        # ----------------------------> prepare data <----------------------------#
        keys, values = list(self.data_dict.keys()), list(self.data_dict.values())
        #x, y, z = [self.data_dict[key][1] for key in keys]
        x, y, z = [self.simp_dict[key][1] for key in keys]
        simp_x, simp_y, simp_z = [self.simp_dict[key][1] for key in keys]
        x_label, y_label, z_label = [keys[i] for i in range(3)]
        x_abbre, y_abbre, z_abbre = [values[i][2] for i in range(3)]
        # Calculate bin size and mid-bin values
        sorted_x, sorted_y, sorted_z = np.sort(x), np.sort(y), np.sort(z)
        bin_size_x, mid_x = (x.max() - x.min()) / 200, sorted_x[np.argmin(np.abs(sorted_x - (x.max() - x.min()) / 2))]
        bin_size_y, mid_y = (y.max() - y.min()) / 200, sorted_y[np.argmin(np.abs(sorted_y - (y.max() - y.min()) / 2))]
        bin_size_z, mid_z = (z.max() - z.min()) / 200, sorted_z[np.argmin(np.abs(sorted_z - (z.max() - z.min()) / 2))]

        data_e = np.column_stack([x, y, z])[(z >= mid_z - bin_size_z / 2) & (z <= mid_z + bin_size_z / 2)]
        data_f = np.column_stack([x, z, y])[(y >= mid_y - bin_size_y / 2) & (y <= mid_y + bin_size_y / 2)]
        data_g = np.column_stack([y, z, x])[(x >= mid_x - bin_size_x / 2) & (x <= mid_x + bin_size_x / 2)]

        unique_coords_e, _, _, counts_e = statistics(data_e)
        unique_coords_f, _, _, counts_f = statistics(data_f)
        unique_coords_g, _, _, counts_g = statistics(data_g)
        #----------------------------> figure path <----------------------------#
        fig_file = os.path.join(f"{self.Path.fig}", fr"{self.variable.name}.Org.({z_abbre},{x_abbre},{y_abbre})")
        if os.path.exists(f"{fig_file}.pdf") and self.jump:
            echo.info(f"==>{fig_file}.pdf is already!")
            return True
        else:
            echo.info(f"{fig_file}.pdf")
        with PdfPages(f"{fig_file}.pdf") as pdf:
            # ----------------------------> plot figure<----------------------------#
            fig, gs = self.set_style((5, 2), (4, 4.5), (0.3, 0.5))
            ax_a = fig.add_subplot(gs[0:2, 0:2], projection='3d')
            ax_b, ax_c, ax_d = [fig.add_subplot(gs[0, i]) for i in [2, 3, 4]]
            ax_e = fig.add_subplot(gs[1, 2], sharex=ax_b, sharey=ax_b)
            ax_f = fig.add_subplot(gs[1, 3], sharex=ax_c, sharey=ax_c)
            ax_g = fig.add_subplot(gs[1, 4], sharex=ax_d, sharey=ax_d)
            # ----------------------------> ax_a <----------------------------#
            sc_a = ax_a.scatter(simp_x, simp_y, simp_z, c=simp_z, cmap='rainbow') #, vmin=df_grp['mean'].min(), vmax=df_grp['mean'].max())
            #ax_a.axhline(y=mid_y, linestyle='--', lw=1.5, color='black')  # Selected Particle ID
            #ax_a.axvline(x=mid_x, linestyle='--', lw=1.5, color='black')  # Selected Time frame
            title = fr'({z_label}, {x_label}, {y_label}) in 3D Space'
            #self.colorbar(fig, ax_a, sc_a, z_label, True)
            self.colorbar(ax_a, simp_z, z_label, True)
            self.set_axes(ax_a, (simp_x, simp_y, simp_z), (x_label, y_label, z_label), title, is_3D=True, rotation=30, loc="center")
            self.adding(ax_a, "(a)")
            ## ----------------------------> ax_bcd <----------------------------#
            for ax, data, axis_labels, note in zip([ax_b, ax_c, ax_d],
                                             [np.column_stack([simp_x, simp_y, simp_z]), np.column_stack([simp_x, simp_z, simp_y]), np.column_stack([simp_y, simp_z, simp_x])],
                                             [(x_label, y_label, z_label), (x_label, z_label, y_label), (y_label, z_label, x_label)],
                                             ['(b)', '(c)', '(d)']):
                unique_coords, mean_values, std_values = statistics(data)[:3]
                sc = ax.scatter(unique_coords[:, 0], unique_coords[:, 1], c=mean_values, s=(std_values + 1) * 10, cmap='rainbow', alpha=0.7)
                title = fr'$\langle$ {axis_labels[2]} $\rangle$ in {axis_labels[0]}-{axis_labels[1]} Space'
                self.colorbar(ax, mean_values, axis_labels[2])
                self.set_axes(ax, (unique_coords[:, 0], unique_coords[:, 1], None), axis_labels, title)
                self.adding(ax, note)

            # ----------------------------> ax_efg <----------------------------#
            bin_set = [(np.around(mid_z - bin_size_z / 2, 2), np.around(mid_z + bin_size_z / 2, 2)),
                            (np.around(mid_y - bin_size_y / 2, 2), np.around(mid_y + bin_size_y / 2, 2)),
                            (np.around(mid_x - bin_size_x / 2, 2), np.around(mid_x + bin_size_x / 2, 2))]
            unique_coords = [unique_coords_e, unique_coords_f, unique_coords_g]
            counts_set = [counts_e, counts_f, counts_g]
            for ax, data, bin, counts, axis_labels, note in zip([ax_e, ax_f, ax_g],
                                                          unique_coords, bin_set, counts_set,
                                                          [(x_label, y_label, z_label), (x_label, z_label, y_label), (y_label, z_label, x_label)],
                                                          ['(e)', '(f)', '(g)']):
                sc = ax.scatter(data[:, 0], data[:, 1], s=counts*50, color="blue", alpha=0.6)
                # axis settings
                ax.set_title(fr'{axis_labels[2]}$_0\ \in$ [{bin[0]}, {bin[1]}]', loc='right', fontsize=20)
                ax.set_xlabel(axis_labels[0], fontsize=20)
                ax.set_ylabel(axis_labels[1], fontsize=20)
                self.adding(ax, note)

            # ----------------------------> save fig <----------------------------#
            pdf.savefig(plt.gcf(), dpi=500, transparent=True)
            # ax.legend(loc='upper left', frameon=False, ncol=int(np.ceil(len(Arg1) / 5.)), columnspacing = 0.1, labelspacing = 0.1, bbox_to_anchor=[0.0, 0.955], fontsize=10)
            #fig.savefig(f"{fig_file}.png", format="png", dpi=1000, transparent=True)
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
        for indices in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
            x_label, y_label, z_label = [keys[i] for i in indices]
            x_abbre, y_abbre, z_abbre = [values[i][2] for i in indices]
            x, y, z = [values[i][1] for i in indices]

            hist_x, x_bins, x_bin_centers, x_range, pdf_x = self.dist_data(x)
            hist_y, y_bins, y_bin_centers, y_range, pdf_y = self.dist_data(y)
            hist_2D = np.histogram2d(x, y, bins=[x_bins, y_bins], density=True)[0]
            hist_x_at_y, hist_y_at_x = hist_2D[:, bin_id]/np.sum(hist_2D[:, bin_id]), hist_2D[bin_id, :]/np.sum(hist_2D[bin_id, :])
            #hist_x_at_y, hist_y_at_x = np.nan_to_num(hist_x_at_y), np.nan_to_num(hist_y_at_x)

            #----------------------------> figure settings <----------------------------#
            fig_file = os.path.join(f"{self.Path.fig}", f"{self.variable.name}.Dist.f^{z_abbre}({x_abbre},{y_abbre})")
            if os.path.exists(f"{fig_file}.pdf") and self.jump:
                echo.info(f"==>{fig_file}.pdf is already!")
                return True
            else:
                echo.info(f"{fig_file}.pdf")
            with PdfPages(f"{fig_file}.pdf") as pdf:
                # ----------------------------> setup: canvas<----------------------------#
                fig, gs = self.set_style((4, 2), (4.5, 4.5), (0.5, 0.5))
                ax_a = fig.add_subplot(gs[0:2, 0:2])
                axes = [[fig.add_subplot(gs[i, j], sharex=ax_a if i == 0 else None, sharey=ax_a if i == 1 else None)
                            for j in range(2, 4)] for i in range(2)]
                # Plot fz(x,y)
                heatmap = ax_a.pcolormesh(x_bins, y_bins, hist_2D.T, shading='auto', cmap='rainbow')
                ax_a.axhline(y=y_bin_centers[bin_id], linestyle='--', lw = 1.5, color='black')  # Selected Particle ID
                ax_a.axvline(x=x_bin_centers[bin_id], linestyle='--', lw = 1.5, color='black')  # Selected Time frame
                title = fr"$f^{{{z_label.replace('$', '')}}}$({x_label},{y_label})"
                self.colorbar(ax_a, hist_2D.T, fr"$f^{{{z_label.replace('$', '')}}}$")
                self.set_axes(ax_a, (x_bin_centers, y_bin_centers, None), (x_label, y_label, None), title, loc="center")

                # Plot Fz(x;y0)
                axes[0][0].bar(x_bin_centers, hist_x_at_y, width=(x_bins[1] - x_bins[0]), alpha = 0.7, label="histogram")
                axes[0][0].set_title(fr"{y_label}$_0$ = {y_bin_centers[bin_id]:.2f}", loc='right', fontsize=20)
                axes[0][0].tick_params(axis='x', rotation=45)
                axes[0][0].set_xlabel(f"{x_label}", fontsize=20)
                axes[0][0].set_ylabel(fr"$f^{{{z_label.replace('$', '')}}}$({x_label}; {y_label}$_0$)", fontsize=20)
                axes[0][0].set_ylim(0, max(hist_x_at_y) * 1.1)

                # Plot Fzy(x)
                axes[0][1].bar(x_bin_centers, hist_x, width=(x_bins[1] - x_bins[0]), alpha = 0.7, label="histogram")
                axes[0][1].plot(x_range, pdf_x, 'r', label='PDF')
                axes[0][1].set_title("Distribution", loc='right', fontsize=20)
                axes[0][1].tick_params(axis='x', rotation=45)
                axes[0][1].set_xlabel(f"{x_label}", fontsize=20)
                axes[0][1].set_ylabel(fr"$f^{{{z_label.replace('$', '')}}}_{{{y_label.replace('$', '')}}}$({x_label})", fontsize=20)
                axes[0][1].set_ylim(0, max(hist_x) *1.1)

                # Plot Fz(y;x0)
                axes[1][0].barh(y_bin_centers, hist_y_at_x, height=(y_bins[1] - y_bins[0]), alpha = 0.7, label="histogram")
                axes[1][0].set_title(fr"{x_label}$_0$ = {x_bin_centers[bin_id]:.2f}", loc='right', fontsize=20)
                axes[1][0].set_xlabel(fr"$f^{{{z_label.replace('$', '')}}}$({y_label}; {x_label}$_0$)", fontsize=20)
                axes[1][0].set_ylabel(f"{y_label}", fontsize=20)
                axes[1][0].set_xlim(0, max(hist_y_at_x)*1.1)

                # Plot Fzx(y)
                axes[1][1].barh(y_bin_centers, hist_y, height=(y_bins[1] - y_bins[0]), alpha = 0.7, label="histogram")
                axes[1][1].plot(pdf_y, y_range, 'r', label='PDF')
                axes[1][1].set_title('Distribution', loc='right', fontsize=20)
                axes[1][1].set_xlabel(fr"$f^{{{z_label.replace('$', '')}}}_{{{x_label.replace('$', '')}}}$({y_label})", fontsize=20)
                axes[1][1].set_ylabel(f"{y_label}", fontsize=20)
                axes[1][1].set_xlim(0, max(hist_y)*1.1)

                # ----------------------------> linewidth <----------------------------#
                for ax, note in zip([ax_a, axes[0][0], axes[0][1], axes[1][0], axes[1][1]], ['(a)', '(b)', '(c)', '(d)', '(e)',]):
                    self.adding(ax, note, (-0.3, 0.9))

                timer.count("saving figure")
                # ----------------------------> save fig <----------------------------#
                pdf.savefig(plt.gcf(), dpi=500, transparent=True)

                #print("saving png......")
                # ax.legend(loc='upper left', frameon=False, ncol=int(np.ceil(len(Arg1) / 5.)), columnspacing = 0.1, labelspacing = 0.1, bbox_to_anchor=[0.0, 0.955], fontsize=10)
                #fig.savefig(f"{fig_file}.png", format="png", dpi=1000, transparent=True)
                plt.show()
                plt.close()
            # -------------------------------Done!----------------------------------------#
        timer.stop()
        return False
    def distics(self):
        print("-----------------------------------Done!--------------------------------------------")
    def relation(self):
        print("Done!")
    ##################################################################
    def org(self, data, variable):
        timer = Timer("Plot")
        timer.start()

        self.data = data
        self.variable = variable #pass to original and distribution
        self.data_dict, self.simp_dict = self.data2dict(), self.data2dict(flag=True)
        #plotting
        self.original()
        self.distribution()
        #self.distics()
        timer.count("plot")
        timer.stop()
#############################################################################################################
class Plotter3D(BasePlot):
    def exp_seprate(self):
        timer = Timer(f"{self.variable.name}: Expand3D")
        timer.start()

        columns_set = permutate([self.variable.name] + self.variable.paras)
        data_set = [tuple(self.df[label].values for label in label_set) for label_set in columns_set]
        labels_set = [list(map(lambda x: var2str(self.variable.name)[0] if x == self.variable.name else x, label_set)) for label_set in columns_set]
        for i, (data, columns, labels) in enumerate(zip(data_set, columns_set, labels_set)):
            f, x, y, z = data
            flabel, xlabel, ylabel, zlabel = labels
            fcolumn, xcolumn, ycolumn, zcolumn = columns
            # ----------------------------> figure.pdf and jump<----------------------------#
            fig_save = os.path.join(os.path.dirname(self.fig_save), f"(r,s,t){fcolumn}({xcolumn},{ycolumn};{zcolumn})")
            fig_file = self.fig_path("Exp", fig_save)
            with PdfPages(f"{fig_file}.pdf") as pdf:
                for iz in z:
                    # ----------------------------> setup: canvas<----------------------------#
                    cols, rows = 2, 1
                    fig, gs = self.set_style((cols, rows))
                    axes_2D = [fig.add_subplot(gs[0, j]) for j in range(0, cols)]
                    # ----------------------------> plotting <----------------------------#
                    self.scatter(fig, axes_2D[0], (x, f, y, iz), (xlabel, flabel, ylabel, zlabel), self.notes[0], log=True)
                    self.scatter(fig, axes_2D[1], (y, f, x, iz), (ylabel, flabel, xlabel, zlabel), self.notes[1], log=True)
                # ----------------------------> save fig <----------------------------#
                fig = plt.gcf()
                pdf.savefig(fig, dpi=500, transparent=True)
                #plt.show()
                plt.close()
        timer.stop()
        # -------------------------------Done!----------------------------------------#
    # -----------------------------------------------------------------------------------------
    def set_axes(self, ax, data, labels, title, is_3D=False, scatter=False, rotation=-60, loc="right", log=False):
        if is_3D:
            rotation, loc = 0, "center"
        # label settings
        ax.set_title(title, loc=loc, fontsize=20)
        axis_labels = {
            "x": labels[0],
            "y": labels[1],
            "z": labels[2] if is_3D else None
        }
        if log:
            lo, hi = 0.5, 2.0
            ax.set_xscale('log')
            ax.set_yscale('log')
            if is_3D:
                ax.set_zscale('log')
        else:
            ax.grid(True)
            if self.variable.name == "Rg":
                lo, hi = 0.2, 1.2
            else:
                lo, hi = 0, 1.2

        # Set axis limits and rotation
        for i, (axis_data, axis_name) in enumerate(zip(data[:2 + is_3D], "xyz"[:2 + is_3D])):
            getattr(ax, f'set_{axis_name}label')(axis_labels[axis_name], fontsize=20, labelpad=3 if axis_name == 'z' else 5)
            min_val, max_val = min(axis_data), max(axis_data)
            if scatter:
                getattr(ax, f'set_{axis_name}lim')((min_val - 1) * lo, max_val * hi)
               #getattr(ax, f'set_{axis_name}lim')(np.min(self.df[self.variable.name])* lo, np.max(self.df[self.variable.name])*hi)
            else:
                getattr(ax, f'set_{axis_name}lim')(min_val, max_val)
            if axis_name == 'x':
                ax.tick_params(axis='x', rotation=rotation)
    def colorbar(self, ax, data, label, is_3D=False):
        loc, pad = ("left", 0.05) if is_3D else ("right", 0.05)
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=Normalize(vmin=data.min(), vmax=data.max()))
        sm.set_array([])
        if HOST == "Linux":
            cbar = plt.colorbar(sm, ax=ax, pad=pad)
        elif HOST == "Darwin":
            cbar = plt.colorbar(sm, ax=ax, location=loc, pad=pad)
        cbar.ax.yaxis.set_ticks_position(loc)
        cbar.ax.set_xlabel(label, fontsize=20, labelpad=10)
    ##################################################################
    # --------------------------------------> scatter: project <-----------------------------------------
    def scatter(self, ax, data, labels, note, scatter=True, log=False):
        '''set_axes'''
        is_3D = False
        x, y, z, w = data
        xlabel, ylabel, zlabel, wlabel = labels
        legend_handles = []
        for idx, uw in enumerate(np.unique(w)):
            mask = (w == uw)
            marker = self.markers[idx % len(self.markers)]
            norm = Normalize(vmin=z.min(), vmax=z.max())
            colors = [plt.get_cmap("rainbow")(norm(zi)) for zi in z[mask]]
            for xi, yi, zi, ci in zip(x[mask], y[mask], z[mask], colors):
                ax.scatter(xi, yi, c='none', s=100, marker=marker, edgecolors=ci, linewidths=3)
            #legend_handles.append(ax.scatter([], [], c='white', s=100, edgecolors='black', facecolor='None', linewidths=2, marker=marker, label=f'{uw}'))
            legend_handles.append(ax.scatter([], [], s=100, edgecolors='black', facecolor='None', linewidths=2, marker=marker, label=f'{uw}'))
        ax.legend(handles=legend_handles, title=wlabel, frameon=False)
        # colorbar
        self.colorbar(ax, z, zlabel, is_3D)
        self.set_axes(ax, (x, y), (xlabel, ylabel), f"({ylabel},{xlabel}) with {zlabel}-{wlabel}", is_3D, scatter, 0, log=log)
        self.adding(ax, note, -0.2, is_3D)
    def project(self):
        '''scatter'''
        timer = Timer(f"{self.variable.name}: Project3D")
        timer.start()
        labels_set = permutate([self.variable.name] + self.variable.paras)
        data_set = [tuple(self.df[label].values for label in label_set) for label_set in labels_set]
        labels_set = [list(map(lambda x: var2str(self.variable.name)[0] if x == self.variable.name else x, label_set)) for label_set in labels_set]
        # ----------------------------> figure.pdf <----------------------------#
        fig_file = self.fig_path("Proj")
        log = True if self.variable.name == "Rg" else False
        with PdfPages(f"{fig_file}.pdf") as pdf:
            for i, (data, labels) in enumerate(zip(data_set, labels_set)):
                # ----------------------------> data and labels <----------------------------#
                f, x, y, z = data
                flabel, xlabel, ylabel, zlabel = labels
                # ----------------------------> setup: canvas<----------------------------#
                cols, rows = 3, 1
                fig, gs = self.set_style((cols, rows), edge=(0.5, 1.0))
                axes_3D = [fig.add_subplot(gs[0, 0], projection='3d')]
                axes_2D = [fig.add_subplot(gs[0, j]) for j in range(1,cols)]
                # ----------------------------> plotting: a <----------------------------#
                sc = axes_3D[0].scatter(x, y, z, c=f, cmap="rainbow", s=100)
                self.colorbar(axes_3D[0], f, flabel, True)
                self.set_axes(axes_3D[0], (x, y, z, f), (xlabel, ylabel, zlabel, flabel), f"{flabel} in {xlabel}-{ylabel}-{zlabel} Space", True, True, 0)
                self.adding(axes_3D[0], self.notes[0], -0.2, True)
                # ----------------------------> plotting: bc <----------------------------#
                self.scatter(axes_2D[0], (x, y, f, z), (xlabel, ylabel, flabel, zlabel), self.notes[1])
                self.scatter(axes_2D[1], (x, f, y, z), (xlabel, flabel, ylabel, zlabel), self.notes[2], log=log)
                # ----------------------------> save fig <----------------------------#
                fig = plt.gcf()
                pdf.savefig(fig, dpi=500, transparent=True)
                plt.show()
                plt.close()
        timer.count("Project3D")
        timer.stop()
        # -------------------------------Done!----------------------------------------#
        return False
    #--------------------------------------> scatter: expand <-----------------------------------------
    def scatter_exp(self, uw, ax, data, labels, note, scatter=True, log=False):
        '''set_axes'''
        is_3D = False
        x, y, z, w = data
        xlabel, ylabel, zlabel, wlabel = labels

        if is_3D:
            sc = ax.scatter(x, y, z, c=w, cmap="rainbow", s=100)
            self.colorbar(ax, w, wlabel, is_3D)
            self.set_axes(ax, data, labels, f"{wlabel} in {xlabel}-{ylabel}-{zlabel} Space", is_3D, scatter, 0)
            self.adding(ax, note, -0.2, is_3D)
        else:
            legend_handles = []
            uz, uw = np.unique(z), np.unique(w)[0]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(uz)))
            color_map = list(zip(self.markers[:len(uz)], colors))

            for zi, (marker, color) in zip(uz, color_map):
                mask = (z==zi)
                coef, x_range, y_range = scale(x[mask], y[mask])
                if coef and log:
                    ax.plot(x_range, y_range, color=color, linestyle='--')
                    title = fr'{zlabel}, ${self.variable.scale}$'
                    label = f'{label2mark(zi, zlabel)}, {coef}'
                else:
                    ax.plot(x[mask], y[mask], color=color, linestyle='--')
                    title = fr'{zlabel}'
                    label = f'{label2mark(zi, zlabel)}'
                for xi, yi in zip(x[mask], y[mask]):
                    ax.scatter(xi, yi, c='none', s=100, marker=marker, edgecolors=color, linewidths=3)
                #legend_handles.append(ax.scatter([], [], c='white', s=100, marker=marker, edgecolors=color, facecolor='None', linewidths=2, label=label))
                legend_handles.append(ax.scatter([], [], s=100, marker=marker, edgecolors=color, facecolor='None', linewidths=2, label=label))
            ax.legend(title=title, frameon=False, title_fontsize=15, ncol=int(len(legend_handles)/5)+1)
            # colorbar
            #self.colorbar(ax, z, zlabel, is_3D)
            self.set_axes(ax, (x, y), (xlabel, ylabel), f"{ylabel}({xlabel},{zlabel};{wlabel}={uw})", is_3D, scatter, 0, log=log)
            self.adding(ax, note, -0.12, is_3D)
    def expand(self):
        '''scatter_exp'''
        timer = Timer(f"{self.variable.name}: Expand3D")
        timer.start()
        # ----------------------------> start <----------------------------#
        columns_set = permutate([self.variable.name] + self.variable.paras)
        #columns_set = [[self.variable.name] + self.variable.paras]
        data_set = [tuple(self.df[label].values for label in label_set) for label_set in columns_set]
        labels_set = [list(map(lambda x: var2str(self.variable.name)[0] if x == self.variable.name else x, label_set)) for label_set in columns_set]
        # ----------------------------> figure.pdf <----------------------------#
        fig_file = self.fig_path("Exp")
        log = True if self.variable.name == "Rg" else False
        with PdfPages(f"{fig_file}.pdf") as pdf:
            for i, (data, labels) in enumerate(zip(data_set, labels_set)):
                f, x, y, z = data
                flabel, xlabel, ylabel, zlabel = labels
                unique_x, unique_y, unique_z = np.unique(x), np.unique(y), np.unique(z)
                suplabel = f"{flabel} with {zlabel} fixed"
                # ----------------------------> setup: canvas<----------------------------#
                cols, rows  = 2, len(unique_z)
                fig, gs = self.set_style((cols, rows), (6, 5.5), (0.3, 0.3), (1.1, 1.5), title=suplabel)
                axes_2D = [fig.add_subplot(gs[i, j]) for i in range(rows) for j in range(cols)]
                for i, iz in enumerate(unique_z):
                    mask = (z == iz)
                    # ----------------------------> plotting <----------------------------#
                    self.scatter_exp(iz, axes_2D[2*i], (x[mask], f[mask], y[mask], z[mask]), (xlabel, flabel, ylabel, zlabel), self.notes[0], log=log)
                    self.scatter_exp(iz, axes_2D[2*i+1], (y[mask], f[mask], x[mask], z[mask]), (ylabel, flabel, xlabel, zlabel), self.notes[1], log=log)
                # ----------------------------> save fig <----------------------------#
                fig = plt.gcf()
                pdf.savefig(fig, dpi=500, transparent=True)
                plt.show()
                plt.close()
                timer.count(f'{self.variable.name}({xlabel}, {ylabel}; {zlabel})')
        timer.stop()
        # -------------------------------Done!----------------------------------------#
        return False
    # --------------------------------------> scatter: scale <-----------------------------------------
    def set_axes3D(self, ax, data, labels, title, rot=0, loc="center", log=False):
        # label settings
        axis_labels = {
            "x": labels[0],
            "y": labels[1],
            "z": labels[2],
        }
        ax.set_title(title, loc=loc, fontsize=20)
        # Set axis limits and rotation
        for i, (axis_data, axis_name) in enumerate(zip(data, "xyz")):
            min_val, max_val = min(axis_data), max(axis_data)
            if log:
                lo, hi = 0.5, 2.0
                getattr(ax, f'set_{axis_name}scale')('log')
            else:
                ax.grid(True)
                lo = 1.2 if min_val < 0 else 0.5
                hi = 0.1 if max_val < 0 else 1.2
            if axis_name == 'x':
                ax.tick_params(axis='x', rotation=rot)

            getattr(ax, f'set_{axis_name}label')(axis_labels[axis_name], fontsize=20, labelpad=12)
            getattr(ax, f'set_{axis_name}lim')(min_val * lo, max_val * hi)
    def set_axes2D(self, ax, data, labels, title, rot=0, loc="right", log=False):
        # label settings
        x, y = data
        xlabel, ylabel = labels
        ax.set_title(title, loc=loc, fontsize=20)
        ax.tick_params(axis='x', rotation=rot)
        # Set axis limits and rotation
        ax.set_xlabel(xlabel, fontsize=20, labelpad=3)
        ax.set_ylabel(ylabel, fontsize=20, labelpad=3)
        lo, hi = 0.98, 1.02
        xlo, xhi = hi if min(x) < 0 else lo, lo if max(x) < 0 else hi
        ylo, yhi = hi if min(y) < 0 else lo, lo if max(y) < 0 else hi
        ax.set_xlim(min(x) * xlo, max(x) * xhi)
        ax.set_ylim(min(y) * ylo, max(y) * yhi)
        if log:
            ax.set_xscale("log")
            xlo, xhi = 0.5, 2.0
            ax.set_xlim(min(x) * xlo, max(x) * xhi)
            ax.set_ylim(min(y) * ylo, max(y) * yhi)
        else:
            ax.grid(True)
    def scatter2D(self, ax, data, labels, note):
        '''set_axes2D'''
        x, y, z = data
        xlabel, ylabel, zlabel = labels
        uz = np.unique(z)
        log = True #if xlabel != "Pe" else False
        legend_handles = []
        colors = plt.cm.rainbow(np.linspace(0, 1, len(uz)))
        color_map = list(zip(self.markers[:len(uz)], colors))
        for zi, (marker, color) in zip(uz, color_map):
            mask = (z == zi)
            ax.plot(x[mask], y[mask], color=color, linestyle="--")
            for xi, yi in zip(x[mask], y[mask]):
                ax.scatter(xi, yi, c='none', s=100, marker=marker, edgecolors=color, linewidths=3)
            #legend_handles.append(ax.scatter([], [], c='white', s=100, marker=marker, edgecolors=color, facecolor='None', linewidths=2, label=f'{label2mark(zi, zlabel)}'))
            legend_handles.append(ax.scatter([], [], s=100, marker=marker, edgecolors=color, facecolor='None', linewidths=2, label=f'{label2mark(zi, zlabel)}'))
        ax.legend(title=fr'{zlabel}', frameon=False, title_fontsize=15, ncol=int(len(legend_handles) / 5) + 1)
        self.set_axes2D(ax, (x, y), (xlabel, ylabel), f"{ylabel}({xlabel}) with {zlabel}", log=log)
        self.adding(ax, note, -0.12, False)
    def cal_nu(self, data, labels):
        unique_y, unique_z = data
        flabel, xlabel, ylabel, zlabel = labels
        df_nu = pd.DataFrame(columns=['nu', ylabel, zlabel])
        for (uy, uz) in [(uy, uz) for uy in unique_y for uz in unique_z]:
            df_yz = self.df[(self.df[ylabel] == uy) & (self.df[zlabel] == uz)]  # query: z, w
            coef = np.float32(scale(df_yz[xlabel].to_numpy(), df_yz[flabel].to_numpy())[0])
            if coef:
                df_nu = df_nu.append({'nu': coef, ylabel: uy, zlabel: uz}, ignore_index=True)
        return df_nu
    def scale(self):
        '''scatter2D'''
        timer = Timer(f"{self.variable.name}: scale")
        timer.start()
        # ----------------------------> setup: data and labels <----------------------------#
        labels_set = permutate([self.variable.name] + self.variable.paras)
        #labels_set = [self.variable.name] + self.variable.paras
        data_set = [tuple(self.df[label].values for label in label_set) for label_set in labels_set]
        # ----------------------------> figure.pdf <----------------------------#
        fig_file = self.fig_path("Scale")
        with PdfPages(f"{fig_file}.pdf") as pdf:
            for data, labels in zip(data_set, labels_set):
                df_xnu = self.cal_nu((np.unique(data[2]), np.unique(data[3])), labels)
                f, x, y = df_xnu['nu'].to_numpy(), df_xnu[labels[2]].to_numpy(), df_xnu[labels[3]].to_numpy()
                #print(f"nu: {f}\n N:{x}\n W:{y}")
                xlabel, ylabel = labels[2:4]
                flabel = fr"$^{{{self.variable.name}}}_{{{labels[1]}}}{self.variable.scale}$"
                suplabel = f"{flabel}({xlabel}, {ylabel})"
                # ----------------------------> setup: canvas<----------------------------#
                cols, rows  = 2, 2
                fig, gs = self.set_style((cols, rows), (6, 5.5), (0.35, 0.5), title=suplabel)
                axe3D = [fig.add_subplot(gs[0, 0], projection='3d')]
                axe2D = [fig.add_subplot(gs[i, j]) for i in range(rows) for j in range(cols) if not (i == 0 and j == 0)]
                # ----------------------------> plotting <----------------------------#
                axe3D[0].scatter(x, y, f, c=f, cmap="rainbow", s=100)
                self.colorbar(axe3D[0], f, flabel, True)
                self.set_axes3D(axe3D[0], (x, y, f), (xlabel, ylabel, ""), f"{flabel}({xlabel},{ylabel})")
                self.adding(axe3D[0], self.notes[0], -0.2)
                # ----------------------------> Fig bd <----------------------------#
                self.scatter2D(axe2D[0], (x, f, y), (xlabel, flabel, ylabel), self.notes[1])
                self.scatter2D(axe2D[2], (y, f, x), (ylabel, flabel, xlabel), self.notes[3])
                # ----------------------------> Fig c <----------------------------#
                axe2D[1].scatter(x, y, c=f, s=100, cmap="rainbow", label=f'{flabel}')
                self.colorbar(axe2D[1], f, flabel)
                self.set_axes2D(axe2D[1], (x, y), (xlabel, ylabel), f"{flabel}({xlabel},{ylabel})")
                self.adding(axe2D[1], self.notes[2], -0.2)
                # ----------------------------> arrange figures <----------------------------#
                plt.tight_layout(pad=3, w_pad=0.3, h_pad=0.5, rect=[0, 0, 0.95, 1])
                pos = axe3D[0].get_position().bounds
                axe3D[0].set_position([pos[0] - 0.02, pos[1], pos[2], pos[3]])
                # ----------------------------> save fig <----------------------------#
                fig = plt.gcf()
                pdf.savefig(fig, dpi=500, transparent=True)
                plt.show()
                plt.close()
                timer.count(f'({labels[1]})n{labels[0]}({xlabel}, {ylabel})')
        timer.stop()
class Plotter4D(BasePlot):
    def colorbar(self, ax, data, label, loc="right"):
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=Normalize(vmin=min(data), vmax=max(data)))
        sm.set_array([])
        if HOST == "Linux":
            cbar = plt.colorbar(sm, ax=ax, pad=0.05)
        elif HOST == "Darwin":
            cbar = plt.colorbar(sm, ax=ax, location=loc, pad=0.05)
        cbar.ax.yaxis.set_ticks_position(loc)
        cbar.ax.set_xlabel(label, fontsize=20, labelpad=10)
    def set_axes2D(self, ax, title, labels, lims, note=None, legends=None, log=False):
        flabel, xlabel, ylabel, zlabel = labels
        xlim, ylim = lims
        # ----------------------------> set axes <----------------------------#
        ax.set_title(title, loc="right", fontsize=20)
        ax.set_xlabel(xlabel, fontsize=20, labelpad=4)
        ax.set_ylabel(flabel, fontsize=20, labelpad=4)
        if log:
            ymin, ymax = 0.2, 3.0
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ymin, ymax = 0.1, 1.2
            ax.grid(True)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0] * ymin, ylim[1] * ymax)
        #sys.exit()
        # ----------------------------> legends <----------------------------#
        if legends:
            ax.legend(handles=legends, title=zlabel, frameon=False, title_fontsize=15, fontsize=15, ncol=int(len(legends)/5)+1)
        else:
            ax.legend(title=ylabel, frameon=False, title_fontsize=15, fontsize=15)
        # ----------------------------> adding <----------------------------#
        ax.annotate(note, (-0.15, 0.9), textcoords="axes fraction", xycoords="axes fraction", va="center", ha="center", fontsize=20)
        ax.tick_params(axis='both', which="both", width=2, labelsize=15, pad=5.0)
        for spine in ["bottom", "left", "right", "top"]:  # axes lines
            ax.spines[spine].set_linewidth(2)
    ##################################################################
    def scatter_exp(self, w, ax, data, labels, note):
        '''set_axes2D'''
        x, unique_y, unique_z, unique_w = data
        flabel, xlabel, ylabel, zlabel, wlabel = labels
        f_flat = np.concatenate(self.df[self.variable.name].values)
        xlim, ylim = (min(x), max(x)), (min(f_flat), max(f_flat))

        df_w = self.df[self.df[wlabel] == w]  # query: w
        data_w = np.concatenate(df_w[self.variable.name].values)
        label_list = []
        plot_line(ax, (x[200], x[-1], max(data_w) * 2.5), 1)
        plot_line(ax, (x[1], x[6], min(data_w[data_w != 0]) * 0.3), 2)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_y)))
        color_map = dict(zip(unique_y, colors))
        marker_map = dict(zip(unique_z, self.markers[:len(unique_z)]))
        for _, wdata in df_w.iterrows():
            if self.variable.name == "MSD":
                # ax.plot(x[1:], wdata[self.variable.name][1:], color=color, linestyle="--")
                logt, logf = np.log10(x[1:]), np.log10(wdata[self.variable.name][1:])
                post_turning, (slope, intercept) = turn_seg((logt, logf))
                ax.plot(10 ** post_turning[0], slope * 10 ** post_turning[0] + intercept, color=color_map[wdata[ylabel]], linestyle="--")
                #ax.axvline(10 ** turning[0][0], color=color_map[wdata[ylabel]], linestyle='--')
                #ax.axvline(10 ** turning[0][-1], color=color_map[wdata[ylabel]], linestyle='--')
            x_samp = np.concatenate([x[:100], x[100::10]])
            y_samp = np.concatenate([wdata[self.variable.name][:100], wdata[self.variable.name][100::10]])
            ax.scatter(x_samp, y_samp, c='none', s=50, edgecolors=color_map[wdata[ylabel]],
                                linewidths=2, facecolors='none', marker=marker_map[wdata[zlabel]])
        # colorbar
        self.colorbar(ax, unique_y, ylabel)
        legends = [plt.Line2D([0], [0], marker=marker_map[z], color='w', label=f"{z}",
                              markerfacecolor='none', markeredgecolor='k') for z in unique_z]
        title = f'{flabel}({xlabel}, {ylabel}, {zlabel}; {wlabel}={w})'
        self.set_axes2D(ax, title, (flabel, xlabel, ylabel, zlabel), (xlim, ylim), note, legends=legends, log=True)
    def expand(self):
        '''scatter_exp'''
        timer = Timer(f"{self.variable.name}: Expand4D1")
        timer.start()
        fig_save = os.path.join(os.path.dirname(self.fig_save), f"(r,s){self.variable.name}(t,Pe,N;W)")
        # ----------------------------> figure.pdf <----------------------------#
        fig_file = self.fig_path("Exp1D", fig_save)
        with PdfPages(f"{fig_file}.pdf") as pdf:
            for columns in permutate([self.variable.name] + self.variable.paras, 't'): #[[self.variable.name, 't', 'Pe', 'N', 'W']]:
                # ----------------------------> prepare: data and labels<----------------------------#
                xlabel, ylabel, zlabel, wlabel = columns[1:]
                flabel = var2str(self.variable.name)[0]
                suplabel = f"{flabel}({xlabel}, {ylabel}, {zlabel}) with {wlabel} fixed"
                x = self.df["dt"][0] * np.arange(len(self.df[self.variable.name][0]))
                unique_y, unique_z, unique_w = self.df[ylabel].unique(), self.df[zlabel].unique(), self.df[wlabel].unique()
                # ----------------------------> setup: canvas<----------------------------#
                cols, rows = 2, len(unique_w)
                fig, gs = self.set_style((cols, rows), title=suplabel)
                axes = [[fig.add_subplot(gs[i, j]) for j in range(cols)] for i in range(rows)]
                # ----------------------------> plotting <----------------------------#
                for i, w in enumerate(unique_w):
                    self.scatter_exp(w, axes[i][0], (x, unique_y, unique_z, unique_w), (flabel, xlabel, ylabel, zlabel, wlabel), self.notes[0])
                    self.scatter_exp(w, axes[i][1], (x, unique_z, unique_y, unique_w), (flabel, xlabel, zlabel, ylabel, wlabel), self.notes[1])
                # ----------------------------> save fig <----------------------------#
                fig = plt.gcf()
                pdf.savefig(fig, dpi=500, transparent=True)
                plt.show()
                plt.close()
                timer.count(f'{self.variable.name}({xlabel}, {ylabel}, {zlabel}; {wlabel})')
        timer.stop()
    def expand2D(self):
        '''scatter2D'''
        timer = Timer(f"{self.variable.name}: Expand4D2")
        timer.start()
        fig_save = os.path.join(os.path.dirname(self.fig_save), f"(r,s){self.variable.name}(t,Pe;N,W)")
        # ----------------------------> figure.pdf <----------------------------#
        fig_file = self.fig_path("Exp2D", fig_save)
        with PdfPages(f"{fig_file}.pdf") as pdf:
            for columns in permutate([self.variable.name] + self.variable.paras, 't'): #[[self.variable.name, 't'] + self.variable.paras]:
                # ----------------------------> prepare: data and labels<----------------------------#
                xlabel, ylabel, zlabel, wlabel = columns[1:]
                flabel = var2str(self.variable.name)[0]
                suplabel = f"{flabel}({xlabel}, {ylabel}) with ({zlabel}, {wlabel}) fixed"
                x, f_flat = self.df["dt"][0] * np.arange(len(self.df[self.variable.name][0])), np.concatenate(self.df[self.variable.name].values)
                unique_y, unique_z, unique_w = self.df[ylabel].unique(), self.df[zlabel].unique(), self.df[wlabel].unique()
                xlim, ylim = (min(x), max(x)), (min(f_flat[f_flat != 0]), max(f_flat))
                # ----------------------------> setup: canvas<----------------------------#
                num_axis = len(unique_z) * len(unique_w)
                cols, rows = 2, num_axis // 2 + num_axis % 2
                fig, gs = self.set_style((cols, rows), (5.5, 5.8), title=suplabel)
                color_map = list(zip(self.markers, plt.cm.rainbow(np.linspace(0, 1, len(unique_y)))))
                axes = [[fig.add_subplot(gs[i, j]) for j in range(cols)] for i in range(rows)]
                axes = [ax for row in axes for ax in row]
                if num_axis % 2:
                    axes[-1].axis('off')
                # ----------------------------> plotting <----------------------------#
                for ax, (z, w) in zip(axes, [(z, w) for z in unique_z for w in unique_w]):
                    df_zw = self.df[(self.df[zlabel] == z) & (self.df[wlabel] == w)]  # query: z, w
                    data_zw = np.concatenate(df_zw[self.variable.name].values)
                    plot_line(ax, (x[200], x[-1], max(data_zw)*2.5), 1)
                    plot_line(ax, (x[1], x[6], min(data_zw[data_zw != 0])*0.3), 2)
                    for (marker, color), y in zip(color_map, unique_y):
                        df_y = df_zw[df_zw[ylabel] == y] # query: y
                        for _, ydata in df_y.iterrows():
                            if self.variable.name == "MSD":
                                #ax.plot(x[1:], ydata[self.variable.name][1:], color=color, linestyle="--")
                                logt, logf = np.log10(x[1:]), np.log10(ydata[self.variable.name][1:])
                                post_turning, (slope, intercept) = turn_seg((logt, logf))
                                pre_point =  int(np.where(logt == post_turning[0][0])[0])
                                #pre_point = 100 if pre_point < 100 else pre_point
                                ax.plot(10 ** post_turning[0], slope * 10 ** post_turning[0] + intercept, color=color, linestyle="--")
                                #ax.axvline(10 ** turning[0][0], color=color, linestyle='--')
                                #ax.axvline(10 ** turning[0][-1], color=color, linestyle='--')
                                if (zlabel == "W" and z == 1.0) or (wlabel == "W" and w == 1.0):
                                    label = f'{y}: {slope/4:.4f}'
                                    labels = (flabel, xlabel, f"{ylabel}, D", zlabel)
                                else:
                                    label = f'{y}: {slope/6:.4f}'
                                    labels = (flabel, xlabel, f"{ylabel}, D", zlabel)
                            else:
                                label = y
                                labels = (flabel, xlabel, ylabel, zlabel)
                            x_samp = np.concatenate([x[:pre_point], x[pre_point::10]])
                            y_samp = np.concatenate([ydata[self.variable.name][:pre_point], ydata[self.variable.name][pre_point::10]])
                            ax.scatter(x_samp, y_samp, c='none', s=50, edgecolors=color, facecolors='none', marker=marker, linewidths=2, label=label)
                    title = f'{flabel}({xlabel}, {ylabel}; {zlabel}={z}, {wlabel}={w})'
                    self.set_axes2D(ax, title, labels, (xlim, ylim), log=True)
                # ----------------------------> save fig <----------------------------#
                fig = plt.gcf()
                pdf.savefig(fig, dpi=500, transparent=True)
                plt.show()
                plt.close()
                timer.count(f'{self.variable.name}({xlabel}, {ylabel}; {zlabel}, {wlabel})')
        timer.stop()
        return
    ##################################################################
    def scale(self):
        timer = Timer("Diff")
        timer.start()
        beta = Property(MSD.scale, "n"+MSD.name)
        dirfile = os.path.join(os.path.dirname(self.fig_save), f"(r,s,t){beta.path}({','.join(beta.paras)})")
        flabel, tlabel, xlabel, ylabel, zlabel = [self.variable.name, "t"] + self.variable.paras
        x, y, z = self.df[xlabel].unique(), self.df[ylabel].unique(), self.df[zlabel].unique()
        beta.df = pd.DataFrame(columns=[MSD.scale, xlabel, ylabel, zlabel])
        for (ix, iy, iz) in [(ux, uy, uz) for ux in x for uy in y for uz in z]:
            df_xyz = self.df[(self.df[xlabel] == ix) & (self.df[ylabel] == iy) & (self.df[zlabel] == iz)]
            f,t = df_xyz[flabel].iloc[0], df_xyz["dt"].iloc[0] * range((len(df_xyz[flabel].iloc[0])))
            if self.variable.name == "MSD":
                logt, logf = np.log10(t[1:]), np.log10(f[1:])
                _, (slope, intercept) = turn_seg((logt, logf))
                if slope:
                    if (xlabel == "W" and ix == 1.0) or (ylabel == "W" and iy == 1.0) or (zlabel == "W" and iz == 1.0):
                        diff = slope / 4
                    else:
                        diff = slope / 6
                    beta.df = beta.df.append({MSD.scale: diff, xlabel: ix, ylabel: iy, zlabel: iz}, ignore_index=True)
        plotter3 = Plotter3D(beta, dirfile)
        plotter3.project()
        plotter3.expand()
        timer.stop()
#############################################################################################################
class Timer:
    '''timer = Timer("Distribution"); timer.start();timer.count("saving figure");timer.stop();'''
    def __init__(self, tip="start", func=time.perf_counter):
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
        print(f"=============={self.tip}==============")
    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        echo.info(f"-------------------------------{self.tip}: Done!----------------------------------------")
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
def convert2array(x):
    # 如果x已经是一个列表，numpy数组，直接转换为numpy数组
    if isinstance(x, (list, np.ndarray, tuple)):
        return np.array(x)
    # 如果x是一个单一的数值，转换为包含该数值的numpy数组
    elif isinstance(x, (int, float, str)):
        return np.array([x])
    else:
        raise ValueError("Unsupported type!")
def label2mark(f, flabel):
    if flabel == "Pe" and f < 1e-6:
        mark = "Passive"
    elif flabel == "W" and f< 1e-6:
        mark = "Free"
    else:
        mark = f
    return mark
def var2str(variable):
    # transform to latex
    if variable.lower() == 'msd':
        return r'$\mathrm{MSD}$', "MSD"
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
def statistics(data):
    '''for plot original'''
    unique_coords, indices, counts = np.unique(data[:, :2], axis=0, return_inverse=True, return_counts=True)
    sum_values = np.bincount(indices, weights=data[:, 2])
    mean_values = sum_values / counts
    sum_values_squared = np.bincount(indices, weights=data[:, 2] ** 2)
    std_values = np.sqrt( (sum_values_squared / counts) - (mean_values ** 2))
    return unique_coords, mean_values, std_values, counts
def describe(dataset, str="data", flag = True):
    '''for debug'''
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
def permutate(array, insert=None):
    list = [array] + [array[:1] + array[i:] + array[1:i] for i in range(2, len(array))]
    if insert:
        return [[elem[0], insert] + elem[1:] for elem in list]
    else:
        return list
def io_data(file_path, data_frame=None):
    if data_frame is not None:
        data_frame.to_pickle(f'{file_path}.pkl')
        if os.path.abspath(__file__) != f"{file_path}.py":
            shutil.copy2(os.path.abspath(__file__), f"{file_path}.py")
        return False
    else:
        return pd.read_pickle(file_path)
def unwrap_x(data, Lx):
    frames = data.shape[1]
    for i in range(1, frames):
        while True:
            dx = data[:, i, :, 0] - data[:, i - 1, :, 0]
            crossed = np.abs(dx) > Lx / 2
            if not np.any(crossed):
                break
            for file_idx in range(data.shape[0]):
                for atom_idx in range(data.shape[2]):
                    if crossed[file_idx, atom_idx]:
                        data[file_idx, i:, atom_idx, 0] -= (np.sign(dx) * Lx * crossed)[file_idx, atom_idx]
    return data

#scale
def scale(x, y):
    if x[0] < 1e-6:
        log_x, log_y = np.log10(x[1:]), np.log10(y[1:])
    else:
        log_x, log_y = np.log10(x), np.log10(y)
    if len(log_x) > 1:
        slop, intercept = np.polyfit(log_x, log_y, 1)
        coef = '{:.3f}'.format(slop)
        poly_fit = np.poly1d((slop, intercept))
        x_range = np.linspace(np.min(log_x) - np.log10(1.41), np.max(log_x) + np.log10(1.41), 100)
        y_range = poly_fit(x_range)
        x_range, y_range = np.power(10, x_range), np.power(10, y_range)
        return coef, x_range, y_range
    else:
        return False, log_x, log_y
def turn_seg(data, bins=20, find_min=True):
    x, y = data
    # 计算一阶和二阶导数
    dx, dy = 0.5 * (x[:-1] + x[1:]), np.diff(y) / np.diff(x)  # 计算中点，以保持导数的维度一致
    d2x, d2y = 0.5 * (dx[:-1] + dx[1:]), np.diff(dy) / np.diff(dx)
    log_intervals = np.linspace(x.min(), x.max(), num=bins)
    tps, segs, slopes = [], [], []
    for i in range(len(log_intervals) - 1):
        start, end = log_intervals[i], log_intervals[i + 1]
        mask = (d2x >= start) & (d2x <= end)
        peaks, _ = find_peaks(d2y[mask])
        if peaks.size > 0:
            peak = peaks[np.argmax(d2y[mask][peaks])]  # 选择最大的极值点
            tps.append(d2x[mask][peak])

    # 创建每个分段的数据
    for i in range(len(tps) + 1):
        if i == 0:
            segment = (x[x <= tps[i]], y[x <= tps[i]])
        elif i == len(tps):
            segment = (x[x > tps[i - 1]], y[x > tps[i - 1]])
        else:
            segment = (x[(x > tps[i - 1]) & (x <= tps[i])],  y[(x > tps[i - 1]) & (x <= tps[i])])
        segs.append(segment)
        slopes.append(linregress(segment[0], segment[1])[0])
    # 继续求二阶导的极大值点：并进一步确定转折区间
    dslopes = np.array(np.diff(slopes))
    if find_min:
        peaks = np.array(find_peaks(-dslopes)[0])
    else:
        peaks = np.array(find_peaks(dslopes)[0])
    peak_point = peaks[np.argmax(np.abs(dslopes[peaks]))]
    if peak_point ==0:
        turning_interval = segs[0:3]
    elif peak_point == len(segs) - 1:
        turning_interval = segs[-3:]
    else:
        turning_interval = segs[peak_point-1: peak_point + 2]
    turning = np.concatenate([segment[0] for segment in turning_interval]), np.concatenate([segment[1] for segment in turning_interval])
    pre_turning = x[x < turning[0][0]], y[x < turning[0][0]]
    post_turning = x[x > turning[0][-1]], y[x > turning[0][-1]]
    slope, intercept = np.polyfit(10 ** post_turning[0], 10 ** post_turning[1], 1)
    return post_turning, (slope, intercept)
def local_slope(data, bins=10):
    x, y = data
    window_size = len(x)//bins
    x_coords, local_slopes = [], []
    for start, end in [(i, i + window_size) for i in range(len(x) - window_size)]:
        local_x, local_y = x[start:end], y[start:end]
        #diff = (np.diff(y) / np.diff(x)).mean()
        local_slopes.append(linregress(local_x, local_y)[0])
        x_coords.append(np.mean(local_x))
    return np.array(x_coords), np.array(local_slopes)
def fit_alpha(x, y, alpha):
    if alpha == 2:
        y_fit = y / (x**2)
    elif alpha == 1:
        y_fit = y / x
    else:
        raise ValueError("Alpha must be 1 or 2.")
    log_y_fit = np.log(y_fit)
    _, log_beta = linregress(np.log(x), log_y_fit)[:2]
    beta = np.exp(log_beta)
    return beta
def plot_line(ax, data, alpha=1):
    if alpha == 1:
        x0, x1, y1 = data
        logx0, logx1, logy1 = np.log10(x0), np.log10(x1), np.log10(y1)
        logy0 = alpha * (logx0 - logx1) + logy1
    elif alpha == 2:
        x0, x1, y0 = data
        logx0, logx1, logy0 = np.log10(x0), np.log10(x1), np.log10(y0)
        logy1 = alpha * (logx1 - logx0) + logy0
    ax.loglog([10 ** logx0, 10 ** logx1], [10 ** logy0, 10 ** logy1], color="k", linewidth=2)
#Cores
def exe_simus(task, path, infile):
    dir_file = os.path.join(path, infile)
    if task == "Run":
        echo.info(f">>> Running jobs: {infile}......\nmpirun -np 1 lmp_wk -i {dir_file}.in")
        subprocess.run(f"mpirun -np 1 lmp_wk -i {dir_file}.in", shell=True)
        echo.info(f"{dir_file}.in ==> Simus: Done!")

    elif task == "Submit":
        echo.info(f">>> Running jobs : {infile}......\nbsub < {dir_file}.lsf")
        subprocess.run(f"bsub < {dir_file}.lsf", shell=True)
        echo.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    else:
        echo.error(f"ERROR: Wrong task => {task} != Run or Submit")
def exe_analysis(Path, data):
    '''describe(data_com, "data_com", flag=False): analyse data[ifile][iframe][iatom][xu,
    yu] -> Rg, MSD, Cee etc.'''
    echo.info(f"analyse and save data => {Path}")
    Trun, frames = params["Trun"][1], params["Frames"]
    Rcom.data = np.linalg.norm(np.mean(data, axis=2), axis=-1) #Rcom[ifile][iframe] 质心位置
    data_com = data - np.expand_dims(np.mean(data, axis=2), axis=2) #质心坐标
    data_Rcom = np.linalg.norm(data_com[..., :], axis=-1) #质心坐标的模
    # ---------------------------------------> Rg <---------------------------------------#
    Rg2_t = np.mean(np.mean(data_Rcom ** 2, axis=-1), axis=0)  # Rg2_t[iframe]
    echo.info(f"Rg2(t)......")
    Rcom.save, Rg.save = os.path.join(Path, f"{Rcom.path}.npy"), os.path.join(Path, f"{Rg.path}.npy")
    Rgp.save = os.path.join(Path, f"{Rgp.path}.npy")
    np.save(Rgp.save, Rgp2_t)
    np.save(Rcom.save, Rcom.data)
    np.save(Rg.save, Rg2_t) #average over time
    # ---------------------------------------> MSD <---------------------------------------#
    # MSD[iframe]
    MSD.save = os.path.join(Path, f"{MSD.path}.npy")
    msd_run = np.zeros((Trun, frames//2))
    for dt in range(1, frames//2):
        msd_run[:, dt] = np.mean((Rcom.data[:, dt:] - Rcom.data[:, :-dt]) ** 2, axis=1)
    MSD.data = np.mean(msd_run, axis=0)
    echo.info(f"MSD(t) ......")
    np.save(MSD.save, MSD.data)
    return data_Rcom
def exe_plot(task, variable, path=CURRENT_DIR):
    '''plotting execute'''
    abbre, file_name = var2str(variable.name)[1], ','.join(variable.paras)
    echo.info(">>> Plotting ......")
    if variable.dtime:
        dirfile = os.path.join(path, f"(r,s){abbre}(t,{file_name})")
        if task == "load":
            variable.df = io_data(dirfile)
        elif task == "save":
            io_data(dirfile, variable.df)
            # self.subfile(f"{abbre}({file_name})_Plot", "Plot: {file_name}", dirfile)
        plotter4 = Plotter4D(variable, dirfile)
        plotter4.expand()
        plotter4.expand2D()
        plotter4.scale()
    else:
        dirfile = os.path.join(path, f"(r,s,t){abbre}({file_name})")
        if task == "load":
            variable.df = io_data(dirfile)
        elif task == "save":
            io_data(dirfile, variable.df)
            # self.subfile(f"{abbre}(t,{file_name})_Plot", "Plot: t, {file_name}", dirfile)
        plotter3 = Plotter3D(variable, dirfile)
        # plotter3.Rg()
        plotter3.project()
        plotter3.expand()
        plotter3.scale()
    print(">>> Plotting -------------------------------Done!----------------------------------------")
    if HOST == "Darwin":
        subprocess.run(["open", path])
# -----------------------------------Jobs-------------------------------------------#
class JobProcessor:
    def __init__(self, params):
        self.params, self.check, self.jump = params, check, jump
        self.queue, self.submitted = "7k83!", False
        #self.queue, self.submitted = "9654!", False
        self.run_on_cluster = os.environ.get("RUN_ON_CLUSTER", "false")
    # -----------------------------------backup:history-------------------------------------------#
    def Rg_job(self, Config, Run, iRin, variable="Rg", var_file="Rg2_time"):
        paras = ['Pe', 'N', 'W']
        if variable == "Rg":
            self.dtime = False
            self.df = pd.DataFrame(columns=paras)
            self.dft = self.df.copy()
        run_on_cluster = os.environ.get("RUN_ON_CLUSTER", "false")
        fig_save = os.path.join(os.path.join(re.match(r"(.*?/Data/)", os.getcwd()).group(1), "Figs"),
                      f"{Config.Dimend}D_{Run.Gamma:.1f}G_{iRin}R_{Run.Temp}T_{Config.Type}{Config.Env}")
        # copy Run.py
        os.makedirs(fig_save, exist_ok=True)
        py_file = os.path.join(f"{fig_save}", f"{variable}.py")
        if os.path.abspath(__file__) != f"{py_file}":
            shutil.copy2(os.path.abspath(__file__), f"{py_file}")
        echo.info(f"dir_figs => {fig_save}")
        # prepare files
        dir_file = os.path.splitext(os.path.abspath(__file__))[0]
        self.subfile(f"{variable}({','.join(paras)})", f"Analysis: {variable}", dir_file)
        # submitting files
        if not self.submitted:
            if HOST == "Linux" and run_on_cluster == "false" and BSUB:  # 登陆节点
                echo.info(">>> Submitting plots......")
                echo.info(f"bsub < {dir_file}.lsf => Submitted: {dir_file}.py")
                subprocess.run(f"bsub < {dir_file}.lsf", shell=True)
                self.submitted = True
            elif "Figs" in CURRENT_DIR: # 计算节点: "Run.py infile" == "bsub < infile.lsf"
                exe_plot("load", variable, ['Pe', 'N', 'W'])
                print(f"==> Rg_job: Done!")
            elif "Codes" in CURRENT_DIR: # HOST == "Darwin":  # macOS
                for iWid in convert2array(params['Wid']):
                    for iN in convert2array(params['N_monos']):
                        Init = _init(Config, Run.Trun, iRin, iWid, iN)
                        if Init.jump:
                            continue
                        for iFa in convert2array(params['Fa']):
                            for iXi in convert2array(params['Xi']):
                                Path = _path(_model(Init, Run, iFa, iXi))
                                for i, path in enumerate([Path.fig, Path.fig0, Path.fig1]):
                                    data_path = os.path.join(path, f"{var_file}.npy")
                                    if os.path.exists(data_path):
                                        break
                                    elif i == 2:
                                        echo.error(f"ERROR: Wrong {variable} path: Anas.job \n==> {data_path},")
                                if variable == "Rg":
                                    Rg2 = np.load(data_path)
                                    Rg, Rgt = np.sqrt(np.mean(Rg2)), np.sqrt(Rg2)
                                    self.df = self.df.append({'Pe': iFa / Run.Temp, 'N': iN, 'W': iWid, variable: Rg}, ignore_index=True)
                                    self.dft = self.dft.append({'Pe': iFa / Run.Temp, 'N': iN, 'W': iWid, variable: Rgt}, ignore_index=True)
                self.dft['dt']=Run.Tdump * 0.001
                # saving, subfile, plotting
                exe_plot("save", variable, paras, fig_save)
                print(f"==> Rg_job: Done! \n==>Please check the results and submit the plots!")
    # -----------------------------------Prepare-------------------------------------------#
    def _initialize(self, Path):
        self.Path = Path
        self.Init, self.Model, self.Run = Path.Init, Path.Model, Path.Run
    def check_params(self):
        print("===> Caveats: Please confirm the following parameters(with Dimend, Type, Env fixed):")
        for key, value in islice(self.params.items(), 1, None):
            user_input = input(f"{key} = {value}    (y/n)?: ")
            if user_input.lower() == "y":
                break
            elif user_input.lower() == '':
                continue
            else:
                new_value = input(f"Please enter the new value for {key}: ")
                self.params[key] = type(value)(new_value)  # 更新值，并尝试保持原来的数据类型
        return self.params
    def read_simp(self, Path):
        '''3 parameters from Path and 2 parameters from dict{params}'''
        Trun, Frames = self.params["Trun"], self.params["Frames"]
        dump_dict, chunk = {2: "xu yu", 3: "xu yu zu"}, 9
        dimend_match, pe_match, atoms_match = re.search(r'(\d+)D', Path), re.search(r'_(\d+\.?\d*)Pe_', Path), re.search(r'_(\d+)N(\d+)_', Path)
        dimend = int(dimend_match.group(1)) if dimend_match else None
        pe = float(pe_match.group(1)) if pe_match else None
        atoms = int(atoms_match.group(1)) * int(atoms_match.group(2)) if atoms_match else None
        # dimend, pe, atoms, dump = 3, 5, 20, dump_dict[3].split(" ")
        dump = dump_dict[dimend].split(" ")

        self.data = np.zeros((Trun[1], Frames + 1, atoms, len(dump)))
        # read data
        for index, ifile in enumerate([f"{i:03}" for i in range(Trun[0], Trun[1] + 1)]):
            filename = os.path.join(CURRENT_DIR, f"{ifile}.lammpstrj")
            echo.info(f"==> Reading {filename} file: ......")
            if index == 0:  # read and check
                natoms = pd.read_csv(filename, skiprows=3, nrows=1, delim_whitespace=True, header=None).iloc[0][0]
                skiprows = np.concatenate([np.arange(chunk) + (atoms + chunk) * x for x in range(Frames + 1)])
                names = list(pd.read_csv(filename, skiprows=7, nrows=0, delim_whitespace=True, header=1).columns[2:])
                xlo, xhi = pd.read_csv(filename, delim_whitespace=True, header=None, skiprows=5, nrows=1).values[0, :2]
                zlo, zhi = pd.read_csv(filename, delim_whitespace=True, header=None, skiprows=7, nrows=1).values[0, :2]
                Lx, Lz = xhi - xlo, zhi - zlo
                if natoms != atoms and (Lz < 1.0001 and dimend == 3):
                    echo.error(f"ERROR: Wrong atoms => {natoms} !={atoms}")
            df = pd.read_csv(filename, skiprows=skiprows, delim_whitespace=True, header=None, names=names, usecols=dump)
            data[index] = df.to_numpy().reshape((Frames + 1, atoms, len(dump)))
        if pe > 50:
            data = unwrap_x(data, Lx)
        #analysis and save data
        return  data
    def subfile(self, jobname, descript, dir_file):
        echo.info(f">>> Preparing subfile for {jobname}......")
        bsub = [
            f'#!/bin/bash',
            f'',
            f'#BSUB -J {jobname}',
            f'#BSUB -Jd "{descript}"',
            f'#BSUB -r',
            f'#BSUB -q {self.queue}',
            f'#BSUB -n 1',
            f'#BSUB -oo {dir_file}.out',
            f'#BSUB -eo {dir_file}.err',
            f'source ~/.bashrc',
            f'export RUN_ON_CLUSTER=true',
            f'cd {os.path.dirname(dir_file)}',
            f'echo "python3 {dir_file}.py"',
            f'python3 {dir_file}.py',
        ]
        with open(f"{dir_file}.lsf", "w") as file:
            for command in bsub:
                file.write(command + "\n")
    # -----------------------------------Simus-------------------------------------------#
    def simus_job(self, Path, **kwargs):
        # Initialize directories and files
        self._initialize(Path)
        if Path.jump:  # jump for repeat: check lammpstrj
            print(f"==>Continue: {Path.lmp_trj} is already!")
            if self.jump:
                return

        # Create simulation directory and copy Run.py
        echo.info(f"dir_simus => {Path.simus}")
        os.makedirs(Path.simus, exist_ok=True)
        if os.path.abspath(__file__) != os.path.join(Path.simus, Path.run_py):
            shutil.copy2(os.path.abspath(__file__), os.path.join(Path.simus, Path.run_py))

        # prepare files
        infiles = [f"{i:03}" for i in range(self.Run.Trun[0], self.Run.Trun[1]+2)]
        self.Init.data_file(Path)
        self.Model.in_file(Path)  # return
        self.Run.sub_file(Path, infiles)
        # excute jobs
        for infile in infiles:
            if HOST == "Linux" and self.run_on_cluster == "false" and BSUB:              # submitting files
                exe_simus("Submit", Path.simus, infile)
            else:
                exe_simus("Run", Path.simus, infile) # running files
    # -----------------------------------Anas-------------------------------------------#
    def anas_job(self, Path, **kwargs):
        self._initialize(Path)
        Pe = kwargs.get("Pe", None)
        ###################################################################
        # Initialize directories and files
        Anas, Plot = _anas(Path, Pe), _plot(Path)
        echo.info(f"dir_figs => {Path.fig}")
        os.makedirs(Path.fig, exist_ok=True)
        py_file = os.path.join(f"{Path.fig}", f"dana.py")
        if os.path.abspath(__file__) != f"{py_file}":
            shutil.copy2(os.path.abspath(__file__), f"{py_file}")
        # prepare files
        self.queue, dir_file = Path.Run.Queue, os.path.splitext(os.path.abspath(__file__))[0]
        self.subfile(f"{Path.Jobname}_Anas", "Analysis: original data", dir_file)
        if Path.jump:  # jump for repeat: check lammpstrj
            if not self.submitted:
                # submitting files
                if HOST == "Linux" and self.run_on_cluster == "false" and BSUB:  # if HOST == "Darwin":  # macOS
                    echo.info(">>> Submitting plots......\nbsub < {dir_file}.lsf")
                    subprocess.run(f"bsub < {dir_file}.lsf", shell=True)
                    echo.info(f"Submitted: {dir_file}.py")
                    self.submitted = True
                else:
                    #self.exe_anas(task, path)
                    if "Codes" in CURRENT_DIR:
                        echo.info(">>> Plotting ......")
                        file_check = os.path.join(Path.fig, f"{JOBS[-1].path}.npy")
                        if os.path.exists(file_check) and jump:
                            echo.info(f"JUMP==>{file_check} is already!")
                        else:
                            data = Anas.read_data()
                            data_Rcom = exe_analysis(Path.fig, data) # saving data
                            Plot.org(data_Rcom, Rcom) #plotting: org(data, variable)
                            # Plot.org(data_Rcom ** 2, "Rcom2")
                        if HOST == "Darwin":
                            subprocess.run(["open", Path.fig])
                        print(f"==> Anas_job: Done! \n==>Please check the results and submit the plots!")
                    elif "Figs" in CURRENT_DIR:
                        echo.info(f">>> Plotting {CURRENT_DIR} ......")
                        path = CURRENT_DIR.replace('Figs', 'Simus')
                        data = self.read_simp(path)
                        data_Rcom = exe_analysis(path, data)
                        #data_Rcom = np.load(os.path.join(CURRENT_DIR, "Rcom.npy"))
                        Plot.org(data_Rcom, "Rcom")  # org(data, variable)
                        print(f"==> Anas_job: Done!")
        else:
            echo.info(f"File doesn't exist in anas_job: {Path.lmp_trj}")
    # -----------------------------------Plot-------------------------------------------#
    def plot_job(self, Config, Run, iRin, variable):
        '''expand Rg(t) -> Rg(t, Pe, W, N)'''

        variable.df = pd.DataFrame(columns=variable.paras)
        run_on_cluster = os.environ.get("RUN_ON_CLUSTER", "false")
        fig_save = os.path.join(os.path.join(re.match(r"(.*?/Data/)", os.getcwd()).group(1), "Figs"),
                      f"{Config.Dimend}D_{Run.Gamma:.1f}G_{iRin}R_{Run.Temp}T_{Config.Type}{Config.Env}")
        # copy Run.py
        os.makedirs(fig_save, exist_ok=True)
        py_file = os.path.join(f"{fig_save}", f"{variable.name}.py")
        if os.path.abspath(__file__) != f"{py_file}":
            shutil.copy2(os.path.abspath(__file__), f"{py_file}")
        echo.info(f"dir_figs => {fig_save}")
        # prepare files
        dir_file = os.path.splitext(os.path.abspath(__file__))[0]
        self.subfile(f"({JOBS})({','.join(variable.paras)})", f"Analysis: {JOBS}", dir_file)
        # submitting files
        if not self.submitted:
            if HOST == "Linux" and run_on_cluster == "false" and BSUB:  # 登陆节点
                echo.info(f">>> Submitting plots......\nbsub < {dir_file}.lsf")
                subprocess.run(f"bsub < {dir_file}.lsf", shell=True)
                self.submitted = True
            elif "Figs" in CURRENT_DIR: # 计算节点: "Run.py infile" == "bsub < infile.lsf"
                exe_plot("load", variable)
                print(f"==> Plot_job: Done!")
            elif "Codes" in CURRENT_DIR: # HOST == "Darwin":  # macOS
                for iWid in convert2array(params['Wid']):
                    for iN in convert2array(params['N_monos']):
                        Init = _init(Config, Run.Trun, iRin, iWid, iN)
                        if Init.jump:
                            continue
                        for iFa in convert2array(params['Fa']):
                            for iXi in convert2array(params['Xi']):
                                Path = _path(_model(Init, Run, iFa, iXi))
                                for i, path in enumerate([Path.fig, Path.fig0, Path.fig1]):
                                    variable.save = os.path.join(path, f"{variable.path}.npy")
                                    #print("Path.fig,0,1:", Path.fig, Path.fig0, Path.fig1)
                                    data_path = variable.save if iFa >= 1 else variable.save.replace(f"{Run.Gamma:.1f}G", f"{10:.1f}G")
                                    if os.path.exists(data_path):
                                        break
                                    elif i == 2:
                                        if HOST == "Darwin":
                                            subprocess.run(["open", path])
                                        echo.error(f"ERROR: Wrong {variable.name} path = {data_path}")
                                if variable.name == "Rg":
                                    data_Rg2 = np.load(data_path)
                                    data_Rg = np.sqrt(np.mean(data_Rg2))
                                    if variable.dtime:
                                        variable.df = variable.df.append({'Pe': iFa / Run.Temp, 'N': iN, 'W': iWid, variable.name: np.sqrt(data_Rg2)}, ignore_index=True)
                                    else:
                                        variable.df = variable.df.append({'Pe': iFa / Run.Temp, 'N': iN, 'W': iWid, variable.name: data_Rg}, ignore_index=True)
                                elif variable.name == "MSD":
                                    if not variable.dtime:
                                        echo.error(f"ERROR: variable: {variable.name} must be combinded with time")
                                    else:
                                        variable.df = variable.df.append({'Pe': iFa / Run.Temp, 'N': iN, 'W': iWid, variable.name: np.load(data_path)}, ignore_index=True)
                                else:
                                    echo.error(f"ERROR: wong variable: {variable.name}")
                # saving, subfile, plotting
                variable.df['dt'] = Run.Tdump * 0.001
                exe_plot("save", variable, fig_save)
                print(f"==> Plot_job: Done! \n==>Please check the results and submit the plots!")
    # -----------------------------------Process-------------------------------------------#
    def process(self, data_job = None, plot_job = None):
        params = self.params
        if platform.system() == "Linux" or "Codes" not in CURRENT_DIR:
            mpl.use("agg")
        for iDimend in convert2array(params['Dimend']):
            for iType in convert2array(params['labels']['Types']):
                for iEnv in convert2array(params['labels']['Envs']):
                    Config = _config(iDimend, iType, iEnv, params)
                    if self.check and (platform.system() == "Linux") and (self.run_on_cluster == 'false'):
                        params = self.check_params()
                        self.check = False
                    for iGamma in convert2array(params['Gamma']):
                        for iTemp in convert2array(params['Temp']):
                            for iRin in convert2array(params['Rin']):
                                # paras for run: Gamma, Temp, Queue, Frames, Trun, Dimend, Temp, Dump
                                Run = _run(Config.Dimend, iGamma, iTemp, params['Trun'])
                                Config.set_dump(Run)
                                if plot_job:
                                    #queue = Run.set_queue()
                                    for ijob in JOBS:
                                        getattr(self, plot_job)(Config, Run, iRin, ijob)
                                elif data_job:
                                    for iWid in convert2array(params['Wid']):
                                        for iN in convert2array(params['N_monos']):
                                            # paras for init config: Rin, Wid, N_monos, L_box
                                            Init = _init(Config, Run.Trun, iRin, iWid, iN)
                                            queue = Run.set_queue()
                                            if Init.jump:  # chains are too long or invalid label
                                                continue
                                            for iFa in convert2array(params['Fa']):
                                                for iXi in convert2array(params['Xi']):
                                                    # paras for model: Pe(Fa), Xi(Kb)
                                                    Model = _model(Init, Run, iFa, iXi)
                                                    Path = _path(Model)  # for directory
                                                    getattr(self, data_job)(Path, Pe=Model.Pe)
# -----------------------------------Main-------------------------------------------#
if __name__ == "__main__":
    print(f"{usage}\n=====>task: {task}......\n###################################################################")

    # Simulations
    run = JobProcessor(params)
    Trun = params['Trun']
    back_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{task}.py")
    if os.path.abspath(__file__) != back_file:
        shutil.copy(os.path.abspath(__file__), back_file)
    if task == "Simus":
        if "Codes" in CURRENT_DIR:
            run.process(data_job="simus_job")
        elif "Simus" in CURRENT_DIR: # 计算节点: "Run.py infile" == "bsub < infile.lsf"
            for infile in [f"{i:03}" for i in range(Trun[0], Trun[1] + 2)]:
                exe_simus("Run", CURRENT_DIR, infile)
    # Analysis: single
    elif task == "Anas":
        run.process(data_job="anas_job")
        if HOST == "Linux":
            print(f"{usage}\n=====>task: Plots......\n###################################################################")
            run.process(plot_job="plot_job")
    # plot: Pe, N, W
    elif task == "Plots":
        run.process(plot_job="plot_job")