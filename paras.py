import os
import subprocess
import shutil
import platform
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from numba import vectorize, float64
from scipy.interpolate import splrep, splev
from scipy.stats import norm

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages as PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms

#-----------------------------------Parameters-------------------------------------------
label = ["Bacteria", "Chain", "Ring"]
tasks = ["Simus", "Anas"]
#-----------------------------------Dictionary-------------------------------------------
#参数字典
params = {
    'marks': {'Labels': label[1], 'config': label[1]},
    'task': tasks[0],
    'restart': False,
    'Queues': {'7k83!': 1.0, '9654!': 1.0},
    # 动力学方程的重要参数
    'Gamma': 100,
    'Trun': 5,
    'Dimend': 2,
    # 障碍物参数：环的半径，宽度和链的长度，数量
    'Rin': [5.0, 10.0, 15.0, 20.0, 30.0],
    #'Rin': [5.0],
    'Wid': [5.0, 10.0, 15.0, 20.0, 30.0], # annulus width
    #'Wid': [10.0],
    'num_chains': 1,
}

class _config:
    def __init__(self, Label, Params = params):
        self.config = {
            '''Pe = Fa / Temp'''
            "Bacteria": {
                'N_monos': 3,
                'Xi': 1000,
                'Fa': [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0],
                'Temp': 1.0,
            },
            "Chain": {
                'N_monos': [20, 40, 80, 100, 150, 200, 250, 300],
                #'N_monos': [100],
                'Xi': 0.0,
                #'Fa': [0.0, 1.0],
                'Fa': [1.0],
                #'Temp': [0.01, 0.1, 1.0],
                'Temp': [0.01, 0.1],
            },
            "Chain":{
                'N_monos': [20, 40, 80, 100, 150, 200, 250, 300],
                # 'N_monos': [100],
                'Xi': 0.0,
                # 'Fa': [0.0, 1.0],
                'Fa': [1.0],
                # 'Temp': [0.01, 0.1, 1.0],
                'Temp': [0.01, 0.1],
            }
        }
        self.Params = Params
        self.Label = Label
        self.Params["marks"]["config"] = Label
        if self.Label in self.config:
            self.Params.update(self.config[self.Label])

    def set_dump(self, Run):
        """计算 Tdump 的值: xu yu"""
        if Run.Dimend == 2:
            dump = "xu yu vx vy"
        elif Run.Dimend == 3:
            dump = "xu yu zu vx vy vz"
        else:
            print(f"Error: Wrong Dimend to run => dimension != {Run.Dimend}")
            exit(1)
        Run.Dump = dump
        #Run.Dump = "xu yu zu"
        Run.Tdump = 2 * 10 ** Run.eSteps // Run.Frames
        Run.Tdump_ref = Run.Tdump // 100
        #if platform.system() == "Darwin":
          #  Run.Tdump = Run.Tdump_ref
            #Run.Damp = 1.0
            
        Run.Tinit = Run.Frames * Run.Tdump_ref
        Run.TSteps = Run.Frames * Run.Tdump
        Run.Tequ = Run.TSteps
        Run.Tref = Run.Tinit
        Run.Params["Total Run Steps"] = Run.TSteps
            
        if self.Label == "Bacteria":
            Run.Tdump = Run.Tdump // 10
            Run.Tequ = Run.Tequ // 100
##########################################END!###############################################################

class _run:
    def __init__(self, Gamma, Temp, Trun, Dimend, Params = params, Frames = 2000):
        self.Params = Params
        self.Queue = "7k83!"
        self.set_queue()
        self.Gamma = Gamma
        self.Trun = Trun
        self.Dimend = Dimend
        self.Frames = Frames
        self.Temp = Temp
        self.SkipRows = 9
        if (self.Dimend == 2):
            self.fix2D = f"fix             2D all enforce2d"
            self.unfix2D = f"unfix             2D"
            self.dump_read = "x y"
        elif (self.Dimend == 3):
            self.dump_read = "x y z"
        
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
        if platform.system() != "Darwin":
            try:
                bqueues = subprocess.check_output(['bqueues']).decode('utf-8') # Decode the output here
                bhosts = subprocess.check_output(['bhosts']).decode('utf-8')
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
                exit(1)
                
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
                    print(f"Error: {e}")
                    exit(1)
                for line in bjobs.strip().split('\n')[1:]:
                    columns = line.split()
                    start_time = datetime.strptime(f"{columns[-4]} {columns[-3]} {datetime.now().year} {columns[-2]}", "%b %d %Y %H:%M")
                    if datetime.now() - start_time > timedelta(hours=24):
                        cores = int(columns[-1].split('*')[0]) if '*' in columns[-1] else 1
                        queue_info[iqueue]["occupy"] += cores
                queue_info[iqueue]["Avail"] = queue_info[iqueue]["cores"] - queue_info[iqueue]["occupy"]
                queue_info[iqueue]["Usage"] = round( (queue_info[iqueue]["PEND"] + queue_info[iqueue]["RUN"] - queue_info[iqueue]["occupy"] ) / queue_info[iqueue]["Avail"], 3)
                self.Params["Queues"][iqueue] = queue_info[iqueue]["Usage"]
                if queue_info[iqueue]["PEND"] == 0:
                    self.Queue = max(myques, key=lambda x: queue_info[x]['cores'] - queue_info[x]['RUN'])
                elif queue_info[iqueue]["PEND"] > 0:
                    self.Queue = min(myques, key=lambda x: queue_info[x]['Usage']) #print(f"queue = {self.Queue}, queue_info: {queue_info}")
            #exit(1)
        return self.Queue
    
    def bsubs(self, Path, test=0):
        Run = Path.Run
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
            if platform.system() == "Darwin" or test:
                print(">>> for test......")
                print(f"mpirun -np 4 lmp_wk -i {dir_file}.in")
                subprocess.run(f"mpirun -np 4 lmp_wk -i {dir_file}.in", shell=True)
                print(f"{dir_file}.in ==> Done! \n ==> Please check the results and submit the jobs!")
            elif platform.system() == "Linux":
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
                print(f"An error occurred: {e}")
        elif task == "Anas":
            try:
                print("Plot")
            except Exception as e:
                print(f"An error occurred: {e}")
##########################################END!###############################################################

class _init:
    def __init__(self, Run, Rin, Wid, N_monos, num_chains = params["num_chains"]):
        self.sigma = 1.0
        self.mass = 1.0
        self.Run = Run
        self.Rin = Rin
        self.Wid = Wid
        self.N_monos = N_monos
        self.num_chains = num_chains
        self.num_monos = self.N_monos * self.num_chains
        self.Rchain = self.Rin + self.sigma
        self.Rout = self.Rin + self.Wid # outer_radius
        self.jump = self.set_box()   #set box
        self.dtheta_chain = self.set_dtheta(self.Rchain)
        self.num_Rin = np.ceil(2 * np.pi * self.Rin / (self.sigma / 2))
        self.num_Rout = np.ceil(2 * np.pi * self.Rout / (self.sigma / 2))
        self.dtheta_in = self.set_dtheta(self.Rin, self.num_Rin)
        self.dtheta_out = self.set_dtheta(self.Rout, self.num_Rout)
        self.num_Ring = self.num_Rin + self.num_Rout

        self.total_particles = self.num_Ring + self.num_monos
        
        if (self.num_chains != 1):
            print(f"ERROR => num_chains = {self.num_chains} is not prepared!\nnum_chains must be 1")
            exit(1)
    def set_box(self):
        """计算盒子大小"""
        if self.Rin != 0.0 and self.Wid != 0.0:
            self. L_box = self.Rout + 1
            if (self.num_monos > np.pi * (self.Wid * self.Wid + self.Wid * ( 2 * self.Rin - 1) - 2 * self.Rin) ):
                print("N_monos is too Long!")
                return True
        elif self.Rin == 0.0 and self.Wid == 0.0:
            self.L_box = self.N_monos/2 + 10
        else:
            raise ValueError("Wrong Rin & Wid: Rin == 0 and Wid == 0")
        
        if self.Run.Dimend == 2:
            self.zlo = -self.sigma/2
            self.zhi = self.sigma/2
        elif self.Run.Dimend == 3:
            self.zlo = 0.0
            self.zhi = self.L_box
        return False
    
    def set_dtheta(self, R, num_R=None):
        """计算角度间隔"""
        if num_R is None:
            return 2 * np.pi / np.floor(2 * np.pi * R / self.sigma)
        elif num_R == 0:
            return 0
        else:
            return 2 * np.pi * R / (num_R * R)
    
    def write_header(self, file):
        # 写入文件头部信息
        file.write("{} LAMMPS data file for initial configuration:\n\n".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
        
        # 写入原子数目和边界信息
        file.write(f"{int(self.total_particles)} atoms\n\n")
        file.write(f"{self.num_monos - self.num_chains*1} bonds\n\n")
        file.write(f"{self.num_monos - self.num_chains*2} angles\n\n")
        file.write("5 atom types\n\n")
        file.write("1 bond types\n\n")
        file.write("3 angle types\n\n")
        file.write(f"-{self.L_box} {self.L_box} xlo xhi\n")
        file.write(f"-{self.L_box} {self.L_box} ylo yhi\n")
        file.write(f"{self.zlo} {self.zhi} zlo zhi\n\n")
        file.write("Masses\n\n")
        file.write(f"1 {self.mass}\n")
        file.write(f"2 {self.mass}\n")
        file.write(f"3 {self.mass}\n")
        file.write(f"4 {self.mass}\n")
        file.write(f"5 {self.mass}\n\n")
        
    def write_atoms(self, file):
        # 写入原子信息
        file.write("Atoms\n\n")
        Rchain = self.Rchain
        dtheta_chain = self.dtheta_chain
        
        # 写入链的原子信息
        circle = 1
        theta = - 2 * self.dtheta_chain
        for i in range(int(self.N_monos)):
            if i == 0:
                x = round(Rchain * np.cos(theta), 2)
                y = round(Rchain * np.sin(theta), 2)
                z = 0
                file.write(f"{i+1} 1 1 {x} {y} {z}\n")
                continue
            if theta >= 2 * np.pi - 5 * circle * dtheta_chain:
                circle += 1
                Rchain += self.sigma
                dtheta_chain = 2 * np.pi / np.floor(2* np.pi * Rchain/self.sigma)
                theta = theta - 2 * np.pi - dtheta_chain
            theta += dtheta_chain
            x = round(Rchain * np.cos(theta), 2)
            y = round(Rchain * np.sin(theta), 2)
            z = 0.0
            if i == self.N_monos-1:
                file.write(f"{i+1} 1 3 {x} {y} {z}\n")
            else:
                file.write(f"{i+1} 1 2 {x} {y} {z}\n")
    
    def write_ring(self, file):
        # 写入第一个环的原子信息
        for i in range(int(self.num_Rin)):
            theta = i * self.dtheta_in
            x = round(self.Rin * np.cos(theta), 4)
            y = round(self.Rin * np.sin(theta), 4)
            z = 0.0
            file.write(f"{self.N_monos+i+1} 1 4 {x} {y} {z}\n")
            
        # 写入第二个环的原子信息
        for i in range(int(self.num_Rout)):
            theta = i * self.dtheta_out
            x =  round(self.Rout * np.cos(theta), 4)
            y =  round(self.Rout * np.sin(theta), 4)
            z = 0.0
            file.write(f"{int(self.N_monos+self.num_Rin+i+1)} 1 5 {x} {y} {z}\n")
    
    def write_potential(self, file):
        #写入bonds and angle
        file.write("\nBonds\n\n")
        for i in range(int(self.N_monos)-1):
            file.write(f"{i+1} 1 {i+1} {i+2}\n")
        file.write("\nAngles\n\n")
        for i in range(int(self.N_monos)-2):
            if i == 0:
                file.write(f"{i+1} 1 {i+1} {i+2} {i+3}\n")
            elif i == int(self.N_monos) - 3:
                file.write(f"{i+1} 3 {i+1} {i+2} {i+3}\n")
            else:
                file.write(f"{i+1} 2 {i+1} {i+2} {i+3}\n")
    
    def data_file(self, Path):
        # 初始构型的原子信息: theta, x, y, z
        print("==> Preparing initial data file......")
        # 打开data文件以进行写入
        with open(f"{Path.data_file}", "w") as file:
            self.write_header(file)
            self.write_atoms(file)
            self.write_ring(file)
            self.write_potential(file)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#############################################################################################################
        
class _model:
    def __init__(self, Init, Run, Fa, Xi):
        self.Init = Init
        self.Run = Run
        self.Fa = Fa
        self.Xi = Xi
        self.Kb = self.Xi * Init.N_monos/4
        #for directory
        self.Pe = self.Fa / self.Run.Temp

    def write_section(self, file, cmds):
        """向文件中写入一个命令区块"""
        for command in cmds:
            file.write(command + "\n")
            
    def in_file(self, Path):
        Model, Init, Run = Path.Model, Path.Init, Path.Run
#        Run.Trun = 1000
        for infile in [f"{i:03}" for i in range(1, Run.Trun + 1)]:
            print(f"==> Writing infile: {infile}......")
            dir_file = os.path.join(f"{Path.dir_data}", infile)
            read = f'read_data       {Path.data_file}'
            if Run.Params["restart"]:
                read = 'read_restart       ${dir_file}.equ.restart'
                
            # Write each command in the list to the file, followed by a newline character
            setup = [
                '# Setup',
                'echo		        screen',
                'units           lj',
                f'dimension       {Run.Dimend}',
                'boundary        p p p',
                'atom_style      angle',
                '',
                f'variable dir_file string {dir_file}',
                read,
                '',
                '#groups',
                'group           head type 1',
                'group           end type 3',
                'group			      chain type 1 2 3',
                'group			      obs type 4 5',
                '',
                'velocity		all set 0.0 0.0 0.0',
                '',
            ]
            initial = [
                '# for initialization',
                '##################################################################',
                '# pair potential and  soft potential',
                'variable	      Pre_soft equal ramp(0.0,10000.0)',
                'pair_style      hybrid/overlay lj/cut 1.12246 soft 1.5480',
                'pair_coeff      *3 *3 lj/cut 1 1.0',
                'pair_modify     shift yes',
                'pair_coeff      *3 4*5 soft 1.5000',
                'pair_coeff      4*5 4*5 lj/cut 1 1.0 0.0',
                'fix 	          SOFT all adapt 1 pair soft a *3 4*5 v_Pre_soft',
                '',
                '# Bond potential',
                'bond_style      harmonic',
                'bond_coeff      1 4000 1.0',
                'special_bonds   lj/coul 1.0 1.0 1.0',
                '',
                '# angle potential',
                'angle_style     hybrid actharmonic_h2 actharmonic actharmonic_t',
                f'angle_coeff     1 actharmonic_h2 {Model.Kb} 180 0.0 0.0',
                f'angle_coeff     2 actharmonic   {Model.Kb} 180 0.0',
                f'angle_coeff     3 actharmonic_t {Model.Kb} 180 0.0',
                '##################################################################',
                '# for communication',
                'comm_style      brick',
                'comm_modify     mode single cutoff 3.0 vel yes',
                '',
                'neighbor	      1.5 bin',
                'neigh_modify	  every 1 delay 0 check yes #exclude none',
                '',
                '# for initialization',
                f'# fix		 	        BOX all deform 1 x final 0.0 {Init.L_box} y final 0.0 {Init.L_box} units box remap x',
                f'fix   	        LANG chain langevin 1.0 1.0 1.0 {Run.set_seed()}',
                'fix			        NVE chain nve/limit 0.1',
                'fix             FREEZE obs setforce 0.0 0.0 0.0',
                Run.fix2D,
                '',
                f'dump	        	INIT all custom {Run.Tinit//20} '
                '${dir_file}.chain_init.lammpstrj id type '
                f'{Run.Dump}',
                'dump_modify     INIT sort id',
                '',
                'reset_timestep	0',
                f'timestep        {Run.dt}',
                f'thermo		      {Run.Tinit//200}',
                f'run	            {Run.Tinit}',
                'write_restart   ${dir_file}.init.restart',
                '',
                '#unfix           BOX',
                'unfix           SOFT',
                'unfix           LANG',
                'unfix           NVE',
                'unfix           FREEZE',
                Run.unfix2D,
                'undump          INIT',
                '',
            ]
            potential = [
                '# for equalibrium',
                '###################################################################',
                '#pair potential',
                'pair_style      lj/cut 1.12246',
                'pair_modify	    shift yes',
                'pair_coeff      *3 *  1 1.0',
                'pair_coeff      4*5 4*5   1 1.0 0.0',
                '',
                '# Bond potential',
                'bond_style      harmonic',
                'bond_coeff      1 4000 1.0',
                'special_bonds   lj/coul 1.0 1.0 1.0',
                '',
                '# Angle potential',
                'angle_style     hybrid actharmonic_h2 actharmonic actharmonic_t',
                f'angle_coeff     1 actharmonic_h2 {Model.Kb} 180 {Model.Fa} {Model.Fa}',
                f'angle_coeff     2 actharmonic   {Model.Kb} 180 {Model.Fa}',
                f'angle_coeff     3 actharmonic_t {Model.Kb} 180 {Model.Fa}',
                '#################################################################',
                'comm_style      brick',
                'comm_modify     mode single cutoff 3.0 vel yes',
                '',
                'neighbor	      1.5 bin',
                'neigh_modify	  every 100 delay 0 check yes #exclude none',
                '',
            ]
            equal_fix = [
                '# for equalibrium',
                f'fix      	      LANG chain langevin {Run.Temp} {Run.Temp} {Run.Damp} {Run.set_seed()}',
                'fix		         	NVE chain nve',
                'fix             FREEZE obs setforce 0.0 0.0 0.0',
                Run.fix2D,
                '',
            ]
            equal_run = [
                f'dump		        EQU chain custom {Run.Tdump} '
                '${dir_file}.chain_equ.lammpstrj id type '
                f'{Run.Dump}',
                'dump_modify     EQU sort id',
                '',
                'reset_timestep  0',
                f'timestep        {Run.dt}',
                f'thermo		      {Run.Tequ//200}',
                'log       ${dir_file}.log',
                f'run		          {Run.Tequ}',
                'write_restart   ${dir_file}.equ.restart',
                '',
                'unfix           LANG',
                'unfix           NVE',
                'unfix           FREEZE',
                Run.unfix2D,
                'undump          EQU',
                '',
            ]
            data = [
                '# for data',
                '#################################################################',
                f'fix      	      LANG all langevin {Run.Temp} {Run.Temp} {Run.Damp} {Run.set_seed()}',
                'fix		         	NVE chain nve',
                'fix             FREEZE obs setforce 0.0 0.0 0.0',
                Run.fix2D,
                '',
                '#output',
                f'dump	    	    DATA chain custom {Run.Tdump} '
                '${dir_file}.lammpstrj id type '
                f'{Run.Dump}',
                'dump_modify     DATA sort id',
                '',
                '# run',
                'reset_timestep  0',
                f'thermo		      {Run.TSteps//200}',
                f'run	        	  {Run.TSteps}',
                'write_restart	  ${dir_file}.end.restart',
                'undump          DATA',
                '',
            ]
            refine = [
                '# for refine',
                '##################################################################',
                'read_dump       ${dir_file}.chain_equ.lammpstrj '
                f'{Run.Tequ} {Run.dump_read} wrapped no format native',
                '',
                f'dump            REFINE chain custom {Run.Tdump_ref} '
                '${dir_file}.chain_refine.lammpstrj id type '
                f'{Run.Dump}',
                'dump_modify     REFINE sort id',
                '',
                'reset_timestep  0',
                f'timestep        {Run.dt}',
                f'thermo          {Run.Tref//200} ',
                f'run             {Run.Tref}',
                'write_restart   ${dir_file}.refine.restart',
                '#############################################################'
            ]
            # Define LAMMPS 参数
            if Run.Params["restart"]:
                with open(f"{dir_file}.in", "w") as file:
                    self.write_section(file, setup)
                    self.write_section(file, potential)
                    self.write_section(file, data)
                    self.write_section(file, refine)
            else:
                with open(f"{dir_file}.in", "w") as file:
                    self.write_section(file, setup)
                    self.write_section(file, initial)
                    self.write_section(file, potential)
                    self.write_section(file, equal_fix)
                    self.write_section(file, equal_run)
                    self.write_section(file, data)
                    self.write_section(file, refine)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>Done!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#############################################################################################################

class _path:
    def __init__(self, Config, Model, Init, Run):
        #host = os.getcwd()
        self.host = os.path.abspath(os.path.dirname(os.getcwd()))
        self.mydirs = ["Codes", "Simus", "Anas", "Figs"]
        self.Config = Config
        self.Model = Model
        self.Init = Init
        self.Run = Run
        self.jump = self.build_paths()
        
    def build_paths(self):
        self.simus = os.path.join(self.host, self.mydirs[1])
        for dir in self.mydirs:
            self.host_dir = os.path.join(self.host, dir)
            subprocess.run(f"mkdir -p {self.host_dir}", shell=True)
        #100G1.0T2D
        self.dir1= f"{self.Run.Gamma}G{self.Run.Temp}T{self.Run.Dimend}D"
        #5.0R5.0_100N1_Chain
        if self.Init.Rin != 0 and self.Init.Wid != 0:
            self.dir2 = f"{self.Init.Rin}R{self.Init.Wid}_{self.Init.N_monos}N{self.Init.num_chains}_{self.Config.Label}"
        elif self.Init.Rin == 0 and self.Init.Wid == 0:
            self.dir2 = f"{self.Init.N_monos}N{self.Init.num_chains}_{self.Config.Label}"
        else:
            raise ValueError("Wrong Rin & Wid: Rin == 0 and Wid == 0")
        #1.0Pe_0.0Xi_8T5
        self.dir3 = f"{self.Model.Pe}Pe_{self.Model.Xi}Xi_{self.Run.eSteps}T{self.Run.Trun}"
        if self.Config.Label == "Bacteria":
            self.Jobname = f"{self.Model.Pe}Pe_{self.Config.Label[:2].upper()}"
        elif self.Config.Label == "Chain":
            self.Jobname = f"{self.Init.N_monos}N_{self.Config.Label[:2].upper()}"
        #/Users/wukong/Data/Simus/100G1.0T2D_5.0R5.0_100N1_Chain/1.0Pe_0.0Xi_8T5
        self.dir_data = os.path.join(self.simus, f"{self.dir1}_{self.dir2}", self.dir3)
        #/Users/wukong/Data/Simus/100G1.0T2D_5.0R5.0_100N1_Chain/1.0Pe_0.0Xi_8T5/5.0R5.0_100N1_Chain.data
        self.data_file = os.path.join(self.dir_data, f"{self.dir2}.data")
        subprocess.run(f"mkdir -p {self.dir_data}", shell=True)
        shutil.copy2(os.path.join(self.host, self.mydirs[0], "paras.py"), os.path.join(self.dir_data, "paras.py"))
        print(f"data_file => {self.data_file}")
        #Figures
        self.fig1 = os.path.join(self.host, self.mydirs[3], f"{self.dir1}_{self.dir2}", self.dir3)
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
        if self.Config.Label == "Bacteria":
            if self.Run.Dimend == 2:
                dump = "xu yu vx vy"
            elif self.Run.Dimend == 3:
                dump = "xu yu zu vx vy vz"
            else:
                print(f"Error: Wrong Dimend to run => dimension != {self.Run.Dimend}")
                exit(1)
        else:
            if self.Run.Dimend == 2:
                dump = "xu yu"
            elif self.Run.Dimend == 3:
                dump = "xu yu zu"
            else:
                print(f"Error: Wrong Dimend to run => dimension != {self.Run.Dimend}")
                exit(1)
        dump = dump.split(" ")
        return ["id"] + dump

    def read_data(self):
        dump = self.set_dump()
        self.data = np.zeros((self.Run.Trun, self.Run.Frames+1, self.Init.num_monos, len(dump)))
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
                print(f"ERROR: Wrong atoms => {natoms} != {self.Init.num_monos}")
            elif lastStep != self.Run.Frames * self.Run.Tdump:
                print(f"ERROR: Wrong timesteps => {lastStep} != {self.Run.Fradfmes} * {self.Run.Tdump}")
            skiprows = np.array(list(map(lambda x: np.arange(self.chunk) + (self.Init.num_monos + self.chunk) * x, np.arange(self.Run.Frames+1)))).ravel()
            try:
                df = pd.read_csv(dir_file, skiprows=skiprows, delim_whitespace=True, header=None, names=names, usecols=dump)
            except Exception as e:
                print(f"ERROR: Could not read the file due to {e}")
                exit(1)
            # data[ifile][iframe][iatom][id, xu, yu]
            self.data[index] = df.to_numpy().reshape((self.Run.Frames+1, self.Init.num_monos, len(dump)))
            #np.save(self.Path.dir_data, data)
        return self.data

    def plot1(self):
        fig_save = os.path.join(f"{self.Path.fig1}", "r(S,T)Dist")
        pdf = PdfPages(f"{fig_save}.pdf")
        print(f"{fig_save}.pdf")
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
