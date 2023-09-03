import os
import time
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import subprocess
import shutil
import platform

#-----------------------------------Parameters-------------------------------------------
label = ["Chain", "Bacteria"]

#-----------------------------------Dictionary-------------------------------------------
#参数字典
params = {
    'Label': label[:],
    'Queue': ["7k83!", "9654!"],
    # 动力学方程的重要参数
    'Gamma': 100,
    'Trun': 5,
    'Dimend': 2,
    # 障碍物参数：环的半径，宽度和链的长度，数量
    #'Rin': [10.0, 15.0],
    'Rin': [10.0], 
    'Wid': 5.0, # annulus width
    'num_chains': 1,
}
class _config:
    def __init__(self, Label, Run, Params = params):
        self.config = {
            "Bacteria": {
                'N_monos': 3,
                'Xi': 1000,
                'Pe': [0.0, 1.0, 10.0],
            },
            "Chain": {
                #'N_monos': [20, 40, 60, 100, 200],
                'N_monos': [20],
                'Xi': 0.0,
                'Pe': [0.0, 1.0],
            },
        }
        self.Params = Params
        self.Run = Run
        self.Label = Label
        self.update()
    
    def update(self):
        if self.Label in self.config:
            self.Params.update(self.config[self.Label])
            if self.Label == "Bacteria":
                self.Run.Dump = "xu yu vx vy fx fy"
                self.Run.Tdump = self.Run.Tdump//10
                self.Run.Tequ = self.Run.Tdump_ref * 10
##########################################END!###############################################################

class _run: 
    def __init__(self, Gamma, Trun, Dimend, Params = params, Frames = 2000):
        self.Params = Params
        self.Queue = self.set_queue()
        self.Gamma = Gamma
        self.Trun = Trun
        self.Dimend = Dimend
        self.Frames = Frames
        self.Temp = 1.0
        self.Dump = "xu yu zu"

        if (self.Dimend == 2):
            self.fix2D = f"fix             2D all enforce2d"
            self.unfix2D = f"unfix             2D"
            self.dump_read = "x y"
        elif (self.Dimend == 3):
            self.dump_read = "x y z"
        
        self.dt = 0.001
        self.Seed = np.random.randint(700000000, 800000001)

        self.eSteps = 9 if self.Gamma == 1000 else 8
        self.Damp = 1.0 / self.Gamma if platform.system() != "Darwin" else 1
        self.Tdump, self.Tdump_ref = self.set_Tdump() # set Tdump
        
        self.Tinit = 10 ** 6
        self.TSteps = self.Frames * self.Tdump
        self.Tequ = self.TSteps
        self.Tref = self.Frames * self.Tdump_ref
        self.Params["Total Run Steps"] = self.TSteps
        
    def set_Tdump(self):
        """计算 Tdump 的值"""
        Tdump = 2 * 10 ** self.eSteps // self.Frames
        Tdump_ref = Tdump // 100
        if platform.system() == "Darwin":
            Tdump = Tdump_ref
        return Tdump, Tdump_ref
    
    def set_queue(self):
        if platform.system() != "Darwin":
            try:
                bqueues = subprocess.check_output(['bqueues']).decode('utf-8') # Decode the output here
                bhosts = subprocess.check_output(['bhosts']).decode('utf-8')
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
                exit(1)
            
            queue_info = {"PEND": 0}
            myques = self.Params["Queue"]
            myhosts = {
                '7k83!': ['g009', 'g008', 'a016'],
                '9654!': ['a017']
            }
            for line in bqueues.strip().split('\n')[1:]:  # Skip the header line
                columns = line.split()
                if columns[0] in myques:
                    queue_info[columns[0]] = {"PEND": int(columns[8])} 
            
            host_info = {"7k83!": {"cores": 0, "run": 0, "suspend": 0}, "9654!": {"cores": 0, "run": 0, "suspend": 0}}
            for line in bhosts.strip().split('\n')[1:]:
                columns = line.split()
                for iqueue in myques:
                    if columns[0] in myhosts[iqueue]:
                        host_info[iqueue]["cores"] +=  int(columns[3])
                        host_info[iqueue]["run"] += int(columns[5])
                        host_info[iqueue]["suspend"] += int(columns[6])
                
            for iqueue in myques:
                    host_info[iqueue]["usage"] = (host_info[iqueue]["suspend"] + host_info[iqueue]["run"]) / host_info[iqueue]["cores"]
            #host_info = {iqueue: {"usage": (info["suspend"] + info["run"]) / info["cores"]} for iqueue, info in host_info.items()}
            self.Params["Queue"] = min(myques, key=lambda x: host_info[x]['usage'])
            #print(f"queue_info: {queue_info}, host_info: {host_info}")
            #exit(1)
        return self.Params["Queue"]
    
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
        self.set_box()   #set box
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
                raise ValueError("N_monos is too Long!")
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
    def __init__(self, Init, Pe, Xi):
        self.Init = Init
        self.Pe = Pe
        self.Xi = Xi
        self.Fa = self.Pe
        self.Kb = self.Xi * Init.N_monos/4
        #self.Fa = self.Pe / Init.N_monos ** 2
    
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
                f'read_data       {Path.data_file}',
                '#read_data		    ${dir_file}.data add append offset 3 0 0 0 0',
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
                f'fix   	        LANG chain langevin 1.0 1.0 1.0 {Run.Seed}',
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
            equal_potential = [
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
                f'fix      	      LANG chain langevin {Run.Temp} {Run.Temp} {Run.Damp} {Run.Seed}',
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
                f'fix      	      LANG chain langevin {Run.Temp} {Run.Temp} {Run.Damp} {Run.Seed}',
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
            with open(f"{dir_file}.in", "w") as file:
                self.write_section(file, setup)
                self.write_section(file, initial)
                self.write_section(file, equal_potential)
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
        print(f"{self.data_file}")
        if os.path.exists(os.path.join(self.dir_data, f"{self.Run.Trun:03}.lammpstrj")):
            return True
        else:
            return False
#############################################################################################################
class _plot:
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

__all__ = [
    "params",
    "_config",
    "_run",
    "_init",
    "_model",
    "_path",
    "Timer",
]
