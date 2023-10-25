from paras import *
import platform
from datetime import datetime
import numpy as np
#from lammps import lammps, PyLammps

#-----------------------------------Part I-------------------------------------------#    
if __name__ == "__main__":
    check = True
    for iDimend in convert2array(params['Dimend']):
        for iType in convert2array(params['labels']['Types']):
            for iEnv in convert2array(params['labels']['Envs']):
                Config = _config(iDimend, iType, iEnv, params)
                if platform.system() != "Darwin": #params['task'] == "Simus" and
                    mpl.use("agg")
                    if check:
                        check_params(params)  # continue
                        check = False

                for iGamma in convert2array(params['Gamma']):
                    for iTemp in convert2array(params['Temp']):
                        # paras for run: Gamma, Temp, Queue, Frames, Trun, Dimend, Temp, Dump
                        Run = _run(Config.Dimend, iGamma, iTemp, params['Trun'])  #print(Run.__dict__)
                        Config.set_dump(Run) #print(f"{params['marks']['config']}, {Run.Dump}, {Run.Tdump}")
                        for iRin in convert2array(params['Rin']):
                            for iWid in convert2array(params['Wid']):
                                for iN in convert2array(params['N_monos']):
                                    # paras for init config: Rin, Wid, N_monos, L_box
                                    Init = _init(Config, Run.Trun, iRin, iWid, iN)
                                    if Init.jump: # chains are too long or invalid label
                                        continue
                                    queue = Run.set_queue()  # print(f"{queue}\n", params["Queues"])
                                    for iFa in convert2array(params['Fa']):
                                        for iXi in convert2array(params['Xi']):
                                            # paras for model: Pe(Fa), Xi(Kb)
                                            Model = _model(Init, Run, iFa, iXi)
                                            Path = _path(Model) # for directory
                                            Plot = _plot(Path)
                                            if params['task'] == "Simus":
                                                if Path.jump: # jump for repeat
                                                    continue
                                                try:
                                                    Init.data_file(Path)
                                                    Model.in_file(Path)
                                                    #continue
                                                    Run.bsub(Path)
                                                    #Run.bsub(Path, 1)
                                                except Exception as e:
                                                    print(f"An error occurred: {e}")

                                            elif params['task'] == "Anas":
                                                if Path.jump:
                                                    #Plot.plot()
                                                    Plot.write_runfile()
##########################################END!################################################################