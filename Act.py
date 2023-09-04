from paras import *
from datetime import datetime
import numpy as np
from lammps import lammps, PyLammps
import matplotlib

def convert2array(x):
    # 如果x已经是一个列表，numpy数组，直接转换为numpy数组
    if isinstance(x, (list, np.ndarray, tuple)):
        return np.array(x)
    # 如果x是一个单一的数值，转换为包含该数值的numpy数组
    elif isinstance(x, (int, float, str)):
        return np.array([x])
    else:
        raise ValueError("Unsupported type!")

def check_params(params):
    print("===> Caveats: Please confirm the following parameters:")
    for key, value in params.items():
        user_input = input(f"{key} = {value}    (y/n)?: ")
        if user_input.lower() == "y":
            break
        elif user_input.lower() != '':
            new_value = input(f"Please enter the new value for {key}: ")
            params[key] = type(value)(new_value)  # 更新值，并尝试保持原来的数据类型
    return params

#-----------------------------------Part I-------------------------------------------#    
if __name__ == "__main__":
    for iGamma in convert2array(params['Gamma']):
        # paras for run: Gamma, Queue, Frames, Trun, Dimend, Temp, Dump
        Run = _run(iGamma, params['Trun'], params['Dimend']) 
        #print(Run.__dict__)
        
        for iLabel in convert2array(params['Label']):
            # configure according to iLabel
            Config = _config(iLabel, Run, params)
            check_params(params) # check
            #print(Run.Dump)
            #exit(1)
            for iRin in convert2array(params['Rin']):
                for iWid in convert2array(params['Wid']):
                    for iN in convert2array(params['N_monos']):
                        # paras for init config: Rin, Wid, N_monos, L_box
                        Init = _init(Run, iRin, iWid, iN)
                        if Init.jump:
                            continue
                        queue = Run.set_queue()
                        print(f"{queue}\n", params["Queues"], params["usages"])
                        exit(1)
                        for iPe in convert2array(params['Pe']):
                            for iXi in convert2array(params['Xi']):
                                # paras for model: Pe(Fa), Xi(Kb)
                                Model = _model(Init, iPe, iXi)
                                Path = _path(Config, Model, Init, Run) # for directory
                                if Path.jump:
                                    continue
                                # prepare files and submit
                                Init.data_file(Path)
                                Model.in_file(Path)
                                Run.bsubs(Path)
                                #Run.bsubs(Path, 1)
##########################################END!################################################################
                                