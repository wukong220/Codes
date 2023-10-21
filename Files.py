import os
import re
import sys
import subprocess
import platform
import shutil
import logging
from datetime import datetime, timedelta

class Reorganize:
    def __init__(self, start_path= 'Simus'):
        self.start_path = os.path.join(os.path.dirname(os.getcwd()), start_path)
        self.backup_path = os.path.join(self.start_path, "Backup")
        self.queues = {
            "7k83!": {"usage": 1.0, "hosts": ['g009', 'g008', 'a016']},
            "9654!": {"usage": 1.0, "hosts": ['a017']}
        }
        self.rename_map = {
            ".refine.lammpstrj": ".equ.lammpstrj",
            ".end.restart": ".init.restart",
            ".refine.restart": ".equ.restart",
        }
        # 检查并创建备份目录
        if not os.path.exists(self.backup_path):
            os.makedirs(self.backup_path)
##############################################################################
    def set_queue(self):
        if platform.system() == "Darwin":
            return None  # Replace with appropriate action
        try:
            bqueues_output = subprocess.check_output(['bqueues']).decode('utf-8')
            bhosts_output = subprocess.check_output(['bhosts']).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            exit(1)

        queue_names = list(self.queues.keys())
        queue_info = {key: {"PEND": 0, "run": 0, "cores": 0, "occupy": 0} for key in queue_names}
        current_year = datetime.now().year

        print(bqueues_output.strip().split('\n')[0])
        for line in bqueues_output.strip().split('\n')[1:]:
            columns = line.split()
            if columns[0] in queue_names:
                queue_info[columns[0]]["PEND"] = int(columns[8])
                queue_info[columns[0]]["run"] = int(columns[9]) 
                print(columns)

        for line in bhosts_output.strip().split('\n')[1:]:
            columns = line.split()
            for queue_name in queue_names:
                if columns[0] in self.queues[queue_name]["hosts"]:
                    #queue_info[queue_name]["run"] += int(columns[5])
                    queue_info[queue_name]["cores"] += int(columns[3])

        for queue_name in queue_names:
            bjobs_output = subprocess.check_output(['bjobs', '-u', 'all', '-q',  queue_name]).decode('utf-8')
            for line in bjobs_output.strip().split('\n')[1:]:
                columns = line.split()
                start_time = datetime.strptime(f"{columns[-4]} {columns[-3]} {current_year} {columns[-2]}", "%b %d %Y %H:%M")
                if datetime.now() - start_time > timedelta(hours=24):
                    cores = int(columns[-1].split('*')[0]) if '*' in columns[-1] else 1
                    queue_info[queue_name]["occupy"] += cores

            usage = round((queue_info[queue_name]["PEND"] + queue_info[queue_name]["run"] - queue_info[queue_name]["occupy"]) / (queue_info[queue_name]["cores"] - queue_info[queue_name]["occupy"]), 3)
            queue_info[queue_name]["usage"] = usage

        optimal_queue = min(queue_names, key=lambda x: queue_info[x]['usage'])
        return optimal_queue, queue_info
##############################################################################
    def rename_files(self):
        print(f"Rename folders……")
        logging.info(f"\n\nRename folders……")
        # 获取当前目录下的所有条目，并过滤出文件夹
        self.rename_map = {"100G": "100.0G",}
        dirnames = [d for d in os.listdir(self.start_path) if os.path.isdir(os.path.join(self.start_path, d))]
        # 遍历当前目录下的所有文件夹
        for dirname in dirnames:
            # Perform the specified file operations
            # delete files
            for num in ["001", "002", "003", "004", "005"]:
                # Delete files
                for ext in [".equ.lammpstrj", ".lammpstrj", ".init.restart", ".equ.restart"]:
                    file_to_delete = os.path.join(dirpath, num + ext)
                    if os.path.exists(file_to_delete):
                        print(f"deleting folder {file_to_delete}")
                        logging.info(f"deleting folder {file_to_delete}")
                        # 执行操作
                        #os.remove(file_to_delete)

            # rename files
            for old_ext, new_ext in rename_map.items():
                new_name = dirname.replace(old_ext, new_ext)
                old_folder_path = os.path.join(self.start_path, dirname)
                new_folder_path = os.path.join(self.start_path, new_name)
                # 检查 new_folder_path 是否已经存在
                if os.path.exists(new_folder_path):
                    print(f"Warning: {new_folder_path} already exists. Exiting.")
                    logging.info(f"Warning: {new_folder_path} already exists. Exiting.")
                    sys.exit(1)
                print(f"Renaming folder {old_folder_path} to {new_folder_path}")
                logging.info(f"Renaming folder {old_folder_path} to {new_folder_path}")
                # 执行实际的重命名操作
                #os.rename(old_folder_path, new_folder_path)

        # 如果运行到这里，说明所有需要重命名的文件夹都已经处理完毕
        print("All folders have been renamed.")
        logging.info("All folders have been renamed.")

##############################################################################
    def merge_files(self):
        logging.basicConfig(filename='Files.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        print(f"Merge folders……")
        logging.info(f"\n\nMerge folders……")
        for dirpath in os.scandir(self.start_path):
            if ('Backup' not in dirpath.path) and (dirpath.is_dir()):
                for dirnames in os.scandir(dirpath.path):
                    if dirnames.is_dir():
                        full_path = dirnames.path
                        if "2D_100.0G_" in full_path:
                            match = re.match(r"(.*)/(\d)D_([\d.]+?)G_([\d.]+?)Pe_(\w+)/([\d.]+?)R([\d.]+?)_(\d+?)N(\d+?)_(\w+)", full_path)
                            if match:
                                path, D, G, Pe, Type, Rin, Wid, N, num, Env = match.groups()
                                #print(full_path)
                                #print(f"path={path}, D={D}, G={G}, Pe={Pe}, Type={Type}, Rin={Rin}, Wid={Wid}, N={N}, num={num}, Env={Env}\n")
                                base_name = os.path.join(f"{path}", f"{D}D_{G}G_{Pe}Pe_{Type}", f"{Rin}R{Wid}_{N}N{num}")
                                anulus_dir = os.path.join(f"{base_name}_Anulus")
                                anlus_dir = os.path.join(f"{base_name}_Anlus")

                                # If both Anulus and Anlus exist, merge them
                                if os.path.exists(anulus_dir) and os.path.exists(anlus_dir):
                                    print(f"Merging {anulus_dir} into {anlus_dir}")
                                    subprocess.run(["rsync", "-av", anulus_dir + "/", anlus_dir + "/"])
                                    #shutil.rmtree(anulus_dir)

                                # If only Anulus exists, rename it to Anlus
                                elif os.path.exists(anulus_dir):
                                    print(f"Renaming {anulus_dir} to {anlus_dir}")
                                    os.rename(anulus_dir, anlus_dir)

    ##############################################################################
    def reorganize_dirs(self):
        for dirpath, dirnames, filenames in os.walk(self.start_path):
            #2D_100G_1.0T_Chain/5.0R5.0_100N1_Anulus
            if 'test' not in dirpath:
                for dirname in dirnames:
                    try:
                        # 分割字符串获取参数
                        match = re.match(r"(\d+?)G([\d.]+?)T(\d)D_([\d.]+?)R([\d.]+?)_(\d+?)N(\d+?)_(.*)", dirname)
                        if match:
                            G, T, D, Rin, Wid, N, num, label = match.groups()
                            # 创建新的母目录和子目录名
                            parent_folder = f"{D}D_{G}G_{T}T"
                            child_folder = f"{Rin}R{Wid}_{N}N{num}_Anlus"

                            # 完整的旧路径和新路径
                            old_path = os.path.join(dirpath, dirname)
                            new_parent_path = os.path.join(dirpath, f'{parent_folder}_{label}')
                            new_child_path = os.path.join(new_parent_path, child_folder)

                            # 创建新的母目录（如果还没有创建）
                            if not os.path.exists(new_parent_path):
                                os.mkdir(new_parent_path)
                                print(f"Created new parent directory: {new_parent_path}")
                                logging.info(f"Created new parent directory: {new_parent_path}")

                            # 移动文件和文件夹到新的目录下
                            shutil.move(old_path, new_child_path)
                            print(f"Moved contents of {old_path} to {new_child_path}")
                            logging.info(f"Moved contents of {old_path} to {new_child_path}")

                    except Exception as e:
                        print(f"An error occurred while processing directory {dirname}: {e}")
##############################################################################
    def remove_empty_dirs(self, dirpath):
        # Remove child directories first (post-order traversal)
        for dirname in os.listdir(dirpath):
            if 'Backup' not in dirpath:
                full_dir_path = os.path.join(dirpath, dirname)
                if os.path.isdir(full_dir_path):
                    self.remove_empty_dirs(full_dir_path)
        # Then remove the parent directory if empty
        if not os.listdir(dirpath):
            if 'Backup' not in dirpath:
                os.rmdir(dirpath)
                print(f"Removed empty directory: {dirpath}")

    def reorganize(self):
        logging.basicConfig(filename='Files.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        #for dirpath, dirnames, filenames in os.walk(self.start_path):
        for dirpath in os.scandir(self.start_path):
            if ('Backup' not in dirpath.path) and (dirpath.is_dir()):
                for dirnames in os.scandir(dirpath.path):
                    if dirnames.is_dir():
                        for dirname in os.scandir(dirnames.path):
                            full_path = dirname.path
                            try:
                                # 使用正则表达式匹配目录名以提取参数
                                match = re.match(r"(.*)/(\d)D_([\d.]+?)G_([\d.]+?)T_(\w+)/([\d.]+?)R([\d.]+?)_(\d+?)N(\d+?)_(\w+)/([\d.]+?)Pe_([\d.]+?)Xi_(\w+)", full_path)
                                #3D_0.1G_0.1T_Ring/250N1_Free/10.0Pe_0.0Xi_8T5
                                Free = re.match(r"(.*)/(\d)D_([\d.]+?)G_([\d.]+?)T_(\w+)/(\d+?)N(\d+?)_(\w+)/([\d.]+?)Pe_([\d.]+?)Xi_(\w+)", full_path)
                                if match:
                                    #print(full_path)
                                    path, D, G, T, Type, Rin, Wid, N, num, Env, Pe, Xi, run = match.groups()
                                    #print(f"path={path}, D={D}, G={G}, T={T}, Type={Type}, Rin={Rin}, Wid={Wid}, N={N}, num={num}, Env={Env}, Pe={Pe}, Xi={Xi}, run={run}\n")
                                    old_folder_path =os.path.join(f"{path}", f"{D}D_{G}G_{T}T_{Type}", f"{Rin}R{Wid}_{N}N{num}_{Env}", f"{Pe}Pe_{Xi}Xi_{run}")
                                    new_folder_path =os.path.join(f"{path}", f"{D}D_{G}G_{Pe}Pe_{Type}", f"{Rin}R{Wid}_{N}N{num}_{Env}", f"{T}T_{Xi}Xi_{run}")
                                    backup_foler_path = os.path.join(f"{self.backup_path}", f"{D}D_{G}G_{T}T_{Type}", f"{Rin}R{Wid}_{N}N{num}_{Env}", f"{Pe}Pe_{Xi}Xi_{run}")
                                elif Free:
                                    path, D, G, T, Type, N, num, Env, Pe, Xi, run = Free.groups()
                                    old_folder_path =os.path.join(f"{path}", f"{D}D_{G}G_{T}T_{Type}", f"{N}N{num}_{Env}", f"{Pe}Pe_{Xi}Xi_{run}")
                                    new_folder_path =os.path.join(f"{path}", f"{D}D_{G}G_{Pe}Pe_{Type}", f"{N}N{num}_{Env}", f"{T}T_{Xi}Xi_{run}")
                                    backup_foler_path = os.path.join(f"{self.backup_path}", f"{D}D_{G}G_{T}T_{Type}", f"{N}N{num}_{Env}", f"{Pe}Pe_{Xi}Xi_{run}")
                                else:
                                    print(f"Could not extract params from {full_path}, error: {e}")
                                    
                                os.makedirs(new_folder_path, exist_ok=True)
                                os.makedirs(backup_foler_path, exist_ok=True)

                                print(f"Reorganized: {old_folder_path}\nTo {new_folder_path}\n")
                                logging.info(f"Reorganized: {old_folder_path}\nTo {new_folder_path}\n")

                                for filename in os.listdir(full_path):
                                    old_file_path = os.path.join(old_folder_path, filename)
                                    new_file_path = os.path.join(new_folder_path, filename)
                                    backup_file_path = os.path.join(backup_foler_path, filename)
                                    # 创建备份并移动文件
                                    if not os.path.exists(new_file_path):
                                        #print(f"old_file_path:{old_file_path}\nnew_file_path:{new_file_path}\nbackup_file_path:{backup_file_path}\n")
                                        try:
                                            #subprocess.run(["cp", f"{old_file_path}", backup_file_path])
                                            subprocess.run(["mv", f"{old_file_path}", new_file_path])
                                        except Exception as e:
                                            print(f"Could not move {old_file_path} to {new_file_path}, error: {e}")
                                    
                            except Exception as e:
                                print(f"Could not extract params from {full_path}, error: {e}")

        self.remove_empty_dirs(self.start_path)
        logging.info("All folders and files have been reorganized and backed up.")
##############################################################################
    def clean_lammpstrj(self):
        logging.basicConfig(filename='Files.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        print(f"Checking and deleting files…")
        logging.info(f"\n\nChecking and deleting files…")
        for dirpath in os.scandir(self.start_path):
            if ('Backup' not in dirpath.path) and (dirpath.is_dir()) and ('3D*Chain' not in dirpath.path):
                for dirnames in os.scandir(dirpath.path):
                    if dirnames.is_dir():
                        for dirname in os.scandir(dirnames.path):
                            full_path = dirname.path
                            if '3D' in full_path and 'Chain' in full_path and 'Rand' in full_path:
                                refine_lammpstrj_files = []
                                for i in range(1, 6):  # 1, 2, 3, 4, 5
                                    file_name = os.path.join(full_path, f"{i:03d}.refine.lammpstrj")
                                    refine_lammpstrj_files.append(file_name)
                                    all_files_exist = all(os.path.exists(file) for file in refine_lammpstrj_files)

                                if all(os.path.exists(file) for file in refine_lammpstrj_files):
                                    continue
                                else:
                                    file_to_delete = os.path.join(full_path, "005.lammpstrj")
                                    print(file_to_delete)
                                    if os.path.exists(file_to_delete):
                                        print(f"Deleting file {file_to_delete}")
                                        logging.info(f"Deleting file {file_to_delete}")
                                        #os.remove(file_to_delete)
##############################################################################

#usage
if __name__ == "__main__":
    reorg = Reorganize()
    #reorg.clean_lammpstrj()
    reorg.merge_files()