import subprocess
import platform
from datetime import datetime, timedelta

def get_bjobs_output(queue_name):
    try:
        return subprocess.check_output(['bjobs', '-u', 'all', '-q',  queue_name]).decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)

def update_queue_info(queue_info, bjobs_output, current_year):
    for line in bjobs_output.strip().split('\n')[1:]:
        columns = line.split()
        start_time = datetime.strptime(f"{columns[-4]} {columns[-3]} {current_year} {columns[-2]}", "%b %d %Y %H:%M")
        if datetime.now() - start_time > timedelta(hours=24):
            cores = int(columns[-1].split('*')[0]) if '*' in columns[-1] else 1
            queue_info["occupy"] += cores

def set_queue():
    queues = {
        "7k83!": {"usage": 1.0, "hosts": ['g009', 'g008', 'a016']},
        "9654!": {"usage": 1.0, "hosts": ['a017']}
    }

    if platform.system() == "Darwin":
        return None  # Replace with appropriate action
    
    try:
        bqueues_output = subprocess.check_output(['bqueues']).decode('utf-8')
        bhosts_output = subprocess.check_output(['bhosts']).decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)

    queue_names = list(queues.keys())
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
            if columns[0] in queues[queue_name]["hosts"]:
                #queue_info[queue_name]["run"] += int(columns[5])
                queue_info[queue_name]["cores"] += int(columns[3])

    for queue_name in queue_names:
        bjobs_output = get_bjobs_output(queue_name)
        update_queue_info(queue_info[queue_name], bjobs_output, current_year)
        
        usage = round((queue_info[queue_name]["PEND"] + queue_info[queue_name]["run"] - queue_info[queue_name]["occupy"]) / (queue_info[queue_name]["cores"] - queue_info[queue_name]["occupy"]), 3)
        queue_info[queue_name]["usage"] = usage

    optimal_queue = min(queue_names, key=lambda x: queue_info[x]['usage'])

    return optimal_queue, queue_info

if __name__ == "__main__":
    optimal_queue, queue_info = set_queue()
    print(f"queue info: {queue_info}")

