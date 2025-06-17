import sys
import os
import time

from vs.environment import Env
from maze_explorer import Explorer
from maze_rescuer import Rescuer

def main(data_folder_name):
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    env = Env(data_folder)

    rescuer_file = os.path.join(data_folder, "rescuer_config.txt")

    resc = Rescuer(env, rescuer_file)

    filename = f"explorer_config.txt"
    explorer_file = os.path.join(data_folder, filename)
    
    Explorer(env, explorer_file, resc)

    env.run()
    
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_408v_94x94")
        
    main(data_folder_name)
