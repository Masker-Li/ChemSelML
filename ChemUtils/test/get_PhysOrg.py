import os
import sys
sys.path.append("/mnt/G16/Scripts")

from ChemUtils.bin.PhysOrg import get_Pre_EQBV, get_Pre_PhysOrg
from ChemUtils.bin.NICS import get_NICS

rootpath = os.getcwd()

EQBV_dir = rootpath+r'/Desc_sp_gas'
os.chdir(EQBV_dir)
print(rootpath)
Merge_table = get_Pre_EQBV(EQBV_dir)

NICS_dir = rootpath+r'/NICS'
if os.path.isdir(NICS_dir):
    os.chdir(NICS_dir)
    path = os.path.abspath(NICS_dir)
    NICS_table = get_NICS(path)
    final_table = get_Pre_PhysOrg(rootpath, Merge_table, NICS_table)
    
    
print ('')
os.system("pause")
