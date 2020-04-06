import os
import numpy as np
import pandas as pd
from glob import glob

def NICS_worker(fname, repeat_num=1) :
    NICS_desc = np.empty([0, 7])
    
    E = '1.0000'
    F = '1.0000'
    No_Bq = 0
    NICS_0 = 0
    NICS_1 = 0
    NICS_0_ZZ = 0
    NICS_1_ZZ = 0
    FileNameStart = -3
    Atom_labels_list = []
    
    ifile = open( fname, 'r' )
    lines = ifile.readlines()
    ifile.close()
    for line in lines :
        words = line.split()
        if len(words) == 2: 
            if words[0] == '(Enter' and words[1][-13:] == 'g16/l101.exe)':
                FileNameStart = -2
        if FileNameStart > -3:
            FileNameStart += 1
            if FileNameStart == 1:
                Atom_labels_list += [int(x) for x in words[1:]]
                FileNameStart = -3
        if len(words) > 3:
            if words[0] == 'SCF' and words[1] == 'Done:':
                E = words[4]
            if  len(words) > 7:
                if words[1] == 'Bq' and words[2] == 'Isotropic' and words[3] == '=':
                    No_Bq += 1
                    if No_Bq == 1:
                        NICS_0 = -float(words[4])
                    if No_Bq == 2:
                        NICS_1 = -float(words[4])
            if No_Bq > 0 and len(words) == 6:
                if words[0] == 'XZ=' and words[2] == 'YZ=' and words[4] == 'ZZ=':
                    if No_Bq == 1:
                        NICS_0_ZZ = -float(words[5])
                    if No_Bq == 2:
                        NICS_1_ZZ = -float(words[5])
            if words[0] == 'Frequencies' and float(words[2]) < 0:
                F = words[2]
    repeat_num = len(Atom_labels_list)
    for i in range(repeat_num):
        fname_ls = os.path.basename(fname).split('-')
        fname_ls[1] = str(Atom_labels_list[i])
        if 'r' in fname_ls[3]:
            fname_ls.pop(3)
        if 'c1' in fname_ls:
            fname_ls.remove('c1')
        if 'NICS' in fname_ls:
            fname_ls.remove('NICS')
        fname_ls.pop()
        fname2 = '-'.join(fname_ls)
        item = np.array([fname2, np.round(float(E),6), F, np.round(float(NICS_0),4), np.round(float(NICS_1),4), np.round(float(NICS_0_ZZ),4), np.round(float(NICS_1_ZZ),4)])
        NICS_desc = np.append(NICS_desc, [item], axis=0)
    
    return NICS_desc
                  
               

def get_NICS(path):
    columns = ['filename', 'E', 'F', 'NICS_0', 'NICS_1', 'NICS_0_ZZ', 'NICS_1_ZZ']
    NICS_Summary = np.empty([0, len(columns)])
    for eachfile in glob('%s/*-NICS-*sp.log'%path) :
        NICS_desc = NICS_worker(eachfile)
        NICS_Summary = np.r_[NICS_Summary, NICS_desc]
    NICS_table_all = pd.DataFrame(NICS_Summary, columns=columns)  
    NICS_table_all = NICS_table_all.round({'NICS_0':4, 'NICS_1':4, 'NICS_0_ZZ':4, 'NICS_1_ZZ':4})
    NICS_table_all.to_csv('NICS_table.csv')
        
    return NICS_table_all
