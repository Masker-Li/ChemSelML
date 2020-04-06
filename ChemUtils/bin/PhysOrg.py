import os
import sys
import numpy as np
import pandas as pd
from glob import glob
import time
from ..bin.GaussianOutputFileReader import Gaussian_Output
from ..bin.BV import BV_engine 


def get_BV_desc(AtomCoordinates, Atom_label):
    AtomCoordinates = np.array(AtomCoordinates)
    BV = list(BV_engine(AtomCoordinates, Atom_label))[:3]
    return BV
    
    
def GetPreDesc(GO, FQtype='N'):
    if GO.is_NormalEnd:
        EQB_desc = np.empty([0, 13])
        BV_desc = np.empty([0, 4])
        Atom_labels_list = GO.Atom_labels_list
        for j in range(len(Atom_labels_list)):
            atomIdx = Atom_labels_list[j]-1
            fn_ls = GO.fn.split('-')
            if 'c1' in fn_ls:
                fn_ls.remove('c1')
            fn_ls.pop()
            fn_ls[1] = str(atomIdx+1)
            fn = '-'.join(fn_ls)
            BondOrderArray = np.empty(0)
            for i in range(GO._nAtoms):
                if i != atomIdx:
                    WibergBondIndex = GO.BondIndex[atomIdx][i]
                    if WibergBondIndex > 0.3:
                        BondOrderArray = np.append(
                            BondOrderArray, np.array([GO.Atoms[i], WibergBondIndex]))
            BondOrderArray = np.append(
                BondOrderArray, np.full(8-len(BondOrderArray), None))
            item = np.append(np.array(
                [fn, GO._nAtoms, GO.E_HOMO, GO.E_LUMO, GO.NPACharge[atomIdx]]), BondOrderArray)
            EQB_desc = np.append(EQB_desc, [item], axis=0)

            if FQtype == 'N':
                t0 = time.clock()
                E_xyz = np.c_[GO.AtomsNum, GO.AtomsCoordinates].tolist()
                BV = get_BV_desc(E_xyz, atomIdx+1)
                BV_desc = np.append(BV_desc, [np.array([fn] + BV)], axis=0)
                print('       Calculating BV of %s cost %.4fs' %
                      (fn, time.clock() - t0))
        return EQB_desc, BV_desc


def GetPOSTDesc(GO, FQtype='N'):
    EQB_desc = np.empty([0, 14])
    BV_desc = np.empty([0, 13])
    fn_ls = GO.fn.split('-')
    if 'c1' in fn_ls:
        fn_ls.remove('c1')
    fn_ls.pop()
    fn = '-'.join(fn_ls)
    if GO._RDKit_Norm:
        KeyAtoms = GO._PostKeyAtoms
        keySelf, keyR, sameRingAtoms, keyH = KeyAtoms['keySelf'], KeyAtoms[
            'keyR'], KeyAtoms['sameRingAtoms'], KeyAtoms['keyH']

        BondOrderArray = np.full(8, None)
        if len(keySelf) > 1:
            WibergBondIndex = GO.BondIndex[keySelf[0]]
            BondOrderArray[0] = keyR[1]
            BondOrderArray[1] = WibergBondIndex[keyR[0]]
            BondOrderArray[2] = keyH[1]
            BondOrderArray[3] = WibergBondIndex[keyH[0]]
            BondOrderArray[4] = sameRingAtoms[0][1]
            BondOrderArray[5] = WibergBondIndex[sameRingAtoms[0][0]]
            BondOrderArray[6] = sameRingAtoms[1][1]
            BondOrderArray[7] = WibergBondIndex[sameRingAtoms[1][0]]

        EQB_item = np.append(np.array(
            [fn, GO._nAtoms, GO.E_HOMO, GO.E_LUMO, GO.NPACharge[keySelf[0]], GO.NPACharge[keyR[0]]]), BondOrderArray)
        EQB_desc = np.append(EQB_desc, [EQB_item], axis=0)

        if FQtype == 'N':
            t0 = time.clock()
            E_xyz = np.c_[GO.AtomsNum, GO.AtomsCoordinates].tolist()
            BV_keySelf = get_BV_desc(E_xyz, keySelf[0]+1)
            BV_keyR = get_BV_desc(E_xyz, keyR[0]+1)
            RH_ratio = GO.Post_Angle_RtoPlane[0]/GO.Post_Angle_HtoPlane[0]

            BV_item = np.array([fn, GO.Post_Bond_R, GO.Post_Bond_H, GO.Post_Angle_RH[0],
                                GO.Post_Angle_RtoPlane[0], GO.Post_Angle_HtoPlane[0], RH_ratio] + BV_keySelf + BV_keyR)
            BV_desc = np.append(BV_desc, [BV_item], axis=0)
            print('       Calculating BV of %s cost %.4fs' %
                  (fn, time.clock() - t0))
    else:
        EQB_item = np.append(
            np.array([fn, GO._nAtoms, GO.E_HOMO, GO.E_LUMO]), np.full(10, None))
        BV_item = np.append(np.array([fn]), np.full(12, None))

        EQB_desc = np.append(EQB_desc, [EQB_item], axis=0)
        BV_desc = np.append(BV_desc, [BV_item], axis=0)

    return EQB_desc, BV_desc


def GetPOSTDesc_noBV(GO, FQtype='N'):
    EQB_desc = np.empty([0, 14])
    BV_desc = np.empty([0, 13])
    fn_ls = GO.fn.split('-')
    if 'c1' in fn_ls:
        fn_ls.remove('c1')
    fn_ls.pop()
    fn = '-'.join(fn_ls)
    if GO._RDKit_Norm:
        KeyAtoms = GO._PostKeyAtoms
        keySelf, keyR, sameRingAtoms, keyH = KeyAtoms['keySelf'], KeyAtoms[
            'keyR'], KeyAtoms['sameRingAtoms'], KeyAtoms['keyH']

        BondOrderArray = np.full(8, None)
        if len(keySelf) > 1:
            WibergBondIndex = GO.BondIndex[keySelf[0]]
            BondOrderArray[0] = keyR[1]
            BondOrderArray[1] = WibergBondIndex[keyR[0]]
            BondOrderArray[2] = keyH[1]
            BondOrderArray[3] = WibergBondIndex[keyH[0]]
            BondOrderArray[4] = sameRingAtoms[0][1]
            BondOrderArray[5] = WibergBondIndex[sameRingAtoms[0][0]]
            BondOrderArray[6] = sameRingAtoms[1][1]
            BondOrderArray[7] = WibergBondIndex[sameRingAtoms[1][0]]

        EQB_item = np.append(np.array(
            [fn, GO._nAtoms, GO.E_HOMO, GO.E_LUMO, GO.NPACharge[keySelf[0]], GO.NPACharge[keyR[0]]]), BondOrderArray)
        EQB_desc = np.append(EQB_desc, [EQB_item], axis=0)

        if FQtype == 'N':
            t0 = time.clock()
            E_xyz = np.c_[GO.AtomsNum, GO.AtomsCoordinates].tolist()
            #BV_keySelf = get_BV_desc(E_xyz, keySelf[0]+1)
            #BV_keyR = get_BV_desc(E_xyz, keyR[0]+1)
            BV_keySelf = [None, None, None]
            BV_keyR = [None, None, None]
            RH_ratio = GO.Post_Angle_RtoPlane[0]/GO.Post_Angle_HtoPlane[0]

            BV_item = np.array([fn, GO.Post_Bond_R, GO.Post_Bond_H, GO.Post_Angle_RH[0],
                                GO.Post_Angle_RtoPlane[0], GO.Post_Angle_HtoPlane[0], RH_ratio] + BV_keySelf + BV_keyR)
            BV_desc = np.append(BV_desc, [BV_item], axis=0)
            print('       Calculating BV of %s cost %.4fs' %
                  (fn, time.clock() - t0))
    else:
        EQB_item = np.append(
            np.array([fn, GO._nAtoms, GO.E_HOMO, GO.E_LUMO]), np.full(10, None))
        BV_item = np.append(np.array([fn]), np.full(12, None))

        EQB_desc = np.append(EQB_desc, [EQB_item], axis=0)
        BV_desc = np.append(BV_desc, [BV_item], axis=0)

    return EQB_desc, BV_desc


def AppendErrorLog(ErrorLogFile, Errors):
    nLines = len(Errors)
    if nLines:
        with open(ErrorLogFile, 'a+') as LogTXT:
            for i in range(nLines):
                line = '%10s, %40s  , %s ,  %s  \n' % (
                    Errors[i][0], Errors[i][1], Errors[i][2], Errors[i][3])
                LogTXT.write(line)


def CatchDscrptr(rootpath, path, RC_State='PRE', FQtype='N'):
    os.chdir(path)
    Errors = np.empty([0, 4])
    if RC_State == 'PRE':
        EQB_Summary = np.empty([0, 13])
        BV_Summary = np.empty([0, 4])
    elif RC_State == 'POST':
        FQtype = 'N'
        EQB_Summary = np.empty([0, 14])
        BV_Summary = np.empty([0, 13])

    if FQtype == 'N':
        filetype = '*sp.log'
    elif FQtype == '+':
        filetype = '*sp+.log'
    elif FQtype == '-':
        filetype = '*sp-.log'

    for eachfile in glob(filetype):
        GO = Gaussian_Output(path, eachfile, RC_State=RC_State)
        if GO.is_NormalEnd:
            if not GO._PostKeyAtoms:
                Error = ['Structure Failure', GO.fn, path, GO.error]
                Errors = np.append(Errors, np.array([Error]),axis=0)
            elif not GO._RDKit_Norm:
                Error = ['Rdkit Failure', GO.fn, path, GO.error]
                Errors = np.append(Errors, np.array([Error]),axis=0)
            if RC_State == 'PRE':
                EQB_desc, BV_desc = GetPreDesc(GO, FQtype)
            elif RC_State == 'POST':
                EQB_desc, BV_desc = GetPOSTDesc(GO, FQtype)
            EQB_Summary = np.r_[EQB_Summary, EQB_desc]
            BV_Summary = np.r_[BV_Summary, BV_desc]
    ErrorLogFile = rootpath + '/FailureFiles.txt'
    AppendErrorLog(ErrorLogFile, Errors)
    return EQB_Summary, BV_Summary

    
def get_Pre_EQBV(rootpath):
    os.chdir(rootpath)
    EB_columns = ['filename', 'NAtoms', 'HOMO', 'LUMO',
                  'QA', 'A1', 'B1', 'A2', 'B2', 'A3', 'B3', 'A4', 'B4']
    FQ_columns = ['filename', 'NAtoms', 'HOMO', 'LUMO', 'QA',
                  'HOMO+', 'LUMO+', 'QA+', 'HOMO-', 'LUMO-', 'QA-']
    BV_columns = ['filename', 'BV_3A', 'BV_4A', 'BV_5A']
    EB_table = np.empty([0, len(EB_columns)])
    FQ_table = np.empty([0, len(FQ_columns)])
    BV_table = np.empty([0, len(BV_columns)])

    EB_dir = r'%s' % rootpath
    EQB_Summary, BV_Summary = CatchDscrptr(
        rootpath, EB_dir, RC_State='PRE', FQtype='N')
    EB_table = np.r_[EB_table, EQB_Summary]
    BV_table = np.r_[BV_table, BV_Summary]

    FQ_dir = r'%s' % rootpath
    EQB_Summary_P = CatchDscrptr(
        rootpath, EB_dir, RC_State='PRE', FQtype='+')[0]
    EQB_Summary_N = CatchDscrptr(
        rootpath, EB_dir, RC_State='PRE', FQtype='-')[0]
    Output = np.c_[np.c_[EQB_Summary[:, :5],
                         EQB_Summary_P[:, 2:5]], EQB_Summary_N[:, 2:5]]
    FQ_table = np.r_[FQ_table, Output]

    EB_table_all = pd.DataFrame(EB_table, columns=EB_columns)
    FQ_table_all = pd.DataFrame(FQ_table, columns=FQ_columns)
    BV_table_all = pd.DataFrame(BV_table, columns=BV_columns)

    os.chdir(rootpath)
    # print(EB_table_all)
    # print(FQ_table_all)
    # print(BV_table_all)
    EB_table_all.to_csv('EB_table.csv')
    FQ_table_all.to_csv('FQ_table.csv')
    BV_table_all.to_csv('BV_table.csv')
    
    FQ_table_all['add1'], FQ_table_all['idx'] = '', BV_table_all.index
    BV_table_all['add2'], BV_table_all['idx'] = '', EB_table_all.index
    
    Merge_table = pd.concat([FQ_table_all,BV_table_all,EB_table_all], axis=1)
    Merge_table.to_csv('Merge_table.csv')
    return Merge_table


def get_POST_EQBV(rootpath):
    os.chdir(rootpath)
    Radical_list = ['CCN', 'CF2', 'CF3', 'CFM', 'CH3', 'CHF',
                    'Et0', 'ipr', 'MF3', 'Ph', 'Py0', 'tBu', 'TF3']
    EB_columns = ['filename', 'NAtoms', 'HOMO', 'LUMO', 'QA_Ar', 'QA_R',
                  'A_R', 'B_R', 'A_H', 'B_H', 'A_S1', 'B_S1', 'A_S2', 'B_S2']
    #FQ_columns = ['filename','NAtoms','HOMO','LUMO', 'QA','HOMO+','LUMO+', 'QA+','HOMO-','LUMO-', 'QA-']
    BV_columns = ['fn', 'BL_R', 'BL_H', 'A_RH', 'A_R2P', 'A_H2P', 'ARatio_RH',
                  'Ar_BV_3A', 'Ar_BV_4A', 'Ar_BV_5A', 'R_BV_3A', 'R_BV_4A', 'R_BV_5A']
    EB_table = np.empty([0, len(EB_columns)])
    #FQ_table = np.empty([0,len(FQ_columns)])
    BV_table = np.empty([0, len(BV_columns)])

    for subdir in os.listdir(rootpath):
        if os.path.isdir(subdir) and subdir in Radical_list:
            EB_dir = r'%s/%s' % (rootpath, subdir)
            EQB_Summary, BV_Summary = CatchDscrptr(
                rootpath, EB_dir, RC_State='POST', FQtype='N')
            EB_table = np.r_[EB_table, EQB_Summary]
            BV_table = np.r_[BV_table, BV_Summary]
            os.chdir(rootpath)

            #FQ_dir = r'%s'%rootpath
            #EQB_Summary_P = CatchDscrptr(rootpath,EB_dir,RC_State='POST',FQtype='+')[0]
            #EQB_Summary_N = CatchDscrptr(rootpath,EB_dir,RC_State='POST',FQtype='-')[0]
            #Output = np.c_[np.c_[EQB_Summary[:,:5],EQB_Summary_P[:,2:5]],EQB_Summary_N[:,2:5]]
            #FQ_table = np.r_[FQ_table,Output]
            # os.chdir(rootpath)

    EB_table_all = pd.DataFrame(EB_table, columns=EB_columns)
    #FQ_table_all = pd.DataFrame(FQ_table,columns=FQ_columns)
    BV_table_all = pd.DataFrame(BV_table, columns=BV_columns)

    os.chdir(rootpath)
    EB_table_all.to_csv('EB2_table.csv')
    # FQ_table_all.to_csv('FQ_table.csv')
    BV_table_all.to_csv('BV2_table.csv')
    
    BV_table_all['add2'], BV_table_all['idx'] = '', EB_table_all.index
    
    Merge_table = pd.concat([BV_table_all,EB_table_all], axis=1)
    Merge_table.to_csv('Merge_table.csv')
    return Merge_table


def get_Pre_PhysOrg(dst_dir, Merge_table, NICS_table):
    Merge_table = Merge_table.T.drop_duplicates(subset=None, keep='first', inplace=False).T
    
    df = pd.DataFrame()
    df['fn'] = Merge_table['filename']
    df[['Ar_n', 'loc_n']] = df[['fn']].apply(lambda x: x['fn'].split('-')[:2],axis=1,result_type="expand")
    
    df[['E_HOMO', 'E_LUMO']] = Merge_table[['HOMO', 'LUMO']]
    df['m'] = (df['E_HOMO'].astype(float) + df['E_LUMO'].astype(float))/2
    df['h'] = (-df['E_HOMO'].astype(float) + df['E_LUMO'].astype(float))/2
    df['s'] = 1/df['h']
    df['w'] = df['m']*df['m']/(2*df['h'])
    
    df['QN'] = Merge_table['QA'].astype(float)
    df['f1'] = Merge_table['QA'].astype(float) - Merge_table['QA-'].astype(float)
    df['f0'] = (Merge_table['QA+'].astype(float) - Merge_table['QA-'].astype(float))/2
    df['f_1'] = Merge_table['QA+'].astype(float) - Merge_table['QA'].astype(float)
    df['Df'] = df['f1'] - df['f_1']
    df['local_S_f0'] = df['s']*df['f0']
    
    tmp_B = Merge_table[['B1','B2', 'B3']].astype(float)
    df['maxC-C'] = np.max(tmp_B,axis=1)
    df['C-H'] = np.min(tmp_B,axis=1)
    df['aveB'] = np.average(tmp_B,axis=1)
    df['midC-C'] = 3*df['aveB'] - df['maxC-C']- df['C-H']
    
    df[['BV_3A', 'BV_4A', 'BV_5A']] = Merge_table[['BV_3A', 'BV_4A', 'BV_5A']].astype(float)
    tmp_NICS = NICS_table[['filename', 'NICS_0', 'NICS_1', 'NICS_0_ZZ', 'NICS_1_ZZ']]
    df = pd.merge(df, tmp_NICS,how='left', left_on='fn', right_on='filename', indicator='left_only')
    df = df.round({'m': 6, 'h': 6, 's': 6, 'w': 6, 
                   'QN': 5, 'f1': 5, 'f0': 5, 'f_1': 5, 'Df': 5, 'local_S_f0': 5, 
                   'maxC-C': 4, 'C-H': 4, 'midC-C': 4, 'aveB': 4, 
                   'BV_3A': 4, 'BV_4A': 4, 'BV_5A': 4})
    
    columns = ['fn', 'Ar_n', 'loc_n', 'E_HOMO', 'E_LUMO', 'm', 'h', 's', 'w', 'QN',
       'f1', 'f0', 'f_1', 'Df', 'local_S_f0', 'maxC-C', 'C-H', 'midC-C', 'aveB',
       'BV_3A', 'BV_4A', 'BV_5A', 'NICS_0', 'NICS_1', 'NICS_0_ZZ', 'NICS_1_ZZ']
    df = df[columns]
    df.to_csv('%s/Ar_phychem.csv'%dst_dir)
    return df

