import os
import sys
import numpy as np
import pandas as pd
from scipy import constants as Const
import openbabel
from rdkit import Chem
import time
from glob import glob


def _isfloat(aString):
    try:
        float(aString)
        return True
    except:
        return False


def Element2AtomNum(element=None, atomnum=None):
    Periodic_Table = {"H":   1, "He":   2, "Li":   3, "Be":   4, "B":   5, "C":   6, "N":   7, "O":   8, "F":   9, "Ne":  10,
                      "Na":  11, "Mg":  12, "Al":  13, "Si":  14, "P":  15, "S":  16, "Cl":  17, "Ar":  18, "K":  19, "Ca":  20,
                      "Sc":  21, "Ti":  22, "V":  23, "Cr":  24, "Mn":  25, "Fe":  26, "Co":  27, "Ni":  28, "Cu":  29, "Zn":  30,
                      "Ga":  31, "Ge":  32, "As":  33, "Se":  34, "Br":  35, "Kr":  36, "Rb":  37, "Sr":  38, "Y":  39, "Zr":  40,
                      "Nb":  41, "Mo":  42, "Tc":  43, "Ru":  44, "Rh":  45, "Pd":  46, "Ag":  47, "Cd":  48, "In":  49, "Sn":  50,
                      "Sb":  51, "Te":  52, "I":  53, "Xe":  54, "Cs":  55, "Ba":  56, "La":  57, "Ce":  58, "Pr":  59, "Nd":  60,
                      "Pm":  61, "Sm":  62, "Eu":  63, "Gd":  64, "Tb":  65, "Dy":  66, "Ho":  67, "Er":  68, "Tm":  69, "Yb":  70,
                      "Lu":  71, "Hf":  72, "Ta":  73, "W":  74, "Re":  75, "Os":  76, "Ir":  77, "Pt":  78, "Au":  79, "Hg":  80,
                      "Tl":  81, "Pb":  82, "Bi":  83, "Po":  84, "At":  85, "Rn":  86, "Fe":  87, "Ra":  88, "Ac":  89, "Th":  90,
                      "Pa":  91, "U":  92, "Np":  93, "Pu":  94, "Am":  95, "Cm":  96, "Bk":  97, "Cf":  98, "Es":  99, "Fm": 100,
                      "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
                      "Rg": 111, "Cn": 112, "Uut": 113, "Fl": 114, "Uup": 115, "Lv": 116, "Uus": 117, "Uuo": 118}
    if element != None and atomnum == None:
        return Periodic_Table[element]
    if element == None and atomnum != None:
        return list(Periodic_Table.keys())[list(Periodic_Table.values()).index(atomnum)]
    if element == None and atomnum == None:
        return "Input Error: No Input"
    if element != None and atomnum != None:
        return "Input Error: Element2AtomNum() takes exactly one argument"


def _GetAngleFromVector(u, v):
    AngleRad = np.arccos(
        np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)))
    AngleDeg = AngleRad*180/np.pi
    return AngleRad, AngleDeg


def _GetNormalVectorFromVector(u, v):
    s0 = u[1]*v[2]-u[2]*v[1]
    s1 = u[2]*v[0]-u[0]*v[2]
    s2 = u[0]*v[1]-u[1]*v[0]
    s = np.array([s0, s1, s2], dtype=float)
    s = s/np.linalg.norm(s)
    return s


def _GetAngleOfVector2PlaneFromVector(keySelfAtomCoord, samePlaneAtomCoord1, samePlaneAtomCoord2, outOfPlaneAtomCoord):
    vectorOfPlane_1 = samePlaneAtomCoord1 - keySelfAtomCoord
    vectorOfPlane_2 = samePlaneAtomCoord2 - keySelfAtomCoord

    u = _GetNormalVectorFromVector(vectorOfPlane_1, vectorOfPlane_2)
    v = outOfPlaneAtomCoord - keySelfAtomCoord

    AngleRad = np.arcsin(np.abs(np.dot(u, v)) /
                         (np.linalg.norm(u)*np.linalg.norm(v)))
    AngleDeg = AngleRad*180/np.pi
    return AngleRad, AngleDeg


def _GetAngle(atomCoord1, atomCoord2, atomCoord3):
    vector21 = atomCoord1 - atomCoord2
    vector23 = atomCoord3 - atomCoord2
    Angle = _GetAngleFromVector(vector21, vector23)
    return Angle


def _GetDihedralAngle(atomCoord1, atomCoord2, atomCoord3, atomCoord4):
    vector21 = atomCoord1 - atomCoord2
    vector23 = atomCoord3 - atomCoord2
    vector34 = atomCoord4 - atomCoord3

    nVector123 = _GetNormalVectorFromVector(vector21, vector23)
    nVector234 = _GetNormalVectorFromVector(vector23, vector34)
    DihedralAngle = _GetAngleFromVector(nVector123, nVector234)
    return DihedralAngle


def checkPostKeyAtoms(rdkitmol):
    ri = rdkitmol.GetRingInfo()

    def keyAtomsCapture(atomIdx, neighborAtoms):
        nNeighborAtoms = len(neighborAtoms)
        keyAtoms = {'keySelf': None, 'sameRingAtoms': [],
                    'keyR': None, 'keyH': None, 'keyRingIdx':None}
        for ring in ri.AtomRings():
            if keyAtoms['sameRingAtoms'] == [] and nNeighborAtoms == 4:
                for idx in range(nNeighborAtoms):
                    if set((atomIdx, neighborAtoms[idx].GetIdx())).issubset(ring):
                        keyAtoms['keyRingIdx'] = ring
                        keyAtoms['sameRingAtoms'].append(
                            (neighborAtoms[idx].GetIdx(), neighborAtoms[idx].GetSymbol()))
                    elif neighborAtoms[idx].GetSymbol() == 'H':
                        keyAtoms['keyH'] = (
                            neighborAtoms[idx].GetIdx(), 'H')
                    else:
                        keyAtoms['keyR'] = (
                            neighborAtoms[idx].GetIdx(), neighborAtoms[idx].GetSymbol())
        return keyAtoms

    for i in range(len(rdkitmol.GetAtoms())):
        atom = rdkitmol.GetAtomWithIdx(i)
        neighborAtoms = atom.GetNeighbors()
        if len(neighborAtoms) == 4 and ri.NumAtomRings(i) == 1:
            keyAtoms = keyAtomsCapture(i, neighborAtoms)
            if len(keyAtoms['sameRingAtoms']) == 2:
                keyAtoms['keySelf'] = (i, atom.GetSymbol())
                return keyAtoms

            
class GaussianOutputFileTypeError(SyntaxError):
    pass


class Gaussian_Output:
    def __init__(self, root='', path='', RC_State=None):
        np.set_printoptions(suppress=True)
        self.RC_State = RC_State
        self._skipOutLine = None

        self._isStable = None
        self._StableIndex = None

        self.is_NormalEnd = False
        self.error = None

        self._mainOutIndexDict = {'Start': [], 'End': []}
        self.taskTypeList = []
        self._MainOutRecording = False
        self._mainOutput_str = None
        self.MainOutput = None

        #self.MainOut = self.MainOut()
        self.__NBO = False
        self.__link_103 = False
        self.__link_601 = False

        self.root = root
        self.path = path
        if os.path.isfile(self.path) == True:
            self.path = path
            self.dir = os.path.split(self.path)[0]
            if self.dir != '':
                os.chdir(r'%s' % self.dir)
            self.fn = os.path.basename(self.path)
            self._readfile()
            os.chdir(r'%s' % self.root)
        else:
            raise GaussianOutputFileTypeError(
                'Please pass in the correct gaussian output file: \n%s' % self.path)

    def getMol(self):
        def _OBmol(self):
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("xyz", "mol2")
            tempdir = os.environ['HOME'] + '/OBTEMP'
            if not os.path.isdir(tempdir):
                os.makedirs(tempdir)
            tempFileName = r'%s/%s.mol2' % (tempdir, self.fn[:-4])
            if not os.path.isfile(tempFileName):
                with open(tempFileName, mode='w+') as temp:
                    temp.write('%d\n' % self._nAtoms)
                    temp.write('%s\n' % self.fn)
                    for i in range(self._nAtoms):
                        xx, yy, zz = self.AtomsCoordinates[i]
                        temp.write(' %-5s%15s%15s%15s\n' %
                                   (self.Atoms[i], xx, yy, zz))
                mol = openbabel.OBMol()
                obConversion.ReadFile(mol, tempFileName)
                obConversion.WriteFile(mol, tempFileName)
                obConversion.CloseOutFile()
            return tempFileName

        def _GO2RdkitMol(self):
            rdkitmol = Chem.rdchem.EditableMol(Chem.rdchem.Mol())
            radicalElectrons = 0
            for Idx, atom in enumerate(self.Atoms):
                rdAtom = Chem.rdchem.Atom(atom)
                rdAtom.SetNumRadicalElectrons(radicalElectrons)
                rdkitmol.AddAtom(rdAtom)

            rdBonds = Chem.rdchem.BondType
            orders = {'S': rdBonds.SINGLE, 'D': rdBonds.DOUBLE, 'T': rdBonds.TRIPLE,
                      'Ar': rdBonds.AROMATIC, 'I': rdBonds.IONIC}
            ArAtomType = ['C', 'N', 'O', 'S']
            for Idx1, atom1 in enumerate(self.Atoms):
                for Idx2, atom2 in enumerate(self.Atoms):
                    if Idx1 < Idx2:
                        bondIdx = self.BondIndex[Idx1][Idx2]
                        if bondIdx > 0.3 and bondIdx <= 0.5:
                            if atom1 == 'C' and atom2 == 'C':
                                order = orders['I']
                            else:
                                order = orders['S']
                        elif bondIdx > 0.5 and bondIdx <= 1.25:
                            order = orders['S']
                        elif bondIdx > 1.25 and bondIdx <= 1.75:
                            order = orders['Ar']
                        elif bondIdx > 1.75 and bondIdx <= 2.25:
                            order = orders['D']
                        elif bondIdx > 2.25 and bondIdx <= 3.1:
                            order = orders['T']
                        if bondIdx > 0.3 and bondIdx <= 3.1:
                            rdkitmol.AddBond(Idx1, Idx2, order)
                    else:
                        continue
            rdkitmol = rdkitmol.GetMol()
            if not rdkitmol.GetNumConformers():
                rdConfMethod = Chem.rdchem.Conformer
                conf = rdConfMethod()
                rdConfMethod.Set3D(conf,True)
                for Idx in range(self._nAtoms):
                    atomCoord3D = self.AtomsCoordinates[Idx]
                    rdConfMethod.SetAtomPosition(conf,Idx,tuple(atomCoord3D))
                rdConfMethod.GetPositions(conf)
                rdkitmol.AddConformer(conf)
                
            Chem.GetSymmSSSR(rdkitmol)
            try:
                rdkitmol.UpdatePropertyCache()
            except ValueError as err:
                print('''UpdatePropertyCache: ***%s'''%err)
                self.error = 'UpdatePropertyCache: %s'%err
            return rdkitmol

        def _GetRdkitMol(self):
            try:
                tempFileName = _OBmol(self)
                with open(tempFileName, 'r') as tempMol2:
                    tempMol2Block = tempMol2.read()
                rdkitmol = Chem.MolFromMol2Block(tempMol2Block, removeHs=False)
                if rdkitmol:
                    pass
                elif hasattr(self, 'BondIndex'):
                    rdkitmol = _GO2RdkitMol(self)
                else:
                    print('***RdkitFailure***')
                    self.error = '***RdkitFailure***\n'
            finally:
                try:
                    # print(tempFileName)
                    os.remove(tempFileName)
                finally:
                    pass
            return rdkitmol

        def _RdkitInfo(self):
            if self.rdkitmol:
                self.SMILES_H = Chem.MolToSmiles(self.rdkitmol)
                self.SMARTS_H = Chem.MolToSmarts(self.rdkitmol)
                try:
                    mol2 = Chem.RemoveHs(self.rdkitmol)
                    self.SMILES = Chem.MolToSmiles(mol2)
                    self.SMARTS = Chem.MolToSmarts(mol2)
                except ValueError as err:
                    print('''***SMILES: %s'''%err)
                    self.error = 'SMILES: ' + str(err).strip('\n')
                if self.RC_State == 'POST' :
                    self._PostKeyAtoms = checkPostKeyAtoms(self.rdkitmol)
                self._RDKit_Norm = True
            else:
                self._RDKit_Norm = False
            return self
        
        self.rdkitmol = _GetRdkitMol(self)
        _RdkitInfo(self)
        
        if self.RC_State == 'POST' and not self._PostKeyAtoms:
            if hasattr(self, 'BondIndex'):
                self.rdkitmol = _GO2RdkitMol(self)
                _RdkitInfo(self)            
        return self

    def getPostKeyBAD(self):
        def GetKeyCoord():
            keyAtoms = self._PostKeyAtoms
            self._keySelfIdx = keyAtoms['keySelf'][0]
            self._sameRingAtomIdx_1 = keyAtoms['sameRingAtoms'][0][0]
            self._sameRingAtomIdx_2 = keyAtoms['sameRingAtoms'][1][0]
            self._keyRIdx = keyAtoms['keyR'][0]
            self._keyHIdx = keyAtoms['keyH'][0]

            self._keySelf_Coord = np.array(
                self.AtomsCoordinates[self._keySelfIdx], dtype=float)
            self._sameRingAtom_Coord_1 = np.array(
                self.AtomsCoordinates[self._sameRingAtomIdx_1], dtype=float)
            self._sameRingAtom_Coord_2 = np.array(
                self.AtomsCoordinates[self._sameRingAtomIdx_2], dtype=float)
            self._keyR_Coord = np.array(
                self.AtomsCoordinates[self._keyRIdx], dtype=float)
            self._keyH_Coord = np.array(
                self.AtomsCoordinates[self._keyHIdx], dtype=float)

        try:
            GetKeyCoord()
            self.Post_Bond_R = np.linalg.norm(self._keyR_Coord-self._keySelf_Coord)
            self.Post_Bond_H = np.linalg.norm(self._keyH_Coord-self._keySelf_Coord)
            self.Post_Angle_RH = _GetAngle(
                self._keyR_Coord, self._keySelf_Coord, self._keyH_Coord)
            self.Post_Angle_RtoPlane = _GetAngleOfVector2PlaneFromVector(
                self._keySelf_Coord, self._sameRingAtom_Coord_1, self._sameRingAtom_Coord_2, self._keyR_Coord)
            self.Post_Angle_HtoPlane = _GetAngleOfVector2PlaneFromVector(
                self._keySelf_Coord, self._sameRingAtom_Coord_1, self._sameRingAtom_Coord_2, self._keyH_Coord)
        except TypeError as err:
            self._RDKit_Norm = False
            print('''***StructureFailure: %s'''%err)
            self.error = 'StructureFailure: ' + str(err).strip('\n')
        return self

    def rename(self, extraname):
        l_e = len(extraname)
        if self.fn[-l_e-5:] != '_%s.log' % extraname:
            new_fn = self.fn[:-4] + '_%s.log' % extraname  # 新的文件名
            os.rename(self.fn, new_fn)  # 重命名
        return self

    def _is_stable(self, line_index, line, words):
        '''
        line_index: the index of textlines
        '''
        if len(self.taskTypeList) == 1:
            if 'Stability' in self.taskTypeList[0]:
                if len(words) >= 5:
                    if line == ' The wavefunction is already stable.':
                        #print('Stable = True')
                        self._isStable = True
                        self._StableIndex = line_index
                else:
                    self._isStable = False
            else:
                self._isStable = None
        return self

    def _normal_end(self):
        '''
        Search the last line of the file to check if it contains "Normal termination"
        '''
        line = self.textlines[-1]
        words = line.split()
        if len(words) > 8:
            if words[0] == 'Normal' and words[1] == 'termination':
                self.is_NormalEnd = True
        return self

    def _main_out_index(self, line_index, line, words):
        '''
        line_index: the index of textlines
        '''
        if self._MainOutRecording == True:
            self._mainOutput_str = line[1:].strip(
                '\n') + str(self._mainOutput_str)
        if self._MainOutRecording == False and self._mainOutput_str != None:
            self.MainOutput = self._mainOutput_str.split('\\')

        if len(line) > 0 and '\\' in line:
            if line[:9] == r' 1\1\GINC':
                self._mainOutIndexDict['Start'].append(line_index)
                subWords = line.split('\\')
                self.taskTypeList.append(subWords[3])
                self._MainOutRecording = False
            elif line[-1] == '@':
                self._mainOutIndexDict['End'].append(line_index)
                if len(self._mainOutIndexDict['End']) == 1:
                    self._MainOutRecording = True
        if line == ' @':
            self._mainOutIndexDict['End'].append(line_index)
            if len(self._mainOutIndexDict['End']) == 1:
                self._MainOutRecording = True
        return self

    # get main information of molecular
    # @classmethod
    def _MainOut(self):
        nBlank = 0
        blankIndex = []

        if self.MainOutput != None:
            self.Atoms = np.empty(0)
            self.AtomsNum = np.empty(0, dtype=int)
            self.AtomsCoordinates = np.empty([0, 3])

            self.FunctionalMethod = self.MainOutput[4]
            self.BasisSet = self.MainOutput[5]
            self.MolecularFormula = self.MainOutput[6]
            self.LastLink0 = self.MainOutput[11]
            self.Title = self.MainOutput[13]
            self.Charge, self.Spin = map(int, self.MainOutput[15].split(','))
            self.Version, self.EleState, self.S2, self.S2A = None, None, None, None
            self.E_HF_short, self.ZeroPoint, self.Thermal, self.Dipole, self.PG = None, None, None, None, None

            for i in range(len(self.MainOutput)):
                if self.MainOutput[i] == '':
                    nBlank += 1
                    blankIndex.append(i)
                if nBlank == 3 and i >= blankIndex[2]+2:
                    self.Atoms = np.append(
                        self.Atoms, [self.MainOutput[i].split(',')[0]])
                    self.AtomsNum = np.append(
                        self.AtomsNum, [Element2AtomNum(element=self.Atoms[-1], atomnum=None)])
                    self.AtomsCoordinates = np.append(self.AtomsCoordinates, [
                        list(map(float, self.MainOutput[i].split(',')[-3:]))], axis=0)
                if nBlank == 4:
                    subWords = self.MainOutput[i].split('=')
                    if subWords[0] == 'Version':
                        self.Version = subWords
                    if subWords[0] == 'State':
                        self.EleState = subWords
                    if subWords[0] == 'HF':
                        self.E_HF_short = subWords
                    if subWords[0] == 'S2':
                        self.S2 = subWords
                    if subWords[0] == 'S2A':
                        self.S2A = subWords
                    if subWords[0] == 'ZeroPoint':
                        self.ZeroPoint = subWords
                    if subWords[0] == 'Thermal':
                        self.Thermal = subWords
                    if subWords[0] == 'Dipole':
                        self.Dipole = subWords
                    if subWords[0] == 'PG':
                        self.PG = subWords
                    if 'CCSD' in self.FunctionalMethod and subWords[0] == 'CCSD':
                        self.CCSD = subWords
                    else:
                        self.CCSD = None
                    if 'CCSD(T)' in self.FunctionalMethod and subWords[0] == 'CCSD(T)':
                        self.CCSD_T = subWords
                    else:
                        self.CCSD_T = None
        return self

    def _link_202(self, line_index, line, words):
        if self.__link_202 == True and len(self.optStepEndLineIndex) == 1 and line_index < self.optStepEndLineIndex[-1]:
            if line[:15] == ' Stoichiometry ':
                DM_record = True
                block_index = (self._nAtoms-1)//5
                column_index = (self._nAtoms-1) % 5+1
                if not hasattr(self, 'DistanceMatrix'):
                    self.DistanceMatrix = np.zeros(
                        (self._nAtoms, self._nAtoms))

                while DM_record:
                    line_index -= 1
                    line = self.textlines[line_index].strip('\n')
                    words = line.split()
                    if len(words) == 3 and words[0] == 'Distance' and words[1] == 'matrix':
                        DM_record = False
                        self._skipOutLine = line_index
                        break
                    elif line[:10] == ' ---------':
                        if self._nAtoms == 2:
                            atom1 = np.array(
                                self.AtomsCoordinates[0], dtype=float)
                            atom2 = np.array(
                                self.AtomsCoordinates[1], dtype=float)
                            distance = np.linalg.norm(atom1-atom2)
                            self.DistanceMatrix[1, 0] = distance
                            DM_record = False
                            self._skipOutLine = line_index
                            break
                    elif block_index >= 0:
                        if line[:8] == '        ' and column_index == 0:
                            block_index -= 1
                            column_index = 5
                            continue
                        else:
                            row_index = int(words[0])
                            if row_index == block_index*5 + column_index:
                                self.DistanceMatrix[row_index-1, block_index *
                                                    5:block_index*5+column_index] = list(map(float, words[2:]))
                                column_index -= 1
                            elif row_index > block_index*5+column_index:
                                self.DistanceMatrix[row_index-1, block_index *
                                                    5:block_index*5+5] = list(map(float, words[2:]))
                    else:
                        self._skipOutLine = line_index
                        break
                self.DistanceMatrix = np.around(
                    self.DistanceMatrix, decimals=6)
        return self

    def _link_601(self, line_index, line, words):
        if self.__link_601 == True:
            if line[:26] == ' Electronic spatial extent':
                if not hasattr(self, 'HeavyAtom'):
                    self.HeavyAtom = {}
                heavy_atom_record = True
                while heavy_atom_record:
                    line_index -= 1
                    line = self.textlines[line_index].strip('\n')
                    words = line.split()
                    if line[:9] == '         ':
                        heavy_atom_record = False
                        self._skipOutLine = line_index
                        break
                    else:
                        self.HeavyAtom.update({int(words[0]): words[1]})
            if line[:36] == ' Total kinetic energy from orbitals=':
                if not hasattr(self, 'HOMO'):
                    self.E_HOMO = None
                    self.E_LUMO = None
                E_MO_record = True
                NMO = 0
                while E_MO_record:
                    line_index -= 1
                    line = self.textlines[line_index].strip('\n')
                    words = line.split()
                    if line[:46] == ' Orbital energies and kinetic energies (alpha)':
                        E_MO_record = False
                        self._skipOutLine = line_index
                        break
                    elif line[:45] == ' Orbital energies and kinetic energies (beta)':
                        continue
                    elif line[:14] == '              ':
                        continue
                    elif len(words) == 4 and words[0].isdigit() == True:
                        if _isfloat(words[2]) == True and _isfloat(words[3]) == True:
                            if words[1][-1] == 'V':
                                self.E_LUMO = float(words[2])
                                NMO = int(words[0])
                            if words[0] == str(NMO - 1) and words[1][-1] == 'O':
                                self.E_HOMO = float(words[2])
                        else:
                            continue
        return self

    def _NBO(self, line_index, line, words):
        if self.__NBO:
            if line[:35] == ' Wiberg bond index, Totals by atom:':
                if not hasattr(self, 'BondIndex'):
                    self.BondIndex = np.zeros((self._nAtoms, self._nAtoms))
                bond_order_record = True
                block_index = (self._nAtoms-1)//9
                column_index = (self._nAtoms-1) % 9+1
                while bond_order_record:
                    line_index -= 1
                    line = self.textlines[line_index].strip('\n')
                    words = line.split()
                    if line == '' or line[:10] == '     ---- ':
                        continue
                    elif line[:10] == '     Atom ':
                        block_index -= 1
                        column_index = 9
                        continue
                    elif line[:43] == ' Wiberg bond index matrix in the NAO basis:':
                        bond_order_record = False
                        self._skipOutLine = line_index
                        break
                    elif block_index >= 0:
                        row_index = int(words[0].strip('.'))
                        self.BondIndex[row_index-1, block_index*9:block_index *
                                       9+column_index] = list(map(float, words[2:]))
                self.BondIndex = np.around(self.BondIndex, decimals=4)
            if line[:12] == '   * Total *':
                if not hasattr(self, 'NPACharge'):
                    self.NPACharge = np.zeros(self._nAtoms)
                npa_charge_record = True
                while npa_charge_record:
                    line_index -= 1
                    line = self.textlines[line_index].strip('\n')
                    words = line.split()
                    if line[:40] == ' Summary of Natural Population Analysis:':
                        npa_charge_record = False
                        self._skipOutLine = line_index
                        break
                    elif line[:10] == ' =========':
                        continue
                    elif line[:10] == ' ---------':
                        continue
                    elif line[:12] == '    Atom  No':
                        continue
                    elif line[:10] == '          ':
                        continue
                    elif len(words) == 7 and words[0].isalpha() == True and words[1].isdigit() == True:
                        AtomIdx = int(words[1])-1
                        self.NPACharge[AtomIdx] = float(words[2])
        return self

    def _link_103(self, line_index, line, words):
        if self.__link_103:
            if line[:14] == ' Trust Radius=':
                initial_para_record = True
                if not hasattr(self, 'InitialPara'):
                    self.InitialPara = {'R': {}, 'A': {}, 'D': {}, 'L': {}}
                while initial_para_record:
                    line_index -= 1
                    line = self.textlines[line_index].strip('\n')
                    words = line.split()
                    if line[:10] == ' ---------':
                        continue
                    elif line[:19] == ' ! Name  Definition':
                        initial_para_record = False
                        self._skipOutLine = line_index
                        break
                    elif len(words) == 8 and words[2][0] == 'R':
                        self.InitialPara['R'].update(
                            {eval(words[2][1:]): float(words[3])})
                    elif len(words) == 8 and words[2][0] == 'A':
                        self.InitialPara['A'].update(
                            {eval(words[2][1:]): float(words[3])})
                    elif len(words) == 8 and words[2][0] == 'D':
                        self.InitialPara['D'].update(
                            {eval(words[2][1:]): float(words[3])})
                    elif len(words) == 8 and words[2][0] == 'L':
                        self.InitialPara['L'].update(
                            {eval(words[2][1:]): float(words[3])})
        return self

    def _LinkByLink(self, line_index, line, words):
        if line[:9] == ' GradGrad':
            self.__link_103 = not self.__link_103

        if self._StableIndex == None:
            if line[:31] == ' NATURAL BOND ORBITAL ANALYSIS:':
                self.__NBO = True
            elif line[:52] == ' *******         Alpha spin orbitals         *******':
                self.__NBO = True
            elif line[:26] == ' Analyzing the SCF density':
                self.__NBO = False

            if len(words) == 8 and words[:2] == ['Anisotropic', 'Spin']:
                self.__link_601 = True
            if line[:16] == ' Leave Link  601':
                self.__link_601 = True
            if line[:20] == ' Orbital symmetries:':
                self.__link_601 = False

        if not hasattr(self, 'optStepEndLineIndex'):
            self.optStepEndLineIndex = []
            self.optStepStartLineIndex = []
            self.__link_202 = False
        if line[:21] == ' Rotational constants' and self.textlines[line_index-1][:5] == ' ----':
            self.optStepEndLineIndex.append(line_index)
            self.__link_202 = True
        if line.split() == ['Input', 'orientation:']:
            self.optStepStartLineIndex.append(line_index)
            self.__link_202 = False

        return self

    def _readfile(self):
        '''
        Reverse search from file
        '''
        with open(self.fn, 'r') as self._f:
            self.textlines = self._f.readlines()
        self._normal_end()
        if self.is_NormalEnd:
            n_textlines = len(self.textlines)
            t0 = time.clock()
            for n in range(n_textlines):
                line = self.textlines[-n].strip('\n')
                words = line.split()

                if self._skipOutLine != None:
                    if -n >= self._skipOutLine:
                        continue
                    else:
                        self._skipOutLine = None

                self._is_stable(line_index=-n, line=line, words=words)
                self._main_out_index(line_index=-n, line=line, words=words)
                if self.MainOutput != None and not hasattr(self, 'Atoms'):
                    self._MainOut()
                    self._nAtoms = len(self.Atoms)
                if len(self._mainOutIndexDict['End']) == 1:
                    self._LinkByLink(line_index=-n, line=line, words=words)
                    if self.__NBO:
                        self._NBO(line_index=-n, line=line, words=words)
                    elif self.__link_601 == True:
                        self._link_601(line_index=-n, line=line, words=words)
                    elif self.__link_202 == True and len(self.optStepEndLineIndex) == 1 and -n < self.optStepEndLineIndex[-1]:
                        self._link_202(line_index=-n, line=line, words=words)
                    elif self.__link_103:
                        self._link_103(line_index=-n, line=line, words=words)
            print('   ____________________\n   ::: Reading %s cost %ss' %
                  (self.fn, time.clock() - t0))
            if hasattr(self, 'DistanceMatrix'):
                self.DistanceMatrix = self.DistanceMatrix + self.DistanceMatrix.T
            self.getMol()
            if self.RC_State == 'POST':
                if not self._RDKit_Norm:
                    pass
                elif not self._PostKeyAtoms:
                    self._RDKit_Norm = False
                elif self._PostKeyAtoms['keySelf']:
                    self.getPostKeyBAD()
            elif self.RC_State == 'PRE':
                words = self.Title.split()
                if len(words) > 1:
                    self.Atom_labels_list = [int(x) for x in words[1:]]
            return self
        else:
            self.rename('False')
            print('Gaussian output file does not end normally: \n%s' % self.fn)

    def __str__(self):
        if self.is_NormalEnd:
            return 'Gaussian Output object (filename: %s, Type: %s)' % (self.fn, self.taskTypeList[-1])
        else:
            return 'Abnormal Gaussian Output object (filename: %s)' % (self.fn)
        
        