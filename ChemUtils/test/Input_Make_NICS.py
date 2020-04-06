import os
from pybel import readstring, Outputfile
import rdkit
from rdkit import Chem
import numpy as np
from glob import glob


def gjfReader(fname):
    '''In the process of file format transformation using openbabel, 
        the coordinate accuracy is reduced, so the coordinates of the incoming file'''
    path = os.getcwd()
    with open(fname, 'r') as ifile:
        textlines = ifile.readlines()
    return textlines


def get_mol(gjf_fn, textlines):
    xyz_list = [0]
    n_Atoms = 0
    xyz_list.append(gjf_fn[:-4]+'\n')
    for line in textlines[7:]:
        xyz_list.append(line)
        if line == '\n':
            break
        n_Atoms += 1
    # return the first line and write the number of this molecule
    xyz_list[0] = '%-5d\n' % n_Atoms

    xyz_list.append("\n")
    xyz_str = ''.join(xyz_list)
    OB_mol = readstring('xyz', xyz_str)

    sdf_file = './%s.sdf' % OB_mol.title
    output_file_writer = Outputfile('sdf', sdf_file, overwrite=True)
    output_file_writer.write(OB_mol)
    output_file_writer.close()

    rdkitmol = Chem.SDMolSupplier(sdf_file, removeHs=False)
    #os.remove(sdf_file)
    return OB_mol, rdkitmol


def Get_Key_atom_num(gjf_fn, textlines, rdkitmol):
    try:
        key_atom_nums = [int(x)-1 for x in textlines[4].split()[1:]]
    except:
        keyAtomNum = int(gjf_fn.split('-')[1])-1

        mol = rdkitmol[0]
        rings = mol.GetRingInfo().AtomRings()
        key_atom_nums = []
        for ring in rings:
            if keyAtomNum in rings:
                atom_in_ring = ring
        for index in atom_in_ring:
            atom = mol.GetAtomWithIdx(index)
            symbol = atom.GetSymbol()
            neighbors = atom.GetNeighbors()
            neighbors_symbols = [neighbor.GetSymbol()
                                 for neighbor in neighbors]
            if symbol == 'C' and 'H' in neighbors_symbols:       # is Carbon and connect with hydrogen
                key_atom_nums.append(index)
    finally:
        return key_atom_nums


def gjfWriter(folder, new_gjf_fn, textlines, key_atom_nums, bq_line):
    Atom_labels_list = [x+1 for x in key_atom_nums]

    # 重新打开文件
    with open('%s/%s' % (folder, new_gjf_fn), 'w') as nf:
        countent = ["%nprocshared=8\n", "%mem=1GB\n",
                    "#p b3lyp/6-311+g(2d,p) nmr\n", "\n"]
        nf.writelines(countent)
        nf.write(new_gjf_fn[:-4]+' '+' '.join(map(str, Atom_labels_list)))
        nf.write("\n")
        nf.write("\n")
        for line in textlines[6:]:
            if line == '\n':
                nf.write(bq_line)
                break
            else:
                nf.write(line)

        nf.write("\n\n")
    return 0


def get_Normal_Vec(P0, P1, P2):
    v1 = P1 - P0
    v2 = P2 - P0
    n = np.array([v1[1]*v2[2]-v2[1]*v1[2],
                  v1[2]*v2[0]-v2[2]*v1[0],
                  v1[0]*v2[1]-v2[0]*v1[1]])
    n /= np.linalg.norm(n)
    return n


def get_NICS_file(i, fname):
    flag = 0
    
    gjf_fn = os.path.basename(fname)
    folder = './NICS'
    if not os.path.isdir(folder):
        os.makedirs(folder)

    tmp = gjf_fn.split('-')
    if glob('%s/%s*-NICS-*sp.com' % (folder, tmp[0])):
        print("%3d  %-30s has already converted!!!" % (i, gjf_fn))
        return
    tmp.insert(-1, 'r0')
    tmp.insert(-1, 'NICS')

    textlines = gjfReader(fname)
    OB_mol, rdkitmol = get_mol(gjf_fn, textlines)
    key_atom_nums = Get_Key_atom_num(gjf_fn, textlines, rdkitmol)
    mol = rdkitmol[0]
    rings = mol.GetRingInfo().AtomRings()
    for r_idx, ring in enumerate(rings):
        key_atom_nums_on_ring = [x for x in key_atom_nums if x in ring]
        if len(key_atom_nums_on_ring):
            tmp[-3] = 'r%s'%(r_idx+1)
            new_gjf_fn = '-'.join(tmp)
            
            ring_pos = np.array(
                [np.array(OB_mol.atoms[x].coords) for x in ring])
            a0, a1, a2 = ring_pos[0], ring_pos[2], ring_pos[-2]
            n = get_Normal_Vec(a0, a1, a2)
            bq0 = np.average(ring_pos, axis=0)
            bq1 = bq0 + n
            bq0_line = ' %-3s%15s%11.8f%3s%11.8f%3s%11.8f\n' % (
                'Bq', ' ', bq0[0], ' ', bq0[1], ' ', bq0[2])
            bq1_line = ' %-3s%15s%11.8f%3s%11.8f%3s%11.8f\n' % (
                'Bq', ' ', bq1[0], ' ', bq1[1], ' ', bq1[2])
            bq_line = bq0_line + bq1_line

            gjfWriter(folder, new_gjf_fn, textlines, key_atom_nums_on_ring, bq_line)
            print("%2d  %-30s is OK and has converted successfully!" % (i, gjf_fn))
            flag += 1
    if not flag:
        print("%2d  %-30s is wrong" % (i, gjf_fn))
    

n = 1
for eachfile in (glob('*sp.gjf')+ glob('*sp.com')):
    n+=1
    get_NICS_file(n, eachfile)

for x in glob('*sp.sdf'):
    os.remove(x)
    
print ('')
os.system("pause")
