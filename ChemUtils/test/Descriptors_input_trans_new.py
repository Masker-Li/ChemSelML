import os
import sys
import numpy as np
import pandas as pd

def Get_C_S_Coord(old_filedir,fname):
    nn = 0
    Charge_Spin_Coordinates = []
    Atom_labels_list = []    
    
    path=os.getcwd()
    fn = os.path.basename(fname)   #获取文件名
    with open( fname, 'r' ) as ifile:
        textlines = ifile.readlines()

    for line in textlines[:] :
        if line == "\n":
            nn +=1
        if nn<1 :
            pass
        elif nn<2 :
            words = line.split()
            if len(words) > 0:
                Atom_labels_list += [int(x) for x in words[1:]]
        elif nn<3 :
            Charge_Spin_Coordinates += [line]
        else:
            break    
    return fn,Atom_labels_list,Charge_Spin_Coordinates

    
def Rewrite(new_file_dir,add_dir,fn,suffix,NewMethods,Atom_labels_list,Charge_Spin_Coordinates,addinput):    
    fn_sep = fn.split('-')
    new_fn = '-'.join(fn_sep[:-1]+['1sp%s.gjf'%suffix])
    if os.path.isdir(r"%s/%s"%(new_file_dir,add_dir))==False:
        os.makedirs(r"%s/%s"%(new_file_dir,add_dir))
    if os.path.isfile(r"%s/%s/%s"%(new_file_dir,add_dir,new_fn))==True:
        print ("[!] %-30s has already converted!!!" %(new_fn))
        #print(new_file_dir,add_dir)
    else:
        #os.chdir(new_file_dir+r"/%s"%(add_dir))    #转换目录，进入sp文件夹
        with open( r"%s/%s/%s"%(new_file_dir,add_dir,new_fn), 'w' ) as nf:  # 打开文件
            countent = ["%nprocshared=8\n","%mem=1GB\n","%chk=",new_fn[:-4],".chk\n",NewMethods,"\n","\n"]
            nf.writelines(countent)
            nf.write(new_fn[:-4]+' '+' '.join(map(str,Atom_labels_list)))
            nf.write("\n")
            for line in Charge_Spin_Coordinates :
                nf.write(line)
                
            nf.write("\n")
            nf.writelines(addinput)
            nf.write("\n")
        print("  %-30s is OK and has converted successfully!" %(new_fn))
        #print(new_file_dir,add_dir)
            
def Desc_transform(path,new_file_dir):   
    os.chdir(path)
    
    from glob import glob
    n=0
    #NewMethods = "#p b3lyp/6-311+g(2d,p) pop=(full,nboread) scrf=(smd,solvent=water) stable=opt"
    NewMethods = "#p b3lyp/6-311+g(2d,p) pop=(full,nboread) stable=opt"
    
    #BasisSet1 = []
    #BasisSet2 = []
    #addinput = BasisSet1 + BasisSet2 + BasisSet2
    addinput = "$nbo bndidx $end \n"
    suffix_ls = ['','+','-']
    
    for eachfile in (glob('*.gjf') + glob('*.com')):
        n+=1
        fn,Atom_labels_list,Charge_Spin_Coordinates = Get_C_S_Coord(path,eachfile)
        Charge_Spin = Charge_Spin_Coordinates[1].split()
        for suffix in suffix_ls:
            if suffix == '+':
                Charge = int(Charge_Spin[0]) + 1
                Spin = int(Charge_Spin[1])%2 +1
                Charge_Spin_Coordinates[1] = str(Charge)+' '+str(Spin)+'\n'
            elif suffix == '-':
                Charge = int(Charge_Spin[0]) - 1
                Spin = int(Charge_Spin[1])%2 +1
                Charge_Spin_Coordinates[1] = str(Charge)+' '+str(Spin)+'\n'
            else:
                pass
            Rewrite(new_file_dir,'Desc_sp_gas',fn,suffix,NewMethods,Atom_labels_list,Charge_Spin_Coordinates,addinput)

        os.chdir(path)
    print ('\n')
    print ('There are %d files have transformed'%n)

#new_file_dir = r'E:\1-G09\M-L\radicals\JOIN\Standard_Coord\Solvent_water\pre\R'

##new_file_dir = input('new_file_dir:')
new_file_dir = ''
if new_file_dir == '':
    new_file_dir = os.getcwd()
#path = r'E:\1-G09\M-L\radicals\JOIN\Standard_Coord\Ar2Fixed\Radical\EB_sp'
path = os.getcwd()
Desc_transform(path,new_file_dir)

print ('')
os.system("pause")
