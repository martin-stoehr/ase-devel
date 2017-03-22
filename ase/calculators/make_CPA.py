from copy import deepcopy as dcopy
from os import system,listdir,remove
from sys import argv


try:
    src_loc = str(argv[1])+'/'
except IndexError:
    print 'No source location specified. Using current location.'
    src_loc = './'

curr_files = listdir(src_loc)
assert ('CPA_recode.f90' in curr_files)

if 'cpa.pyf' in curr_files:
    remove(src_loc+'cpa.pyf')
if 'CPA_recode.so' in curr_files:
    remove(src_loc+'CPA_recode.so')

if (system('dpkg --list | grep ifort') == 256):
    f2pycomp = 'gfortran'
else:
    f2pycomp = 'intelem'

system('f2py '+src_loc+'CPA_recode.f90 -m CPA_recode -h '+src_loc+'cpa.pyf')

f = open(src_loc+'cpa.pyf', 'r')
text = f.readlines()
f.close()

tnew = dcopy(text)
for i, line in enumerate(text):
    if 'integer, optional,intent(in),check(len(filenames)>=n_files),depend(filenames) :: n_files=len(filenames)' in line:
        tnew[i] = '                integer intent(in) :: n_files\n'
    elif 'character(len=*) dimension(n_files),intent(in) :: filenames' in line:
        tnew[i] = '                character(len=*) dimension(n_files),intent(in),depend(n_files) :: filenames\n'
    elif 'integer, optional,intent(in),check(len(wk)>=n_k_tot),depend(wk) :: n_k_tot=len(wk)' in line:
        tnew[i] = '                integer intent(in) :: n_k_tot\n'
    elif 'integer, optional,intent(in),check(len(orb2atom)>=n_basis),depend(orb2atom) :: n_basis=len(orb2atom)' in line:
        tnew[i] = '                integer intent(in) :: n_basis\n'
    elif 'integer dimension(n_basis),intent(in) :: orb2atom' in line:
        tnew[i] = '                integer dimension(n_basis),intent(in),depend(n_basis) :: orb2atom\n'
    elif 'double precision dimension(n_k_tot),intent(in) :: wk' in line:
        tnew[i] = '                double precision dimension(n_k_tot),intent(in),depend(n_k_tot) :: wk\n'

f = open(src_loc+'cpa.pyf', 'w')
for line in tnew:
    f.write(line)
f.close()

system('f2py -c --fcompiler='+f2pycomp+' --f90flags="-O3" '+src_loc+'cpa.pyf '+src_loc+'CPA_recode.f90')

#--EOF--#
