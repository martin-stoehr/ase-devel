import numpy as np
from ase.io import read
from ase.io.py import write_py


pref = 'x40'

f = open('x40_data.txt', 'r')
lines = f.readlines()
f.close()

energies = {}
for line in lines:
    sysname = line.split()[0][:-4]
    iae = float(line.split()[-1])
    energies[sysname] = iae

sysnames = energies.keys()

f = open(pref+'.py', 'w')
f.write('import numpy as np\n')
f.write('from ase.atoms import Atoms\n\n\n')

f.write(pref+'_names = [')
for iname, name in enumerate(sysnames[:-1]):
    f.write("'"+name+"', ")
    if (np.mod(iname, 4) == 3): f.write("\\\n    ")

f.write("'"+sysnames[-1]+"']\n\n")

f.write(pref+'_interaction_energy = {}\n')
for name, iae in energies.iteritems():
    f.write(pref+"_interaction_energy['"+name+"'] = "+str(iae)+'\n')

f.close()

f = open(pref+'.py', 'a')
f.write('\n\n'+pref+'_systems = {}\n')
for name in sysnames:
    atoms = read(name+'.xyz')
    write_py(f, atoms, name=pref+"_systems['"+name+"']", write_header=False)
    

f.write('\n\n\nclass '+pref+':\n')
f.write('    def get_names(self):\n        return '+pref+'_names\n        \n    \n')
f.write('    def get_interaction_energy(self, name):\n')
f.write('        return '+pref+'_interaction_energy[name]\n        \n    \n')
f.write('    def create_'+pref+'_system(self, name):\n')
f.write('        return '+pref+'_systems[name]\n        \n    \n')

f.close()

