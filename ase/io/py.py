
def write_py(fileobj, images, name='images', write_header=True):
    """Write to ASE-compatible python script."""
    if isinstance(fileobj, str):
        isfilename = True
        fileobj = open(fileobj, 'w')
    else:
        isfilename = False
    
    if write_header:
        fileobj.write('from ase import Atoms\n')
        fileobj.write('import numpy as np\n\n')
    
    if hasattr(images, 'get_positions'):
        fileobj.write(name+" = Atoms(symbols='%s',\n"
                          "          pbc=np.%s,\n"
                          "          cell=np.array(\n      %s,\n"
                          "          positions=np.array(\n      %s)\n" % (
                              images.get_chemical_formula(mode='reduce'),
                              repr(images.pbc),
                              repr(images.cell)[6:],
                              repr(images.positions)[6:]))
    else:
        fileobj.write('images = [\n')

        for image in images:
            fileobj.write("    Atoms(symbols='%s',\n"
                          "          pbc=np.%s,\n"
                          "          cell=np.array(\n      %s,\n"
                          "          positions=np.array(\n      %s),\n" % (
                              image.get_chemical_formula(mode='reduce'),
                              repr(image.pbc),
                              repr(image.cell)[6:],
                              repr(image.positions)[6:]))
        
        fileobj.write(']')
    
    if isfilename: fileobj.close()

