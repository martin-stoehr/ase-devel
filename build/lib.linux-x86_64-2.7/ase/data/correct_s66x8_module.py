from sys import argv


s66x8_dir = str(argv[1])
f = open(s66x8_dir+'s66x8.py','r')
textold = f.read()
f.close()

textnew = textold.replace('!INSERTDIRECTORYHERE!', s66x8_dir)
f = open(s66x8_dir+'s66x8.py','w')
f.write(textnew)
f.close()

