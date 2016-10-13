import pickle

import numpy as np

class STM:
    def __init__(self, atoms, symmetries=None):
        """Scanning tunneling microscope.

        atoms: Atoms object or filename
            Atoms to scan or name of file to read LDOS from.
        symmetries: list of int
            List of integers 0, 1, and/or 2 indicating which surface
            symmetries have been used to reduce the number of k-points
            for the DFT calculation.  The three integers correspond to
            the following three symmetry operations::

                 [-1  0]   [ 1  0]   [ 0  1]
                 [ 0  1]   [ 0 -1]   [ 1  0]
        """

        if isinstance(atoms, str):
           with open(atoms, 'rb') as f:
               self.ldos, self.bias, self.cell = pickle.load(f)
           self.atoms = None
        else:
            self.atoms = atoms
            self.cell = atoms.cell
            self.bias = None
            self.ldos = None
            assert not self.cell[2, :2].any() and not self.cell[:2, 2].any()

        self.symmetries = symmetries or []

    def calculate_ldos(self, bias):
        """Calculate local density of states for given bias."""
        if self.ldos is not None and bias == self.bias:
            return

        if bias < 0:
            emin = bias
            emax = 0.0
        else:
            emin = 0
            emax = bias

        calc = self.atoms.calc

        nbands = calc.get_number_of_bands()
        weights = calc.get_k_point_weights()
        nkpts = len(weights)
        nspins = calc.get_number_of_spins()
        eigs = np.array([[calc.get_eigenvalues(k, s)
                          for k in range(nkpts)]
                         for s in range(nspins)])
        eigs -= calc.get_fermi_level()
        ldos = 0.0
        for s in range(nspins):
            for k in range(nkpts):
                for n in range(nbands):
                    e = eigs[s, k, n]
                    if emin < e < emax:
                        psi = calc.get_pseudo_wave_function(n, k, s)
                        ldos += weights[k] * (psi * np.conj(psi)).real

        if 0 in self.symmetries:
            # (x,y) -> (-x,y)
            ldos[1:] += ldos[:0:-1].copy()
            ldos[1:] *= 0.5

        if 1 in self.symmetries:
            # (x,y) -> (x,-y)
            ldos[:, 1:] += ldos[:, :0:-1].copy()
            ldos[:, 1:] *= 0.5

        if 2 in self.symmetries:
            # (x,y) -> (y,x)
            ldos += ldos.transpose((1, 0, 2)).copy()
            ldos *= 0.5

        self.ldos = ldos
        self.bias = bias

    def write(self, filename='stm.pckl'):
        """Write local density of states to pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump((self.ldos, self.bias, self.cell), f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def get_averaged_current(self, bias, z):
        """Calculate avarage current at height z.

        Use this to get an idea of what current to use when scanning."""

        if self.ldos is None:
            self.calculate_ldos(bias)
        nz = self.ldos.shape[2]

        # Find grid point:
        n = z / self.cell[2, 2] * nz
        dn = n - np.floor(n)
        n = int(n) % nz

        # Average and do linear interpolation:
        return ((1 - dn) * self.ldos[:, :, n].mean() +
                dn * self.ldos[:, :, (n + 1) % nz].mean())

    def scan(self, bias, current, z0=None, repeat=(1, 1)):
        """Constant current 2-d scan.

        Returns three 2-d arrays (x, y, z) containing x-coordinates,
        y-coordinates and heights.  These three arrays can be passed to
        matplotlibs contourf() function like this:

        >>> import matplotlib.pyplot as plt
        >>> plt.gca(aspect='equal')
        >>> plt.contourf(x, y, z)
        >>> plt.show()

        """
        if self.ldos is None:
            self.calculate_ldos(bias)

        L = self.cell[2, 2]
        nz = self.ldos.shape[2]
        h = L / nz

        ldos = self.ldos.reshape((-1, nz))

        heights = np.empty(ldos.shape[0])
        for i, a in enumerate(ldos):
            heights[i] = find_height(a, current, h, z0)

        s0 = heights.shape = self.ldos.shape[:2]
        heights = np.tile(heights, repeat)
        s = heights.shape

        ij = np.indices(s, dtype=float).reshape((2, -1)).T
        x, y = np.dot(ij / s0, self.cell[:2, :2]).T.reshape((2,) + s)

        return x, y, heights

    def linescan(self, bias, current, p1, p2, npoints=50, z0=None):
        """Constant current line scan.

        Example::

            stm = STM(...)
            z = ...  # tip position
            c = stm.get_averaged_current(-1.0, z)
            stm.linescan(-1.0, c, (1.2, 0.0), (1.2, 3.0))
        """

        heights = self.scan(bias, current, z0)[2]

        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        d = p2 - p1
        s = np.dot(d, d)**0.5

        cell = self.cell[:2, :2]
        shape = np.array(heights.shape, float)
        M = np.linalg.inv(cell)
        line = np.empty(npoints)
        for i in range(npoints):
            p = p1 + i * d / (npoints - 1)
            q = np.dot(p, M) * shape
            line[i] = interpolate(q, heights)
        return np.linspace(0, s, npoints), line

    def do_rolling_ball(self, c=1.0e-6 , c_tol=0.02, r=1.0, z0=None):
        """
            Routine does a 'rolling ball smoothing' of the LDOS, where
            r is the radius of the ball.
            following J. Mol. Biol. 79, pp. 351-364

            c ... isodensity value for surface
            c_tol ... relative tolerance on isovalue
            r ... radius of ball
            z0...vertical maximal cutoff (z direction)
            ncpus...number of CPUs that are used (shared memory)

            written by Reinhard J. Maurer (TUM) 2014

        """

        iso = c
        iso_tol = c_tol
        ldos = self.ldos
        ldos_new = np.zeros(self.ldos.shape)
        ldos_new = np.copy(ldos)
        a, b, c = self.atoms.cell
        na, nb, nc = ldos_new.shape
        da = a / na
        db = b / nb
        dc = c / nc
        da_len = np.linalg.norm(da)
        db_len = np.linalg.norm(db)
        dc_len = np.linalg.norm(dc)

        imax = iso + iso_tol*iso
        imin = iso - iso_tol*iso

        #number of neighbor voxels to include for given radius
        nna = max(1,int(np.round(r / da_len)))
        nnb = max(1,int(np.round(r / db_len)))
        nnc = max(1,int(np.round(r / dc_len)))

        print nna,nnb,nnc

        r2 = r*r
        da_len2 = da_len*da_len
        db_len2 = db_len*db_len
        dc_len2 = dc_len*dc_len

        if z0 is None:
            nc_lim = nc
        else:
            nc_lim = int(z0/dc_len)

######SERIAL VERSION############################

        try:
            import rollingball
            do_fortran = True
        except:
            print 'FORTRAN module not available..using very slow Python version'
            do_fortran = False

        if do_fortran:
            print 'Cycling nx (and ny and nz) voxels'
            print 'Using external FORTRAN module for rollingball'

            import rollingball
            roll = rollingball.rollingball

            roll.a = a
            roll.b = b
            roll.c = c
            roll.na = na
            roll.nb = nb
            roll.nc = nc
            roll.iso = iso
            roll.iso_tol = iso_tol
            roll.r = r
            if z0 is None:
                roll.z0 = 0.0
            else:
                roll.z0 = z0
            roll.ldos = ldos
            roll.ldos_new = ldos_new

            roll.calculate_rollingball()

            ldos_new = roll.ldos_new
        else:
            print 'Cycling nx (and ny and nz) voxels'
            for l in range(na):
                #print 'nx voxel no. ', l,'of', na
                for k in range(nb):
                    #for h in range(nc_lim_down,nc_lim):
                    for h in range(nc_lim):
                        if ldos[l,k,h]< imax and ldos[l,k,h]>imin:
                            value = ldos[l,k,h]
                            for ll in range(l-nna,l+nna):
                                for kk in range(k-nnb,k+nnb):
                                    for hh in range(h-nnc,h+nnc):
                                        if ldos[ll%na,kk%nb,hh%nc]<value:
                                            hhh = (hh-h)*(hh-h)*da_len2
                                            lll = (ll-l)*(ll-l)*db_len2
                                            kkk = (kk-k)*(kk-k)*dc_len2
                                            nsum = lll+kkk+hhh
                                            if nsum<r2:
                                                ldos_new[ll%na,kk%nb,hh%nc] += value

####PARALLEL VERSION###############################

            #nprocess = 0
            #pool=mp.Pool(ncpus)
            #results = []

            ##ldosbig
            #ldosbig = np.concatenate([ldos,ldos,ldos],axis=0)
            #ldosbig = np.concatenate([ldosbig,ldosbig,ldosbig],axis=1)
            #ldosbig = np.concatenate([ldosbig,ldosbig,ldosbig],axis=2)
            #ldosbig = ldosbig[na/2:5*na/2,nb/2:5*nb/2,nc/2:5*nc/2]
            #for l in range(na):
                #for k in range(nb):
                    #for h in range(nc_lim):
                        #if ldos[l,k,h]< imax and ldos[l,k,h]>imin:
                            #ldoss = ldosbig[(l-nna+na/2):(l+nna+na/2),\
                                    #(k-nnb+nb/2):(k+nnb+nb/2),\
                                    #(h-nnc+nc/2):(h+nnc+nc/2)]
                            #ind = (l, k, h, na, nb, nc, nna, nnb, nnc, \
                                    #imin, imax, da_len2, db_len2, dc_len2, r2)
                            #results.append(pool.apply_async(\
                                    #rolling_ball_in_parallel, \
                                    #args=(ind,ldoss,nprocess,)))
                            #nprocess +=1
            #pool.close()
            #pool.join()

            #for res in results:
                #l,k,h, ldos_tmp = res.get()
                #for ll in range(l-nna,l+nna):
                    #for kk in range(k-nnb,k+nnb):
                        #for hh in range(h-nnc,h+nnc):
                            #ldos_new[ll%na,kk%nb,hh%nc] += \
                                    #ldos_tmp[(ll-l),\
                                    #(kk-k),\
                                    #(hh-h)]

######################################################
#def rolling_ball_in_parallel(indices,ldos, nnp):
    #l , k, h, na, nb, nc, nna, nnb, nnc, imin, imax, \
            #da_len2, db_len2, dc_len2, r2 = indices
    #ldos_small = np.zeros([2*nna,2*nnb,2*nnc])
    #value = ldos[nna,nnb,nnc]
    #for ll in range(-nna,nna):
        #for kk in range(-nnb,nnb):
            #for hh in range(-nnc,nnc):
                #if ldos[ll+nna,kk+nnb,hh+nnc]<value:
                    #hhh = (hh)*(hh)*da_len2
                    #lll = (ll)*(ll)*db_len2
                    #kkk = (kk)*(kk)*dc_len2
                    #nsum = lll+kkk+hhh
                    #if nsum<r2:
                        #ldos_small[ll+nna,\
                                #kk+nnb,\
                                #hh+nnc] += value
    #print  l,k,h
    #return (l,k,h,ldos_small)
    ##return [l,k,h,ldos_new]

        return ldos_new

    def smoothing_filter(self, ldos=None, s = (1.00,1.00,0.00)):
        """
            Routine does Gaussian smoothing and
            returns new LDOS

            s ... smoothing value, number of appended zeros in multiples of shape

            written by Reinhard J. Maurer TUM (2014)

        """

        print 'Performing Gaussian filter smearing on LDOS topograph'

        s = s
        if ldos is None:
            ldos = self.ldos
        else:
            ldos=ldos

        try:
            from scipy.signal import medfilt
            from scipy.ndimage.filters import gaussian_filter
            from scipy.ndimage.filters import uniform_filter
        except:
            raise ImportError("Could not import scipy.fftpack.fftn")

        ldos_new = gaussian_filter(ldos, sigma=s, order=0, mode='wrap')

        return ldos_new

def interpolate(q, heights):
    qi = q.astype(int)
    f = q - qi
    g = 1 - f
    qi %= heights.shape
    n0, m0 = qi
    n1, m1 = (qi + 1) % heights.shape
    z = (g[0] * g[1] * heights[n0, m0] +
         f[0] * g[1] * heights[n1, m0] +
         g[0] * f[1] * heights[n0, m1] +
         f[0] * f[1] * heights[n1, m1])
    return z


def find_height(ldos, current, h, z0=None):
    if z0 is None:
        n = len(ldos) - 1
    else:
        n = int(z0 / h)
    while n >= 0:
        if ldos[n] > current:
            break
        n -= 1
    else:
        raise RuntimeError('Tip crash!')

    try:
        c2, c1= ldos[n:n + 2]
    except:
        raise RuntimeError('Tip too far away!')
    return (n + 1 - (current - c1) / (c2 - c1)) * h

