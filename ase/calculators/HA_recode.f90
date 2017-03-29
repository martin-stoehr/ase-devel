module HA_recode
use splines_alt
use spherical_harmonics

implicit none

double precision,parameter   :: pi = 3.14159265

contains
    subroutine hirshfeld_main(nAtoms,nkpts,nOrbs,nThetas,nPhis,positions,coeff,wk,f,lRnl_max, &
                      &dr,nr,len_r,rmins,Rnl_id,occ_free,otypes,orb2at,at2orbs, hirsh_volrat)

        !!! return volume ratios as obtained by Hirshfeld analysis !!!
        !!! input in Bohr !!!
        
        !! default input data defined in python wrapper
        integer,intent(in)                                  :: nAtoms,nkpts,nOrbs,nThetas,nPhis,nr,lRnl_max
        !! (numeric) ID for Rnl data (system specific, required for parallelism)
        character(len=16),intent(in)                        :: Rnl_id
        double precision,intent(in)                         :: dr
        double precision,dimension(3,nAtoms),intent(in)     :: positions
        !! density matrix elements
        complex,dimension(nOrbs,nOrbs,nkpts),intent(in)     :: coeff
        !! occupations state k,a
        double precision,dimension(nOrbs,nkpts),intent(in)  :: f
        !! length of radial grids, minimal value of radial grid
        integer,dimension(nAtoms),intent(in)                :: len_r
        !! minimal value of radial grids
        double precision,dimension(nAtoms),intent(in)       :: rmins
        !! free occupations
        double precision,dimension(nOrbs),intent(in)        :: occ_free
        !! kpt-weighting factors
        double precision,dimension(nkpts),intent(in)        :: wk
        !! orbital(iOrb) belongs to atom j
        !! orbital types (e.g. otypes(iOrb)=2 -> 'py', see module spherical_harmonics for further details)
        integer,dimension(nOrbs),intent(in)                 :: orb2at,otypes
        !! orbitals with index [at2orbs(1,j) to at2orbs(2,j)] belong to atom j
        integer,dimension(2,nAtoms),intent(in)              :: at2orbs
        
        !! increments, number of radial grid points, and loop indices for integration
        double precision                                    :: Vincr,dTheta,dPhi,rc,Theta,Phi,sin_t
        double precision,parameter                          :: ln_rmax=dlog(100.d0)
        integer                                             :: ir,iT,iP,i
        double precision,dimension(nThetas,nr)              :: wV
        double precision,dimension(3,nThetas,nPhis,nr)      :: sph2cart
        double precision,dimension(nAtoms)                  :: ln_rmin,len_lnrmin_lnrmax
        !! Rnl arrays for free and confined atoms (to be read-in)
        double precision,dimension(nOrbs,lRnl_max)          :: Rnls_free,Rnls_conf
        !! spline coefficients for Rnl functions sp*1(iOrb), sp*2(iOrb), and sp*3(iOrb)
        double precision,dimension(nOrbs,4,lRnl_max)        :: spf,spc
        
        double precision,dimension(nAtoms),intent(out)      :: hirsh_volrat
        
        
        !! for arbitrary distance p: largest index with grid(index) < p on logarithmic grid
        !! is given by (log(p)-log(rmin)) * (len(grid)-1)/(log(rmax)-log(rmin))
        !! => pre-calculate the latter ratio for each atom
        do i=1,nAtoms
            ln_rmin(i) = log(rmins(i))
            len_lnrmin_lnrmax(i) = (len_r(i)-1)/(ln_rmax-ln_rmin(i))
        enddo
        
        call get_Rnls(Rnls_free, Rnls_conf, spf, spc)
        
        dTheta = pi/nThetas
        dPhi = 2.*pi/nPhis
        Vincr = dr*dTheta*dPhi
        !! evaluate trigonometric functions for radial -> cartesian and integration
        do iT=1, nThetas
            Theta = (iT-1)*dTheta
            sin_t = sin(Theta)
            do ir=1, nr
                rc = ir*dr
                !! calculate integration weights wV = r^3*sin(Theta)*r^2*dr*dTheta*dPhi
                !! => V = rho*wV
                wV(iT,ir) = rc*rc*rc*sin_t*rc*rc*Vincr
                do iP=1, nPhis
                    Phi = (iP-1)*dPhi
                    !! transformation spherical grid to cartesian coordinates
                    sph2cart(:,iT,iP,ir) = (/sin_t*cos(Phi)*rc, sin_t*sin(Phi)*rc, cos(Theta)*rc/)
                enddo
            enddo
        enddo
        
        !! get Hirshfeld volume ratios
        do i=1, nAtoms
            hirsh_volrat(i) =  get_hvr_atom(i)
        enddo
        
    
    contains
        subroutine get_Rnls(Rnls_free, Rnls_conf, spf, spc)
            !! Rnl data
            double precision,dimension(nOrbs,lRnl_max),intent(inout)    :: Rnls_free,Rnls_conf
            !! spline coefficients free, conf
            double precision,dimension(nOrbs,4,lRnl_max),intent(inout)  :: spf,spc
            
            !! I/O-Stuff (file identifier, filenames), loop indices (atoms, orbitals)
            integer                                                     :: fO_id=37,fR_id=73,a_idx,iOrb
            character(len=7)                                            :: int2char
            character(len=35)                                           :: fname
            
            
            do iOrb=1,nOrbs
                a_idx = orb2at(iOrb)
                write(int2char, '(I7)') iOrb
                !! filename: '<Rnl_id>_O<orb_state>_<orb_idx>.unf' (e.g. '9452165498321654_Ofree_21.unf')
                fname = adjustl(Rnl_id//'_Ofree_'//trim(adjustl(int2char))//'.unf')
                open (unit=fO_id, file=trim(fname), form='unformatted')
                read (fO_id) Rnls_free(iOrb,1:len_r(a_idx))
                close(fO_id)
                !! filename: '<Rnl_id>_O<orb_state>_<orb_idx>.unf' (e.g. '9452165498321654_Oconf_21.unf')
                fname = adjustl(Rnl_id//'_Oconf_'//trim(adjustl(int2char))//'.unf')
                open (unit=fO_id, file=trim(fname), form='unformatted')
                read (fO_id) Rnls_conf(iOrb,1:len_r(a_idx))
                close(fO_id)
                
                !! get (3rd order) spline interpolation parameters sp*1,sp*2,sp*3 for Rnl_free and Rnl_conf
                !! see module splines_alt for further information
                call cubic_spline(Rnls_free(iOrb,1:len_r(a_idx)), len_r(a_idx), spf(iOrb,1:4,1:len_r(a_idx)))
                call cubic_spline(Rnls_conf(iOrb,1:len_r(a_idx)), len_r(a_idx), spc(iOrb,1:4,1:len_r(a_idx)))
            enddo
        end subroutine get_Rnls
        
        
        function get_hvr_atom(iAtom) result (hvr)
            integer, intent(in)            :: iAtom
            
            !! integration-stuff (indices, coordinates, increment, densities)
            integer                        :: iTheta,iPhi,iRad
            double precision               :: r,x,y,z,dV,Vf,Ve,rhoA,rho_aim,hvr
            double precision,dimension(3)  :: Rcart
            
            
            !! Initialize volumes
            Ve = 0.
            Vf = 0.
            !! INTEGRATE
            do iTheta=2, nThetas     !! sin_t(1)=sin(0)=0 => dV=0 -> skip iTheta=1
                do iPhi=1, nPhis
                    do iRad=1,nr
                        !! define actual position in cartesian coordinates
                        Rcart = positions(:,iAtom)+sph2cart(:,iTheta,iPhi,iRad)
                        !! get free atomic and atom-in-molecule density at Rcart
                        call get_rhos(Rcart,iAtom, rho_aim,rhoA)
                        !! integration free volume (Vf) and effective volume (Ve)
                        Ve = Ve + rho_aim*wV(iTheta,iRad)
                        Vf = Vf + rhoA*wV(iTheta,iRad)
                    enddo
                enddo
            enddo
            hvr = Ve/Vf
        end function get_hvr_atom
        
        
        subroutine get_rhos(Rcart,iAtom, rho_aim,rhoA)
            !! position for density, atom index
            double precision,dimension(3),intent(in)  :: Rcart
            integer,intent(in)                        :: iAtom
            
            !! atomic positions
            double precision,dimension(3)             :: pos
            !! radial and angular parts for orbitals {iOrb} at current grid point, Rcart
            double precision,dimension(nOrbs)         :: Rnl_f,Rnl_c,Ylm
            !! actual wave function at Rcart
            complex(kind=8)                           :: wf
            !! densities, approximate index of current point in radial grid
            double precision                          :: rho_pro,rho_k,rho_sys,grid_idx
            !! loop indices (orbs, kpts, states, atoms)
            integer                                   :: iOrb,ik,a,a_idx
            !! boundaries of orbitals located at atom iAtom
            integer,dimension(2)                      :: orblim
            
            !! free atomic and promolecular density
            double precision,intent(inout)            :: rho_aim,rhoA
            
            
            !! initialize densities
            rho_pro = 0.
            rho_sys = 0.
            do iOrb=1,nOrbs
                !! get atom index and position
                a_idx = orb2at(iOrb)
                pos = positions(:,a_idx)
                
                !! interpolate Rnl functions (free,conf) at current grid point for all orbitals
                grid_idx = ( log(norm2(Rcart-pos)) - ln_rmin(a_idx))*len_lnrmin_lnrmax(a_idx)
                Rnl_c(iOrb) = val_spline(grid_idx, spc(iOrb,1:4,1:len_r(a_idx)), len_r(a_idx))
                Rnl_f(iOrb) = val_spline(grid_idx, spf(iOrb,1:4,1:len_r(a_idx)), len_r(a_idx))

                !! evaluate spherical harmonics at current grid point (see module spherical_harmonics)
                Ylm(iOrb) = Ylm_real(Rcart-pos, otypes(iOrb))
                !! sum up promolecular density = sum_{iOrb} occ_free(iOrb)*|Rnl_f(iOrb)*Ylm(iOrb)|^2
                rho_pro = rho_pro + occ_free(iOrb)*abs(Rnl_f(iOrb)*Ylm(iOrb))*abs(Rnl_f(iOrb)*Ylm(iOrb))
            enddo
            
            !! density of system = sum_k wk*sum_a f_a*|sum_{iOrb} c(k,a,iOrb)*Rnl_c(iOrb)*Ylm(iOrb)|^2
            do ik=1,nkpts     !! loop over k-points, ik
                rho_k = 0.
                do a=1,nOrbs     !! loop over states, a
                    wf = 0.
                    do iOrb=1,nOrbs     !! loop over orbitals, iOrb
                        !! build wave function in state a
                        wf = wf + coeff(iOrb,a,ik)*Rnl_c(iOrb)*Ylm(iOrb)
                    enddo
                    !! sum up densities of all states at current k-point
                    rho_k = rho_k + f(a,ik)*abs(wf)*abs(wf)
                enddo
                !! sum up densities at all k-points
                rho_sys = rho_sys + wk(ik)*rho_k
            enddo
            
            !! get indices for orbitals on atom iAtom and its position
            orblim = at2orbs(:,iAtom)
            pos = positions(:,iAtom)
            !! free atomic density = sum_{iOrb@iAtom} occ_free(iOrb)*|Rnl_f(iOrb)*Ylm(iOrb)|^2
            rhoA = 0.
            do iOrb=orblim(1),orblim(2)
                rhoA = rhoA + occ_free(iOrb)*abs(Rnl_f(iOrb)*Ylm(iOrb))*abs(Rnl_f(iOrb)*Ylm(iOrb))
            enddo
            
            !! atom-in-molecule density = (free atomic density/promolecular density)*system's density
            rho_aim = (rhoA/rho_pro)*rho_sys
        end subroutine get_rhos
        
    end subroutine hirshfeld_main
    
end module HA_recode
