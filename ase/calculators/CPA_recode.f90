module cpa_recode
implicit none

contains
    subroutine get_APT(n_files, n_k_tot, n_basis, nAtoms, Orb2Atom, wk, do_pbc, type_evecs, APT)
        !! number of restart_files, k-points, basis orbitals, and atoms, and flag for pbc
        integer,intent(in)                               :: n_files,n_k_tot,n_basis,nAtoms,do_pbc
        !! assign orbital index to atom center and number of orbitals
        integer,dimension(n_basis),intent(in)            :: Orb2Atom
        !! k-points weighting factors
        double precision,dimension(n_k_tot),intent(in)   :: wk
        !! specifier for eigenvector type ('R':real, 'C':complex)
        character(len=1),intent(in)                      :: type_evecs
        
        !! variables for number of orbitals, states, spins, and k-points in each restart file
        integer                                          :: nOrbs,nStates,nSpins,n_k_task,i_file,i_k_start
        !! dump variable for eigenvalue
        double precision                                 :: KS_eigenvalue
        !! restart file identifier
        character(len=3)                                 :: int2char
        
        !! (intermediate) results arrays |c(i,a,s,k)|^2 and f(a,s,k)
        double precision,dimension(:,:,:,:),allocatable  :: abs_KS_evec_sqrd
        double precision,dimension(:,:,:),allocatable    :: occ_numbers
        
        !! results array (Atom-Projected Trace of density-matrix)
        double precision,dimension(nAtoms),intent(out)   :: APT
        
        
        !! get info periodic calculations
        if (do_pbc == 1) then
            open(file = 'wvfn.dat000', unit = 7, status = 'old', form = 'unformatted')
            read(7) nOrbs
            read(7) nStates
            read(7) nSpins
            close(unit = 7)
            
            !! allocate |eigenvector(s)|^2 and occupation(s)
            allocate(abs_KS_evec_sqrd(nOrbs, nStates, nSpins, n_k_tot))
            allocate(occ_numbers(nStates, nSpins, n_k_tot))
            
            !! (k_point index at which to start read-in) - 1
            i_k_start = 0
            do i_file = 1, n_files
                write(int2char, '(I3.3)') (i_file - 1)
                open(file = 'wvfn.dat'//int2char, unit = 7, status = 'old', form = 'unformatted')
                read(7) nOrbs
                read(7) nStates
                read(7) nSpins
                read(7) n_k_task
                
                if (n_k_task == 0) then
                    close(unit = 7)
                else
                    call read_KS_pbc(nOrbs, nStates, nSpins, n_k_tot, n_k_task, i_k_start, type_evecs)
                    close(unit = 7)
                    i_k_start = i_k_start + n_k_task
                endif
            enddo
        else
            !! get info cluster calculation (one single restart file)
            open(file = 'wvfn.dat', unit = 7, status = 'old', form = 'unformatted')
            read(7) nOrbs
            read(7) nStates
            read(7) nSpins
            
            !! allocate eigenvector(s) and occupation(s)
            allocate(abs_KS_evec_sqrd(nOrbs, nStates, nSpins, 1))
            allocate(occ_numbers(nStates, nSpins, 1))
            
            !! read restart file cluster calculation
            call read_KS(nOrbs, nStates, nSpins, type_evecs)
            close(unit = 7)
        endif
        
        call calculate_APT(nOrbs, nStates, nSpins, n_k_tot, nAtoms, Orb2Atom, wk, APT)
        call deallocations
        
        
    contains
        subroutine read_KS(n_orbs, n_states, n_spins, type_evecs)
            !! number of orbitals, states, spins, and k-points,
            !! number of k-points in current file, and (first k_point index to read-in) - 1
            integer, intent(in)          :: n_orbs,n_states,n_spins
            !! specifier for type of eigenvectors ('R':real, 'C':complex)
            character(len=1),intent(in)  :: type_evecs
            !! loop indices
            integer                      :: i_basis,i_state,i_spin
            !! dummy variables for eigenvector value(s)
            double complex               :: cmplxval
            double precision             :: realval
            
            
            if (type_evecs=='C') then
            !! read complex eigenvector(s)
                do i_basis = 1, n_orbs
                    do i_state = 1, n_states
                        do i_spin = 1, n_spins
                            read(7) cmplxval
                            abs_KS_evec_sqrd(i_basis, i_state, i_spin, 1) = abs(cmplxval*conjg(cmplxval))
                        enddo
                    enddo
                enddo
            elseif (type_evecs=='R') then
                !! read real eigenvector(s)
                do i_basis = 1, n_orbs
                    do i_state = 1, n_states
                        do i_spin = 1, n_spins
                            read(7) realval
                            abs_KS_evec_sqrd(i_basis, i_state, i_spin, 1) = abs(realval)*abs(realval)
                        enddo
                    enddo
                enddo
            endif
            
            !! read occupation(s)
            do i_state = 1, n_states
                do i_spin = 1, n_spins
                    read(7) KS_eigenvalue, occ_numbers(i_state, i_spin, 1)
                enddo
            enddo
        
        end subroutine read_KS
        
        
        subroutine read_KS_pbc(n_orbs, n_states, n_spins, n_k_total, n_k_file, i_k_begin, type_evecs)
            !! number of orbitals, states, spins, and k-points,
            !! number of k-points in current file, and (first k_point index to read-in) - 1
            integer, intent(in)          :: n_orbs,n_states,n_spins,n_k_total,n_k_file,i_k_begin
            !! specifier for type of eigenvectors ('R':real, 'C':complex)
            character(len=1),intent(in)  :: type_evecs
            !! loop indices
            integer                      :: i_basis,i_state,i_spin,i_k
            !! dummy variables for eigenvector value(s)
            double complex               :: cmplxval
            double precision             :: realval
            
            
            if (type_evecs=='C') then
                !! read complex eigenvector(s)
                do i_k = 1, n_k_file
                    do i_spin = 1, n_spins
                        do i_state = 1, n_states
                            do i_basis = 1, n_orbs
                                read(7) cmplxval
                                abs_KS_evec_sqrd(i_basis, i_state, i_spin, i_k+i_k_begin) = real(cmplxval*conjg(cmplxval))
                            enddo
                        enddo
                    enddo
                enddo
            elseif (type_evecs=='R') then
                !! read real eigenvector(s)
                do i_k = 1, n_k_file
                    do i_spin = 1, n_spins
                        do i_state = 1, n_states
                            do i_basis = 1, n_orbs
                                read(7) realval
                                abs_KS_evec_sqrd(i_basis, i_state, i_spin, i_k+i_k_begin) = abs(realval)*abs(realval)
                            enddo
                        enddo
                    enddo
                enddo
            endif
            
            !! read occupation(s)
            do i_k = 1, n_k_total
                do i_spin = 1, n_spins
                    do i_state = 1, n_states
                        read(7) KS_eigenvalue, occ_numbers(i_state, i_spin, i_k)
                    enddo
                enddo
            enddo
        
        end subroutine read_KS_pbc
        
        
        subroutine calculate_APT(n_orbs, n_states, n_spins, n_k_total, n_atoms, Orb2Atom, wk, APT)
            !! number of orbitals, states, spins, k-points, and atoms
            integer,intent(in)                                 :: n_orbs,n_states,n_spins,n_k_total,n_atoms
            !! assign orbital index to atom center
            integer,dimension(n_basis),intent(in)              :: Orb2Atom
            !! k-point weighting factors
            double precision,dimension(n_k_total),intent(in)   :: wk
            
            !! dummy arguments for occupations and eigenvector(s)
            double precision                                   :: occ,abs_ci2
            !! loop indices
            integer                                            :: i_basis,i_state,i_spin,i_k,iAtom
            
            !! result array (Atom-Projected Trace)
            double precision,dimension(n_atoms),intent(inout)  :: APT
            
            
            APT(:) = 0.
            do i_basis = 1, n_orbs
                iAtom = Orb2Atom(i_basis)
                do i_state = 1, n_states
                    do i_spin = 1, n_spins
                        do i_k = 1, n_k_total
                            occ = occ_numbers(i_state, i_spin, i_k)
                            abs_ci2 = abs_KS_evec_sqrd(i_basis, i_state, i_spin, i_k)
                            APT(iAtom) = APT(iAtom) + wk(i_k)*occ*abs_ci2
                        enddo
                    enddo
                enddo
            enddo
            
        end subroutine calculate_APT
        
        
        subroutine deallocations
            
            !! deallocate (intermediate) results arrays
            deallocate(abs_KS_evec_sqrd)
            deallocate(occ_numbers)
            
        end subroutine deallocations
    
    end subroutine get_APT
    

end module cpa_recode
