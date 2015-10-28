module opa_recode
implicit none

contains
    subroutine get_wvfn_info(filename, n_basis, nAtoms, Orb2Atom, ONOP)
        character(*), intent(in)                        :: filename
        integer,intent(in)                              :: n_basis,nAtoms
        !! assign orbital index to atom center and number of orbitals
        integer,dimension(n_basis),intent(in)           :: Orb2Atom
        integer                                         :: n_states,n_spin,nOrbs
        double precision                                :: KS_eigenvalue
        
        !! (intermediate) result arrays
        double precision,dimension(:,:,:),allocatable   :: KS_eigenvector
        double precision,dimension(:,:),allocatable     :: occ_numbers
        
        !! results array
        double precision,dimension(nAtoms),intent(out)  :: ONOP
        
        
        open(file = filename, unit = 7, status = 'old', form = 'unformatted')
        read(7) nOrbs
        read(7) n_states
        read(7) n_spin
        
        call read_KS(nOrbs, n_states, n_spin)
        close(unit = 7)

        call calculate_ONOP(nOrbs, n_states, n_spin, nAtoms, Orb2Atom, ONOP)
        call deallocations
        
        
    contains
        subroutine read_KS(n_Orbs, n_states, n_spin)
            integer, intent(in)    :: n_Orbs,n_states,n_spin
            integer                :: i_basis,i_states,i_spin
            
            allocate(KS_eigenvector(n_Orbs, n_states, n_spin))
            allocate(occ_numbers(n_states, n_spin))
            
            do i_basis = 1, n_Orbs
                do i_states = 1, n_states
                    do i_spin = 1, n_spin
                        read(7) KS_eigenvector(i_basis, i_states, i_spin)
                    enddo
                enddo
            enddo
            
            do i_states = 1, n_states
                do i_spin = 1, n_spin
                    read(7) KS_eigenvalue, occ_numbers(i_states, i_spin)
                enddo
            enddo
        
        end subroutine read_KS
        
        
        subroutine calculate_ONOP(n_Orbs, n_states, n_spin, nAtoms, Orb2Atom, ONOP)
            !! number of orbitals, states and spins
            integer,intent(in)                                :: n_Orbs,n_states,n_spin,nAtoms
            !! assign orbital index to atom center
            integer,dimension(n_basis),intent(in)             :: Orb2Atom
            
            !! dummy arguments for occupations and eigenvector(s)
            double precision                                  :: occ,abs_ci
            !! loop indices
            integer                                           :: i_basis,i_states,i_spin,iAtom
            
            !! result array
            double precision,dimension(nAtoms),intent(inout)  :: ONOP
            
            
            ONOP(:) = 0.
            do i_basis = 1, n_Orbs
                iAtom = Orb2Atom(i_basis)
                do i_states = 1, n_states
                    do i_spin = 1, n_spin
                        occ = occ_numbers(i_states, i_spin)
                        abs_ci = abs(KS_eigenvector(i_basis, i_states, i_spin))
                        ONOP(iAtom) = ONOP(iAtom) + occ*abs_ci*abs_ci
                    enddo
                enddo
            enddo
            
        end subroutine calculate_ONOP
        
        
        subroutine deallocations
            
            deallocate(KS_eigenvector)
            deallocate(occ_numbers)
            
        end subroutine deallocations
    
    end subroutine get_wvfn_info
    

end module opa_recode
