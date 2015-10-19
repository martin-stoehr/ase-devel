!=============================================================================!
!                       rolling ball module                                   !
!=============================================================================!
!                                                                             !
!                                                                             !
!-----------------------------------------------------------------------------!
! Descriptor                                                                  ! 
!-----------------------------------------------------------------------------!
! Written by Reinhard J. Maurer (TUM), version 1.0, 2014/05/23                !
!=============================================================================!

module types
implicit none
integer,parameter,public                                 :: dp_rk = kind(1D0)
end module

module rollingball 

  use types, only: dp_rk

  implicit none

  !private

  !---------------------------------------------------------------------------!
  !                        P u b l i c   V a r i a b l e s                    !
  !---------------------------------------------------------------------------!
  
  !Input parameters
  integer, public, save :: na
  integer, public, save :: nb
  integer, public, save :: nc

  real(kind=dp_rk), dimension(3)  , save, public :: a
  real(kind=dp_rk), dimension(3)  , save, public :: b
  real(kind=dp_rk), dimension(3)  , save, public :: c

  real(kind=dp_rk), save, public :: iso, iso_tol 
  real(kind=dp_rk), save, public :: r, z0 
  
  real(kind=dp_rk), dimension(:,:,:), allocatable, save, public :: ldos 
  real(kind=dp_rk), dimension(:,:,:), allocatable, save, public :: ldos_new

  !---------------------------------------------------------------------------!
  !                        P r i v a t e  V a r i a b l e s                   !
  !---------------------------------------------------------------------------!
  real(kind=dp_rk), parameter    :: pi = 3.141592653589793_dp_rk

  !--------------------------------------------------------------------------!
  !                       The public subroutines                             !
  !--------------------------------------------------------------------------!

  public :: calculate_rollingball

contains

subroutine calculate_rollingball() 
  !=========================================================================!
  ! Performs a rolling ball smearing on a given cube grid                   !
  !-------------------------------------------------------------------------!
  ! written by Reinhard Maurer, TUM, 2014/05/23                             !
  !=========================================================================!

  implicit none

  real(kind=dp_rk)  :: da_len, db_len, dc_len, imax, imin, eps
  real(kind=dp_rk), dimension(3)  :: da, db, dc 
  real(kind=dp_rk) :: da_len2, db_len2, dc_len2, r2, value
  
  integer :: nc_lim, nna, nnb, nnc
  integer :: i, j, l, ll, k, kk, h, hh
  integer :: lls, kks, hhs
  real(kind=dp_rk) :: lll, hhh, kkk, nsum

  da = a / na
  db = b / nb
  dc = c / nc

  da_len = sqrt(da(1)*da(1)+da(2)*da(2)+da(3)*da(3))
  db_len = sqrt(db(1)*db(1)+db(2)*db(2)+db(3)*db(3))
  dc_len = sqrt(dc(1)*dc(1)+dc(2)*dc(2)+dc(3)*dc(3))

  imax = iso + iso_tol*iso
  imin = iso - iso_tol*iso

  nna = max(1,nint(r/da_len))
  nnb = max(1,nint(r/db_len))
  nnc = max(1,nint(r/dc_len))

  r2 = r*r
  da_len2 =  da_len*da_len
  db_len2 =  db_len*db_len
  dc_len2 =  dc_len*dc_len

  eps = epsilon(1.0_dp_rk)
  if (z0 <= 0.0_dp_rk+eps) then
    nc_lim = nc
  else
    nc_lim = int(z0/dc_len)
  end if

  do l=1, na
    do k=1,nb
      do h=nc_lim+1,nc
        ldos(l,k,h) = 0.0_dp_rk
      end do
    end do
  end do

  do l=1, na
    do k=1, nb
      do h=1, nc_lim
        if (ldos(l,k,h)< imax .and. ldos(l,k,h)>imin) then
          value = ldos(l,k,h)
          !write(*,*) l, k, h
          do ll=l-nna, l+nna
            lls = ll
            if (ll < 1) lls = ll + na
            if (ll > na) lls = ll - na
            do kk=k-nnb, k+nnb
              kks = kk
              if (kk < 1) kks = kk + nb
              if (kk > nb) kks = kk - nb
              do hh=h-nnc, h+nnc
                hhs = hh
                if (hh < 1) hhs = hh + nc
                if (hh > nc) hhs = hh - nc
                if (ldos(lls,kks,hhs)<value) then
                  lll = (ll-l)*(ll-l)*da_len2
                  kkk = (kk-k)*(kk-k)*db_len2
                  hhh = (hh-h)*(hh-h)*dc_len2
                  nsum = lll+kkk+hhh
                  if (nsum<r2) then
                      ldos_new(lls,kks,hhs) = ldos_new(lls,kks,hhs) + value
                  end if
                end if
              end do
            end do 
          end do
        end if
      end do
    end do
  end do

  return

end subroutine calculate_rollingball


end module rollingball 

