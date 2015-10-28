module splines
!!! Adapted from spline.f90 !!!

contains
    subroutine spline (x, y, b, c, d, n)
    !======================================================================
    !  Calculate coefficients b,c,d (each of length n)
    !  for cubic spline interpolation
    !  s(x) = y(i) + b(i)*(x-x(i)) + c(i)*(x-x(i))**2 + d(i)*(x-x(i))**3
    !  for  x(i) <= x <= x(i+1)
    !----------------------------------------------------------------------
    !  spline.f90 program is based on fortran version of program spline.f
    !  the accompanying function fspline can be used for interpolation
    !======================================================================
    implicit none
    integer                        :: n
    double precision,dimension(n)  :: x, y, b, c, d
    integer                        :: i, j, gap
    double precision               :: h
    
    gap = n-1
    !! step 1: preparation
    d(1) = x(2) - x(1)
    c(2) = (y(2) - y(1))/d(1)
    do i = 2, gap
        d(i) = x(i+1) - x(i)
        b(i) = 2.0*(d(i-1) + d(i))
        c(i+1) = (y(i+1) - y(i))/d(i)
        c(i) = c(i+1) - c(i)
    enddo
    
    !! step 2: end conditions 
    b(1) = -d(1)
    b(n) = -d(n-1)
    c(1) = 0.0
    c(n) = 0.0
    c(1) = c(3)/(x(4)-x(2)) - c(2)/(x(3)-x(1))
    c(n) = c(n-1)/(x(n)-x(n-2)) - c(n-2)/(x(n-1)-x(n-3))
    c(1) = c(1)*d(1)**2/(x(4)-x(1))
    c(n) = -c(n)*d(n-1)**2/(x(n)-x(n-3))
    
    !! step 3: forward elimination 
    do i = 2, n
        h = d(i-1)/b(i-1)
        b(i) = b(i) - h*d(i-1)
        c(i) = c(i) - h*c(i-1)
    enddo
    
    !! step 4: back substitution
    c(n) = c(n)/b(n)
    do j = 1, gap
        i = n-j
        c(i) = (c(i) - d(i)*c(i+1))/b(i)
    enddo
    
    !! step 5: compute spline coefficients
    b(n) = (y(n) - y(gap))/d(gap) + d(gap)*(c(gap) + 2.0*c(n))
    do i = 1, gap
        b(i) = (y(i+1) - y(i))/d(i) - d(i)*(c(i+1) + 2.0*c(i))
        d(i) = (c(i+1) - c(i))/d(i)
        c(i) = 3.*c(i)
    enddo
    c(n) = 3.0*c(n)
    d(n) = d(n-1)
    end subroutine spline

    function ispline(u, x, y, b, c, d, n)
    !======================================================================
    ! function ispline evaluates the cubic spline interpolation at point u
    ! ispline = y(i)+b(i)*(u-x(i))+c(i)*(u-x(i))**2+d(i)*(u-x(i))**3
    ! where  x(i) <= u <= x(i+1)
    !=======================================================================
    implicit none
    double precision               :: u, ispline, dx
    integer                        :: n, i, j, k
    double precision,dimension(n)  :: x, y, b, c, d
        
    !!  binary search for for i, such that x(i) <= u <= x(i+1)
    i = 1
    j = n+1
    do while (j > i+1)
        k = (i+j)/2
        if (u < x(k)) then
            j=k
        else
            i=k
        endif
    enddo
    
    !!  evaluate spline interpolation
    dx = u - x(i)
    ispline = y(i) + dx*(b(i) + dx*(c(i) + dx*d(i)))
    end function ispline

end module splines
