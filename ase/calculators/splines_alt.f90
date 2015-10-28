module splines_alt

implicit none

!! "vector" and "matrix" are the objects to be handled by LAPACK.

integer,save,private                                    :: spline_dimension=0
double precision,dimension(:),allocatable,save,private  :: vector
double precision,dimension(:),allocatable,save,private  :: matrix_diag
double precision,dimension(:),allocatable,save,private  :: matrix_upper
double precision,dimension(:),allocatable,save,private  :: matrix_lower

contains
    subroutine cubic_spline ( f_grid, n_points, spl_param )
        !! Subroutine cubic_spline splines a function f_grid, given on grid points 1,2,3,...,n_points.
        integer,intent(in)                                    :: n_points
        double precision,dimension(n_points),intent(in)       :: f_grid
        double precision,dimension(4,n_points),intent(inout)  :: spl_param
        
        !! INPUT
        !!   . n_points -- number of points
        !!   . f_grid -- values of data on grid
        
        !! OUTPUT
        !!   . spl_param -- spline parameters
        integer  :: i_info, i_point
        
        !! general setup for spline
        
        !! check whether spline dimension changed since last call
        if (n_points.ne.spline_dimension) then
            !! set up spline matrix from scratch
            if (allocated(vector)) then
                deallocate(vector)
            endif
            if (allocated(matrix_diag)) then
                deallocate(matrix_diag)
            endif
            if (allocated(matrix_upper)) then
                deallocate(matrix_upper)
            endif
            if (allocated(matrix_lower)) then
                deallocate(matrix_lower)
            endif
            
            spline_dimension = n_points
            
            allocate(vector(spline_dimension),stat=i_info)
!            call check_allocation(i_info, 'vector                        ') 
            allocate(matrix_diag(spline_dimension),stat=i_info)
!            call check_allocation(i_info, 'matrix_diag                   ') 
            allocate(matrix_upper(spline_dimension-1),stat=i_info)
!            call check_allocation(i_info, 'matrix_upper                  ') 
            allocate(matrix_lower(spline_dimension-1),stat=i_info)
!            call check_allocation(i_info, 'matrix_lower                  ') 
        end if
        
        !! first grid point
        vector(1) = 3d0*(f_grid(2) - f_grid(1))
        matrix_diag(1) = 2d0
        matrix_upper(1) = 1d0
        matrix_lower(1) = 1d0
        
        !! intermediate grid points
        do i_point=2, n_points-1
            vector(i_point) = 3d0*(f_grid(i_point+1) - f_grid(i_point-1))
            matrix_diag(i_point) = 4d0
            matrix_upper(i_point) = 1d0
            matrix_lower(i_point) = 1d0
        enddo
        
        !! last grid point
        vector(n_points) = 3d0*(f_grid(n_points) - f_grid(n_points-1))
        matrix_diag(n_points) = 2d0
        !! no matrix_upper or matrix_lower elements.
        
        !! solve system to obtain all spline derivatives, using lapack
        call dgtsv(n_points, 1, matrix_lower, matrix_diag, matrix_upper, &
                  &vector, n_points, i_info)
        
        if (i_info.ne.0) then
            write (6,'(1X,A)') "* spline.f : A cubic spline failed - investigate!"
            stop
        endif
        
        !! calculate spline parameters
        do i_point=1, n_points-1
            spl_param(1,i_point) = f_grid(i_point)
            spl_param(2,i_point) = vector(i_point)
            spl_param(3,i_point) = 3d0*(f_grid(i_point+1)-f_grid(i_point)) &
                                   &- 2d0*vector(i_point) - vector(i_point+1)
            spl_param(4,i_point) = 2d0*(f_grid(i_point)-f_grid(i_point+1)) &
                                   &+ vector(i_point) + vector(i_point+1)
        enddo
        
        !! value and derivative at the last point
        spl_param(1,n_points) = f_grid(n_points)
        spl_param(2,n_points) = vector(n_points)
        
    end subroutine cubic_spline
    
    
    function val_spline(r_output, spl_param, n_points) result(val)
        !! Function val_spline
        !! takes a set of spline parameters spl_param for a function, given on grid points
        !! 1, ... n_points, and produce the splined interpolation onto some intermediate point
        
        double precision,intent(in)                        :: r_output
        integer,intent(in)                                 :: n_points
        double precision,dimension(4,n_points),intent(in)  :: spl_param
        
        !! INPUT
        !!   . r_output -- distance for interpolation
        !!   . n_points -- number of grid points
        !!   . spl_param -- spline parameters
        
        !! OUTPUT
        !!   . val -- value of a splined function at asked distance.
        
        integer           :: i_spl,i_term
        double precision  :: val
        
        
        i_spl = int(r_output)
        val = spl_param(1,i_spl)
        
        do i_term=2, 4
            val = val + spl_param(i_term,i_spl)
        enddo
        
    end function val_spline
    
    
    subroutine cleanup_spline (  )
        !! Deallocate module variables !!
        spline_dimension = 0
        if (allocated(vector)) then
            deallocate(vector)
        endif
        if (allocated(matrix_diag)) then
            deallocate(matrix_diag)
        endif
        if (allocated(matrix_upper)) then
            deallocate(matrix_upper)
        endif
        if (allocated(matrix_lower)) then
            deallocate(matrix_lower)
        endif
        
    end subroutine cleanup_spline
    
end module splines_alt
