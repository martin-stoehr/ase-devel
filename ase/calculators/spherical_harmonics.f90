module spherical_harmonics
implicit none

double precision,parameter,private  :: pi=3.14159265

contains
    function Ylm_real_trafo(Rcart, otype) result(val)
        !!! Evaluates real spherical harmonics at Rcart in spherical coordinates
        !!! (l,m) specified by otype, e.g. otype = 1, up to l=3
        !!! orbital type corresponds to states(otype+1):
        !!! states = (/'s','px','py','pz','dxy','dyz','dzx','dx2-y2','d3z2-r2',&
        !!!          &'f3yx2-y3','fxyz','fyz2','fxz2','fzx2-zy2','fx3-3xy2','fz3'/)

        double precision,dimension(3),intent(in)  :: Rcart
        integer,intent(in)                        :: otype
        double precision                          :: dist,th,ph,val
        
        dist = norm2(Rcart)
        th = acos(Rcart(3)/dist)
!        ph = xy2phi(Rcart(1), Rcart(2))
        ph = atan2(Rcart(2),Rcart(1))
        
        select case(otype)
        !! s orbital
        case (0)
            val = 1./sqrt(4.*pi)
        !! p orbitals
        case (1)
            val = sqrt(3./(4.*pi))*sin(th)*cos(ph)
        case (2)
            val = sqrt(3./(4.*pi))*sin(th)*sin(ph)
        case (3)
            val = sqrt(3./(4.*pi))*cos(th)
        !! d orbitals
        case (4)
            val = sqrt(15./(4.*pi))*cos(ph)*sin(ph)*sin(th)*sin(th)
        case (5)
            val = sqrt(15./(4.*pi))*sin(th)*cos(th)*sin(ph)
        case (6)
            val = sqrt(15./(4.*pi))*sin(th)*cos(th)*cos(ph)
        case (7)
            val = 0.5*sqrt(15./(4.*pi))*cos(2.*ph)*sin(th)*sin(th)
        case (8)
            val = 0.5*sqrt(5./(4.*pi))*(3.*cos(th)*cos(th)-1.)
        !! f orbitals
        case (9)
            val = ( sqrt(35./(2.*pi))*sin(3.*ph)*sin(th)*sin(th)*sin(th) )/4.
        case (10)
            val = ( sqrt(105./pi)*cos(th)*sin(2.*ph)*sin(th)*sin(th) )/4.
        case (11)
            val = ( sqrt(21./(2.*pi))*sin(th)*(5.*cos(th)*cos(th)-1)*sin(ph) )/4.
        case (12)
            val = ( sqrt(21./(2.*pi))*sin(th)*(5.*cos(th)*cos(th)-1)*cos(ph) )/4.
        case (13)
            val = ( sqrt(105./pi)*cos(th)*cos(2.*ph)*sin(th)*sin(th) )/4.
        case (14)
            val = ( sqrt(35./(2.*pi))*cos(3.*ph)*sin(th)*sin(th)*sin(th) )/4.
        case (15)
            val = ( sqrt(7./pi)*cos(th)*(5.*cos(th)*cos(th)-3.) )/4.
        end select
    end function Ylm_real_trafo
    
    
    function xy2phi(x,y) result(phival)
        double precision,intent(in)  :: x,y
        double precision             :: min_nr=1e-16,phival
        
        
        if (x>min_nr .and. y>min_nr) then
            phival = atan(y/x)
        elseif (x<-min_nr .and. y>min_nr) then
            phival = atan(y/x) + pi
        elseif (x<-min_nr .and. y<-min_nr) then
            phival = atan(y/x) + pi
        elseif (x>min_nr .and. y<-min_nr) then
            phival = atan(y/x)+2*pi
        elseif (abs(x)<=min_nr .and. abs(y)<=min_nr) then
            phival = 0.
        elseif (x>min_nr .and. abs(y)<=min_nr) then
            phival = 0.
        elseif (y>min_nr .and. abs(x)<=min_nr) then
            phival = pi/2.
        elseif (x<-min_nr .and. abs(y)<=min_nr) then
            phival = pi
        elseif (y<-min_nr .and. abs(x)<=min_nr) then
            phival = 3*pi/2
        endif
    end function xy2phi
    
    
    function Ylm_real(Rcart, otype) result(val)
        !!! Evaluates real spherical harmonics at Rcart in cartesian coordinates
        !!! (l,m) specified by otype, e.g. otype = 1, up to l=3
        !!! orbital type corresponds to states(otype+1):
        !!! states = (/'s','px','py','pz','dxy','dyz','dzx','dx2-y2','d3z2-r2',&
        !!!          &'f3yx2-y3','fxyz','fyz2','fxz2','fzx2-zy2','fx3-3xy2','fz3'/)

        double precision,dimension(3),intent(in)  :: Rcart
        integer,intent(in)                        :: otype
        double precision                          :: dist,xc,yc,zc, val
        
        dist = norm2(Rcart)
        xc = Rcart(1)
        yc = Rcart(2)
        zc = Rcart(3)
        
        select case(otype)
        !! s orbital
        case (0)
            val = 1./sqrt(4.*pi)
        !! p orbitals
        case (1)
            val = sqrt(3./(4.*pi))*xc/dist
        case (2)
            val = sqrt(3./(4.*pi))*yc/dist
        case (3)
            val = sqrt(3./(4.*pi))*zc/dist
        !! d orbitals
        case (4)
            val = sqrt(15./(4.*pi))*( xc*yc )/(dist*dist)
        case (5)
            val = sqrt(15./(4.*pi))*( yc*zc )/(dist*dist)
        case (6)
            val = sqrt(15./(4.*pi))*( xc*zc )/(dist*dist)
        case (7)
            val = ( sqrt(15./(4.*pi))*( xc*xc - yc*yc )/(dist*dist) )/2
        case (8)
            val = ( sqrt(5./(4.*pi))*( 3.*zc*zc - dist*dist )/(dist*dist) )/2.
        !! f orbitals
        case (9)
            val = ( sqrt(35./(2.*pi))*( 3.*yc*xc*xc - yc*yc*yc )/(dist*dist*dist) )/4.
        case (10)
            val = sqrt(105./(4.*pi))*( xc*yc*zc )/(dist*dist*dist)
        case (11)
            val = ( sqrt(21./(2.*pi))*( yc*(5.*zc*zc - dist*dist) )/(dist*dist*dist) )/4.
        case (12)
            val = ( sqrt(21./(2.*pi))*( xc*(5.*zc*zc - dist*dist) )/(dist*dist*dist) )/4.
        case (13)
            val = ( sqrt(105./pi)*zc*(xc*xc - yc*yc)/(dist*dist*dist) )/4.
        case (14)
            val = ( sqrt(35./(2.*pi))*xc*(xc*xc - 3.*yc*yc)/(dist*dist*dist) )/4.
        case (15)
            val = ( sqrt(7./pi)*zc*(5.*zc*zc - 3.*dist*dist)/(dist*dist*dist) )/4.
        end select
    end function Ylm_real
    

end module spherical_harmonics
