!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE SIMULATE(x,y,steps,dt,mG,kbT,XX,YY,AA,sx,sy,sk,n_wells)
  IMPLICIT NONE
  INTEGER(kind=4),INTENT(in) :: steps, n_wells
  REAL(kind=8),INTENT(in) :: dt, mG, kbT
  REAL(kind=8),INTENT(in) :: AA(1:n_wells)
  REAL(kind=8),INTENT(in) :: XX(1:n_wells), YY(1:n_wells)
  REAL(kind=8),INTENT(in) :: sx(1:n_wells), sy(1:n_wells), sk(1:n_wells)
  REAL(kind=8),INTENT(inout) :: x, y

  INTEGER(kind=4) :: step
  INTEGER(kind=4) :: i, clock, N_seed
  INTEGER(kind=4), ALLOCATABLE :: seed(:)
  REAL(kind=8) :: dVx, dVy, Fx_random, Fy_random
  REAL(kind=8) :: mean, std
  REAL(kind=8) :: randNum, pi
  pi = 3.141592653589793E+00

  ! Generate random seeds
  CALL RANDOM_SEED(size = N_seed)
  ALLOCATE(seed(N_seed))
  CALL SYSTEM_CLOCK(COUNT_RATE = clock)
  seed = clock + 37*(/ (i-1, i=1, N_seed) /)
  CALL RANDOM_SEED(put = seed)

  Fx_random = 0.d0
  Fy_random = 0.d0
  mean = 0.d0
  std = SQRT((2*kbT*dt)/mG)

  !	OPEN(unit=1, file='walk.b', form='unformatted', action='readwrite', &
  !		& access='direct',recl=16)
  !	WRITE(1,rec=1)x,y

  OPEN(unit=1, file='walk',status='REPLACE')
  WRITE(1,'(F16.6,A,F16.6)')x,"    ",y
  DO step=2,steps
     ! Calculate force along potential
     dVx = 0.d0
     dVy = 0.d0
     CALL POTENTIAL_FORCE(dVx, dVy, x, y, XX, YY, AA, sx, sy, sk, n_wells)

     ! Generate random force along X
     CALL RANDOM_NUMBER(randNum)
     Fx_random = mean+std*DSIN(2.0E+00*pi*randNum)

     ! Generate random force along Y
     CALL RANDOM_NUMBER(randNum)
     Fy_random = mean+std*DSIN(2.0E+00*pi*randNum)

     x = x - (dt/mG)*dVx + Fx_random
     y = y - (dt/mG)*dVy + Fy_random

     !WRITE(1,rec=step)x,y
     WRITE(1,'(F16.6,A,F16.6)')x,"    ",y
  END DO
  CLOSE(1)
END SUBROUTINE SIMULATE

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE POTENTIAL_FORCE(dVx, dVy, x, y, XX, YY, AA, sx, sy, sk, n_wells)
  IMPLICIT NONE
  INTEGER(kind=4),INTENT(in) :: n_wells
  REAL(kind=8),INTENT(in) :: AA(1:n_wells)
  REAL(kind=8),INTENT(in) :: x, y, XX(1:n_wells), YY(1:n_wells)
  REAL(kind=8),INTENT(in) :: sx(1:n_wells), sy(1:n_wells), sk(1:n_wells)
  REAL(kind=8),INTENT(out) :: dVX, dVy

  INTEGER(kind=8) :: state
  REAL(kind=8) :: a(1:n_wells), b(1:n_wells), c(1:n_wells), ee

  ! Prepare any asymmetries in Gaussian potential
  a = 0.d0
  b = 0.d0
  c = 0.d0
  DO state=1,n_wells
     a(state) = DCOS(sk(state))**2/(2*sx(state)**2) + &
          & DSIN(sk(state))**2/(2*sy(state)**2)
     b(state) = -DSIN(2*sk(state))/(4*sx(state)**2) + &
          & DSIN(2*sk(state))/(4*sy(state)**2)
     c(state) = DSIN(sk(state))**2/(2*sx(state)**2) + &
          & DCOS(sk(state))**2/(2*sy(state)**2)
  END DO

  ! Calculate gradient at position x and y
  dVx = 0.d0
  dVy = 0.d0
  ee = 0.d0
  DO state=1,n_wells
     ee = AA(state)*DEXP(-(a(state)*(x-XX(state))**2 - &
          & 2*b(state)*(x-XX(state))*(y-YY(state)) + &
          & c(state)*(y-YY(state))**2))
     dVx = dVx - 2*(a(state)*(x-XX(state))-b(state)*(y-YY(state)))*ee
     dVy = dVy - 2*(-b(state)*(x-XX(state))+c(state)*(y-YY(state)))*ee
  END DO
END SUBROUTINE POTENTIAL_FORCE

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE RANDOM_FORCE(mean, std, value_out)
  IMPLICIT NONE
  REAL(kind=8),INTENT(in) :: mean, std
  REAL(kind=8),INTENT(out) :: value_out
  REAL(kind=8) :: randNum, x, pi
  INTEGER(kind=4) :: i, clock, N_seed
  INTEGER(kind=4), ALLOCATABLE :: seed(:)
  pi = 3.141592653589793E+00

  ! Generate random seeds
  CALL RANDOM_SEED(size = N_seed)
  ALLOCATE(seed(N_seed))
  CALL SYSTEM_CLOCK(COUNT_RATE = clock)
  seed = clock + 37*(/ (i-1, i=1, N_seed) /)
  CALL RANDOM_SEED(put = seed)

  ! Generate random number 1 & 2
  CALL RANDOM_NUMBER(randNum)

  x = DSIN(2.0E+00*pi*randNum)
  value_out = mean + std * x
  WRITE(*,*) clock, randNum, value_out
  RETURN
END SUBROUTINE RANDOM_FORCE
