!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	SUBROUTINE SIMULATE(x,y,steps,dt,mG,kbT,XX,YY,AA,sx,sy,sk,states)
	IMPLICIT NONE
	INTEGER(kind=4),INTENT(in) :: steps, states
	REAL(kind=8),INTENT(in) :: dt, mG, kbT
	REAL(kind=8),INTENT(in) :: AA(1:states)
	REAL(kind=8),INTENT(in) :: XX(1:states), YY(1:states)
	REAL(kind=8),INTENT(in) :: sx(1:states), sy(1:states), sk(1:states)
	REAL(kind=8),INTENT(inout) :: x, y

	INTEGER(kind=4) :: step
	INTEGER(kind=4) :: i, clock, N_seed
	INTEGER(kind=4), ALLOCATABLE :: seed(:)
	REAL(kind=8) :: dVx, dVy, Fx_random, Fy_random
	REAL(kind=8) :: mean, std
	REAL(kind=8) :: r1, r2, r3, pi
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

	OPEN(unit=1, file="walk")
	WRITE(1,*)x,y
	DO step=1,steps
		! Calculate force along potential
		dVx = 0.d0
		dVy = 0.d0
		CALL POTENTIAL_FORCE(dVx, dVy, x, y, XX, YY, AA, sx, sy, sk, states)

		! Generate random force along X
		CALL RANDOM_NUMBER(r1)
		CALL RANDOM_NUMBER(r2)
		Fx_random = mean+std*SQRT(-2.0E+00*LOG(r1)) * COS(2.0E+00*pi*r2)

		! Generate random force along Y
		CALL RANDOM_NUMBER(r1)
		CALL RANDOM_NUMBER(r2)
		Fy_random = mean+std*SQRT(-2.0E+00*LOG(r1)) * COS(2.0E+00*pi*r2)

		!WRITE(*,'(I4,A,6(F16.8,A))')step," ", x, " ", y, " ", &
		! 	& -(dt/mG)*dVx, " ", -(dt/mG)*dVy, " ", Fx_random," ", Fy_random
		! Update coordinates
		x = x - (dt/mG)*dVx + Fx_random
		y = y - (dt/mG)*dVy + Fy_random

		WRITE(1,'(2(F8.4))')x,y
		!CALL PROGRESS(step,steps)
	END DO
	CLOSE(1)
	END !SUBROUTINE SIMULATE

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	SUBROUTINE POTENTIAL_FORCE(dVx, dVy, x, y, XX, YY, AA, sx, sy, sk, states)
	IMPLICIT NONE
	INTEGER(kind=4),INTENT(in) :: states
	REAL(kind=8),INTENT(in) :: AA(1:states)
	REAL(kind=8),INTENT(in) :: x, y, XX(1:states), YY(1:states)
	REAL(kind=8),INTENT(in) :: sx(1:states), sy(1:states), sk(1:states)
	REAL(kind=8),INTENT(out) :: dVX, dVy

	INTEGER(kind=8) :: state
	REAL(kind=8) :: a(1:states), b(1:states), c(1:states), ee

	! Prepare any asymmetries in Gaussian potential
	a = 0.d0
	b = 0.d0
	c = 0.d0
	DO state=1,states
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
	DO state=1,states
		ee = AA(state)*DEXP(-(a(state)*(x-XX(state))**2 - &
		   & 2*b(state)*(x-XX(state))*(y-YY(state)) + &
		   & c(state)*(y-YY(state))**2))
		dVx = dVx - 2*(a(state)*(x-XX(state))-b(state)*(y-YY(state)))*ee
		dVy = dVy - 2*(-b(state)*(x-XX(state))+c(state)*(y-YY(state)))*ee
	END DO
	END !SUBROUTINE POTENTIAL_FORCE

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	SUBROUTINE RANDOM_FORCE(mean, std, value_out)
	IMPLICIT NONE
	REAL(kind=8),INTENT(in) :: mean, std
	REAL(kind=8),INTENT(out) :: value_out
	REAL(kind=8) :: r1, r2, x, pi
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
	CALL RANDOM_NUMBER(r1)
	CALL RANDOM_NUMBER(r2)

	x = SQRT(-2.0E+00*LOG(r1)) * COS(2.0E+00*pi*r2)
	value_out = mean + std * x
	WRITE(*,*) clock, r1, r2, value_out
	return
	END !SUBROUTINE RANDOM_FORCE

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	SUBROUTINE PROGRESS(step,steps)
	IMPLICIT NONE
	INTEGER(kind=4)::step,steps, k
	CHARACTER(len=21)::bar="??????% |          |"
	WRITE(unit=bar(1:6),fmt="(F6.2)") 100*(REAL(step)/REAL(steps))

	!IF(MOD(INT(steps/step),10).eq.0)THEN
	!	k = INT(steps/step)/10
	!	bar(10+k:10+k)="*"
	!END IF

	! print the progress bar.
	WRITE(unit=6,fmt="(a1,a1,a17)") '+',char(13), bar
	RETURN
	END !SUBROUTINE PROGRESS