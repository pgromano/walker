!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE RUN(X, mu, icov, A, dt, friction, T0, Tf, n_features, n_frames, n_peaks)
    IMPLICIT NONE
    INTEGER(kind=4),INTENT(in) :: n_features, n_frames, n_peaks
    DOUBLE PRECISION,INTENT(in) :: dt, friction, T0, Tf
    DOUBLE PRECISION,INTENT(in) :: A(n_peaks)
    DOUBLE PRECISION,INTENT(in) :: mu(n_peaks, n_features)
    DOUBLE PRECISION,INTENT(in) :: icov(n_peaks, n_features, n_features)
    DOUBLE PRECISION,INTENT(inout) :: X(n_frames, n_features)

    INTEGER(kind=4) :: i, k, clock, frame, N_seed
    INTEGER(kind=4), ALLOCATABLE :: seed(:)
    DOUBLE PRECISION :: dV(n_features), F_random(n_features)
    DOUBLE PRECISION :: randNum, pi, kT
    pi = 4*DATAN(1.d0)

    ! Generate random seeds
    CALL RANDOM_SEED(size = N_seed)
    ALLOCATE(seed(N_seed))
    CALL SYSTEM_CLOCK(COUNT_RATE = clock)
    seed = clock + 37*(/ (i-1, i=1, N_seed) /)
    CALL RANDOM_SEED(put = seed)

    F_random = 0.d0
    DO frame=1,n_frames-1
        ! Calculate force along potential
        dV = 0.d0
        CALL CALCULATE_FORCE(dV, X(frame,:), mu, icov, A, n_features, n_peaks)

        ! Generate random force along X
        kT = DBLE(frame)*(Tf-T0)/DBLE(n_frames)
        DO k=1,n_features
            CALL RANDOM_NUMBER(randNum)
            F_random(k) = DSQRT((2*kT*dt)/friction)*DSIN(2.d0*pi*randNum)
        END DO

        X(frame+1,:) = X(frame,:) - (dt/friction)*dV + F_random
    END DO
END SUBROUTINE RUN

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE CALCULATE_FORCE(dV, X, mu, icov, A, n_features, n_peaks)
    IMPLICIT NONE
    INTEGER(kind=4),INTENT(in) :: n_features, n_peaks
    DOUBLE PRECISION,INTENT(in) :: A(n_peaks)
    DOUBLE PRECISION,INTENT(in) :: mu(n_peaks, n_features)
    DOUBLE PRECISION,INTENT(in) :: icov(n_peaks, n_features, n_features)
    DOUBLE PRECISION,INTENT(in) :: X(n_features)
    DOUBLE PRECISION,INTENT(inout) :: dV(n_features)

    INTEGER(kind=4) :: n, k
    DOUBLE PRECISION :: V, delX(n_features)

    ! Calculate gradient at position X
    V = 0.d0
    dV = 0.d0
    DO n=1,n_peaks
        delX = X-mu(n,:)
        V = A(n)*DEXP(-DOT_PRODUCT(delX, MATMUL(icov(n,:,:), delX))/2.0)
        dV = dV - V*(MATMUL(icov(n,:,:), delX))
    END DO
END SUBROUTINE CALCULATE_FORCE
