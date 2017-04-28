
SUBROUTINE DISTRIBUTION(P, dX, A, icov, n_features, n_samples, n_peaks)
    IMPLICIT NONE
    INTEGER(kind=8),INTENT(in) :: n_features, n_samples, n_peaks
    DOUBLE PRECISION,INTENT(in) :: dX(n_samples, n_peaks, n_features), A(n_peaks), &
    & icov(n_peaks,n_features,n_features)
    DOUBLE PRECISION,INTENT(inout) :: P(n_samples)
    INTEGER(kind=8) :: i,t

    P = 0.d0
    DO t=1,n_samples
        DO i=1,n_peaks+1
            IF(i == 1)THEN
                P(t) = A(i)*DEXP(-DOT_PRODUCT(dX(t,i,:), MATMUL(icov(i,:,:), dX(t,i,:)))/2)
            ELSE
                P(t) = P(t) + A(i)*DEXP(-DOT_PRODUCT(dX(t,i,:), MATMUL(icov(i,:,:), dX(t,i,:)))/2)
            END IF
        END DO
    END DO
END
