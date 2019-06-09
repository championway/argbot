import numpy as np

class KF(object):
    def __init__(self, n, m, pval=0.1, qval=1e-4, rval=0.1):
    def kalman_filter(self, x, P):
        x, P = measurement_update(x, P, measurements[n])
        x, P = prior_update(x,P,u)
        return x, P

    def prior_update(x,P,u):
        x = F * x + u
        P = F * P * F.T
        return x, P
        
    def measurement_update(x,P,measurement):
        I = np.matrix([[1.,0.], [0.,1.]])
        z = np.matrix([[measurement]])
        y = z - H * x
        S = H * P * H.T + R
        K = P * H.T * S.I
        
        x = x + K * y
        P = (I - K * H) * P
        return x, P


    measurements = [5, 6, 7]

    x = np.matrix([[0.], [0.]]) # initial state (location and velocity)
    P = np.matrix([[1000., 0.], [0., 1000.]]) # initial uncertainty
    u = np.matrix([[0.], [0.]]) # external motion
    F = np.matrix([[1., 1.], [0, 1.]]) # next state function
    H = np.matrix([[1., 0.]]) # measurement function
    R = np.matrix([[1.]]) # measurement uncertainty
    I = np.matrix([[1., 0.], [0., 1.]]) # identity matrix

    x, P = kalman_filter(x, P)
    print "Estimate of position: {}, Estimate of velocity: {}".format(x[0],x[1])
    print "Covariance matrix {}".format(P)