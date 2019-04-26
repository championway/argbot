import numpy as np

class KalmanFilter(object):
    def __init__(self):
        self.dt = 0.05  # delta time

        self.A = np.array([[1, 0], [0, 1]])  # matrix in observation equations
        self.u = np.zeros((2, 1))  # previous state vector

        # (x,y) tracking object center
        self.b = np.array([[0], [255]])  # vector of observations

        self.P = np.diag((3.0, 3.0))  # covariance matrix
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat

        self.Q = np.eye(self.u.shape[0])  # process noise matrix
        self.R = np.eye(self.b.shape[0])  # observation noise matrix
        self.lastResult = np.array([[0], [255]])

    def set_dt(self, dt):
        self.dt = dt
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.u = np.dot(self.F, self.u)
        # Predicted estimate covariance
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u  # same last predicted result
        return self.u

    def correct(self, b, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        if not flag:  # update using prediction
            self.b = self.lastResult
        else: # update using detection
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = self.u + np.dot(K, (self.b - np.dot(self.A, self.u)))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        self.lastResult = self.u
        return self.u

class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of objecMt to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path

class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh=100, max_trace_length = 10):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_trace_length = max_trace_length
        self.track = None

    def set_dt(self, dt):
        if (self.track is None):
            return
        self.track.KF.set_dt(dt)

    def Update(self, detection):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detection: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if (self.track is None):
            track = Track(detection)
            self.track = track

        
        # Update KalmanFilter state, lastResults and tracks trace
        self.track.KF.predict()

        if detection is None:
            self.track.prediction = self.track.KF.correct(
                                        np.array([[0], [0]]), 0)
        else:
            self.track.prediction = self.track.KF.correct(
                                        detection, 1)

        if(len(self.track.trace) > self.max_trace_length):
            for j in range(len(self.track.trace) - self.max_trace_length):
                del self.track.trace[j]

        self.track.trace.append([self.track.prediction[0][0], self.track.prediction[0][1]])
        self.track.KF.lastResult = self.track.prediction
    
    def get_prediction(self):
        if self.track is None:
            return None
        return self.track.prediction