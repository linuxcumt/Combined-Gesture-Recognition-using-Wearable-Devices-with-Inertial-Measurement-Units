import numpy as np
from calibraxis import Calibraxis

c = Calibraxis()
points = np.array([[-4772.38754098, 154.04459016, -204.39081967],
                   [3525.0346179, -68.64924886, -34.54604833],
                   [-658.17681729, -4137.60248854, -140.49377865],
                   [-564.18562092, 4200.29150327, -130.51895425],
                   [-543.18289474, 18.14736842, -4184.43026316],
                   [-696.62532808, 15.70209974, 3910.20734908],
                   [406.65271419, 18.46827992, -4064.61085677],
                   [559.45926413, -3989.69513798, -174.71879106],
                   [597.22629169, -3655.54153041, -1662.83257031],
                   [1519.02616089, -603.82472204, 3290.58469588]])
# Add points to calibration object's storage.
c.add_points(points)
# Run the calibration parameter optimization.
c.calibrate_accelerometer()

# Applying the calibration parameters to the calibration data.
c.apply(points[0 :])

c.batch_apply(points)
