import cv2
import yaml
import time
import numpy as np
from datetime import datetime

from functions.camera_sensor import Camera_sensor, Camera_test

class Noise_characterization(Camera_test):
  """
  This test characterizes the electronic noise of the camera by capturing multiple dark frames at different exposure times with the lens fully covered. The function computes per-pixel temporal statistics (mean and variance) and aggregates them into global metrics, including read noise and dark-signal non-uniformity (DSNU). By analyzing how dark variance behaves across exposure times, the function isolates fixed-pattern noise and temporal noise sources inherent to the sensor. This test provides the fundamental noise parameters required for realistic camera simulation and for correcting downstream flat-field measurements.
  """

  def __init__(self, camera:Camera_sensor, num_frames:int=100, exposure_list=[], *args, **kwargs):
    super().__init__(camera, num_frames)
    self.t_array = np.array(sorted(exposure_list), dtype=float)

  def setup_configuration(self):
    print("\nINSTRUCTIONS:")
    print('To perform this test, ensure the lens is completely covered.')
    print('Press [Y] when you are certain the entire image is black.')
    print('Press [Q] to quit.')

    isReady:bool = False
    while True:
      ret, frame = self.cam.read()
      if not ret:
        break

      cv2.imshow('Camera Feed - Press Y when ready, Q to quit', frame)

      key = cv2.waitKey(1) & 0xFF

      if key == ord('y') or key == ord('Y'):
        print("\nThe test is starting,\n⚠️\tDo not move the camera or experimental setup until the test is finished.")
        isReady = True
        break
      elif key == ord('q') or key == ord('Q'):
        print("\nTest cancelled.")
        break

    cv2.destroyAllWindows()
    return isReady

  def get_data(self):
    D = []
    for _ in range(self.N):
      ret, frame = self.cam.read()
      if not ret:
        print("Failed to read frame from camera.")
        break

      d = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      D.append(d)

    return D

  def run_test(self):
    dark_mean_point = []
    dark_var_point = []
    read_std_dn_point = []
    dsnu_dn_point = []

    min_dark_var = 0
    for attempt, exposure in enumerate(self.t_array):
      # Convert real exposure time (seconds or ms) into EV
      ev = np.log2(exposure)
      # Apply exposure
      self.cam.set(cv2.CAP_PROP_EXPOSURE, ev)
      time.sleep(0.2)

      # Read back EV from camera (driver may clamp or round)
      ev_applied = self.cam.get(cv2.CAP_PROP_EXPOSURE)
      exposure = 2 ** ev_applied
      if (ev != ev_applied) and attempt > 0:
        break
      print(f"Getting {self.N} frames for exposure time: {exposure} seg ")

      D = np.array(self.get_data())

      print(f"Analizing data set {attempt + 1}")
      # Pixel mean
      mu = np.mean(D, axis=0).astype(np.float32)
      # Pixel variance
      var = np.var(D, axis=0).astype(np.float32)
      # Global mean
      dark_mean = float(np.mean(mu))
      # Global variance
      dark_var = float(np.mean(var))
      if attempt == 0:
        min_dark_var = dark_var

      # Read noise
      read_std_dn = np.sqrt(min_dark_var)
      # Dark Signal Non-Uniformity
      dsnu_dn = np.sqrt(np.mean((mu - dark_mean) ** 2))

      now = datetime.now()
      self.add_to_output(f"test_{attempt}", {
          "time": now.strftime("%Y-%m-%d %H:%M:%S"),
          "exposure": exposure,
          "dark_mean": float(dark_mean),
          "dark_var": float(dark_var),
          "read_std_dn": float(read_std_dn),
          "dsnu_dn": float(dsnu_dn),
        })
      dark_mean_point.append(dark_mean)
      dark_var_point.append(dark_var)
      read_std_dn_point.append(read_std_dn)
      dsnu_dn_point.append(dsnu_dn)

    dark_mean_data = np.array(dark_mean_point)
    dark_var_data = np.array(dark_var_point)
    read_std_dn_data = np.array(read_std_dn_point)
    dsnu_dn_data = np.array(dsnu_dn_point)
    self.add_to_output("results",{
      "dark_mean": float(np.mean(dark_mean_data)),
      "dark_var": float(np.mean(dark_var_data)),
      "read_std_dn": float(np.mean(read_std_dn_data)),
      "dsnu_dn": float(np.mean(dsnu_dn_data)),
    })
    self.save_data()

