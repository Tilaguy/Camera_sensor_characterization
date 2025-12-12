import cv2
import yaml
import time
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

from functions.camera_sensor import Camera_sensor, Camera_test

class Response_function(Camera_test):
  """
  This test analyzes uniformly illuminated flat-field images captured under varying exposure-lighting combinations. It computes global mean and variance for each condition, constructs the photon-transfer curve (PTC), and fits a linear model in the shot-noise-limited region to estimate the conversion gain (shot_e_per_dn). The function also computes photo-response non-uniformity (PRNU) by removing dark offsets and measuring spatial variance across normalized flat-field data. This experiment quantifies the sensor's sensitivity, linearity, and pixel-level response variability, producing key photometric parameters used in accurate camera modeling and illumination-dependent noise simulation.
  """

  def __init__(self, camera:Camera_sensor, num_frames:int=100, dark_var:float=0.0, dark_mean=0, exposure_list=[10], *args, **kwargs):
    super().__init__(camera, num_frames)
    self.t_array = np.array(sorted(exposure_list), dtype=float)

    ev = self.cam.get(cv2.CAP_PROP_EXPOSURE)
    self.exposure = 2 ** ev
    self.F = np.array([])

    self.var_read = dark_var
    self.d_p = dark_mean

  def setup_configuration(self):
    print("\nINSTRUCTIONS:")
    print("To perform this test, make sure the camera is pointing at a flat field with uniform lighting below the camera's saturation level.")
    print('Press [Y] when you are certain the entire image is uniform.')
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
    F = []
    for _ in range(self.N):
      ret, frame = self.cam.read()
      if not ret:
        print("Failed to read frame from camera.")
        break

      f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      F.append(f)

    return F

  def run_test(self):
    prnu_points = []
    ptc_points = []

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

      F = np.array(self.get_data())

      print(f"Analizing data set {attempt + 1}")
      # Pixel mean
      mu = np.mean(F, axis=0).astype(np.float32)
      # Pixel variance
      var = np.var(F, axis=0).astype(np.float32)
      # Global mean
      flat_mean = float(np.mean(mu))
      if flat_mean >= 0.95 * 255:#max_dn:
        break
      # Global variance
      flat_var = float(np.mean(var))

      # Shot-Noise Variance in DN
      var_shot = flat_var - self.var_read
      if var_shot <= 0:
        break

      # Global signal mean
      s_p = mu - self.d_p
      bar_s = np.mean(s_p).astype(np.float32)
      if bar_s <= 0:
        break
      # Spatial Variance of Pixel Response
      var_spat = np.mean((s_p - bar_s) ** 2).astype(np.float32)
      # Temporal Noise Contribution Correction
      var_temp_spat = flat_var / self.N
      # PRNU
      var_prnu = np.max([var_spat - var_temp_spat, 0])
      prnu = np.sqrt(var_prnu) / bar_s

      now = datetime.now()
      self.add_to_output(f"test_{attempt}", {
          "time": now.strftime("%Y-%m-%d %H:%M:%S"),
          "exposure": exposure,
          "flat_mean": float(flat_mean),
          "flat_var": float(flat_var),
          "prnu": float(prnu),
        })
      prnu_points.append(prnu)
      ptc_points.append([flat_mean, var_shot])

    ptc_data = np.array(ptc_points)
    x = ptc_data[:,0].reshape(-1, 1)
    y = ptc_data[:,1].reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)

    slope = float(model.coef_[0][0])
    if slope == 0:
      shot_e_per_dn=0
    else:
      shot_e_per_dn = 1/slope
    prnu_data = np.array(prnu_points)


    self.add_to_output("results",{
      "shot_e_per_dn": float(shot_e_per_dn),
      "prnu": float(np.mean(prnu_data)),
    })
    self.save_data()
