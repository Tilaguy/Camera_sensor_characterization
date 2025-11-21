import cv2
import yaml
import time
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

from functions.camera_sensor import Camera_sensor

class Response_function():
  """
  This test analyzes uniformly illuminated flat-field images captured under varying exposure-lighting combinations. It computes global mean and variance for each condition, constructs the photon-transfer curve (PTC), and fits a linear model in the shot-noise-limited region to estimate the conversion gain (shot_e_per_dn). The function also computes photo-response non-uniformity (PRNU) by removing dark offsets and measuring spatial variance across normalized flat-field data. This experiment quantifies the sensor's sensitivity, linearity, and pixel-level response variability, producing key photometric parameters used in accurate camera modeling and illumination-dependent noise simulation.
  """

  def __init__(self, camera:Camera_sensor, num_frames:int=100, dark_var=0, dark_mean=0, exposure_list=[10], *args, **kwargs):
    self.cam = camera
    self.N = num_frames
    self.t_array = np.array(sorted(exposure_list), dtype=float)

    ev = self.cam.get(cv2.CAP_PROP_EXPOSURE)
    self.exposure = 2 ** ev
    self.F = np.array([])

    self.var_read = dark_var
    self.d_p = dark_mean
    self.ptc_points = []
    self.prnu_points = []

    self.output_data = {
      "Response_function":{
        "Camera":{
          "name": self.cam.ref,
          "W": self.cam.W,
          "H": self.cam.H,
          "FPS": self.cam.fps,
          "brightness": self.cam.brightness,
          },
        "num_frames": self.N,
        "test_A":[],
      }
    }

    self.setup_configuration()

  def setup_configuration(self):
    print("To perform this test, make sure the camera is pointing at a flat field with uniform lighting below the camera's saturation level.")
    print('Press [Y] when you are certain the entire image is uniform.')
    print('Press [Q] to quit.')

    while True:
      ret, frame = self.cam.read()
      if not ret:
        break

      cv2.imshow('Camera Feed - Press Y when ready, Q to quit', frame)

      key = cv2.waitKey(1) & 0xFF

      if key == ord('y') or key == ord('Y'):
        print("Test starting...")
        cv2.destroyAllWindows()

        for t in self.t_array:
          # Convert real exposure time (seconds or ms) into EV
          ev = np.log2(t)

          # Apply exposure
          self.cam.set(cv2.CAP_PROP_EXPOSURE, ev)
          time.sleep(0.2)

          # Read back EV from camera (driver may clamp or round)
          ev_applied = self.cam.get(cv2.CAP_PROP_EXPOSURE)
          t_applied = 2 ** ev_applied

          print(f"Requested: {t:.6f}s | Applied: {t_applied:.6f}s (EV={ev_applied:.3f})")

          self.exposure = t_applied

          self.get_data()
          self.run_test_part_A()

        self.run_test_part_B()
        self.save_data()
        break
      elif key == ord('q') or key == ord('Q'):
        print("Test cancelled.")
        cv2.destroyAllWindows()
        break

  def get_data(self):
    array = []
    for _ in range(self.N):
      ret, frame = self.cam.read()
      if not ret:
        print("⚠️ Failed to read frame from camera.")
        break

      f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      array.append(f)
    self.F = np.array(array)

  def run_test_part_A(self):
    # Pixel mean
    mu = np.mean(self.F, axis=0).astype(np.float32)
    # Pixel variance
    var = np.var(self.F, axis=0).astype(np.float32)
    # Global mean
    flat_mean = float(np.mean(mu))
    if flat_mean >= 0.95 * 255:#max_dn:
      return
    # Global variance
    flat_var = float(np.mean(var))

    # Shot-Noise Variance in DN
    var_shot = flat_var - self.var_read
    if var_shot <= 0:
      return
    self.ptc_points.append([flat_mean, var_shot])

    # Global signal mean
    s_p = mu - self.d_p
    bar_s = np.mean(s_p).astype(np.float32)
    if bar_s <= 0:
      return
    # Spatial Variance of Pixel Response
    var_spat = np.mean((s_p - bar_s) ** 2).astype(np.float32)
    # Temporal Noise Contribution Correction
    var_temp_spat = flat_var / self.N
    # PRNU
    var_prnu = np.max([var_spat - var_temp_spat, 0])
    prnu = np.sqrt(var_prnu) / bar_s
    self.prnu_points.append(prnu)

    now = datetime.now()
    self.output_data['Response_function']["test_A"].append({
          "time": now.strftime("%Y-%m-%d %H:%M:%S"),
          "flat_mean": float(flat_mean),
          "flat_var": float(flat_var),
          "prnu": float(prnu),
        })

  def run_test_part_B(self):
    ptc_data = np.array(self.ptc_points)
    x = ptc_data[:,0].reshape(-1, 1)
    y = ptc_data[:,1].reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    print(model.coef_)
    slope = float(model.coef_[0][0])

    shot_e_per_dn = 1/slope
    self.output_data['Response_function']["shot_e_per_dn"] = float(shot_e_per_dn)

    prnu_data = np.array(self.prnu_points)
    self.output_data['Response_function']["prnu"] = float(np.mean(prnu_data))

  def save_data(self):
    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d%H%M%S")
    output_filename = f'data/Response_function_{self.cam.ref}_{formatted_datetime}.yaml'
    with open(output_filename, 'w') as file:
      yaml.dump(self.output_data, file, default_flow_style=False, sort_keys=False)

