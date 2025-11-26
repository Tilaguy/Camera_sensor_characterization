import cv2
import yaml
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

from functions.camera_sensor import Camera_sensor

class Vignette():
  """
  This test estimates optical vignetting by analyzing a uniformly illuminated flat-field image. It identifies the optical center, computes each pixel's radial distance, and normalizes intensities using a central reference region. Radial bins are formed to create a smooth radial brightness profile, which is then fitted to a cos⁴-based vignetting model. The resulting parameters describe how image brightness decreases from center to corners, and a scalar vignette_strength is derived to quantify the falloff. This function provides the optical shading characteristics required for rendering realistic peripheral darkening in simulated camera images.
  """

  def __init__(self, camera:Camera_sensor, *args, **kwargs):
    self.cam = camera

    self.W = camera.W
    self.H = camera.H
    self.F = np.zeros((self.H, self.W)).astype(np.float32)

    self.rho = np.zeros((self.H, self.W))
    self.F_norm = np.zeros((self.H, self.W))
    self.center_threshold = 0.1
    self.K = 100
    self.bar_I = np.zeros(self.K, dtype=np.float32)
    self.rho_bins = np.zeros(self.K, dtype=np.float32)

    self.output_data = {
      "Vignette":{
        "Camera":{
          "name": self.cam.ref,
          "W": self.W,
          "H": self.H,
          "FPS": self.cam.fps,
          },
        "num_frames": 1,
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
        self.get_data()
        self.run_test()

        self.output_data["Vignette"]["K"] = self.K
        self.output_data["Vignette"]["center_threshold"] = self.center_threshold
        self.output_data["Vignette"]["rho_bins"] = self.rho_bins.tolist()
        self.output_data["Vignette"]["bar_I"] = self.bar_I.tolist()
        self.save_data()
        break
      elif key == ord('q') or key == ord('Q'):
        print("Test cancelled.")
        cv2.destroyAllWindows()
        break

  def get_data(self):
    ret, frame = self.cam.read()
    if not ret:
      print("⚠️ Failed to read frame from camera.")
      return
    self.F = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

  def optical_center(self):
    cx = (self.W - 1)/2
    cy = (self.H - 1)/2

    yy, xx = np.indices((self.H, self.W))

    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    rmax = np.sqrt(cx**2 + cy**2)

    self.rho = r / rmax

  def intensity_norm(self):
    c_mask = self.rho < self.center_threshold
    F_filtered = self.F[c_mask]
    if F_filtered.size == 0:
      raise ValueError("Center region mask is empty. Check center_threshold.")

    c_mean = float(np.mean(F_filtered))
    if c_mean <= 0:
      raise ValueError("Central mean intensity is non-positive. Check illumination / exposure.")

    self.F_norm = self.F / c_mean

  def radial_profile(self):
    bins = np.linspace(0, 1, self.K+1)

    rho_flat = self.rho.flatten()
    I_flat = self.F_norm.flatten()

    for k in range(self.K):
      mask = (rho_flat >= bins[k]) & (rho_flat < bins[k+1])
      if np.any(mask):
        self.bar_I[k] = np.mean(I_flat[mask])
        self.rho_bins[k] = 0.5 * (bins[k] + bins[k+1])
      else:
        self.bar_I[k] = np.nan
        self.rho_bins[k] = 0.5 * (bins[k] + bins[k+1])

  def vignette_model(self, r, c1):
    return np.cos(r * c1) ** 4

  def run_test(self):
    self.optical_center()
    self.intensity_norm()
    self.radial_profile()

    mask = ~np.isnan(self.bar_I)
    r = self.rho_bins[mask]
    I_rho = self.bar_I[mask]
    params, covariance = curve_fit(self.vignette_model, r, I_rho, p0=[1])
    c1 = params[0]

    I_edge = np.cos(c1) ** 4
    vignette_strength = float(1.0 - I_edge)
    if vignette_strength < 0:
      # Interpret: no detectable vignetting, probably flat sensor
      vignette_strength = 0.0

    now = datetime.now()
    self.output_data['Vignette']["test"] = {
          "time": now.strftime("%Y-%m-%d %H:%M:%S"),
          "brightness": self.cam.brightness,
          "vignette_strength": vignette_strength,
          "c1": float(c1),
          "covariance": covariance.tolist()
        }

  def save_data(self):
    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d%H%M%S")
    output_filename = f'data/Vignette_{self.cam.ref}_{formatted_datetime}.yaml'
    with open(output_filename, 'w') as file:
      yaml.dump(self.output_data, file, default_flow_style=False, sort_keys=False)

