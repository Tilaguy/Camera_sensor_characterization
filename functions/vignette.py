import cv2
import yaml
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

from functions.camera_sensor import Camera_sensor, Camera_test

class Vignette(Camera_test):
  """
  This test estimates optical vignetting by analyzing a uniformly illuminated flat-field image. It identifies the optical center, computes each pixel's radial distance, and normalizes intensities using a central reference region. Radial bins are formed to create a smooth radial brightness profile, which is then fitted to a cos⁴-based vignetting model. The resulting parameters describe how image brightness decreases from center to corners, and a scalar vignette_strength is derived to quantify the falloff. This function provides the optical shading characteristics required for rendering realistic peripheral darkening in simulated camera images.
  """

  def __init__(self, camera:Camera_sensor, num_frames:int=100, *args, **kwargs):
    super().__init__(camera, num_frames)
    self.W = camera.W
    self.H = camera.H
    self.F = np.zeros((self.H, self.W)).astype(np.float32)

    self.rho = np.zeros((self.H, self.W))
    self.F_norm = np.zeros((self.H, self.W))
    self.center_threshold = 0.1
    self.K = 100

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
    ret, frame = self.cam.read()
    if not ret:
      print("⚠️ Failed to read frame from camera.")
      return []

    return [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)]

  def __optical_center(self):
    cx = (self.W - 1)/2
    cy = (self.H - 1)/2

    yy, xx = np.indices((self.H, self.W))

    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    rmax = np.sqrt(cx**2 + cy**2)

    return r / rmax

  def __intensity_norm(self, rho, frame):
    c_mask = rho < self.center_threshold
    F_filtered = frame[c_mask]
    if F_filtered.size == 0:
      raise ValueError("Center region mask is empty. Check center_threshold.")

    c_mean = float(np.mean(F_filtered))
    if c_mean <= 0:
      raise ValueError("Central mean intensity is non-positive. Check illumination / exposure.")

    return frame / c_mean

  def __radial_profile(self, rho, frame_norm):
    bins = np.linspace(0, 1, self.K + 1)
    bar_I = np.zeros(self.K, dtype=np.float32)
    rho_bins = np.zeros(self.K, dtype=np.float32)

    rho_flat = rho.flatten()
    I_flat = frame_norm.flatten()

    for k in range(self.K):
      mask = (rho_flat >= bins[k]) & (rho_flat < bins[k+1])
      if np.any(mask):
        bar_I[k] = np.mean(I_flat[mask])
        rho_bins[k] = 0.5 * (bins[k] + bins[k+1])
      else:
        bar_I[k] = np.nan
        rho_bins[k] = 0.5 * (bins[k] + bins[k+1])

    return bar_I, rho_bins

  def __vignette_model(self, r, c1):
    return np.cos(r * c1) ** 4

  def run_test(self):
    vignette_strength_array = []
    rho = self.__optical_center()

    for attempt in range(self.N):
      print(f"Getting {attempt + 1}/{self.N} frame")
      f = self.get_data()[0]
      f_norm = self.__intensity_norm(rho, f)
      bar_I, rho_bins = self.__radial_profile(rho, f_norm)

      mask = ~np.isnan(bar_I)
      r = rho_bins[mask]
      I_rho = bar_I[mask]
      params, covariance = curve_fit(self.__vignette_model, r, I_rho, p0=[1])
      c1 = params[0]

      I_edge = np.cos(c1) ** 4
      vignette_strength = float(1.0 - I_edge)
      if vignette_strength < 0:
        # Interpret: no detectable vignetting, probably flat sensor
        vignette_strength = 0.0

      now = datetime.now()
      self.add_to_output(f"test_{attempt}", {
          "time": now.strftime("%Y-%m-%d %H:%M:%S"),
          "brightness": self.cam.brightness,
          "vignette_strength": float(vignette_strength),
          "c1": float(c1),
          "covariance": covariance.tolist()
        })
      vignette_strength_array.append(vignette_strength)

    vignette_strength_mean = np.mean(np.array(vignette_strength_array))
    self.add_to_output("results",{
      "vignette_strength": float(vignette_strength_mean),
    })
    self.save_data()
