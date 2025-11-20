import cv2
import yaml
import numpy as np
from datetime import datetime

from functions.camera_sensor import Camera_sensor

class Noise_characterization():
  """This test characterizes the electronic noise of the camera by capturing multiple dark frames at different exposure times with the lens fully covered. The function computes per-pixel temporal statistics (mean and variance) and aggregates them into global metrics, including read noise and dark-signal non-uniformity (DSNU). By analyzing how dark variance behaves across exposure times, the function isolates fixed-pattern noise and temporal noise sources inherent to the sensor. This test provides the fundamental noise parameters required for realistic camera simulation and for correcting downstream flat-field measurements."""

  def __init__(self, camera:Camera_sensor, num_frames:int=100, exposure=10):
    self.cam = camera
    self.N = num_frames
    self.t = exposure
    self.D = np.array([])

    # Run test
    self.get_data()
    self.setup_configuration()

  def get_data(self):
    array = []
    for _ in range(self.N):
      ret, frame = self.cam.read()
      d = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      array.append(d)
    self.D = np.array(array)

  def setup_configuration(self):
    print('To perform this test, ensure the lens is completely covered.')
    print('Press [Y] when you are certain the entire image is black.')
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
        self.run_test()
        break
      elif key == ord('q') or key == ord('Q'):
        print("Test cancelled.")
        cv2.destroyAllWindows()
        break

  def run_test(self):
    # Pixel mean
    mu = np.mean(self.D, axis=0).astype(np.uint8)
    # Pixel variance
    var = np.var(self.D, axis=0).astype(np.uint8)
    # Global mean
    dark_mean = np.mean(mu)
    # Global variance
    dark_var = np.mean(var)
    print(f"⚠️\tThis information will be nesesary for futere tests.\n\t- dark_mean: {dark_mean}\n\t- dark_var: {dark_var}")

    # Read noise
    read_std_dn = np.sqrt(dark_var)
    # Darck Signal Non-Uniformity
    dsnu_dn = np.sqrt(np.mean((mu - dark_mean) ** 2))

    output_data = {
      "Noise_characterization":{
        "Camera":{
          "name": self.cam.ref,
          "W": self.cam.W,
          "H": self.cam.H,
          "FPS": self.cam.fps,
          "brightness": self.cam.brightness,
          },
        "test":{
          "exposure": self.cam.exposure,
          "dark_mean": float(dark_mean),
          "dark_var": float(dark_var),
          "read_std_dn": float(read_std_dn),
          "dsnu_dn": float(dsnu_dn),
        }
      }
    }

    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d%H%M%S")
    output_filename = f'data/Noise_characterization_{self.cam.ref}_{formatted_datetime}.yaml'
    with open(output_filename, 'w') as file:
      yaml.dump(output_data, file, default_flow_style=False, sort_keys=False)

