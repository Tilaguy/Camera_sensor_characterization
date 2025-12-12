import screeninfo
import cv2
import yaml
import numpy as np
from datetime import datetime
import time
from matplotlib import pyplot as plt

from functions.camera_sensor import Camera_sensor, Camera_test

class Rolling_shutter(Camera_test):
  '''
  This test measures the rolling-shutter line time by capturing an image of a vertically moving high-contrast bar with known motion speed. Due to row-wise exposure delays, the bar appears tilted in the captured frame. The function detects the bar’s edge, extracts points along it, and fits a linear model to estimate its slope. Given the known pattern speed, the slope converts directly into the line_time_us parameter, representing the temporal delay between consecutive rows. This test is essential for simulating rolling-shutter distortions and reproducing realistic motion-dependent geometric artifacts.
  '''

  def __init__(self, camera: Camera_sensor, num_frames:int=100,
              speed_px: int = 5, aux_screen_hz: float = 144.0,
              *args, **kwargs):
    super().__init__(camera, num_frames)
    self.W = camera.W
    self.H = camera.H
    self.speed_px = speed_px

    self.static_slope = 0.0

    self.window_name = "Rolling Shutter Test - MOVING BAR"
    self.aux_screen_hz = aux_screen_hz
    self.aux_screen_w:int = 0
    self.aux_screen_h:int = 0
    self.__select_auxiliar_screen()

    self.bar_width = 30

  def __select_auxiliar_screen(self):
    screens = screeninfo.get_monitors()
    screen = max(screens, key=lambda s: s.height)

    self.aux_screen_w = screen.width
    self.aux_screen_h = screen.height

    # Create window for animation
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(self.window_name, screen.x, screen.y)
    cv2.setWindowProperty(self.window_name,
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

  def __create_calibration_pattern(self):
    static_img = np.ones((self.aux_screen_h, self.aux_screen_w), dtype=np.uint8) * 255
    center_x = self.aux_screen_w // 2
    cv2.rectangle(static_img,
                  (center_x - self.bar_width//2, 0),
                  (center_x + self.bar_width//2, self.aux_screen_h),
                  (0, 0, 0), -1)
    return static_img

  def __clean_img(self, frame, debug:bool = False):
    frame_u8 = frame.astype(np.uint8)
    if debug:
      cv2.imshow("frame_u8", frame_u8)
      cv2.waitKey(0)

      hist, bins = np.histogram(frame_u8.flatten(), 256, [0,256]) # type: ignore
      cdf = hist.cumsum()
      cdf_normalized = cdf * float(hist.max()) / cdf.max()
      plt.plot(cdf_normalized, color = 'b')
      plt.hist(frame_u8.flatten(), 256, [0, 256], color = 'r') # type: ignore
      plt.show()

  #   # 1. Enhance contrast
    kernel = np.ones((15, 15), np.uint8)
    erosion = cv2.erode(frame_u8, kernel, iterations = 1)
    if debug:
      cv2.imshow("erosion", erosion)
      cv2.waitKey(0)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    if debug:
      cv2.imshow("opening", opening)
      cv2.waitKey(0)

      hist, bins = np.histogram(opening.flatten(), 256, [0, 256]) # type: ignore
      cdf = hist.cumsum()
      cdf_normalized = cdf * float(hist.max()) / cdf.max()
      plt.plot(cdf_normalized, color = 'b')
      plt.hist(opening.flatten(), 256, [0, 256], color = 'r') # type: ignore
      plt.show()

    # 3. Threshold to get strong edges
    _, edges = cv2.threshold(opening, 40, 255, cv2.THRESH_BINARY)
    if debug:
      cv2.imshow("edges", edges)
      cv2.waitKey(0)

    # 2. Detect edges with Sobel (vertical edges)
    grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
    if debug:
      cv2.imshow("grad_x", grad_x)
      cv2.waitKey(0)

      hist, bins = np.histogram(grad_x.flatten(), 256, [0, 256]) # type: ignore
      cdf = hist.cumsum()
      cdf_normalized = cdf * float(hist.max()) / cdf.max()
      plt.plot(cdf_normalized, color = 'b')
      plt.hist(grad_x.flatten(), 256, [0, 256], color = 'r') # type: ignore
      plt.show()

    grad_pos = np.zeros_like(grad_x, dtype=np.uint8)
    grad_pos[grad_x > 10] = 255
    if debug:
      cv2.imshow("Positive edges (right)", grad_pos)
      cv2.waitKey(0)

    # grad_x_abs = cv2.convertScaleAbs(grad_x)
    # if debug:
    #   cv2.imshow("grad_x_abs", grad_x_abs)
    #   cv2.waitKey(0)

    return grad_pos

  def __measure_slope_from_frames(self, frames):
    """Measure slope from frames using robust method"""
    slopes = []
    for frame in frames:
      grad_x_abs = self.__clean_img(frame)

      # 4. Find contours
      contours, _ = cv2.findContours(grad_x_abs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      if contours:
        # Find largest contour (should be our bar)
        largest_contour = max(contours, key=cv2.contourArea)

        # Fit line to contour
        if len(largest_contour) >= 2:
          [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
          if abs(vy) > 0.001:  # Avoid division by zero
            slope = vx[0] / vy[0] # type: ignore
            slopes.append(slope)

    if slopes:
        return float(np.median(slopes)), slopes
    return 0.0, []

  def __camera_calibration(self):
    print("Capturing static bar for calibration...")
    static_frames = []
    for _ in range(30):
      ret, frame = self.cam.read()
      if not ret:
        break

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      static_frames.append(gray)

    if not static_frames:
      print("ERROR: Could not capture static frames")
      return None

    print("Measuring geometric distortion...")
    slope, _ = self.__measure_slope_from_frames(static_frames)
    self.static_slope = slope
    print(f"Geometric slope: {self.static_slope:.6f}\n")

  def setup_configuration(self):
    print("\nINSTRUCTIONS:")
    print("This test requires a pre-calibration routine to correct any inconsistencies or errors in the experimental setup.")
    print("⚠️\tOnce the calibration routine has started, do not move the camera until the test is completely finished.")
    print('Press [Y] to start calibration.')
    print('Press [Q] to quit.')

    static_img = self.__create_calibration_pattern()
    isReady:bool = False
    while True:
      cv2.imshow("Rolling Shutter Test - MOVING BAR", static_img)

      ret, frame = self.cam.read()
      if not ret:
        break
      cv2.imshow('Camera Feed - Press Y when ready, Q to quit', frame)

      key = cv2.waitKey(1) & 0xFF

      if key == ord('y') or key == ord('Y'):
        self.__camera_calibration()
        print("\nThe test is starting,\n⚠️\tDo not move the camera or experimental setup until the test is finished.")
        isReady = True
        break
      elif key == ord('q') or key == ord('Q'):
        print("\nTest cancelled.")
        break

    cv2.destroyAllWindows()
    return isReady

  def __create_moving_bar_animation(self, x_position:int = 0):
    moving_img = np.ones((self.aux_screen_h, self.aux_screen_w), dtype=np.uint8) * 255
    current_x = int(x_position) % self.aux_screen_w
    cv2.rectangle(moving_img,
                  (current_x, 0),
                  (current_x + self.bar_width, self.aux_screen_h),
                  (0, 0, 0), -1)
    return moving_img

  def __measure_velocity_simple(self, frames):
    """Simple but robust velocity measurement"""
    if len(frames) < 2:
      return 0.0, []

    # Track bar position using intensity centroid
    positions = []

    for frame in frames:
      frame_u8 = self.__clean_img(frame)

      contours, _ = cv2.findContours(frame_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      if contours:
        # Find largest contour (should be our bar)
        largest_contour = max(contours, key=cv2.contourArea)

        # Fit line to contour
        if len(largest_contour) >= 2:
          [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
          vx = float(vx[0])   # type: ignore
          vy = float(vy[0])   # type: ignore
          x = float(x[0])   # type: ignore
          y = float(y[0])   # type: ignore

          if abs(vy) > 0.001:  # Avoid division by zero
            slope = vx / vy # type: ignore
            if np.abs(slope) <= np.abs(self.static_slope + 0.5):
              height, width = frame_u8.shape[:2]
              y_fixed = height // 2  # Middle of image
              x_at_fixed_y = x + (y_fixed - y) * (vx / vy) if vy != 0 else x
              positions.append(x_at_fixed_y)

              # # DELETE show centroid of contour
              # frame_color = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
              # cv2.drawContours(frame_color, [largest_contour], -1, (0, 255, 0), 2)
              # cv2.circle(frame_color, (int(x), int(y)), 5, (255, 0, 0), -1)
              # cv2.circle(frame_color, (int(x_at_fixed_y), int(y_fixed)), 5, (0, 0, 255), -1)
              # cv2.imshow("Fitted Line", frame_color)
              # cv2.waitKey(0)

    if len(positions) < 2:
        return 0.0, []

    # Calculate displacements
    positions = np.array(positions)
    filtered_positions = [positions[0] if positions[0] > 0 else 0.0]
    for i in range(1, len(positions)):
      if positions[i] >= filtered_positions[-1] and positions[i] > 0:
        filtered_positions.append(positions[i])

    displacements = np.diff(filtered_positions)

    median_disp = np.median(displacements)
    valid_mask = np.abs(displacements - median_disp) < self.bar_width

    if np.any(valid_mask):
      valid_displacements = displacements[valid_mask]
      median_dx = float(np.median(valid_displacements))
      return median_dx, valid_displacements.tolist()

    return 0.0, []

  def save_results(self, real_fps, dx_per_frame, velocity_px_per_s,
                  moving_slope, static_slope, corrected_slope,
                  line_time_us, frame_time, max_line_time_us,
                  displacements):
      """Save results to YAML file"""
      results = {
          "Rolling_shutter_test": {
              "camera": {
                  "name": self.cam.ref,
                  "resolution": f"{self.W}x{self.H}",
                  "nominal_fps": self.cam.fps,
                  "measured_fps": float(real_fps),
              },
              "test_parameters": {
                  "screen_hz": float(self.aux_screen_hz),
                  "bar_speed_px_per_frame": self.speed_px,
                  "num_frames": self.N,
              },
              "measurements": {
                  "dx_per_frame_camera_px": float(dx_per_frame),
                  "velocity_camera_px_per_s": float(velocity_px_per_s),
                  "slope_moving": float(moving_slope),
                  "slope_static": float(static_slope),
                  "slope_corrected": float(corrected_slope),
              },
              "results": {
                  "line_time_us": float(line_time_us),
                  "frame_period_ms": float(frame_time * 1000),
                  "max_line_time_us_from_fps": float(max_line_time_us),
                  "lines_per_frame_calculated": float(frame_time / (line_time_us * 1e-6)) if line_time_us > 0 else 0,
                  "lines_per_frame_nominal": self.H,
              },
              "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          }
      }

      if displacements:
          results["Rolling_shutter_test"]["displacement_stats"] = {
              "count": len(displacements),
              "mean": float(np.mean(displacements)),
              "median": float(np.median(displacements)),
              "std": float(np.std(displacements)),
              "min": float(np.min(displacements)),
              "max": float(np.max(displacements)),
          }

      # Save to file
      timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
      filename = f"rolling_shutter_results_{timestamp}.yaml"

      with open(filename, 'w') as f:
          yaml.dump(results, f, default_flow_style=False)

      print(f"\nResults saved to: {filename}")

  def get_data(self):
    ret, frame = self.cam.read()
    if not ret:
      print("⚠️ Failed to read frame from camera.")
      return []

    return [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)]

  def run_test(self):
    """Main test execution"""
    # Capture frames
    print(f"\nBar speed: {self.speed_px} px/frame = {self.speed_px * self.aux_screen_hz} px/s")
    F = []
    x_position = 0

    moving_img = self.__create_moving_bar_animation(x_position)
    cv2.imshow("Rolling Shutter Test - MOVING BAR", moving_img)
    cv2.waitKey(1)
    time.sleep(1)
    while True:
      moving_img = self.__create_moving_bar_animation(x_position)
      cv2.imshow("Rolling Shutter Test - MOVING BAR", moving_img)
      cv2.waitKey(1)

      F.append(self.get_data()[0])

      if len(F) == self.N:
        break
      x_position += self.speed_px

    # Analyze and compute
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # 1. Measure actual FPS
    print("\n1. Measuring camera FPS...")
    test_frames = 30
    start = time.time()
    for _ in range(test_frames):
        self.cam.read()
    end = time.time()
    real_fps = test_frames / (end - start)
    print(f"   Nominal FPS: {self.cam.fps}")
    print(f"   Measured FPS: {real_fps:.2f}")

    # 2. Measure velocity from moving frames
    print("\n2. Measuring bar velocity...")
    dx_per_frame, displacements = self.__measure_velocity_simple(F)

    if dx_per_frame == 0 and displacements:
        dx_per_frame = np.mean(displacements) # type: ignore

    velocity_px_per_s = dx_per_frame * real_fps

    print(f"   Displacement per frame: {dx_per_frame:.3f} camera-px")
    print(f"   Velocity: {velocity_px_per_s:.2f} camera-px/s")

    if len(displacements) > 0: # type: ignore
      print(f"   Valid measurements: {len(displacements)}") # type: ignore
      print(f"   Displacement range: [{min(displacements):.2f}, {max(displacements):.2f}]") # type: ignore

      # 3. Measure slope from moving frames
      print("\n3. Measuring slope from moving bar...")
      _, moving_slopes = self.__measure_slope_from_frames(F)
      corrected_slopes = np.abs(np.array(moving_slopes, dtype=float) - self.static_slope)

      # print(f"   Moving bar slope: {moving_slope:.6f} px/row")
      print(f"   Static bar slope: {self.static_slope:.6f} px/row")
      # print(f"   Corrected slope: {corrected_slope:.6f} px/row")

      # 5. Theoretical limits
      frame_time = 1.0 / real_fps
      max_line_time_us = (frame_time / self.H) * 1e6
      print(f"\n4. Theoretical limits:")
      print(f"   Theoretical max: {max_line_time_us:.2f} µs")

      print("\n5. Calculating line time...")
      if abs(velocity_px_per_s) > 1e-6:
          line_time_s = corrected_slopes / abs(velocity_px_per_s)
          filtered_line_time_s = []
          for t in line_time_s:
            if t < max_line_time_us:
              filtered_line_time_s.append(t)
          filtered_line_time_s = np.array(filtered_line_time_s)
          line_time_us = np.median(filtered_line_time_s) * 1e6
      else:
          line_time_us = 0
          print("   WARNING: Could not calculate line time (zero velocity or slope)")

      print(f"   line time: {line_time_us:.2f} µs")

      if line_time_us > 0:
          ratio = line_time_us / max_line_time_us
          print(f"   Ratio: {ratio:.3f}")

          if ratio > 1.5:
              print(f"   WARNING: Line time exceeds theoretical maximum!")
          elif ratio < 0.1:
              print(f"   WARNING: Line time suspiciously small!")
          else:
              print(f"   ✓ Result looks reasonable")

      # 6. Lines per frame calculation
      lines_per_frame_actual = frame_time / (line_time_us * 1e-6) if line_time_us > 0 else 0

      print(f"\n6. Additional info:")
      print(f"   Frame period: {frame_time*1000:.2f} ms")
      print(f"   Lines per frame (calculated): {lines_per_frame_actual:.0f}")
      print(f"   Lines per frame (nominal): {self.H}")

    self.add_to_output("results",{
      "real_fps": float(real_fps),
      "static_slope": float(self.static_slope),
      "velocity_px_per_s": float(velocity_px_per_s),
      "moving_slopes": np.array(moving_slopes, dtype=float).tolist(), # type: ignore
      "max_line_time_us": float(max_line_time_us),
      "line_time_us": float(line_time_us),
    })

    self.save_data()
