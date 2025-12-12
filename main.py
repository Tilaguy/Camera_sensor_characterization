import argparse
import yaml
import cv2
from functions.camera_sensor import Camera_sensor, Camera_test
from functions.noise_characterization import Noise_characterization
from functions.response_function import Response_function
from functions.vignette import Vignette
from functions.rolling_shutter import Rolling_shutter
from typing import Callable, Dict, Any

parser = argparse.ArgumentParser(prog='Camera_sensor_characterization',
                                 description='Script to run a selected group of tests for a specific camera.')

# script arguments
def list_available_cameras():
  """
  Tests potential camera device indices and returns a list of working camera indices.
  """
  available_cameras = []
  index = 0
  while True:
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:  # Attempt to read a frame
      cap.release()
      break  # No more cameras found
    else:
      available_cameras.append(index)
      cap.release()  # Release the camera after checking
      index += 1
  return available_cameras

cameras = []
with open("config/lists.yml") as stream:
  lists = dict(yaml.safe_load(stream))['cameras']
  cameras = list(lists.keys())
  cameras.append('WEBCAM')

parser.add_argument('--camera', '-c', metavar='<camera_ref>',
                    action='store', dest='camera_ref', choices=cameras,
                    required=True,
                    help='Type of camera you want to characterize.')
parser.add_argument('--port', '-p', metavar='<camera_port>', default=0,
                    action='store', dest='camera_port', choices=list_available_cameras(), type=int,
                    help='Computer port where the camera is actually connected.')

parser.add_argument('--test', '-t', metavar='<test_number>',
                    action='store', dest='test', choices=[1, 2, 3, 4, 5, 6],
                    required=True, type=int,
                    help='Number of tests you want to run.')

def create_test(test_number: int, camera: Camera_sensor, **kwargs) -> Camera_test:
  """
  Simple factory that creates test instances.
  Handles different parameter requirements automatically.
  """
  test_map = {
    1: (Noise_characterization, ['camera', 'num_frames', 'exposure_list']),
    2: (Response_function, ['camera', 'num_frames', 'dark_var', 'exposure_list']),
    3: (Vignette, ['camera', 'num_frames']),
    4: (Rolling_shutter, ['camera', 'num_frames', 'speed_px', 'screen_hz']),
    # 5: (Chromatic_aberration),
    # 6: (MTF),
    }
  TestClass, required_params = test_map[test_number]

  test_kwargs:Dict[str, Any] = {'camera': camera}
  for param in required_params:
    if param == 'camera':
      continue
    if param in kwargs:
      test_kwargs[param] = kwargs[param]
    elif param == 'exposure_list':
      test_kwargs[param] = []
    elif param == 'dark_var':
      test_kwargs[param] = 0
    elif param == 'num_frames':
      test_kwargs[param] = kwargs.get('num_frames', 100)
    elif param == 'speed_px':
      test_kwargs[param] = 10
    elif param == 'screen_hz':
      test_kwargs[param] = 60

  return TestClass(**test_kwargs)

if __name__=='__main__':
  args = parser.parse_args()

  # Create camera obj
  camera = Camera_sensor(ref=args.camera_ref, port=args.camera_port)

  test = create_test(args.test, camera, num_frames=100,
                     exposure_list=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                     dark_var=0.22906899452209473,
                     speed_px=10, screen_hz=144,
                     )

  if test.setup_configuration():
    test.run_test()

