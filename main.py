import argparse
import yaml
import cv2
from functions.camera_sensor import Camera_sensor
from functions.noise_characterization import Noise_characterization
from functions.response_function import Response_function

parser = argparse.ArgumentParser(prog='Camera_sensor_characterization',
                                 description='Script to run a selected group of tests for a specific camera.')

# script arguments
cameras = []
with open("config/lists.yml") as stream:
  lists = dict(yaml.safe_load(stream))['cameras']
  cameras = lists.keys()

parser.add_argument('--camera', '-c', metavar='<camera_ref>',
                    action='store', dest='camera_ref', choices=cameras,
                    required=True,
                    help='Type of camera you want to characterize.')

parser.add_argument('--test', '-t', metavar='<test_number>',
                    action='store', dest='test', choices=[1, 2, 3, 4, 5, 6],
                    required=True, type=int,
                    help='Number of tests you want to run.')

def test_directory(test_number):
  index = test_number-1
  func = [
    Noise_characterization,
    Response_function,
    # Vignetting,
    # Rolling_shutter_timing,
    # Chromatic_aberration,
    # MTF,
  ]
  return func[index]

if __name__=='__main__':
  args = parser.parse_args()

  # Create camera obj
  camera = Camera_sensor(ref=args.camera_ref)
  print(camera.W, camera.H, camera.fps, camera.brightness, 2 ** camera.exposure )

  test_cls = test_directory(args.test)
  params = test_cls(camera, exposure_list=[0.01,
                                           0.02, 0.03, 0.04, 0.05, 0.06, 0.07
                                           ])

  cv2.destroyAllWindows()

