import argparse
import yaml
import cv2
from functions.camera_sensor import Camera_sensor
from functions.noise_characterization import Noise_characterization
from functions.response_function import Response_function
from functions.vignette import Vignette

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
                    action='store', dest='camera_port', type=int,
                    help='Computer port where the camera is actually connected.')

parser.add_argument('--test', '-t', metavar='<test_number>',
                    action='store', dest='test', choices=[1, 2, 3, 4, 5, 6],
                    required=True, type=int,
                    help='Number of tests you want to run.')

def test_directory(test_number):
  index = test_number-1
  func = [
    Noise_characterization,
    Response_function,
    Vignette,
    # Rolling_shutter_timing,
    # Chromatic_aberration,
    # MTF,
  ]
  return func[index]

if __name__=='__main__':
  args = parser.parse_args()
  print(list_available_cameras())

  # Create camera obj
  print(args)
  camera = Camera_sensor(ref=args.camera_ref, port=args.camera_port)
  print(camera.W, camera.H, camera.fps, camera.brightness, 2 ** camera.exposure )

  test_cls = test_directory(args.test)
  params = test_cls(camera, exposure_list=[0.01,
                                           0.02, 0.03, 0.04, 0.05, 0.06, 0.07
                                           ])

  cv2.destroyAllWindows()

