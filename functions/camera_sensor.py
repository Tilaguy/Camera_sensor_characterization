import yaml
import cv2
from abc import ABC, abstractmethod
from typing import ClassVar
from datetime import datetime
from typing import ClassVar, Dict, Any, List
import os

class Camera_sensor(cv2.VideoCapture):
  '''Class to create a camera instance, connect it, and read the image in real time.'''
  def __init__(self, ref, port=0):
    self.port = port
    self.ref = ""
    self.resolution = []
    self.W = 0
    self.H = 0
    self.fps = 0
    self.brightness = 0
    self.exposure = 0

    self.__validate_reference(ref)

    if self.__validate_reference(ref):
      if self.ref == "IMX219":
        super().__init__(self.port)
        self.__configure_imx219()
      elif self.ref == "OAKD_RGB":
        self.__oakd_rgb_config()
      elif self.ref == "OAKD_DEPTH":
        self.__oakd_depth_config()
      else:
        super().__init__(self.port)
        self.__configure_webcam()
    else:
      super().__init__(self.port)

    if not self.isOpened():
      raise Exception("Error: Could not open camera.")
    else:
      # Configurar propiedades de la cámara
      self.__configure_camera_properties()
      print(f'{self.ref} is already connected...')

  def __validate_reference(self, ref):
    if ref=='WEBCAM':
      return True

    with open("config/lists.yml") as stream:
      lists = dict(yaml.safe_load(stream))['cameras']
      if ref in lists.keys():
        self.ref = ref
        self.resolution = lists[ref]['resolution']
        return True
      else:
        raise Exception("Camera not configured")

    return False

  def __configure_webcam(self):
    return cv2.VideoCapture(self.port)

  def __configure_imx219(self):
    # TODO: configuracion e inicializacion de camara RGB para captura de datos y ejecucion de test
    # delete
    return cv2.VideoCapture(self.port)

  def __oakd_rgb_config(self):
    # TODO: configuracion e inicializacion de camara RGB para captura de datos y ejecucion de test
    # delete
    return cv2.VideoCapture(self.port)

  def __oakd_depth_config(self):
    # TODO: configuracion e inicializacion de camara depth para captura de datos y ejecucion de test
    # delete
    return cv2.VideoCapture(self.port)

  def __configure_camera_properties(self):
    self.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode

    # Get frame width and height
    self.H = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.W = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Get Frames Per Second (FPS)
    self.fps = int(self.get(cv2.CAP_PROP_FPS))

    # Get Brightness (if supported by the camera)
    self.brightness = self.get(cv2.CAP_PROP_BRIGHTNESS)

    # Get exposure time (if supported by la cámara)
    self.exposure = self.get(cv2.CAP_PROP_EXPOSURE)



class Camera_test(ABC):
  _required_params: ClassVar[list] = ['camera', 'num_frames']

  def __init__(self, camera: Camera_sensor, num_frames: int):
        self.cam = camera
        self.N = num_frames
        self.test_name = self.__class__.__name__

        self.output_data: Dict[str, Any] = {
          "test_name": self.test_name,
          "Camera":{
            "name": self.cam.ref,
            "W": self.cam.W,
            "H": self.cam.H,
            "FPS": self.cam.fps,
            "brightness": self.cam.brightness,
            },
          "num_frames": self.N,
        }
        self._validate_params()

  def _validate_params(self) -> None:
        """Validate the common parameters"""
        if not isinstance(self.cam, Camera_sensor):
            raise TypeError("Camera must be an instance of Camera_sensor")
        if not isinstance(self.N, int) or self.N <= 0:
            raise ValueError("num_frames must be a positive integer")

  def save_data(self) -> None:
    '''It saves all the data used and processed while running, in a YAML file.'''
    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d%H%M%S")
    output_filename = f'data/{self.test_name}_{self.cam.ref}_{formatted_datetime}.yaml'

    os.makedirs('data', exist_ok=True)

    with open(output_filename, 'w') as file:
      yaml.dump(self.output_data, file, default_flow_style=False, sort_keys=False)

    print(f"Data saved to: {output_filename}")

  def add_to_output(self, key: str, value: Any):
    """Helper method to add data to output_data dictionary"""
    self.output_data[key] = value

  @abstractmethod
  def setup_configuration(self) -> bool:
    '''Wait until the user confirms that the experimental setup is ready to run the test.'''
    pass
  @abstractmethod
  def get_data(self) -> List:
    '''Read and return the current frame read by the camera.'''
    pass
  @abstractmethod
  def run_test(self) -> dict[str, Any]:
    '''Run the main test algorithm.'''
    pass
