import yaml
import cv2

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

    if self.ref in ["IMX219", "OAKD_RGB", "OAKD_DEPTH"]:
      if self.ref == "IMX219":
        super().__init__(self.port)
        self.__configure_imx219()
      elif self.ref == "OAKD_RGB":
          self.__oakd_rgb_config()
      elif self.ref == "OAKD_DEPTH":
          self.__oakd_depth_config()
    else:
      super().__init__(self.port)

    if not self.isOpened():
      raise Exception("Error: Could not open camera.")
    else:
      # Configurar propiedades de la cámara
      self.__configure_camera_properties()
      print(f'{self.ref} is already connected...')

  def __validate_reference(self, ref):
    with open("config/lists.yml") as stream:
      lists = dict(yaml.safe_load(stream))['cameras']
      if ref in lists.keys():
        self.ref = ref
        self.resolution = lists[ref]['resolution']
      else:
        raise Exception("Camera not configured")

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
