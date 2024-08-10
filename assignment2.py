import cv2
import time
import modi_plus.module
from modi_plus.module.input_module import env, imu
from modi_plus.module.output_module import speaker, display, led

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

def make_image():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 1)

    ret, frame = cam.read()

    for _ in range(10):
        ret, frame = cam.read()

    cv2.imwrite('baby.jpg', frame)

    cam.release()
    cv2.destroyAllWindows()

def nature_number(a: int):
    return not a if a < 0 else a

np.set_printoptions(suppress=True)
model = load_model("./baby_keras/keras_Model.h5", compile=False)
class_names = open("./baby_keras/labels.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

bundle = modi_plus.MODIPlus()
bundle.modules

envi: env.Env = bundle.envs[0]
imuu: imu.Imu = bundle.imus[0]
speak: speaker.Speaker = bundle.speakers[0]
disp: display.Display = bundle.displays[0]
leds: led.Led = bundle.leds[0]

while True:
    print("습도:", envi.humidity, "%")
    print("Y 각도:", imuu.pitch)
    if envi.humidity >= 85 and (nature_number(imuu.pitch) <= 140 or nature_number(imuu.roll) <= 140):
        make_image()

        image = Image.open("baby.jpg").convert("RGB")

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        image_array = np.asarray(image)

        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        print(class_name[2:])

        if "1" in class_name[2:]:
            disp.write_text("아이가 위험합니다")
            leds.turn_on()
            for _ in range(10):
                speak.play_music('Warning 2', 100)
                time.sleep(3)
            time.sleep(60)
    time.sleep(0.5)
