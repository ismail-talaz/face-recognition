from programclass import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

encoder = FaceNet().model
facetracker = load_model('facedetector-guzelkv4.h5')

newinstance = Program(facetracker,encoder)

newinstance.capture()
