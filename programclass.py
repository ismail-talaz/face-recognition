import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_facenet import FaceNet
import cv2
import time
import os

class Program():


    def __init__(self,facetracker,encoder):
        self.facetracker = facetracker
        self.encoder = encoder
    

    def img_to_encoding(self,image):
        if(isinstance(image,str)):
            img = tf.keras.preprocessing.image.load_img(image, target_size=(160, 160)) # from image path
        else:
            img = tf.image.resize(image, (160,160))  # from opencv 

        img = np.around(np.array(img) / 255.0, decimals=12)
        img = np.expand_dims(img, axis=0)
        embedding = self.encoder.predict(img)[0]
        return embedding / np.linalg.norm(embedding, ord=2)
    

    def capture(self):
        
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('output.avi', fourcc, 10.0, (450,450))

        while cap.isOpened():
            _ , frame = cap.read()
            frame = frame[50:500, 50:500,:]
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = tf.image.resize(rgb, (120,120))
            
            yhat = self.facetracker.predict(np.expand_dims(resized/255,0))
            sample_coords = yhat[1][0]

            cv2.rectangle(frame, 
                                tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                                tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                                        (255,0,0), 2)
            
            if yhat[0] > 0.5: 
                

                x1,y1 = np.multiply(sample_coords[:2], [450,450]).astype(int)
                x2,y2 = np.multiply(sample_coords[2:], [450,450]).astype(int)
                cropped = rgb[y1:y2,x1:x2]

                current_embedding = self.img_to_encoding(cropped)
                
                identity = self.traverseDatabase(current_embedding)

                if (identity != "None"):

                    
                    cv2.rectangle(frame, 
                                tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                                [0,-30])),
                                tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                                [80,0])), 
                                        (255,0,0), -1)
                    
                    cv2.putText(frame, f'{identity}', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                                        [0,-5])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.imshow('Camera', frame)
            frame = cv2.resize(frame,(450,450))
            out.write(frame)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    
    def traverseDatabase(self, embedding):
        
        detected = "None"
        mindist = 10000

        for identity in os.listdir('identities'):

            identity_embedding = self.img_to_encoding(os.path.join('identities',identity))
            current_dist = np.linalg.norm(embedding - identity_embedding)
            if( current_dist < 0.9 and mindist > current_dist):

                detected = identity
                mindist = current_dist

        return detected



