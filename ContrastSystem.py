import skfuzzy as skfuzz
from skfuzzy import control as ctrl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import random

matplotlib.use("GTK3Agg")

IMAGE_FILES = [
            "./Resource/enishtein.png",
            "./Resource/earth.jpeg",
            "./Resource/lake.jpg",
            "./Resource/lena.jpg",
            "./Resource/woman.jpeg",]



class ContrastEnhancemer:

    def __init__(self):
        self.image = cv2.imread(IMAGE_FILES[random.randint(0,len(IMAGE_FILES)-1)])
        self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)


    def load_image(self, filename:str=None):
        if not filename is None:
            self.image = cv2.imread(filename)
            self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)

    def Init_Input(self): 
        self.pixel = ctrl.Antecedent(np.arange(0,256,1),'pixel')
        self.new_pixel = ctrl.Consequent(np.arange(-50,51,1),'new_pixel')
        self.pixel['vdark'] = skfuzz.trapmf(self.pixel.universe,[0,0,20,45])
        self.pixel['dark'] = skfuzz.gaussmf(self.pixel.universe,50,10)
        self.pixel['DG'] = skfuzz.gaussmf(self.pixel.universe,80,14)
        self.pixel['gray'] = skfuzz.gaussmf(self.pixel.universe,120,6)
        self.pixel['LG'] = skfuzz.gaussmf(self.pixel.universe,150,14)
        self.pixel['bright'] = skfuzz.gaussmf(self.pixel.universe,180,10)
        self.pixel['vbright'] = skfuzz.trapmf(self.pixel.universe,[200,225,255,255])
        
        self.new_pixel['VD'] = skfuzz.gaussmf(self.new_pixel.universe,-50,6)
        self.new_pixel['SD'] = skfuzz.gaussmf(self.new_pixel.universe,-30,8)
        self.new_pixel['NC'] = skfuzz.gaussmf(self.new_pixel.universe,0,4)
        self.new_pixel['SB'] = skfuzz.gaussmf(self.new_pixel.universe,15,5)
        self.new_pixel['VB'] = skfuzz.gaussmf(self.new_pixel.universe,50,6)
    
