import skfuzzy as skfuzz
from skfuzzy import control as ctrl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import random

matplotlib.use("GTK3Agg")

IMAGE_FILES = [
    "./Resource/einstein.png",
    "./Resource/earth.jpeg",
    "./Resource/lena.jpg",
    "./Resource/woman.jpeg",
]


class ContrastEnhancemer:
    def __init__(self):
        self.image = cv2.imread(IMAGE_FILES[random.randint(0, len(IMAGE_FILES) - 1)])
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.Init_Input()
        self.InitRules()

    def load_image(self, filename: str = None):
        if not filename is None:
            self.image = cv2.imread(filename)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def Init_Input(self):
        self.pixel = ctrl.Antecedent(np.arange(0, 256, 1), "pixel")
        self.new_pixel = ctrl.Consequent(np.arange(-50, 51, 1), "new_pixel")
        self.pixel["vdark"] = skfuzz.trapmf(self.pixel.universe, [0, 0, 20, 45])
        self.pixel["dark"] = skfuzz.gaussmf(self.pixel.universe, 50, 10)
        self.pixel["DG"] = skfuzz.gaussmf(self.pixel.universe, 80, 14)
        self.pixel["gray"] = skfuzz.gaussmf(self.pixel.universe, 120, 6)
        self.pixel["LG"] = skfuzz.gaussmf(self.pixel.universe, 150, 14)
        self.pixel["bright"] = skfuzz.gaussmf(self.pixel.universe, 180, 10)
        self.pixel["vbright"] = skfuzz.trapmf(self.pixel.universe, [200, 225, 255, 255])

        self.new_pixel["VD"] = skfuzz.gaussmf(self.new_pixel.universe, -50, 6)
        self.new_pixel["SD"] = skfuzz.gaussmf(self.new_pixel.universe, -30, 8)
        self.new_pixel["NC"] = skfuzz.gaussmf(self.new_pixel.universe, 0, 4)
        self.new_pixel["SB"] = skfuzz.gaussmf(self.new_pixel.universe, 15, 5)
        self.new_pixel["VB"] = skfuzz.gaussmf(self.new_pixel.universe, 50, 6)

    def InitRules(self):
        r1 = ctrl.Rule(self.pixel["vdark"], self.new_pixel["SD"])
        r2 = ctrl.Rule(self.pixel["DG"], self.new_pixel["SD"])
        r3 = ctrl.Rule(self.pixel["gray"], self.new_pixel["SD"])
        r4 = ctrl.Rule(self.pixel["bright"], self.new_pixel["SB"])
        r5 = ctrl.Rule(self.pixel["dark"], self.new_pixel["VD"])
        r6 = ctrl.Rule(self.pixel["vbright"], self.new_pixel["NC"])
        r7 = ctrl.Rule(self.pixel["LG"], self.new_pixel["SD"])
        self.ContrastEnhancementSystem = ctrl.ControlSystem(
            [r1, r2, r3, r4, r5, r6, r7]
        )
        self.ContrastEnhancementSystemSimulation = ctrl.ControlSystemSimulation(
            self.ContrastEnhancementSystem
        )
    
    def compute(self,value):
        self.ContrastEnhancementSystemSimulation.input['pixel'] = value
        self.ContrastEnhancementSystemSimulation.compute()
        return self.ContrastEnhancementSystemSimulation.output['new_pixel']
        
    def show_pltots(self):
        self.pixel.view()
        self.new_pixel.view()
        plt.show()
    
    def clip(self,value):
        if value>=255:return 255
        elif value<=0:return 0
        return value
    
    def apply(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.new_image = np.zeros(gray_image.shape)        
        for i in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1]):
                new_value = self.compute(gray_image[i,j])
                clip_val = self.clip(gray_image[i,j] + new_value)
                self.new_image[i,j] = clip_val

    def show(self):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(self.image)
        plt.title("Before")
        plt.subplot(1,2,2)
        plt.imshow(self.new_image,'gray')
        plt.title("After")
        plt.show()