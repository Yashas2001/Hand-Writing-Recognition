import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255, 0, 0)
IMAGESAVE = False
MODEL = load_model('bestmodel.h5')

LABELS = {0:'zero', 1:'one', 
        2:'two', 3:'three',     
        4:'four', 5:'five',     
        6:'six', 7:'seven', 
        8:'eight', 9:'nine'}

pygame.init()

FONT = pygame.font.Font('freesansbold.ttf', 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
number_xcord = []
number_ycord = []

pygame.display.set_caption('HandWriting Recognition')

iswriting = False
PREDICT = True

img_cnt = 1

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit() 
        
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)
        
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x , rect_max_x = max(number_xcord[0]-BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1]+BOUNDRYINC)
            rect_min_y , rect_max_y = max(number_ycord[0]-BOUNDRYINC,0), min(number_ycord[-1]+BOUNDRYINC, WINDOWSIZEY)
            
            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x : rect_max_x, rect_min_y: rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                img_cnt += 1

            if PREDICT:
                img = cv2.resize(img_arr, (28,28))
                img = np.pad(img, (10,10), 'constant', constant_values= 0)
                img = cv2.resize(img, (28,28))/255
                label = str(LABELS[np.argmax(MODEL.predict(img.reshape(1, 28, 28, 1)))])

                txt_sur = FONT.render(label, True, RED, WHITE)
                txt_recobj = txt_sur.get_rect()
                txt_recobj.left, txt_recobj.bottom = rect_min_x, rect_min_y

                DISPLAYSURF.blit(txt_sur, txt_recobj)

            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    DISPLAYSURF.fill(BLACK)

        pygame.display.update()