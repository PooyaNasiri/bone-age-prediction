from scipy import ndimage
import mediapipe as mp
import numpy as np
import random
import time
import sys
import cv2
import os
import platform


def clahe3(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def handRec(img):
    global suc
    result = mp.solutions.hands.Hands().process(img)
    h, w, c = img.shape
    hand_landmarks = result.multi_hand_landmarks
    x_max = 0
    x_min = w
    y_min = h
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
            # cv2.rectangle(img, (x_min, y_min), (x_max, h), (0, 255, 0), 2)
            # mp.solutions.drawing_utils.draw_landmarks(img, handLMs, mphands.HAND_CONNECTIONS)
        offset = int((h * offset_percent) / 100)
        y_min_new = y_min - offset
        x_min_new = x_min - offset
        x_max_new = x_max + offset
        # pad = int(
        #     (abs(y_min_new - h) - abs(x_min_new - x_max_new)) / 2)
        # if (pad > 0):
        #     x_min_new -= pad
        #     x_max_new += pad
        # if (pad < 0):
        #     y_min_new -= pad
        if (y_min_new < 0):
            y_min_new = 0
        if (x_min_new < 0):
            x_min_new = 0
        if (x_max_new > w):
            x_max_new = w
        # print(f"%s) offest: %s  | crop(y1,y2,x1,x2): %s, %s, %s, %s "%(count, offset, y_min_new, y_max_new, x_min_new, x_max_new))
        suc += 1
        return img[y_min_new:h, x_min_new:x_max_new]
    else:
        # print(f"%s) coudln't find the hand!"%count)
        return img


def rescale(img):
    height, width = img.shape[:2]
    max_dim = max(height, width)
    square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    x_pad = (max_dim - width) // 2
    y_pad = (max_dim - height) // 2
    square_img[y_pad:y_pad + height, x_pad:x_pad + width] = img
    resized = cv2.resize(square_img, (output_size, output_size),
                         interpolation=cv2.INTER_AREA)
    return resized


def getx(img):
    w, h, c = img.shape
    result = mp.solutions.hands.Hands().process(img)
    mx, wx = 0, 0
    if (result.multi_hand_landmarks):
        for hand_lm in result.multi_hand_landmarks:
            mx = hand_lm.landmark[
                mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x * w
            wx = hand_lm.landmark[mp.solutions.hands.HandLandmark.WRIST].x * w
    return wx, mx


def rotate(img):
    w, h, c = img.shape
    x1, x2 = getx(img)
    while (abs(x1 - x2) > ((w * rotation_percent) / 100)):
        img = ndimage.rotate(img, 2 if x1 < x2 else -2, reshape=True)
        x1, x2 = getx(img)
    return img


def flipAndRotate(img):
    if (random.random() > 0.5): img = cv2.flip(img, 1)
    if (random.random() > 0.5): img = cv2.flip(img, -1)
    rnd = random.random()
    if rnd < 0.25: img = cv2.rotate(img, cv2.ROTATE_180)
    elif rnd < 0.5: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rnd < 0.75: img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _print():
    print("%c[%d;%df" % (0x1B, 0, 0), end='')
    print(
        "PREPROCESS ON HAND X-RAY IMAGES  |  2023"
        "\nInput directory:  ", image_dir, "\nOutput directory: ", output_dir,
        "\nOutput Size: ", output_size,
        ("px | 3" if output_channel == 1 else "px | 1"), "Output channels | ",
        ("With augmentations" if data_type == 1 else "Without augmentations"),
        "\nRunning time: %ss" % ("{:.2f}".format(time.time() - start_time)),
        f"\nproccessed images: %s/%s (%%%s)" %
        (count, img_count, int(
            (count / img_count * 100))), "\nHand detection succsess rate: %",
        int(suc / count * 100))


offset_percent = 5  # offset percentage for croping the detected hand
rotation_percent = 20  # offset percentage for hand straightening
count = 0  # count of images which has been processed
suc = 0  # count of images wihcn successfully found a hand in it

if (len(sys.argv) < 2):
    print("You should give the input directory as an argument!",
          "\nExample: py preProcess.py \'...path_to_images\'")
    exit(0)
image_dir = os.path.abspath(str(
    sys.argv[1][:-1]))  # path to the directory containing the images
if (str(platform.system()).lower() == "windows"):
    output_dir = os.path.abspath(os.path.join(
        image_dir, os.pardir)) + "\\" + os.path.basename(
            image_dir) + " (preprocessed)\\"  # path to the output directory
else:
    output_dir = os.path.abspath(os.path.join(
        image_dir, os.pardir)) + "/" + os.path.basename(
            image_dir) + " (preprocessed)/"  # path to the output directory

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
img_count = len([  # number of total images in the <image_dir> directory
    entry for entry in os.listdir(image_dir)
    if os.path.isfile(os.path.join(image_dir, entry))
])
output_size = int(input("Output size (ex: 256) :\n"))
output_channel = int(
    input(
        "Output channels :\n 1) RGB (3channels)\n 2) Grayscale (1channel)\n"))
data_type = int(
    input("Which data :\n 1) Training data\n 2) Test/Validation data\n"))

os.system('cls')
start_time = time.time()
for filename in os.listdir(image_dir):
    count += 1
    img = cv2.imread(os.path.join(image_dir, filename))
    if (img is None):
        print("couldn't open image!")
    else:
        if (data_type == 1):
            img = rotate(img)  #rotate the hand to get a strait upward hand
            img = handRec(img)  #detect the hand for croping the image
            img = clahe3(img)  #adjust the brightness and contrast of the image
        img = rescale(img)  #rescale the image to <output_size>
        img = flipAndRotate(img)  #randomly flip and rotate the images
        if (output_channel == 2):
            img = cv2.cvtColor(
                img, cv2.COLOR_BGR2GRAY)  #Reduce the channels from 3 to 1
        cv2.imwrite(os.path.join(output_dir, filename), img)  #save
        _print()