import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


input_images_directory = "C:\\Users\\home\\Google Drive\\IO\\Sudoku Project"

def get_images(directory):
    files = os.listdir(directory)
    result = []
    extensions = ['.png', '.jpg', '.jpeg']
    for f in files:
        if any(e in f.lower() for e in extensions):
            result.append(os.path.join(directory, f))
    return result

def resize_and_pad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

imgs = get_images(input_images_directory)

def detect_cluster(xs, window):
    xs = np.sort(xs)

    current = xs[0]
    cluster = [xs[0]]
    points = []

    for i in range(0, len(xs)):
        x = xs[i]

        if abs(x - np.mean(cluster)) < window:
            cluster.append(x)
        else:
            points.append(np.mean(cluster))
            cluster = [x]
    
    points.append(np.mean(cluster))

    return points

def extract_cells(xs, ys):
    result = []
    for i in range(len(xs)-1):
        x1, x2 = xs[i:i+2]
        for j in range(len(ys)-1):
            y1, y2 = ys[j:j+2]
            result.append((int(x1), int(y1), int(x2), int(y2), 9*j+i))
    return result

import itertools

def img_to_grid(w, h, margin, imgs, file_name):
    n = w*h

    if len(imgs) != n:
        print(len(imgs))
        raise ValueError('Number of images  does not match.')

    img_h, img_w, img_c = imgs[0].shape

    m_x = 0
    m_y = 0
    if margin is not None:
        margin = margin[0]
        m_x = int(margin)
        m_y = m_x

    imgmatrix = np.zeros((img_h * h + m_y * (h - 1),
                          img_w * w + m_x * (w - 1),
                          img_c),
                         np.uint8)

    imgmatrix.fill(255)    

    positions = itertools.product(range(w), range(h))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w, :] = img

    cv2.imwrite(file_name, imgmatrix) 

for img_path in imgs:

    file_name = os.path.basename(img_path)
    output_directory = os.path.join('puzzles', file_name[:file_name.index('.')])
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    img = cv2.imread(img_path)
    img_orig = cv2.imread(img_path)

    cv2.imwrite(os.path.join(output_directory, "puzzle.png"), img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)


    xs, ys = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        xs.append(x1), xs.append(x2)
        ys.append(y1), ys.append(y2)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imwrite(os.path.join(output_directory, "hough_lines.png"), img)

    k = 0.1
    x_window = k * np.std(xs)
    y_window = k * np.std(ys)
    points_x = detect_cluster(xs, x_window)
    points_y = detect_cluster(ys, y_window)

    for x in points_x:
        cv2.line(img, (int(x), 0), (int(x), 512), (0, 255, 0), thickness=1)
    for y in points_y:
        cv2.line(img, (0, int(y)), (512, int(y)), (0, 255, 0), thickness=1)

    cv2.imwrite(os.path.join(output_directory, "grid.png"), img)

    print(points_x, points_y)

    cells = extract_cells(points_x, points_y)
    print(cells)

    if len(cells) == 81:
        print('Detected grid!')
    else:
        print('Incorrect cell count.')
        continue

    cell_imgs = []

    for x1, y1, x2, y2, i in cells:
        w, h = x2 - x1, y2 - y1
        roi = gray[y1:y1+h, x1:x1+w]
        crop_size = 4
        #Tamaño orignal size = 128
        size = 28
        roi_resized = resize_and_pad(roi, (size + 2*crop_size, size + 2*crop_size), 255)
        roi_cropped = roi_resized[crop_size:crop_size+size,crop_size:crop_size+size]
        _, roi_threshold = cv2.threshold(roi_cropped, 127, 255, cv2.THRESH_BINARY_INV)

        fp = 2
        cv2.floodFill(roi_threshold, None, (fp, fp), (0, 0, 0))
        cv2.floodFill(roi_threshold, None, (fp, size-fp), (0, 0, 0))
        cv2.floodFill(roi_threshold, None, (size-fp, fp), (0, 0, 0))
        cv2.floodFill(roi_threshold, None, (size-fp, size-fp), (0, 0, 0))

        pixel_count = cv2.countNonZero(roi_threshold)
        roi_threshold_gray = cv2.cvtColor(roi_threshold, cv2.COLOR_BGR2RGB)
        cell_imgs.append(roi_threshold_gray)
        cell_name = "cell_{}".format(str(i).rjust(2, '0'))

        aux = (255-roi_threshold_gray)
        cv2.imwrite(os.path.join(output_directory, "{}.png".format(cell_name)), aux)

        # if pixel_count > 250:
        #     plt.figure(dpi=200)
        #     plt.imshow(cv2.cvtColor(roi_threshold, cv2.COLOR_BGR2RGB))
        #     plt.axis('off')
        #     # plt.show()
        #     digit_name = "digit_{}".format(str(i).rjust(2, '0'))
        #     cv2.imwrite(os.path.join(output_directory, "{}.png".format(digit_name)), roi_threshold)

    img_to_grid(9, 9, (8, 8), cell_imgs, os.path.join(output_directory, "extracted.png"))

        


  
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(dpi=200)
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.figure(figsize=(8, 3), dpi=200)
    plt.hist(xs, bins=100, color="lightgray")
    plt.xlim([0, 512])
    plt.title("X-axis")
    for x in points_x:
        plt.axvline(x, lw=2, c='salmon')
    plt.savefig(os.path.join(output_directory, "histogram_x.png"))
    # plt.show()

    plt.figure(figsize=(8, 3), dpi=200)
    plt.hist(ys, bins=100, color="lightgray")
    plt.xlim([0, 512])
    plt.title("Y-axis")
    for y in points_y:
        plt.axvline(y, lw=2, c='salmon')
    plt.savefig(os.path.join(output_directory, "histogram_y.png"))
    # plt.show()