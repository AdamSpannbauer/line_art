from collections import Counter
import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_dominant_color(image, k=3, avoid=(255, 255, 255), avoid_thresh=40):
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(pixels)
    label_counts = Counter(labels)

    dominant_color = None
    for i in range(k):
        c = kmeans.cluster_centers_[label_counts.most_common(k)[i][0]]
        d = (avoid - c).sum()
        if d > avoid_thresh:
            dominant_color = c
            break

    return tuple(dominant_color)


def get_contour_color(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y + h, x:x + w]
    dominant_color = get_dominant_color(roi)

    return dominant_color


def image_color_masks(image, background_color=(0, 0, 0), min_contour_area=10, *thresh_args):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not thresh_args:
        thresh_args = (5, 255, 0)
    _, threshed = cv2.threshold(gray, *thresh_args)

    contours, _ = cv2.findContours(threshed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    color_mask = np.zeros_like(image)
    mask = np.zeros_like(gray)
    h, w = image.shape[:2]
    max_contour_area = 0.9 * h * w
    for c in contours:
        area = cv2.contourArea(c)

        if area < min_contour_area or area > max_contour_area:
            continue

        color = get_contour_color(c, image)
        if np.all(color != background_color):
            color_mask = cv2.drawContours(color_mask, [c], -1, color, -1)
            mask = cv2.drawContours(mask, [c], -1, 255, -1)

    return color_mask, mask


if __name__ == '__main__':
    import imutils
    from rand_walk_draw import draw_random_walk

    img = cv2.imread('images/py_pandas_2.png')
    img = imutils.resize(img, width=600)

    canvas_size = img.shape

    canvas_color = (200, 200, 200)
    canvas = np.ones(canvas_size) * canvas_color
    canvas = canvas.astype('uint8')

    color_mask, mask = image_color_masks(img, canvas_color, 10, 220, 255, cv2.THRESH_BINARY)

    draw_random_walk(canvas, canvas_color=canvas_color,
                     n_starts=50, n_restarts=1000,
                     mask=mask, color_mask=color_mask,
                     output='color.avi')
