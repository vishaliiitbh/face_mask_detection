import cv2
import numpy as np
import tensorflow as tf


labels_dict = {0: 'No mask', 1: 'Mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

imgsize = 4

camera = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cnn = tf.keras.models.load_model('mask_detector_model.h5')

while True:
    rval, im = camera.read()
    if not rval:
        break

    im = cv2.flip(im, 1, 1)

    imgs = cv2.resize(im, (im.shape[1] // imgsize, im.shape[0] // imgsize))

    face_rec = classifier.detectMultiScale(imgs)

    for i in face_rec:
        (x, y, l, w) = [v * imgsize for v in i]

        face_img = im[y:y + w, x:x + l]
        resized = cv2.resize(face_img, (150, 150))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))

        result = cnn.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(im, (x, y), (x + l, y + w), color_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + l, y), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', im)

    key = cv2.waitKey(10)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
