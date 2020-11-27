import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml" )

video_capture = cv2.VideoCapture(0)
mst = cv2.imread('moustache.png')
hat = cv2.imread('cowboy_hat.png')
dog = cv2.imread('dog_filter.png')
hair = cv2.imread('short-hair 1.png')
glasses = cv2.imread('glassesred.png')

def put_hair(hair, fc, x, y, w, h):
    face_width = w
    face_height = h

    hair_width = face_width + 1
    hair_height = int(0.35 * face_height) + 1

    hair = cv2.resize(hair, (hair_width, hair_height))
    for i in range(hair_height):
        for j in range(hair_width):
            for k in range(3):
                if hair[i][j][k] < 235:
                    fc[y + i - int(0.25 * face_height)][x + j][k] = hair[i][j][k]
    return fc


def put_moustache(mst, fc, x, y, w, h):
    face_width = w
    face_height = h

    mst_width = int(face_width * 0.4166666) + 1
    mst_height = int(face_height * 0.142857) + 1

    mst = cv2.resize(mst, (mst_width, mst_height))

    for i in range(int(0.62857142857 * face_height), int(0.62857142857 * face_height) + mst_height):
        for j in range(int(0.29166666666 * face_width), int(0.29166666666 * face_width) + mst_width):
            for k in range(3):
                if mst[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k] < 235:
                    fc[y + i][x + j][k] = \
                    mst[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k]
    return fc


def put_hat(hat, fc, x, y, w, h):
    face_width = w
    face_height = h

    hat_width = face_width + 1
    hat_height = int(0.35 * face_height) + 1

    hat = cv2.resize(hat, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k] < 235:
                    fc[y + i - int(0.25 * face_height)][x + j][k] = hat[i][j][k]
    return fc

def put_glasses(glass, fc, x, y, w, h):
    face_width = w
    face_height = h

    hat_width = face_width + 1
    hat_height = int(0.50 * face_height) + 1

    glass = cv2.resize(glass, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if glass[i][j][k] < 235:
                    fc[y + i - int(-0.20 * face_height)][x + j][k] = glass[i][j][k]
    return fc

def put_dog_filter(dog, fc, x, y, w, h):
    face_width = w
    face_height = h

    dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.75)))
    for i in range(int(face_height * 1.75)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if dog[i][j][k] < 235:
                    fc[y + i - int(0.375 * h) - 1][x + j - int(0.25 * w)][k] = dog[i][j][k]
    return fc

def cartoon(frame):
    return cv2.stylization(frame,sigma_s=200, sigma_r=0.3)

def sketch(frame):
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayImageInv = 255 - grayImage
    grayImageInv = cv2.GaussianBlur(grayImageInv, (21, 21), 0)
    return cv2.divide(grayImage, 255 - grayImageInv, scale=256.0)


print("Select Filter:1) Moustache 2) Hat 3) Hat and Moustache 4) Dog Filter 5) Glasses 6) Cartoon 7) Sketch")
ch = int(input())

while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,5,)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:

        if ch == 1:
            frame = put_moustache(mst, frame, x, y, w, h)
        elif ch == 2:
            frame = put_hat(hat, frame, x, y, w, h)
        elif ch == 3:
            frame = put_moustache(mst, frame, x, y, w, h)
            frame = put_hat(hat, frame, x, y, w, h)
        elif ch == 4:
            frame = put_dog_filter(dog, frame, x, y, w, h)
        elif ch == 5:
            frame = put_glasses(glasses, frame, x, y, w, h)
        elif ch == 6:
            frame = cartoon(frame)
        elif ch == 7:
            frame = sketch(frame)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()