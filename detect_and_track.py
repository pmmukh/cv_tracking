import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

def of_tracker(cap, file_name) :
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    frameCounter = 0

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=10,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    r, h, c, w = detect_one_face(old_frame)
    # old_frame = old_frame[r:r + h, c:c + w]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(old_gray)
    mask[r:r+h, c:c+w] = 1
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    sumx = 0
    sumy = 0
    totalwts = 0
    for i, new in enumerate(zip(p0)):
        a,b = new[0].ravel()
        wdist = (c + w / 2 - a) ** 2 + (r + h / 2 - b) ** 2
        sumx = sumx + (1 / wdist) * a
        sumy = sumy + (1 / wdist) * b
        totalwts = totalwts + (1 / wdist)
    output.write("%d,%d,%d\n" % (frameCounter, sumx/totalwts, sumy/totalwts))
    frameCounter+=1
    while (1):
        ret, frame = cap.read()
        if ret == True:

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            r, h, c, w = detect_one_face(frame)
            measurement_valid = r + h + c + w

            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # draw the tracks
            sumx = 0
            sumy = 0
            totalwts = 0
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                if measurement_valid != 0:
                    wdist = (c + w/2 - a)**2 + (r + h/2 - b)**2
                    sumx = sumx + (1/wdist)*a
                    sumy = sumy + (1/wdist)*b
                    totalwts = totalwts + (1/wdist)
                else :
                    sumx = sumx + a
                    sumy = sumy + b
                    totalwts = totalwts + 1
            output.write("%d,%d,%d\n" % (frameCounter, sumx / totalwts, sumy / totalwts))
            frameCounter += 1

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = p1.reshape(-1, 1, 2)
        else :
            break

def kalman_tracker(cap, file_name) :

    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    frameCounter = 0
    ret, frame = cap.read()
    r, h, c, w = detect_one_face(frame)
    output.write("%d,%d,%d\n" % (frameCounter, c + (w / 2), r + (h / 2)))
    frameCounter += 1
    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
    kalman = cv2.KalmanFilter(4, 2, 0)  # 4 state/hidden, 2 measurement, 0 control
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                        [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)  # you can tweak these to make the tracker
    kalman.processNoiseCov = 1e-4 * np.eye(4, 4)  # respond faster to change and be less smooth
    kalman.measurementNoiseCov = 1e-2 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    while(1):
        ret, frame = cap.read()
        if ret == True:
            prediction = kalman.predict()
            r, h, c, w = detect_one_face(frame)
            measurement_valid = c + w + r + h
            if measurement_valid != 0:  # e.g. face found
                measurement = np.array([c + w / 2, r + h / 2], dtype='float64')
                posterior = kalman.correct(measurement)
                nextstate = posterior
            else :
                nextstate = prediction

            # frame = cv2.circle(frame, (int(nextstate[0]), int(nextstate[1])), 1, (0, 255, 0), -1)
            output.write("%d,%d,%d\n" % (frameCounter, int(nextstate[0]), int(nextstate[1])))  # Write as 0,pt_x,pt_y
            frameCounter = frameCounter + 1
        else:
            break



# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1], particle[0]]

def pf_tracker(cap, file_name):
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    frameCounter = 0
    ret, frame = cap.read()
    # hist_bp: obtain using cv2.calcBackProject and the HSV histogram
    r, h, c, w = detect_one_face(frame)
    n_particles = 200

    roi_hist = hsv_histogram_for_window(frame, (c, r, w, h))

    init_pos = np.array([c + w / 2.0, r + h / 2.0], int)  # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos  # Init particles to init position
    weights = np.ones(n_particles) / n_particles  # weights are uniform (at first)
    pos = np.sum(particles.T * weights, axis=1).astype(int)  # expected position: weighted average
    output.write("%d,%d,%d\n" % (frameCounter, pos[0], pos[1]))
    frameCounter = frameCounter + 1
    while (1):
        ret, frame = cap.read()
        if ret == True:

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist_bp = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            im_h,im_w,chanels = frame.shape
            # Particle motion model: uniform step (TODO: find a better motion model)
            np.add(particles, np.random.uniform(-10, 10, particles.shape), out=particles, casting="unsafe")

            # Clip out-of-bounds particles
            particles = particles.clip(np.zeros(2), np.array((im_w, im_h)) - 1).astype(int)

            f = particleevaluator(hist_bp, particles.T)  # Evaluate particles
            weights = np.float32(f.clip(1))  # Weight ~ histogram response
            weights /= np.sum(weights)  # Normalize w
            pos = np.sum(particles.T * weights, axis=1).astype(int)  # expected position: weighted average
            output.write("%d,%d,%d\n" % (frameCounter, pos[0], pos[1]))
            frameCounter = frameCounter + 1
            if 1. / np.sum(weights ** 2) < n_particles / 2.:  # If particle cloud degenerate:
                particles = particles[resample(weights), :]  # Resample particles according to weights
        else:
            break

def camshift_tracker(cap, file_name):
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")
    frameCounter = 0
    # take first frame of the video
    ret, frame = cap.read()
    # setup initial location of window
    r, h, c, w = detect_one_face(frame)
    track_window = (c, r, w, h)
    output.write("%d,%d,%d\n" % (frameCounter, c + (w/2), r + (h/2)))  # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h))
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (1):
        ret, frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)

            x, y, w, h = track_window
            output.write("%d,%d,%d\n" % (frameCounter, x + (w / 2), y + (h / 2)))  # Write as 0,pt_x,pt_y
            frameCounter = frameCounter + 1
        else:
            break


def skeleton_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # perform the tracking
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()

        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        # write the result to the output file
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        camshift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        pf_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        kalman_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        of_tracker(video, "output_of.txt")
