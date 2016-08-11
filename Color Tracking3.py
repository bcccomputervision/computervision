#######################Color Tracking###########################START
import array
import numpy
import cv2


RED = 0
BLUE = 1
YELLOW = 2
MAGENTA = 3

H_MIN = 159
H_MAX = 256
S_MIN = 127
S_MAX = 256
V_MIN = 176
V_MAX = 256

HSV = 0


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_NUM_OBJECTS = 10
MIN_OBJECT_AREA = 20*20
MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5

trackbarWindowName = "HSV Threshold Values"
trackbarSwitchName = "HSV Red Yellow Blue Switch"

key = 0

showPixels = 0

#######################Color Tracking###########################END
##############################################
##############################################
##############################################
##############################################
#######################Color Tracking Functions###########################START
def on_trackbar(self):
    
    global H_MIN
    global H_MAX
    global S_MIN
    global S_MAX
    global V_MIN
    global V_MAX
    
    H_MIN = cv2.getTrackbarPos("H_MIN", trackbarWindowName)
    H_MAX = cv2.getTrackbarPos("H_MAX", trackbarWindowName)
    S_MIN = cv2.getTrackbarPos("S_MIN", trackbarWindowName)
    S_MAX = cv2.getTrackbarPos("S_MAX", trackbarWindowName)
    V_MIN = cv2.getTrackbarPos("V_MIN", trackbarWindowName)
    V_MAX = cv2.getTrackbarPos("V_MAX", trackbarWindowName)
    
    msg = "H_MIN: " + str(H_MIN) + " H_MAX: " + str(H_MAX) + " S_MIN: " + str(S_MIN) + " S_MAX: " + str(S_MAX) + " V_MIN: " + str(V_MIN) + " V_MAX: " + str(V_MAX)
    print(msg)

def on_trackbar_switch(self):
    '''
    global H_MIN
    global H_MAX
    global S_MIN
    global S_MAX
    global V_MIN
    global V_MAX
    global HSV
    '''
    HSV = cv2.getTrackbarPos("R/B/Y/M", trackbarSwitchName)
    
    if HSV == RED:
        print("Red")
        H_MIN = 159
        H_MAX = 256
        S_MIN = 127
        S_MAX = 256
        V_MIN = 176
        V_MAX = 256
    if HSV == BLUE:
        print("Blue")
        H_MIN = 96
        H_MAX = 255
        S_MIN = 253
        S_MAX = 255
        V_MIN = 140
        V_MAX = 253
    if HSV == YELLOW:
        print("Yellow")
        H_MIN = 20
        H_MAX = 30
        S_MIN = 183  #100
        S_MAX = 255
        V_MIN = 100
        V_MAX = 255
    if HSV == MAGENTA:
        print("Magenta Outside")
        H_MIN = 111
        H_MAX = 156
        S_MIN = 137
        S_MAX = 255
        V_MIN = 163
        V_MAX = 256

    msg = "H_MIN: " + str(H_MIN) + " H_MAX: " + str(H_MAX) + " S_MIN: " + str(S_MIN) + " S_MAX: " + str(S_MAX) + " V_MIN: " + str(V_MIN) + " V_MAX: " + str(V_MAX)
    print(msg)
    cv2.setTrackbarPos("H_MIN", trackbarWindowName, H_MIN)
    cv2.setTrackbarPos("H_MAX", trackbarWindowName, H_MAX)
    cv2.setTrackbarPos("S_MIN", trackbarWindowName, S_MIN)
    cv2.setTrackbarPos("S_MAX", trackbarWindowName, S_MAX)
    cv2.setTrackbarPos("V_MIN", trackbarWindowName, V_MIN)
    cv2.setTrackbarPos("V_MAX", trackbarWindowName, V_MAX)

def on_trackbar_switch_pixels(self):
        showPixels = cv2.getTrackbarPos("Cm/Pixels", trackbarSwitchName)

def intToString(number):
    msg = str(number)
    return msg

def drawObject(x_3D, y_3D, x, y, z, frame):
    cv2.circle(frame,(x,y),20,cv2.cv.Scalar(0,255,0),2)
    if y-25 > 0:
        cv2.line(frame, (x,y), (x,y-25), cv2.cv.Scalar(0,255,0), 2)
    else:
        cv2.line(frame,(x,y), (x,0), cv2.cv.Scalar(0,255,0),2)
    if y+25<FRAME_HEIGHT:
        cv2.line(frame, (x,y), (x,y+25), cv2.cv.Scalar(0,255,0),2)
    else:
        cv2.line(frame, (x,y), (x,FRAME_HEIGHT),cv2.cv.Scalar(0,255,0),2)
    if x-25>0:
        cv2.line(frame, (x,y), (x-25,y),cv2.cv.Scalar(0,255,0),2)
    else:
        cv2.line(frame, (x,y), (0,y), cv2.cv.Scalar(0,255,0),2)
    if x+25<FRAME_WIDTH:
        cv2.line(frame,(x,y),(x+25,y),cv2.cv.Scalar(0,255,0),2)
    else:
        cv2.line(frame, (x,y), (FRAME_WIDTH,y),cv2.cv.Scalar(0,255,0),2)
    
    if showPixels:
        cv2.putText(frame,intToString(x)+", "+intToString(y)+", "+intToString(z),(x,y+30),2,2,cv2.cv.Scalar(0,255,0),2)
        cv2.putText(frame,"x = "+intToString(x)+","+"y = "+intToString(y)+", z = "+intToString(z)+" centimeters",(5,50),1,1,cv2.cv.Scalar(0,255,0),2)
    else:
        cv2.putText(frame,intToString(x_3D)+", "+intToString(y_3D)+", "+intToString(z),(x,y+30),1,1,cv2.cv.Scalar(0,255,0),2)
        cv2.putText(frame,"x = "+intToString(x_3D)+","+"y = "+intToString(y_3D)+", z = "+intToString(z)+" centimeters",(5,50),0,1,cv2.cv.Scalar(0,255,0),2)
    

    if x < 300 and y > 600:
        cv2.putText(frame,"Tracking Object",(600,700),1,2,cv2.cv.Scalar(0,255,0),4)
    else:
        cv2.putText(frame,"Tracking Object",(5,700),1,2,cv2.cv.Scalar(0,255,0),4)
    return frame

def morphOps(thresh):
    erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilateElement = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))

    thresh = cv2.erode(thresh,erodeElement)
    thresh = cv2.erode(thresh,erodeElement)
	
    thresh = cv2.dilate(thresh,dilateElement)
    thresh = cv2.dilate(thresh,dilateElement)
    return thresh

def do_nothing(self):
    global matrixMath
    matrixMath = cv2.getTrackbarPos("Mat Math Off/On", trackbarSwitchName)

def trackFilteredObject(x, y, threshold, cameraFeed):
    temp = threshold.copy()
    contours, hierarchy = cv2.findContours(temp, cv2.cv.CV_RETR_CCOMP, cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    refArea = float(0)
    objectFound = False
    
    if not hierarchy is None:
        if len(hierarchy[0]):
            hi=len(hierarchy[0])
    else:
        hi=0

    if hi > 0:
        numObjects = len(hierarchy[0])
        if numObjects<MAX_NUM_OBJECTS:
            index = int(0)
            while index >= 0:
                cnt = contours[index]
                moment = cv2.moments(cnt)
                area = float(moment['m00'])
                
                if area>MIN_OBJECT_AREA and area<MAX_OBJECT_AREA and area>refArea:
                    x = moment['m10'] / area
                    y = moment['m01'] / area
                    objectFound = True
                else:
                    objectFound = False
                index = hierarchy[0][index][0]
        else:
            cv2.putText(cameraFeed, "TOO MUCH NOISE!", (0, 50), 2, 2, cv2.cv.Scalar(0, 0, 255), 4)
    return objectFound,int(x),int(y), cameraFeed
#######################Color tracking functions###########################END
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
#######################Dronekit###########################START
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
from pymavlink import mavutil # Needed for command message definitions



#Set up option parsing to get connection string
import argparse
parser = argparse.ArgumentParser(description='Print out vehicle state information. Connects to SITL on local PC by default.')
parser.add_argument('--connect', default='127.0.0.1:14550',
                  help="vehicle connection target. Default '127.0.0.1:14550'")
args = parser.parse_args()

connection_string = args.connect
sitl = None


#Start SITL if no connection string specified
if not connection_string:
    import dronekit_sitl
    sitl = dronekit_sitl.start_default()
    connection_string = sitl.connection_string()


# Connect to the Vehicle
print 'Connecting to vehicle on: %s' % args.connect
vehicle = connect(args.connect, wait_ready=True)

#######################Dronekit###########################END
##############################################
##############################################
##############################################
##############################################
#######################Dronekit functions###########################START
def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print "Basic pre-arm checks"
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print " Waiting for vehicle to initialise..."
        time.sleep(1)

        
    print "Arming motors"
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True    

    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:      
        print " Waiting for arming..."
        time.sleep(1)

    print "Taking off!"
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command 
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print " Altitude: ", vehicle.location.global_relative_frame.alt 
        #Break and return from function just below target altitude.        
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: 
            print "Reached target altitude"
            break
        time.sleep(1)

def goto_position_target_local_ned(north, east, down):
    """	
    Send SET_POSITION_TARGET_LOCAL_NED command to request the vehicle fly to a specified 
    location in the North, East, Down frame.

    It is important to remember that in this frame, positive altitudes are entered as negative 
    "Down" values. So if down is "10", this will be 10 metres below the home altitude.

    Starting from AC3.3 the method respects the frame setting. Prior to that the frame was
    ignored. For more information see: 
    http://dev.ardupilot.com/wiki/copter-commands-in-guided-mode/#set_position_target_local_ned

    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.

    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111111000, # type_mask (only positions enabled)
        north, east, down, # x, y, z positions (or North, East, Down in the MAV_FRAME_BODY_NED frame
        0, 0, 0, # x, y, z velocity in m/s  (not used)
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 
    # send command to vehicle
    vehicle.send_mavlink(msg)

#######################Dronekit functions###########################END
##############################################
##############################################
##############################################
##############################################
#######################Dronekit###########################START
arm_and_takeoff(10)

#######################Dronekit###########################END
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################








trackObjects = bool(1)
useMorphOps = bool(1)
matrixMath = int(1)
trackObjectCamera1 = bool(0)
trackObjectCamera2 = False
xRight = 0
yRight = 0
xLeft = 0
yLeft = 0
x_3D = 0
y_3D = 0
z_3D = 0
w = 0.0
d = 0

captureRight = cv2.VideoCapture(0)
captureLeft = cv2.VideoCapture(1)
if not captureRight.isOpened():
    print("Failed to open video device 1 or video file!")

#if (!captureLeft.isOpened())
if not captureLeft.isOpened():
    print("Failed to open video device 2 or video file!")

captureRight.set(3, FRAME_WIDTH)
captureRight.set(4, FRAME_HEIGHT)
captureLeft.set(3, FRAME_WIDTH)
captureLeft.set(4, FRAME_HEIGHT)

cv2.namedWindow("Left Camera HSV Smoothed and Thresholded Video", 0)
cv2.namedWindow("Right Camera HSV Smoothed and Thresholded Video", 0)

cv2.namedWindow(trackbarWindowName, 0)
cv2.namedWindow(trackbarSwitchName, 0)

cv2.namedWindow("Right Camera Tracking", 0)
cv2.namedWindow("Left Camera Tracking", 0)

cv2.moveWindow("Right Camera HSV Smoothed and Thresholded Video", 0, 450)
cv2.moveWindow("Left Camera HSV Smoothed and Thresholded Video", 800, 450)

cv2.moveWindow("Right Camera Tracking", 700, 0)
cv2.moveWindow("Left Camera Tracking", 0, 0)

cv2.createTrackbar("H_MIN", trackbarWindowName, H_MIN, 255, on_trackbar)
cv2.createTrackbar("H_MAX", trackbarWindowName, H_MAX, 255, on_trackbar)
cv2.createTrackbar("S_MIN", trackbarWindowName, S_MIN, 255, on_trackbar)
cv2.createTrackbar("S_MAX", trackbarWindowName, S_MAX, 255, on_trackbar)
cv2.createTrackbar("V_MIN", trackbarWindowName, V_MIN, 255, on_trackbar)
cv2.createTrackbar("V_MAX", trackbarWindowName, V_MAX, 255, on_trackbar)

cv2.createTrackbar("R/B/Y/M", trackbarSwitchName, HSV, 3, on_trackbar_switch)
cv2.createTrackbar("Cm/Pixels", trackbarSwitchName, showPixels, 1, on_trackbar_switch_pixels)
cv2.createTrackbar("Mat Math Off/On", trackbarSwitchName, matrixMath, 1, do_nothing)

cv2.moveWindow(trackbarWindowName, 400, 775)
cv2.moveWindow(trackbarSwitchName, 1000, 730)

cv2.startWindowThread()

Q = cv2.cv.Load("Q.xml")
mx1 = cv2.cv.Load("mx1.xml")
my1 = cv2.cv.Load("my1.xml")
mx2 = cv2.cv.Load("mx2.xml")
my2 = cv2.cv.Load("my2.xml")

if not Q or not mx1 or not my1 or not mx2 or not my2:
    print("Error loading 1 or more matrix xml files\n")

Mx1 = numpy.asmatrix(mx1)
My1 = numpy.asmatrix(my1)
Mx2 = numpy.asmatrix(mx2)
My2 = numpy.asmatrix(my2)
Q_mat = numpy.asmatrix(Q)

pointsXYD = numpy.array([0.,0.,0.])
result3DPoints = numpy.array([0.,0.,0.])


zchange = None
zchange2 = 0
zchange3 = 0
ychange = None
ychange2 = 0
ychange3 = 0
xchange = None
xchange2 = 0
xchange3 = 0

tcount = 0

while not key == 113:
    ret, frame0 = captureRight.read()
    ret, frame1 = captureLeft.read()
    captureFeedRight = frame0
    captureFeedLeft = frame1

    captureFeedRightR = cv2.remap(captureFeedRight, Mx2, My2, cv2.INTER_LINEAR)
    captureFeedLeftR = cv2.remap(captureFeedLeft, Mx1, My1, cv2.INTER_LINEAR)

    HSV1 = cv2.cvtColor(captureFeedRightR, cv2.COLOR_BGR2HSV)
    HSV2 = cv2.cvtColor(captureFeedLeftR, cv2.COLOR_BGR2HSV)

    thresholdRight = cv2.inRange(HSV1, cv2.cv.Scalar(H_MIN, S_MIN, V_MIN), cv2.cv.Scalar(H_MAX, S_MAX, V_MAX))
    thresholdLeft = cv2.inRange(HSV2, cv2.cv.Scalar(H_MIN, S_MIN, V_MIN), cv2.cv.Scalar(H_MAX, S_MAX, V_MAX))

    if useMorphOps:
        thresholdRight = morphOps(thresholdRight)
        thresholdLeft = morphOps(thresholdLeft)

    if trackObjects:
        trackObjectCamera1,xRight,yRight,captureFeedRightR = trackFilteredObject(xRight, yRight, thresholdRight, captureFeedRightR)
        trackObjectCamera2,xLeft,yLeft,captureFeedLeftR = trackFilteredObject(xLeft, yLeft, thresholdLeft, captureFeedLeftR)

    pointsXYD = numpy.array([0.,0.,0.])
    if trackObjectCamera1 and trackObjectCamera2:
        if not matrixMath:
            d = xLeft - xRight
            pointsXYD = (cv2.cv.Scalar(xLeft, yLeft, d))
            print pointsXYD
            print result3DPoints
            z_3D = result3DPoints[2]
            y_3D = result3DPoints[0]
            z_3D = result3DPoints[1]
        else:
                d = xLeft - xRight
                x_3D = xLeft * Q_mat[0, 0] + Q_mat[0, 3]
                y_3D = yLeft * Q_mat[1, 1] + Q_mat[1, 3]
                z_3D = Q_mat[2, 3]
                w = d * Q_mat[3, 2] + Q_mat[3, 3]
                x_3D = x_3D / w
                y_3D = y_3D / w
                z_3D = z_3D / w

        captureFeedRightR = drawObject(x_3D, y_3D, xRight, yRight, z_3D, captureFeedRightR)
        captureFeedLeftR = drawObject(x_3D, y_3D, xLeft, yRight, z_3D, captureFeedLeftR)

    cv2.imshow("Right Camera Tracking", captureFeedRightR)
    cv2.imshow("Left Camera Tracking", captureFeedLeftR)
    cv2.imshow("Right Camera HSV Smoothed and Thresholded Video", thresholdRight)
    cv2.imshow("Left Camera HSV Smoothed and Thresholded Video", thresholdLeft)

    if not zchange == z_3D:
        zchange = z_3D
        zchange2 = int(zchange)
        zchange2 = zchange2/10
        if not zchange3 == zchange2 and not zchange2 == None and zchange2>0 and zchange2<10:
            zchange3 = zchange2
            point1 = goto_position_target_local_ned(-ychange3*10,-xchange3*10,-(zchange3*10))
    if not ychange == y_3D:
        ychange = y_3D
        ychange2 = int(ychange)
        #ychange2 = ychange2/10
        if not ychange3 == ychange2 and not ychange2 == None and ychange2>0 and ychange2<10:
            ychange3 = ychange2
            point1 = goto_position_target_local_ned(-ychange3*10,-xchange3*10,-(zchange3*10))
    if not xchange == x_3D:
        xchange = x_3D
        xchange2 = int(xchange)
        #xchange2 = xchange2/10
        if not xchange3 == xchange2 and not xchange2 == None and xchange2>0 and xchange2<10:
            xchange3 = xchange2
            point1 = goto_position_target_local_ned(-ychange3*10,-xchange3*10,-(zchange3*10))
        
    key = cv2.cv.WaitKey(20)

    level = vehicle.battery
    tcount = tcount + 1
    if tcount > 9:
        print " Altitude: ", int(vehicle.location.global_relative_frame.alt)
        print " Longitude: ", int(100000*(vehicle.location.global_relative_frame.lon + 74.085467))
        print " Latitude: ", int(100000*(vehicle.location.global_relative_frame.lat - 40.9505997))
        print "Battery: ", level
        tcount = 0
    
captureRight.release()
captureLeft.release()
cv2.destroyAllWindows()
'''
print "Returning to Launch"
vehicle.mode = VehicleMode("RTL")

#Close vehicle object before exiting script
print "Close vehicle object"
vehicle.close()

# Shut down simulator if it was started.
if sitl is not None:
    sitl.stop()
'''

print " Vehicle Heading: %s" % vehicle.heading
time.sleep(3)

"""
The example is completing. LAND at current location.
"""

print "Returning to Launch"
vehicle.mode = VehicleMode("RTL")

#print("Setting LAND mode...")
#vehicle.mode = VehicleMode("LAND")

print " \nVehicle is Landing..."

while vehicle.armed:
    print " \nWaiting for disarming..."
    time.sleep(1)

print " \nVehicle is armed: %s" % vehicle.armed

print " \nVehicle Heading: %s" % vehicle.heading


#Close vehicle object before exiting script
time.sleep(3)
print "\nClose vehicle object"
time.sleep(3)
vehicle.close()


print("Completed")
time.sleep(3)
