Use commands in terminal (working directory):

python tello_aruco_test.py                       (starts connection to the drone and displays video stream)

t                                                (makes the drone takeoff -- enters STATE: FLYING)
l                                                (makes the drone land -- enters STATE: LANDING/LANDED -- only possible when in STATE: FLYING)
a                                                (makes the drone automated -- enters STATE: AUTO -- only possible when in STATE: FLYING)
q                                                (shuts down connection to the drone -- a little bit buggy)
