"""
Click on the image to define your ROI polygon points.
Press:
  - Left click  : add a point
  - Z           : undo last point
  - Enter/Space : finish and print the config.yml points
  - Q / Escape  : quit without saving
"""

import cv2
import yaml
import numpy as np
from utils.queue_utils import get_config

config = get_config('./config.yml')

ip_address = config.get('ip_address', '')
username   = config.get('username', '')
password   = config.get('password', '')

if len(username) > 0 and len(password) > 0 and len(ip_address) > 0:
    if ip_address.startswith('http'):
        link = ip_address
    else:
        link = 'rtsp://' + username + ':' + password + '@' + ip_address
else:
    link = ip_address if ip_address else 'live_test_video.mp4'

print(f'Connecting to: {link}')
cap = cv2.VideoCapture(link)

grabbed, frame = cap.read()
cap.release()

if not grabbed or frame is None:
    print('ERROR: Could not grab a frame. Check your config.yml ip_address/username/password.')
    exit(1)

frame = cv2.resize(frame, (640, 480))
points = []


def draw(img):
    overlay = img.copy()
    for i, pt in enumerate(points):
        cv2.circle(overlay, pt, 5, (0, 255, 0), -1)
        cv2.putText(overlay, str(i+1), (pt[0]+6, pt[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if len(points) > 1:
        cv2.polylines(overlay, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
    if len(points) > 2:
        cv2.polylines(overlay, [np.array(points, dtype=np.int32)], True, (0, 255, 255), 2)
    cv2.putText(overlay, 'Left click: add point | Z: undo | Enter: done | Q: quit',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.imshow('Pick Zone', draw(frame.copy()))


cv2.namedWindow('Pick Zone')
cv2.setMouseCallback('Pick Zone', on_mouse)
cv2.imshow('Pick Zone', draw(frame.copy()))

while True:
    key = cv2.waitKey(20) & 0xFF
    if key in (13, 32):  # Enter or Space — done
        break
    elif key == ord('z') and points:  # undo
        points.pop()
        cv2.imshow('Pick Zone', draw(frame.copy()))
    elif key in (ord('q'), 27):  # Q or Escape
        print('Quit without saving.')
        cv2.destroyAllWindows()
        exit(0)

cv2.destroyAllWindows()

if len(points) < 3:
    print('Need at least 3 points to define a zone.')
    exit(1)

print('\n--- Copy this into your config.yml (replace the existing points section) ---\n')
print('points:')
for pt in points:
    print(f'- - {pt[0]}')
    print(f'  - {pt[1]}')
print('- []')  # empty list signals end of polygon to the parser in main.py
print('\n--- Done ---')
