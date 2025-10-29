#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import insightface
import onnxruntime


# In[ ]:


import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Инициализация модели для CPU
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Получаем лица и лендмарки
    faces = app.get(frame)
    for face in faces:
        # Рисуем bounding box
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Рисуем лендмарки лица (106 точек)
        for x, y in face.landmark_2d_106:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)

    cv2.imshow("Face Mesh (InsightFace)", frame)

    # Выход по ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

