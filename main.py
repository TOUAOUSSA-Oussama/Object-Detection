# 1 - Introduction : 
## Dans ce projet, on essaye de detecter les objets et pour chaque objets detecter predire son nom (c'est de la segmentation)
## Tout d'abord, on a besoin de coco.names = fichier qui contient toutes les objets qu'on peut detecter (dispo dans https://github.com/pjreddie/darknet/blob/master/data/coco.names)
## On besoin de l'architecture et des poids du modele mobile net ssd qui va faire la prediction.
## Avantage de mobile net : rapidite et precision (predition on temps reel)

# 2 - Bibliotheques requises :
from sre_constants import SUCCESS
import cv2

# 3 - Importer les noms des objets
objNames = []
pathNames = 'coco.names'
with open(pathNames, 'rt') as f:
    objNames = f.read().rstrip('\n').split('\n') # rstrip pour supprimer les retours a la ligne, split pour diviser le resultat
## test :   
# print(objNames)

# 4 - Importer les fichiers de configuration du modele 
pathConfig = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
pathWeights = 'frozen_inference_graph.pb'

# 5 - Creation du modele avec les hyperparameters (les memes que ceux utilises dans la documentation) :
net = cv2.dnn_DetectionModel(pathWeights, pathConfig)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# 6 - Tester le modele sur l'image lena :
# img = cv2.imread("lena.PNG")
# ## detection et segmentation (confThreshold regle la valeur de proba on dit qu'un objet est detecte)
# objetsID, confs, boxs = net.detect(img, confThreshold=0.5)
# ## test :
# # print(objetsID, boxs) # ceci donne : [1] [[ 21  16 168 197]]
# ## Dessiner le box + ecrire le nom de chaque objet (rectangle) sur les objets detectes : parcourir toutes les objets
# for objetID, confidence, box in zip(objetsID, confs, boxs):
#     # tracer le rectangle :
#     cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
#     # afficher le nom :
#     cv2.putText(
#         img, 
#         objNames[objetID-1].upper(),
#         (box[0]+10, box[1]+20),
#         cv2.FONT_HERSHEY_COMPLEX,
#         0.5,
#         (178,34,34),
#         2
#         )
# cv2.imshow("output", img)
# cv2.waitKey(0)

# 7 - appliquer le modele sur les frames du camera :
frames = cv2.VideoCapture(0) # 0 est l'id du camera
frames.set(3, 640)
frames.set(4, 480)
# parcourir les frames 
while True :
    success, frame = frames.read()
    objetsID, confs, boxs = net.detect(frame, confThreshold=0.5)
    ## Dessiner le box + ecrire le nom de chaque objet (rectangle) sur les objets detectes : parcourir toutes les objets
    if len(objetsID) != 0 :
        for objetID, confidence, box in zip(objetsID, confs, boxs):
            # tracer le rectangle :
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            # afficher le nom :
            cv2.putText(
                frame, 
                objNames[objetID-1].upper(),
                (box[0]+10, box[1]+20),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (178,34,34),
                2
                )
            # afficher la precision :
            cv2.putText(
                frame, 
                str(round(confidence*100, 2))+"%",
                (box[0]+150, box[1]+20),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (178,34,34),
                2
                )
        cv2.imshow("output", frame)
        cv2.waitKey(1)