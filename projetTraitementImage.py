import cv2
import tkinter as tk
from tkinter import ttk, Button
from PIL import Image, ImageTk
import numpy as np
import random
import mediapipe as mp
from pathlib import Path
import imageio

class AppTraitementImage:

    def __init__(self, root):
        self.root = root
        self.root.title("Projet Traitement Image")

        # Chargement de la vidéo
        self.cap = cv2.VideoCapture(1)
        self.ret, self.frame = self.cap.read()
        
        # Chargement des cascades
        self.cheminCascades = str (Path (__file__).resolve ().parent) + "/haarcascades/"

        self.eye_cascade = cv2.CascadeClassifier(self.cheminCascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.face_cascade = cv2.CascadeClassifier(self.cheminCascades +'haarcascade_frontalface_alt.xml')
        self.mouth_cascade = cv2.CascadeClassifier(self.cheminCascades + 'haarcascade_mcs_mouth.xml')
        
        # Chargement du modèle de segmentation
        self.segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        # Chargement des images
        self.cheminImages = str (Path (__file__).resolve ().parent) + "/images/"

        self.background_image = cv2.imread (self.cheminImages + "back_etoile2.jpg")
        self.hat = cv2.imread (self.cheminImages + "chapeau.png", cv2.IMREAD_UNCHANGED)
        # Chargement des lunettes avec le canal alpha
        self.lunettes = cv2.imread (self.cheminImages + "lunette.png", cv2.IMREAD_UNCHANGED)
        # Image d'un diplome
        self.diplome_img = cv2.imread (self.cheminImages + "diplome.png", cv2.IMREAD_UNCHANGED)
        self.diplome_img = cv2.resize(self.diplome_img, (30, 30))
        # Image d'une etoile
        self.etoile_img = cv2.imread (self.cheminImages + "etoile.png", cv2.IMREAD_UNCHANGED)
        self.etoile_img = cv2.resize(self.etoile_img, (30, 30))

        # # Charger le gif du sifflet
        # self.gif_path = self.cheminImages + "siflet.gif"
        # self.gif = imageio.get_reader(self.gif_path)
        
        # Création d'un label pour la video
        self.video_label = ttk.Label(root)
        self.video_label.pack()
        
        # Création du bouton pour activer ou désactiver le filtre sepia
        self.toggle_sepia_button = Button(root, text="Filtre Sepia", width=10, height=2, command=self.toggle_sepia)
        self.toggle_sepia_button.pack(side=tk.LEFT, padx=20, pady=10)

        # Création du bouton pour activer ou désactiver le filtre lunettes
        self.toggle_lunettes_button = Button(root, text="Lunettes", width=10, height=2, command=self.toggle_lunettes)
        self.toggle_lunettes_button.pack(side=tk.LEFT, padx=20, pady=10)

        # Création du bouton pour activer ou désactiver le filtre chapeau
        self.toggle_hat_button = Button(root, text="Chapeau", width=10, height=2, command=self.toggle_hat)
        self.toggle_hat_button.pack(side=tk.LEFT, padx=20, pady=10)

        # # Création du bouton pour activer ou désactiver le filtre chapeau
        # self.toggle_siflet_button = Button(root, text="Sifflet", width=10, height=2, command=self.toggle_siflet)
        # self.toggle_siflet_button.pack(side=tk.LEFT, padx=20, pady=10)

        # Création du bouton pour activer ou désactiver le filtre d'une image interactive
        self.toggle_interactive_diplome_button = Button(root, text="Pluie de diplome", width=15, height=2, command=self.toggle_interactiveDiplome)
        self.toggle_interactive_diplome_button.pack(side=tk.LEFT, padx=20, pady=10)

        # Création du bouton pour activer ou désactiver le filtre d'une image interactive
        self.toggle_interactive_etoile_button = Button(root, text="Pluie d'étoile", width=15, height=2, command=self.toggle_interactiveEtoile)
        self.toggle_interactive_etoile_button.pack(side=tk.LEFT, padx=20, pady=10)

        # Création du bouton pour activer ou désactiver le filtre arrière plan
        self.toggle_background_button = Button(root, text="Arrière plan", width=10, height=2, command=self.toggle_background)
        self.toggle_background_button.pack(side=tk.LEFT, padx=20, pady=10)

        #Initialisation des boutons 
        self.sepia_enabled = False  # Activer/désactiver le filtre sépia
        self.lunette_enabled = False # Activer/désactiver le filtre lunettes
        self.hat_enabled = False  # Activer/désactiver le filtre chapeau
        #self.siflet_enabled = False  # Activer/désactiver le filtre sifflet
        self.interactiveDiplome_enabled = False  # Activer/désactiver le fond interactif diplome
        self.interactiveEtoile_enabled = False  # Activer/désactiver le fond interactif etoile
        self.background_enabled = False  # Activer/désactiver l'arrière plan
        
        # Initialisation des diplomes
        self.diplomes = []  # stocke les diplome
        num_initial_diplomes = 100
        for _ in range(num_initial_diplomes):
            self.diplomes.append(self.initialize_interactif(int(self.cap.get(3)), int(self.cap.get(4))))

        # Initialisation des etoiles
        self.etoiles = []  # stocke les etoiles
        num_initial_etoiles = 100
        for _ in range(num_initial_etoiles):
            self.etoiles.append(self.initialize_interactif(int(self.cap.get(3)), int(self.cap.get(4))))

        # Stocke une copie de la frame original
        self.original_frame = self.frame.copy()
        # Start the video processing loop
        self.activeFiltre_video()

    #########################################################################################
    ###### Change la couleur du bouton en fonction de son activation ########################
    #########################################################################################
    
    def update_button_state(self, button, enabled):
        if enabled:
            button.configure(bg="#7BB544", fg="white")
        else:
            button.configure(bg="#C70039", fg="white")
    
    
    #########################################################################################
    #### Appliquez un filtre au choix  : Sépia ##############################################
    #########################################################################################

    def toggle_sepia(self):
        self.sepia_enabled = not self.sepia_enabled
        self.update_button_state(self.toggle_sepia_button, self.sepia_enabled)

    
    def apply_sepia(self, frame):
        # Convertir en float pour éviter la perte d'informations
        sepia_frame = np.array(frame, dtype=np.float64)
        
        # Appliquer la transformation sepia
        sepia_frame = cv2.transform(sepia_frame, np.matrix([[0.272, 0.543, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]))
        
        # Clip les valeurs pour s'assurer qu'elles restent dans la plage [0, 255]
        sepia_frame = np.clip(sepia_frame, 0, 255)
        
        # Convertir de nouveau en uint8 pour obtenir une image valide
        sepia_frame = np.array(sepia_frame, dtype=np.uint8)

        return sepia_frame
 
    #########################################################################################
    ####### Incrustez une image sur le visage et les yeux : Lunettes ########################
    #########################################################################################
    
    def toggle_lunettes(self):
        self.lunette_enabled = not self.lunette_enabled
        self.update_button_state(self.toggle_lunettes_button, self.lunette_enabled)

    def apply_lunettes(self, frame):
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) for face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                # Choose the two eyes with the highest y-coordinate
                eyes = sorted(eyes, key=lambda x: x[1], reverse=True)[:2]

                # Calculate the position and size of the sunglasses
                x1, y1, w1, h1 = eyes[0]
                x2, y2, w2, h2 = eyes[1]
                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y

                # Resize the sunglasses image to fit the face region
                lunettesResize = cv2.resize(self.lunettes, (w, h))

                # Get the alpha channel from the sunglasses image
                lunettesAlpha = lunettesResize[:, :, 3] / 255.0

                # Blend the sunglasses with the face
                for c in range(0, 3):
                    roi_color[y:y + h, x:x + w, c] = (1 - lunettesAlpha) * roi_color[y:y + h, x:x + w, c] + lunettesAlpha * lunettesResize[:, :, c]

        return frame

    #########################################################################################
    ####### Incrustez une image sur la tete : Chapeau #######################################
    #########################################################################################
    def toggle_hat(self):
        self.hat_enabled = not self.hat_enabled
        self.update_button_state(self.toggle_hat_button, self.hat_enabled)

    def apply_hat(self, frame):
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) for face
            roi_color = frame[y:y + h, x:x + w]

            # Resize the hat image to fit the face region
            hat_resized = cv2.resize(self.hat, (w, h))

            # Get the alpha channel from the hat image
            alpha_hat = hat_resized[:, :, 3] / 255.0

            # Calculate the position to place the hat on the head
            x_offset = x
            y_offset = max(y - int(h * 0.8), 0)  # Adjust the height of the hat

            # Blend the hat with the face
            for c in range(0, 3):
                frame[y_offset:y_offset + h, x_offset:x_offset + w, c] = (
                        (1 - alpha_hat) * frame[y_offset:y_offset + h, x_offset:x_offset + w, c]
                        + alpha_hat * hat_resized[:, :, c]
                )

        return frame

    #########################################################################################
    ####### Incrustez une image sur la bouche : siflet de fete ##############################
    #########################################################################################
    
    # def toggle_siflet(self):
    #     self.siflet_enabled = not self.siflet_enabled
    #     self.update_button_state(self.toggle_siflet_button, self.siflet_enabled)

    # def apply_siflet(self, frame):
    #     # Convertir l'image en niveaux de gris pour la détection de visages
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     # Détecter les visages dans l'image
    #     faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    #     # Incruster le gif animé sur chaque bouche détectée
    #     for (x, y, w, h) in faces:
    #         # Récupérer le prochain frame du gif
    #         try:
    #             gif_frame = cv2.cvtColor(cv2.resize(self.gif.get_next_data(), (w, h)), cv2.COLOR_RGB2BGR)
    #         except:
    #             # Si le gif est terminé, revenir au début
    #             self.gif.seek(0)
    #             gif_frame = cv2.cvtColor(cv2.resize(self.gif.get_next_data(), (w, h)), cv2.COLOR_RGB2BGR)

    #         # Incruster le gif sur la bouche
    #         roi = frame[y:y+h, x:x+w]
    #         mask = cv2.cvtColor(gif_frame, cv2.COLOR_BGR2GRAY)
    #         _, mask_inv = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)
    #         img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    #         img_fg = cv2.bitwise_and(gif_frame, gif_frame, mask=mask)
    #         frame[y:y+h, x:x+w] = cv2.add(img_bg, img_fg)

    #     return frame


    #########################################################################################
    ####### Incrustez une image interactive dans le fond : diplome ##########################
    #########################################################################################
   
    def toggle_interactiveDiplome(self):
        self.interactiveDiplome_enabled = not self.interactiveDiplome_enabled
        self.update_button_state(self.toggle_interactive_diplome_button, self.interactiveDiplome_enabled)

    def initialize_interactif(self, video_width, video_height):
        x = random.randint(0, video_width - 30)
        y = random.randint(0, video_height - 30)
        return {"x": x, "y": y}
    
    # Fonction pour vérifier si un point se trouve dans la région du visage détectée
    def is_point_inside_face(self, point, face_regions):
        for (x, y, w, h) in face_regions:
            if x - 30 < point["x"] < x + w + 30 and y - 50 < point["y"] < y + h:
                return True
        return False
    
    def apply_diplome_interactive(self, frame):
        # Convertir le frame en niveaux de gris.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectection du visage
        face_regions = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Mettre à jour et dessiner chaque diplome
        for diplome in self.diplomes:
            alpha_channel = self.diplome_img[:, :, 3] / 255.0

            # Vérifier si le diplome est à l'intérieur de la région du visage détectée
            if self.is_point_inside_face(diplome, face_regions):
                # If yes, reset its position to the top
                diplome["y"] = 0
                diplome["x"] = random.randint(0, int(self.cap.get(3)) - 30)

            for c in range(0, 3):
                frame[diplome["y"]:diplome["y"] + 30, diplome["x"]:diplome["x"] + 30, c] = \
                    (1 - alpha_channel) * frame[diplome["y"]:diplome["y"] + 30, diplome["x"]:diplome["x"] + 30, c] + \
                    alpha_channel * self.diplome_img[:, :, c]

            # Mettre à jour la position du diplome pour la prochaine itération
            diplome["y"] += 1
            # Si le diplome atteint le bas, réinitialiser sa position en haut
            if diplome["y"] > self.cap.get(4) - 30:
                diplome["y"] = 0
                diplome["x"] = random.randint(0, int(self.cap.get(3)) - 30)

        return frame
    
    ##################### Etoile ############################################################

    def toggle_interactiveEtoile(self):
        self.interactiveEtoile_enabled = not self.interactiveEtoile_enabled
        self.update_button_state(self.toggle_interactive_etoile_button, self.interactiveEtoile_enabled)

    def apply_etoile_interactive(self, frame):
        # Convertir le frame en niveaux de gris.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectection du visage
        face_regions = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Mettre à jour et dessiner chaque diplome
        for etoile in self.etoiles:
            alpha_channel =  self.etoile_img[:, :, 3] / 255.0

            # Vérifier si le diplome est à l'intérieur de la région du visage détectée
            if self.is_point_inside_face(etoile, face_regions):
                # If yes, reset its position to the top
                etoile["y"] = 0
                etoile["x"] = random.randint(0, int(self.cap.get(3)) - 30)

            for c in range(0, 3):
                frame[etoile["y"]:etoile["y"] + 30, etoile["x"]:etoile["x"] + 30, c] = \
                    (1 - alpha_channel) * frame[etoile["y"]:etoile["y"] + 30, etoile["x"]:etoile["x"] + 30, c] + \
                    alpha_channel *  self.etoile_img[:, :, c]

            # Mettre à jour la position du diplome pour la prochaine itération
            etoile["y"] += 1
            # Si le diplome atteint le bas, réinitialiser sa position en haut
            if etoile["y"] > self.cap.get(4) - 30:
                etoile["y"] = 0
                etoile["x"] = random.randint(0, int(self.cap.get(3)) - 30)

        return frame
    

    #########################################################################################
    ####### Changer le fond de la vidéo #####################################################
    #########################################################################################

    def toggle_background(self):
        self.background_enabled = not self.background_enabled
        self.update_button_state(self.toggle_background_button, self.background_enabled)

    def apply_background(self, frame):
        height, width, _ = frame.shape
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.segmentation.process(RGB)
        mask = results.segmentation_mask

        background = cv2.resize(self.background_image, (width, height))
        
        # Mélanger directement le filtre d'arrière-plan avec la vidéo
        output = np.where(mask[:, :, None] > 0.6, frame, background)

        return output



    #########################################################################################
    ####### Activer les filtres #############################################################
    #########################################################################################

    def activeFiltre_video(self):
        ret, frame = self.cap.read()

        if ret:
            # Activer le filtre sepia
            if self.sepia_enabled:
                frame = self.apply_sepia(frame)

            # Activer le filtre des lunettes
            if self.lunette_enabled:
                frame = self.apply_lunettes(frame)
                        
            # Activer le filtre du chapeau
            if self.hat_enabled:
                frame = self.apply_hat(frame)
            
            # # Activer le filtre du siflet
            # if self.siflet_enabled:
            #     frame = self.apply_siflet(frame)

            # Activer le filtre des diplomes
            if self.interactiveDiplome_enabled:
                frame = self.apply_diplome_interactive(frame)

             # Activer le filtre des etoiles
            if self.interactiveEtoile_enabled:
                frame = self.apply_etoile_interactive(frame)
            
            # Activer le filtre de l'arriere plan
            if self.background_enabled:
                frame = self.apply_background(frame)

            # Afficher le résultat
            frame_flip = cv2.flip(frame, 1) # inverse la video 
            img = Image.fromarray(cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.img = img_tk
            self.video_label.configure(image=img_tk)

        # Reprend la video apres l'application du filtre
        self.root.after(10, self.activeFiltre_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = AppTraitementImage(root)
    root.mainloop()
