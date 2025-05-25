import time
import tkinter as tk
from tkinter import Button, Label, filedialog

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from torch.autograd import Variable

from model import decoder, encoder

GPU = 0

class VideoApp:
    def __init__(self, root, model, histogram_equalize = True):
        self.root = root
        self.root.title("Underwater Video Enhancement")
        self.root.geometry("1100x600")
        self.root.configure(bg="black")
        self.model = model
        self.histogram_equalize = histogram_equalize

        # Set up frames
        self.input_frame_label = Label(
            root, text="Input Video", font=("Arial", 16), bg="black", fg="white")
        self.input_frame_label.grid(row=0, column=0, padx=10, pady=10)

        self.output_frame_label = Label(
            root, text="Output Video", font=("Arial", 16), bg="black", fg="white")
        self.output_frame_label.grid(row=0, column=1, padx=10, pady=10)

        self.input_video = Label(
            root, highlightbackground="#944dff", highlightthickness=2)
        self.input_video.grid(row=1, column=0, padx=10, pady=10)

        self.output_video = Label(
            root, highlightbackground="#944dff", highlightthickness=2)
        self.output_video.grid(row=1, column=1, padx=10, pady=10)

        self.upload_button = Button(
            root, text="Upload Video", command=self.upload_video, font=("Arial", 14))
        self.upload_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Video variables
        self.video_path = None
        self.cap = None

    def upload_video(self):
        """Handle video upload and start processing."""
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov"),
                       ("All files", "*.*")]
        )
        if self.video_path:
            # Open video file
            self.cap = cv2.VideoCapture(self.video_path)
            self.update_frames()

    def equalize(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        if not self.histogram_equalize:
            return lab
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_image = cv2.merge((l, a, b))
        return enhanced_image

    def update_frames(self):
        """Update video frames in the UI."""
        ret, frame = self.cap.read()
        if ret:
            # Resize frame for the input display
            input_resized = cv2.resize(frame, (480, 360))

            # Convert input frame to ImageTk format
            input_image = ImageTk.PhotoImage(Image.fromarray(
                cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)))
            self.input_video.configure(image=input_image)
            self.input_video.image = input_image

            enhanced_image = self.equalize(input_resized)

            # Process the frame for the output display
            output_frame = self.model.enhance(cv2.cvtColor(
                enhanced_image, cv2.COLOR_LAB2RGB))
            output_resized = cv2.resize(output_frame, (480, 360))

            # Convert output frame to ImageTk format
            output_bgr = cv2.cvtColor(output_resized, cv2.COLOR_RGB2BGR)
            output_image = ImageTk.PhotoImage(Image.fromarray(
                cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)))
            self.output_video.configure(image=output_image)
            self.output_video.image = output_image

        # Schedule the next frame update
        self.root.after(10, self.update_frames)

    def on_close(self):
        # Release resources when the app is closed.
        if self.cap:
            self.cap.release()
        self.root.destroy()


class UVSVENet:
    def __init__(self):
        self.encoder_1 = encoder.Encoder().cuda(GPU)
        self.encoder_2 = encoder.Encoder().cuda(GPU)
        self.encoder_3 = encoder.Encoder().cuda(GPU)
        self.decoder_1 = decoder.Decoder().cuda(GPU)
        self.decoder_2 = decoder.Decoder().cuda(GPU)
        self.decoder_3 = decoder.Decoder().cuda(GPU)
        self.iteration = 0
        self.test_time = 0
        self.mv_runtime = 0

    def load(self):
        self.encoder_1.load_state_dict(torch.load(
            f"./saved/encoder_lv1.pkl"))
        self.encoder_2.load_state_dict(torch.load(
            f"./saved/encoder_lv2.pkl"))
        self.encoder_3.load_state_dict(torch.load(
            f"./saved/encoder_lv3.pkl"))
        self.decoder_1.load_state_dict(torch.load(
            f"./saved/decoder_lv1.pkl"))
        self.decoder_2.load_state_dict(torch.load(
            f"./saved/decoder_lv2.pkl"))
        self.decoder_3.load_state_dict(torch.load(
            f"./saved/decoder_lv3.pkl"))

    def enhance(self, image):
        with torch.no_grad():
            image = image.astype(np.float32) / 255.0
            images_lv1 = torch.from_numpy(image).permute(2, 0, 1)
            start = time.time()

            images_lv1 = Variable(images_lv1).unsqueeze(0).cuda(GPU)
            images_lv1 = images_lv1.data - 0.5

            H = images_lv1.size(2)
            W = images_lv1.size(3)

            # Split the image with a horizontal cut
            images_lv2_1 = images_lv1[:, :, 0:int(H/2), :]
            images_lv2_2 = images_lv1[:, :, int(H/2):H, :]

            # Split the image with vertical cuts
            images_lv3_1 = images_lv2_1[:, :, :, 0:int(W/2)]
            images_lv3_2 = images_lv2_1[:, :, :, int(W/2):W]
            images_lv3_3 = images_lv2_2[:, :, :, 0:int(W/2)]
            images_lv3_4 = images_lv2_2[:, :, :, int(W/2):W]

            # Extract features from level 3 patches (x4)
            feature_lv3_1 = self.encoder_3(images_lv3_1)
            feature_lv3_2 = self.encoder_3(images_lv3_2)
            feature_lv3_3 = self.encoder_3(images_lv3_3)
            feature_lv3_4 = self.encoder_3(images_lv3_4)

            # Concat level 3 features for level 3 decoder
            feature_lv3_top = torch.cat(
                (feature_lv3_1, feature_lv3_2), 3)
            feature_lv3_bot = torch.cat(
                (feature_lv3_3, feature_lv3_4), 3)

            # Concat level 3 features to obtain residuals for level 2 decoder
            feature_lv3 = torch.cat(
                (feature_lv3_top, feature_lv3_bot), 2)

            # Calculate residuals for level 2 encoder
            residual_lv3_top = self.decoder_3(feature_lv3_top)
            residual_lv3_bot = self.decoder_3(feature_lv3_bot)

            # Obtain level 2 features from encoder
            feature_lv2_1 = self.encoder_2(
                images_lv2_1 + residual_lv3_top[:images_lv2_1.shape[0], :images_lv2_1.shape[1], :images_lv2_1.shape[2], :images_lv2_1.shape[3]])
            feature_lv2_2 = self.encoder_2(
                images_lv2_2 + residual_lv3_bot[:images_lv2_2.shape[0], :images_lv2_2.shape[1], :images_lv2_2.shape[2], :images_lv2_2.shape[3]])

            # Concat level 2 features for decoder
            feature_lv2_pre = torch.cat(
                (feature_lv2_1, feature_lv2_2), 2)
            feature_lv2 = feature_lv2_pre + feature_lv3[:feature_lv2_pre.shape[0],
                                                        :feature_lv2_pre.shape[1], :feature_lv2_pre.shape[2], :feature_lv2_pre.shape[3]]

            # Calculate residuals for level 1 encoder
            residual_lv2 = self.decoder_2(feature_lv2)

            # Perform enhancements with level 1 network
            feature_lv1_pre = self.encoder_1(
                images_lv1 + residual_lv2[:images_lv1.shape[0], :images_lv1.shape[1], :images_lv1.shape[2], :images_lv1.shape[3]])
            feature_lv1 = feature_lv1_pre + feature_lv2[:feature_lv1_pre.shape[0],
                                                        :feature_lv1_pre.shape[1], :feature_lv1_pre.shape[2], :feature_lv1_pre.shape[3]]
            dehazed_image = self.decoder_1(feature_lv1)

            stop = time.time()

            self.mv_runtime = 0.9 * self.mv_runtime + 0.1 * (stop - start)
            self.iteration += 1
            if self.iteration % 50 == 0:
                print(f'RunTime:{self.mv_runtime}')
            output_image = dehazed_image.data + 0.5
            tensor_cpu = output_image.cpu()[0].permute(1, 2, 0)
            image_np = tensor_cpu.numpy() * 255
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            return image_np


if __name__ == "__main__":
    root = tk.Tk()
    model = UVSVENet()
    model.load()
    app = VideoApp(root, model, histogram_equalize=True)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
