import fitz
import keyboard
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

import cv2
import mediapipe as mp
import time

def get_gradient(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)

class PDFViewer:
    def __init__(self):
        self.doc = None
        self.current_page = 0
        self.tilt_start_time = None
        self.root = tk.Tk()

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        select_button = tk.Button(self.root, text="Select PDF", command=self.load_pdf)
        select_button.pack()

        self.update_display()

        self.root.bind("<Right>", self.next_page)
        self.root.bind("<Left>", self.previous_page)
        self.root.bind("<Escape>", self.close_viewer)

    def load_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.doc = fitz.open(file_path)
            self.current_page = 0
            self.update_display()

    def update_display(self):
        if self.doc:
            page = self.doc[self.current_page]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = ImageTk.PhotoImage(img)

            self.image_label.config(image=img)
            self.image_label.image = img

    def next_page(self, event):
        if self.doc:
            self.current_page = min(self.current_page + 1, self.doc.page_count - 1)
            self.update_display()

    def previous_page(self, event):
        if self.doc:
            self.current_page = max(self.current_page - 1, 0)
            self.update_display()

    def close_viewer(self, event):
        if self.doc:
            self.doc.close()
        self.root.destroy()

    def check_and_draw_line(self, frame, x_69, y_69, x_299, y_299):
        gradient = get_gradient(x_69, y_69, x_299, y_299)

        if abs(gradient) > 0.4:
            if self.tilt_start_time is None:
                self.tilt_start_time = time.time()
            elif time.time() - self.tilt_start_time > 1.0:
                if gradient > 0.4:
                    cv2.line(frame, (x_69, y_69), (x_299, y_299), (255, 0, 0), 2)
                    print("Previous page")
                    self.previous_page(None)
                elif gradient < -0.4:
                    cv2.line(frame, (x_69, y_69), (x_299, y_299), (0, 0, 255), 2)
                    print("Next page")
                    self.next_page(None)
                self.tilt_start_time = None
        else:
            self.tilt_start_time = None

    def run(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        # Initialize VideoCapture
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe FaceMesh Image processing
            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # specific landmarks on forehead
                    landmark_69 = face_landmarks.landmark[69]
                    landmark_299 = face_landmarks.landmark[299]

                    # coordinates for point 69
                    x_69 = int(landmark_69.x * frame.shape[1])
                    y_69 = int(landmark_69.y * frame.shape[0])

                    # coordinates for point 299
                    x_299 = int(landmark_299.x * frame.shape[1])
                    y_299 = int(landmark_299.y * frame.shape[0])

                    # Draw points at the specified points
                    cv2.circle(frame, (x_69, y_69), 5, (0, 255, 0), -1)  # Point 69
                    cv2.circle(frame, (x_299, y_299), 5, (0, 255, 0), -1)  # Point 299

                    # Check head tilt and draw line
                    self.check_and_draw_line(frame, x_69, y_69, x_299, y_299)

            # Display output
            cv2.imshow('Face Mesh Detection', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.root.update()

            if not self.root.winfo_exists():
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pdf_viewer = PDFViewer()
    pdf_viewer.load_pdf()
    pdf_viewer.run()
