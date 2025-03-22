import os
import pickle
import tkinter as tk
from tkinter import messagebox
import face_recognition


def get_button(window, text, color, command, fg='white'):
    """
    Creates and returns a styled button.
    """
    button = tk.Button(
        window,
        text=text,
        activebackground="black",
        activeforeground="white",
        fg=fg,
        bg=color,
        command=command,
        height=2,
        width=20,
        font=('Helvetica bold', 20))
    return button


def get_img_label(window):
    """
    Creates and returns an image label.
    """
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    """
    Creates and returns a text label.
    """
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    """
    Creates and returns a text input field.
    """
    inputtxt = tk.Text(window, height=2, width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    """
    Displays a message box with the given title and description.
    """
    messagebox.showinfo(title, description)


def recognize(img, db_path):
    """
    Recognizes a face in the given image by comparing it with the faces in the database.

    Args:
        img: The image to recognize (in RGB format).
        db_path: Path to the directory containing the face embeddings.

    Returns:
        str: The name of the recognized person, 'unknown_person', or 'no_persons_found'.
    """
    # Ensure the image is in RGB format
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Image must be in RGB format.")

    # Get face encodings for the input image
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'  # No faces detected
    embeddings_unknown = embeddings_unknown[0]  # Use the first face found

    # Load all embeddings from the database
    db_dir = sorted(os.listdir(db_path))
    match = False
    j = 0

    # Compare the input face with all faces in the database
    while not match and j < len(db_dir):
        path_ = os.path.join(db_path, db_dir[j])
        with open(path_, 'rb') as file:
            embeddings = pickle.load(file)
            match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]
        j += 1

    # Return the matched person's name or 'unknown_person'
    if match:
        return db_dir[j - 1][:-7]  # Remove '.pickle' from the filename
    else:
        return 'unknown_person'