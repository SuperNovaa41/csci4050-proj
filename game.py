import tkinter as tk
from PIL import Image, ImageTk

# First scene - intro describing the game

# Text label title
# Text label authors 
# Info/how to play button revealing textbox detailing what game is and how model works with it
# with button allowing user to start the game

# Set root window and window size (width x height)
root = tk.Tk()
root.geometry("1000x650")

image = Image.open('titanic.jpg')
image = ImageTk.PhotoImage(image)

# Put image in a label - basic - not good for background image
image_label = tk.Label(root, image=image)
#image_label.pack()
image_label.place(x=0,y=0,relwidth=1,relheight=1)
# Place behind other widgets
image_label.lower()

# For rules label
rules = """
The sinking of the Titanic in 1912 was a tragic event that took hundreds of lives. 
It turns out that many factors influenced whether someone would survive the crash.
You will be introduced to former passengers of the Titanic. You
will listen to their stories, investigate the scene, and determine whether
you think they would survive the crash.
This is a Machine Learning game. Your geuss will be run through a trained ML Model.
This model will decide whether you are correct, based on its training data.
"""

# When rules button is clicked, want to make a label visible
# describing the game + model
def on_click():
    rule_txt.pack(anchor="w")

# frame1 = tk.Frame(root, bg="#096bcd")
# frame1.pack()

title_txt = tk.Label(root, text="Titanic Geusser", font=("Helvetica", 30), fg="white",bg="#217fdd") 
title_txt.pack()
author_txt = tk.Label(root, text="Olly Love, Nathan Singer, David Kelly", font=("Helvetica", 15), fg="white",bg="#217fdd")
author_txt.pack()
rule_btn = tk.Button(root, text="Rules", font=("Helvetica", 15), fg="white", bg="#217fdd",command=on_click)
rule_btn.pack(anchor="w", pady=20)
rule_txt = tk.Label(root, text=rules, font=("Helvetica", 12), fg="white",bg="#217fdd",wraplength=400,justify="left")
#rule_txt.pack(anchor="w")
# Want to add an on click to change scenes
start_btn = tk.Button(root, text="Start", font=("Helvetica", 20), fg="white", bg="#217fdd")
start_btn.pack(pady=20)

# Sample showing frame of buttons

# Can have frame for bunch of buttons in a row, think can have frames within frames
# effectively a frame can be used as a container
my_frame = tk.Frame(root,bg="#096bcd")
my_frame.pack(pady=200)

# Add some buttons
my_button1 = tk.Button(my_frame, text="Exit")
# When adding to a frame must use grid
my_button1.grid(row=0,column=0,padx=20)

my_button2 = tk.Button(my_frame, text="Start")
# When adding to a frame must use grid
my_button2.grid(row=0,column=1,padx=20)

my_button3 = tk.Button(my_frame, text="Reset")
# When adding to a frame must use grid
my_button3.grid(row=0,column=2,padx=20)

root.mainloop()