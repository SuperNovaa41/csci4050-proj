import tkinter as tk
from PIL import Image, ImageTk

# For now forget about formatting and just set up 
# page switching with different scenarios

class MyApp(tk.Frame):
    def __init__(self, root):
        super().__init__(
            root
        )

        self.main_frame = self
        self.main_frame.pack(fill="both", expand=True)
        # Frame is 1 column, 0=column index, weight=1 means can stretch
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0,weight=1)

        self.load_main_widgets()

    def load_main_widgets(self):
        self.create_page_container()
        #self.create_pager()

    # Container for whole page - set up right amount of columns and rows for titles
    # Authors, rules label, rules text box, and start button, with background image
    def create_page_container(self):
        self.page_container = tk.Frame(
            self.main_frame
        )

        # Background image
        self.bg_image = ImageTk.PhotoImage(Image.open("titanic.jpg"))
        image_label = tk.Label(self.page_container, image=self.bg_image)
        image_label.place(x=0, y=0, relwidth=1, relheight=1)
        image_label.lower()

        nrows = 5

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

        self.page_container.columnconfigure(0,weight=0)
        # 2nd column fills rest of screen
        self.page_container.columnconfigure(1,weight=1)

        for i in range(nrows):
            # Add padding to start button
            if i == 4:
                self.page_container.rowconfigure(i,weight=1)
            else:
                self.page_container.rowconfigure(i,weight=0)

        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # This stuff would be moved to a different function, this not structure but what
        # goes in the structure, and root should be changed
        title_txt = tk.Label(self.page_container, text="Titanic Geusser", font=("Helvetica", 30), fg="white",bg="#217fdd") 
        title_txt.grid(row=0, column=1)
        author_txt = tk.Label(self.page_container, text="Olly Love, Nathan Singer, David Kelly", font=("Helvetica", 15), fg="white",bg="#217fdd")
        author_txt.grid(row=1,column=1)
        rule_btn = tk.Button(self.page_container, text="Rules", font=("Helvetica", 15), fg="white", bg="#217fdd")
        rule_btn.grid(row=2,column=0,sticky="w")
        # FORMAT ERROR - Why not going on the left?
        rule_txt = tk.Label(self.page_container, text=rules, font=("Helvetica", 12),fg="white",bg="#217fdd",wraplength=400,anchor="w",justify="left")
        rule_txt.grid(row=3,column=0,sticky="w")
        # Want to add an on click to change scenes
        start_btn = tk.Button(self.page_container, text="Start", font=("Helvetica", 20), fg="white", bg="#217fdd")
        start_btn.grid(row=4,column=1)

    # Will have new background with some image of a character, textbox with there story
    # survive or not buttons, next button. Each survive/not will run the model, use placeholder for now.
    def page1():
        pass

root = tk.Tk()
root.title('Titanic Guesser')
root.geometry("1000x650")
root.resizable(width=False,height=False)
# Passing root to a frame that packs it on the screen
app_instance = MyApp(root)
root.mainloop()

# First scene - intro describing the game

# Text label title
# Text label authors 
# Info/how to play button revealing textbox detailing what game is and how model works with it
# with button allowing user to start the game

# Set root window and window size (width x height)
# root = tk.Tk()
# root.geometry("1000x650")

# image = Image.open('titanic.jpg')
# image = ImageTk.PhotoImage(image)

# # Put image in a label - basic - not good for background image
# image_label = tk.Label(root, image=image)
# #image_label.pack()
# image_label.place(x=0,y=0,relwidth=1,relheight=1)
# # Place behind other widgets
# image_label.lower()

# # For rules label
# rules = """
# The sinking of the Titanic in 1912 was a tragic event that took hundreds of lives. 
# It turns out that many factors influenced whether someone would survive the crash.
# You will be introduced to former passengers of the Titanic. You
# will listen to their stories, investigate the scene, and determine whether
# you think they would survive the crash.
# This is a Machine Learning game. Your geuss will be run through a trained ML Model.
# This model will decide whether you are correct, based on its training data.
# """

# # When rules button is clicked, want to make a label visible
# # describing the game + model
# # Don't know how to make button visible and not visible
# # def on_click():
# #     rule_txt.pack(anchor="w")

# # frame1 = tk.Frame(root, bg="#096bcd")
# # frame1.pack()

# title_txt = tk.Label(root, text="Titanic Geusser", font=("Helvetica", 30), fg="white",bg="#217fdd") 
# title_txt.pack()
# author_txt = tk.Label(root, text="Olly Love, Nathan Singer, David Kelly", font=("Helvetica", 15), fg="white",bg="#217fdd")
# author_txt.pack()
# rule_btn = tk.Button(root, text="Rules", font=("Helvetica", 15), fg="white", bg="#217fdd")
# rule_btn.pack(anchor="w", pady=20)
# rule_txt = tk.Label(root, text=rules, font=("Helvetica", 12), fg="white",bg="#217fdd",wraplength=400,justify="left")
# rule_txt.pack(anchor="w")
# # Want to add an on click to change scenes
# start_btn = tk.Button(root, text="Start", font=("Helvetica", 20), fg="white", bg="#217fdd")
# start_btn.pack(pady=20)

# Sample showing frame of buttons

# Can have frame for bunch of buttons in a row, think can have frames within frames
# effectively a frame can be used as a container
# my_frame = tk.Frame(root,bg="#096bcd")
# my_frame.pack(pady=200)

# # Add some buttons
# my_button1 = tk.Button(my_frame, text="Exit")
# # When adding to a frame must use grid
# my_button1.grid(row=0,column=0,padx=20)

# my_button2 = tk.Button(my_frame, text="Start")
# # When adding to a frame must use grid
# my_button2.grid(row=0,column=1,padx=20)

# my_button3 = tk.Button(my_frame, text="Reset")
# # When adding to a frame must use grid
# my_button3.grid(row=0,column=2,padx=20)

#root.mainloop()