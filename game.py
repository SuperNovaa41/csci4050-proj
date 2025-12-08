import tkinter as tk
from PIL import Image, ImageTk

# For now forget about formatting and just set up 
# page switching with different scenarios

class MyApp(tk.Frame):
    def __init__(self, root):

        self.current_page_index = 0
        self.pages = [self.create_page_container, self.hint_page, self.game_page1, self.game_page2,self.game_page3,self.game_page4,self.game_page5]

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
        self.pages[self.current_page_index]()

    # When switching pages, clear page first
    def clear_frame(self,frame):
        for child in frame.winfo_children():
            child.destroy()

    # Container for whole page - set up right amount of columns and rows for titles
    # Authors, rules label, rules text box, and start button, with background image
    def create_page_container(self):
        self.page_container = tk.Frame(
            self.main_frame
        )

        def change_page():
            self.clear_frame(self.page_container)
            self.current_page_index = 1
            self.pages[self.current_page_index]()

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

        # Makes container take up the whole screen
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
        start_btn = tk.Button(self.page_container, text="Start", font=("Helvetica", 20), fg="white", bg="#217fdd",command=change_page)
        start_btn.grid(row=4,column=1)

    # Will have new background (or same image) with some image of a character, textbox with there story
    # survive or not buttons, next button. Each survive/not will run the model, use placeholder for now.

    # This page black with textbox of hints to aid users in geussing.
    def hint_page(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#000000"
        )

        def change_page():
            self.clear_frame(self.page_container)
            self.current_page_index = 2
            self.pages[self.current_page_index]()

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        hint = """
        To aid you in making your choices - women and children were prioritized
        access to lifeboats. The nicer ($$$) living quarters were closer to the top of the ship.
        Lifeboats were deployed from the top of the ship.
        """

        hint_txt = tk.Label(self.page_container,text=hint, font=("Helvetica", 20),wraplength=500,fg="white",bg="#000000",justify="center")
        hint_txt.grid(row=0, column=0,sticky="s",columnspan=2)
        continue_btn = tk.Button(self.page_container, text="Continue", font=("Helvetica", 20), fg="white", bg="#000000",command=change_page)
        continue_btn.grid(row=1,column=1, sticky="s")

    # Finish format for game page, and other pages
    # Then connect to model to actually run geusses
    def game_page1(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )

        def change_page():
            self.clear_frame(self.page_container)
            self.current_page_index += 1
            self.pages[self.current_page_index]()

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)
        self.page_container.columnconfigure(2,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # Didn't Survive - PassengerId = 1, info from dataset + made up details on top
        # related to the dataset info
        # Will store an array in here to and run it through dataset later
        story = """
        Story 1:
        Hi, I'm Mr. Owen Harris Braund. I heard about the Titanic all over the news
        and just had to go. I worked extra hours as a server to make just enough
        for a 3rd class ticket. Its hard finding work at my age as I'm only 22, not
        many people want to hire someone like me, especially with no university education.
        I worked very hard to be here, I even managed to grab an extra ticket for my brother Lewis. 
        I see all these families around, I'm so grateful I don't have any kids to look
        after, that seems like a tough job. Though, some of those families are living on 
        the upper decks, I'm in a shared cabin at the bottom of the ship.
        Theres always a compromise, but I'm enjoying my time here anyways.
        """
        
        # Titanic man 1
        self.portrait = ImageTk.PhotoImage(Image.open("titanicman1.jpg").resize((450, 600), Image.Resampling.LANCZOS))
        image_label = tk.Label(self.page_container, image=self.portrait)
        image_label.grid(row=0,column=0,rowspan=4)
        story_txt = tk.Label(self.page_container,text=story, font=("Helvetica", 12),wraplength=500,fg="white",bg="#217fdd",anchor="w")
        story_txt.grid(row=0, column=1,columnspan=2,sticky="w")
        question_txt = tk.Label(self.page_container,text="Did they survive the Titanic?", font=("Helvetica", 12),fg="white",bg="#217fdd")
        question_txt.grid(row=1, column=1,columnspan=2,sticky="n")

        # Here needs to connect to the model - must create functions storing user input 
        # yes = 1, no = 0, if yes_btn.click() or something like that
        no_btn = tk.Button(self.page_container, text="No", font=("Helvetica", 20), fg="white", bg="#7A0A0A")
        no_btn.grid(row=2,column=1,sticky="s")
        yes_btn = tk.Button(self.page_container, text="Yes", font=("Helvetica", 20), fg="white", bg="#0b711a")
        yes_btn.grid(row=2,column=2,sticky="s")

        next_btn = tk.Button(self.page_container, text="Next", font=("Helvetica", 20), fg="white", bg="#217fdd",command=change_page)
        next_btn.grid(row=3,column=2,sticky="se")
    
    def game_page2(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )

        def change_page():
            self.clear_frame(self.page_container)
            self.current_page_index += 1
            self.pages[self.current_page_index]()

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)
        self.page_container.columnconfigure(2,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # Survived - PassengerId = 3
        story = """
        Story 2:
        How do you do? My names Miss. Laina Heikkinen. As you can tell, I'm not married. 
        In fact I'm here all alone. My grandma gifted me a ticket here for my birthday.
        We aren't wealthy so I'm staying in a shared cabin on the lower decks, but
        it doesn't matter, its such a beautiful ship and I'm having a great time.
        I can't wait to tell my family all about the experience!
        """
        
        # Titanic woman 2
        self.portrait = ImageTk.PhotoImage(Image.open("titanicwoman2.jpg").resize((450, 600), Image.Resampling.LANCZOS))
        image_label = tk.Label(self.page_container, image=self.portrait)
        # Change to grid
        # image_label.place(x=0, y=0, relwidth=1, relheight=1)
        # image_label.lower()
        image_label.grid(row=0,column=0,rowspan=4)
        story_txt = tk.Label(self.page_container,text=story, font=("Helvetica", 12),wraplength=500,fg="white",bg="#217fdd",anchor="w")
        story_txt.grid(row=0, column=1,columnspan=2,sticky="w")
        question_txt = tk.Label(self.page_container,text="Did they survive the Titanic?", font=("Helvetica", 12),fg="white",bg="#217fdd")
        question_txt.grid(row=1, column=1,columnspan=2,sticky="n")
        no_btn = tk.Button(self.page_container, text="No", font=("Helvetica", 20), fg="white", bg="#7A0A0A")
        no_btn.grid(row=2,column=1,sticky="s")
        yes_btn = tk.Button(self.page_container, text="Yes", font=("Helvetica", 20), fg="white", bg="#0b711a")
        yes_btn.grid(row=2,column=2,sticky="s")
        next_btn = tk.Button(self.page_container, text="Next", font=("Helvetica", 20), fg="white", bg="#217fdd",command=change_page)
        next_btn.grid(row=3,column=2,sticky="se")

    def game_page3(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )

        def change_page():
            self.clear_frame(self.page_container)
            self.current_page_index += 1
            self.pages[self.current_page_index]()

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)
        self.page_container.columnconfigure(2,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # Suvived - PassengerId = 18
        story = """
        Story 3:
        Hello, I'm Mr. Charles Eugene Williams, but you can call me Charles.
        I'm taking a vacation from my ungrateful family, I work so hard, buy them
        tons of nice things, but all they do is complain. I heard about the Titanic
        on the news and quickly bought this 2nd class ticket, all the first class
        ones were sold out, or I woulda bought 2, one for me, and one for my bags.
        I'm close enough to all the amenities on the top floor, so I don't mind being
        where I am. 
        """
        
        # Titanic man 3
        # Width  * height
        self.portrait = ImageTk.PhotoImage(Image.open("titanicman3.jpg").resize((450, 600), Image.Resampling.LANCZOS))
        image_label = tk.Label(self.page_container, image=self.portrait)
        image_label.grid(row=0,column=0,rowspan=4,sticky="w")
        story_txt = tk.Label(self.page_container,text=story, font=("Helvetica", 12),wraplength=500,fg="white",bg="#217fdd",anchor="w")
        story_txt.grid(row=0, column=1,columnspan=2,sticky="w")
        question_txt = tk.Label(self.page_container,text="Did they survive the Titanic?", font=("Helvetica", 12),fg="white",bg="#217fdd")
        question_txt.grid(row=1, column=1,columnspan=2,sticky="n")


        no_btn = tk.Button(self.page_container, text="No", font=("Helvetica", 20), fg="white", bg="#7A0A0A")
        no_btn.grid(row=2,column=1,sticky="s")
        yes_btn = tk.Button(self.page_container, text="Yes", font=("Helvetica", 20), fg="white", bg="#0b711a")
        yes_btn.grid(row=2,column=2,sticky="s")


        next_btn = tk.Button(self.page_container, text="Next", font=("Helvetica", 20), fg="white", bg="#217fdd",command=change_page)
        next_btn.grid(row=3,column=2,sticky="se")

    def game_page4(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )

        def change_page():
            self.clear_frame(self.page_container)
            self.current_page_index += 1
            self.pages[self.current_page_index]()

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)
        self.page_container.columnconfigure(2,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # Survived - PassengerId = 44
        story = """
        Story 4:
        Goo goo ga ga. I'm Miss. Simone Marie Anne Andree Laroche! I'm only 3! My parents 
        are amazing and buy me tons of nice things!
        Like this 2nd class ticket to the Titanic! This ship is amazing!
        """
        
        self.portrait = ImageTk.PhotoImage(Image.open("titanicbaby.jpg").resize((450, 600), Image.Resampling.LANCZOS))
        image_label = tk.Label(self.page_container, image=self.portrait)
        image_label.grid(row=0,column=0,rowspan=4,sticky="w")
        story_txt = tk.Label(self.page_container,text=story, font=("Helvetica", 12),wraplength=500,fg="white",bg="#217fdd",anchor="w")
        story_txt.grid(row=0, column=1,columnspan=2,sticky="w")
        question_txt = tk.Label(self.page_container,text="Did they survive the Titanic?", font=("Helvetica", 12),fg="white",bg="#217fdd")
        question_txt.grid(row=1, column=1,columnspan=2,sticky="n")


        no_btn = tk.Button(self.page_container, text="No", font=("Helvetica", 20), fg="white", bg="#7A0A0A")
        no_btn.grid(row=2,column=1,sticky="s")
        yes_btn = tk.Button(self.page_container, text="Yes", font=("Helvetica", 20), fg="white", bg="#0b711a")
        yes_btn.grid(row=2,column=2,sticky="s")


        next_btn = tk.Button(self.page_container, text="Next", font=("Helvetica", 20), fg="white", bg="#217fdd",command=change_page)
        next_btn.grid(row=3,column=2,sticky="se")

    def game_page5(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )

        def change_page():
            self.clear_frame(self.page_container)
            self.current_page_index += 1
            self.pages[self.current_page_index]()

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)
        self.page_container.columnconfigure(2,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # Survived - PassengerId = 86
        story = """
        Story 5:
        Well hey there, my names Mrs. Karl Alfred. My husband thought it would be nice to surprise 
        the family with Titanic tickets, so here we are. Our quarters are cramped, but its a fabulous
        ship with tons to do. If only the kid didn't keep running away, at least were getting
        good exercise.
        """
        
        self.portrait = ImageTk.PhotoImage(Image.open("titanicmom.jpg").resize((450, 600), Image.Resampling.LANCZOS))
        image_label = tk.Label(self.page_container, image=self.portrait)
        image_label.grid(row=0,column=0,rowspan=4,sticky="w")
        story_txt = tk.Label(self.page_container,text=story, font=("Helvetica", 12),wraplength=500,fg="white",bg="#217fdd",anchor="w")
        story_txt.grid(row=0, column=1,columnspan=2,sticky="w")
        question_txt = tk.Label(self.page_container,text="Did they survive the Titanic?", font=("Helvetica", 12),fg="white",bg="#217fdd")
        question_txt.grid(row=1, column=1,columnspan=2,sticky="n")


        no_btn = tk.Button(self.page_container, text="No", font=("Helvetica", 20), fg="white", bg="#7A0A0A")
        no_btn.grid(row=2,column=1,sticky="s")
        yes_btn = tk.Button(self.page_container, text="Yes", font=("Helvetica", 20), fg="white", bg="#0b711a")
        yes_btn.grid(row=2,column=2,sticky="s")


        next_btn = tk.Button(self.page_container, text="Next", font=("Helvetica", 20), fg="white", bg="#217fdd",command=change_page)
        next_btn.grid(row=3,column=2,sticky="se")

root = tk.Tk()
root.title('Titanic Guesser')
root.geometry("1000x650")
root.resizable(width=False,height=False)
# Passing root to a frame that packs it on the screen
app_instance = MyApp(root)
root.mainloop()