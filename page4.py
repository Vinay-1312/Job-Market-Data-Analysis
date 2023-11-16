

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
import webbrowser  
from datetime import datetime

def plotDisplay(title):
    zoom_factor = 1  
    
    def on_canvas_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    def resize_image(image, width, height):
        resized_image = image.resize((width, height), Image.ANTIALIAS)
        return resized_image
    
    def zoom_in():
        nonlocal zoom_factor
        zoom_factor += 0.1
        display_selected_plot()
    
    def zoom_out():
        nonlocal zoom_factor  
        zoom_factor = max(0.1, zoom_factor - 0.1)
        display_selected_plot()
    
    def display_selected_plot():
        selected_plot = selected_plot_var.get()
        today_date = datetime.today().strftime('%Y-%m-%d')
        file_name = f"{title}_{selected_plot}_{today_date}.png"
        file_path = f"Plots/{file_name}"  
        image = Image.open(file_path)
        image = resize_image(image, 1500, 1500)
        
        width = int(image.width * zoom_factor)
        height = int(image.height * zoom_factor)
        resized_image = resize_image(image, width, height)
        
        tk_image = ImageTk.PhotoImage(resized_image)
        
        image_canvas.delete("all")
        
        image_canvas.create_image(0, 0, anchor="nw", image=tk_image)
        image_canvas.image = tk_image
    
    def open_html_file():  
        file_path = filedialog.askopenfilename(filetypes=[("HTML Files", "*.html")])
        if file_path:
            webbrowser.open(file_path)
    
    root = tk.Tk()
    root.title("Select Plot and Load Image")
    root.state("zoomed")
    
    zoom_factor = 1.0
    
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True)
    
    canvas = tk.Canvas(frame)
    canvas.pack(side="left", fill="both", expand=True)
    
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    image_canvas = tk.Canvas(canvas, bg="white", height=2000, width=2000)
    canvas.create_window((0, 0), window=image_canvas, anchor="nw")
    
    plot_frame = ttk.Frame(root)
    plot_frame.pack(padx=20, pady=10)
    plot_list = ["Azure_Clustering", "NER_Clustering",   "Skills Cloud NER","Skills Cloud Azure Analytics", "Top 10 skills using Azure Analytics", "Top 10 skills using Named Entity Recognition", "Job type", "Salary Location Scatter", "Salary Location", "Location Count", "Region Count", "Job experience distribution", "Job type distribution","Resume Matching Score"]

    selected_plot_var = tk.StringVar()
    selected_plot_var.set(plot_list[0])
    plot_dropdown = ttk.Combobox(plot_frame, textvariable=selected_plot_var, values=plot_list,width=50)
    plot_dropdown.pack(side="left", pady=10, padx=5)
    
    plot_button = ttk.Button(plot_frame, text="Display Selected Plot", command=display_selected_plot)
    plot_button.pack(side="left", pady=10, padx=5)
    
    zoom_in_button = ttk.Button(plot_frame, text="Zoom In", command=zoom_in)
    zoom_in_button.pack(side="left", pady=10, padx=5)
    
    zoom_out_button = ttk.Button(plot_frame, text="Zoom Out", command=zoom_out)
    zoom_out_button.pack(side="left", pady=10, padx=5)
    
    open_html_button = ttk.Button(plot_frame, text="Open HTML File", command=open_html_file)  
    open_html_button.pack(side="left", pady=10, padx=5) 
    
    root.mainloop()


#plotDisplay("Title")
