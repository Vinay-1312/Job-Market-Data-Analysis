import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import time
import Data_Processing
import page4
import resumeMatch
import Main
def load_pdf():
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    num_plots = plot_count.get()
    #print(num_plots)

    if pdf_path:
        
        root.destroy()
        Main.data_collection(num_plots)
        resumeMatch.main(num_plots,pdf_path)
      
        Data_Processing.mainfunction( num_plots)
        
     
        time.sleep(5)
        
   
        page4.plotDisplay(num_plots)

root = tk.Tk()
root.title("Select Plot and Load Image")


plot_frame = ttk.Frame(root)
plot_frame.pack(padx=20, pady=10)


image_frame = ttk.Frame(root)
image_frame.pack(padx=20, pady=10)
plot_count_label = tk.Label(image_frame, text="Job Ttile:")
plot_count_label.pack(side=tk.LEFT)

plot_count = tk.Entry(image_frame)
plot_count.pack(side=tk.LEFT)

generate_button = tk.Button(image_frame, text="Upload CV", command=load_pdf)
generate_button.pack(side=tk.LEFT, padx=10)

original_image = None
scale = 1.0


root.mainloop()

