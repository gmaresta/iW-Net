

import sys
import pickle
#from tkinter import Tk, ttk, Label, Button, Entry, StringVar, DISABLED, NORMAL, END, N, S, E, W, HORIZONTAL 

import tkinter as tk

from tkinter.filedialog import askopenfilename
from skimage.transform import resize
from PIL import Image, ImageTk
from matplotlib import pyplot
from IPython import get_ipython
import numpy as np
from skimage.segmentation import find_boundaries
from utils import normFunc, attractionField3D
from scipy.ndimage.interpolation import zoom
from create_model import load_model
from scipy.ndimage.measurements import label

from matplotlib import pyplot as plt
import time




from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

def filterSegmentation(x):
    lab, n_label = label(x[0,:,:,:,0],np.ones((3,3,3)))
    
    dim = 0
    best_label = 1
    
    if n_label > 1: #keep the largest segmented object also
        for y in range(1,n_label+1):
            temp = lab==y
            temp_dim = np.sum(temp)
            if temp_dim > dim:
                dim = temp_dim
                best_label = y
    return np.expand_dims(np.expand_dims(lab==best_label,axis=0),axis=-1)

def createOverlay(im,mask,color=(0,1,0),contour=True):
    if len(im.shape)==2:
        im = np.expand_dims(im,axis=-1)
        im = np.repeat(im,3,axis=-1)
    elif len(im.shape)==3:
        if im.shape[-1] != 3:
            ValueError('Unexpected image format. I was expecting either (X,X) or (X,X,3), instead found', im.shape)

    else:
        ValueError('Unexpected image format. I was expecting either (X,X) or (X,X,3), instead found', im.shape)
   
    if contour:
        bw = find_boundaries(mask,mode='inner') #thick
    else:
        bw = mask
    for i in range(0,3):
        im_temp = im[:,:,i]
        im_temp = np.multiply(im_temp,np.logical_not(bw)*1)
        im_temp += bw*color[i]
        im[:,:,i] = im_temp
    return im

def preprocess(image,order=2, norm=True):
    image = np.load(image)
    target_size = (64,64,64)
    if norm: image = normFunc(image)
    h2,w2,z2 = image.shape
    image = zoom(image, (target_size[0]/h2,
                target_size[1]/w2,
                target_size[2]/z2),
                order=order)
    
    image = np.expand_dims(image,axis=0)
    image = np.expand_dims(image,axis=-1)
    return image


pyplot.close("all")
get_ipython().run_line_magic('matplotlib', 'inline') 

ResultsPath="//192.168.106.134/Projects/LNDetector/Pulmonary Nodule Segmentation/Python Codes/"

# ########################## Segmentation Class ###############################

class Segmentation_Function:  
    
    def seg (self):   
        
        from DataReadingMain import DRM        
        [BestCost, FeatureMatrix, Size, Slice, SliceM, MaxIt, SI, GT3D, ImageM, SegmentedNoduleLabel, Measure] = DRM (self)
        return BestCost, FeatureMatrix, Size, Slice, SliceM, MaxIt, SI, GT3D, ImageM, SegmentedNoduleLabel, Measure
        
    def rep (BestCost, FeatureMatrix, Size, Slice, SliceM, MaxIt, SI, GT3D, ImageM, SegmentedNoduleLabel, Measure, self):
        
        from ResultRep1 import RR        
        RR (BestCost, FeatureMatrix, Size, Slice, SliceM, MaxIt, SI, GT3D, ImageM, SegmentedNoduleLabel, Measure, self)
        
        
# ############################## Main Class ###################################
        
class NoduleSegmentation:
    
    def __init__(self, master):        
        
        # Calls the constructor for the parent class.
        self.master = master
        
        
        self.model = load_model()
        
        self.colors = [(253/255,174/255,97/255),(215/255,25/255,28/255),(0,168/255,188/255),(197/255,0,188/255)]

        
        self.w = 420
        self.h = 420
        self.z = 32
        
        #set up the title of the window
        self.master.title("iW-Net")         
        
        #self.master.configure(bg="black")  
        # get screen width and height
#        ws = self.master.winfo_screenwidth() # width of the screen
#        hs = self.master.winfo_screenheight() # height of the screen
#        self.master.geometry('%dx%d+%d+%d' % (ws, hs, 0, 0))
        #self.master.geometry('+205+15')
        
        
#        self.data_path = ""   
#        self.message = "Path to file"
#        self.label_text = StringVar()
#        self.label_text.set(self.message)
#        self.label = Label(self.master, textvariable=self.label_text, bg="white")
#        self.label.grid(row=0, column=0, columnspan=1, sticky=(N, E, W), pady=5, padx=5)
        

        #self.error_message = "Select a .npy containing a nodule"
        #self.label_text_error = tk.StringVar()
        #self.label_text_error.set(self.error_message)
        #self.label_error = tk.Label(self.master, textvariable=self.label_text_error,justify='left',anchor='w')
        #self.label_error.grid(row=0, column=1, columnspan=1, sticky=(tk.N, tk.E, tk.W), pady=5, padx=0)
        

        #text field
        #vcmd = self.master.register(self.validate) # we have to wrap the command        
        #self.entry = tk.Entry(self.master, validate="key", validatecommand=(vcmd, '%P'))
        #self.entry.grid(row=0, column=0, columnspan=2, sticky=(tk.N, tk.E, tk.W), pady=5, padx=5)
        

        self.browse_button = tk.Button(self.master, text="Browse", command=self.browse, state=tk.NORMAL)
        self.browse_button.grid(row=0, column=0, sticky=(tk.N, tk.W))
        self.browse_button.config(width=20)
        
        
        self.segmentation_button = tk.Button(self.master, text="Segment", command=self.segmentation, state=tk.DISABLED)
        self.segmentation_button.grid(row=0, column=3, sticky=(tk.N, tk.W))
        self.segmentation_button.config(width=20)
        
        
        self.correction_button = tk.Button(self.master, text="Select points", command=self.select_points, state=tk.DISABLED)
        self.correction_button.grid(row=0, column=6, sticky=(tk.N, tk.W))
        self.correction_button.config(width=20)


#        self.correction_button.config()
        
        
        #self.reset_button = tk.Button(self.master, text="Reset", command=self.reset, state=tk.DISABLED)
        #self.reset_button.grid(row=10, column=0, sticky=(tk.N, tk.E), pady=5, padx=5)
        #self.reset_button.config(width=20)
        
        
        #self.exit_button = tk.Button(self.master, text="Exit", command=self.exit, state=tk.NORMAL)
        #self.exit_button.grid(row=11, column=5, sticky=(tk.S, tk.E), pady=5, padx=5) 
        #self.exit_button.config(width=20)
        
        
        #INITIAL IMAGE
        self.message1 = "Central slice of the nodule (axial view)"
        self.label_text1 = tk.StringVar()
        self.label_text1.set(self.message1)
        self.label1 = tk.Label(self.master, textvariable=self.label_text1,anchor='w')        
        self.label1.grid(row=0, column=2, columnspan=1, sticky=(tk.N, tk.E, tk.W), pady=5, padx=0)

        #INITIAL SEGMENTATION
        self.message2 = "Initial segmentation"
        self.label_text2 = tk.StringVar()
        self.label_text2.set(self.message2)
        self.label2 = tk.Label(self.master, textvariable=self.label_text2,anchor='w')        
        self.label2.grid(row=0, column=5, columnspan=1, sticky=(tk.N, tk.E, tk.W), pady=5, padx=0)
        #holder to place the imshow




        #CORRECTED SEGMENTATION
        self.message3 = "Corrected segmentation"
        self.label_text3 = tk.StringVar()
        self.label_text3.set(self.message3)
        self.label3 = tk.Label(self.master, textvariable=self.label_text3,anchor='w')        
        self.label3.grid(row=0, column=8, columnspan=1, sticky=(tk.N, tk.E, tk.W), pady=5, padx=0)
        
        self.label_author = tk.Label(self.master, text='iW-Net: an automatic and minimalistic interactive lung nodule segmentation deep network. 2018. G. Aresta et al.',fg='gray',anchor='w')        
        self.label_author.grid(row=10, column=0, columnspan=5, sticky=(tk.N, tk.E, tk.W), pady=5, padx=0)
        
        #holder to place the imshow
        
        self.__ititFrames__()
        
        
   
    
    def validate(self,path):
        
        print("Inside validate")
        
        if not path: # the field is being cleared
            
            """
            self.data_path = ""
            self.message = "Enter the path of the nodule to be segmented"
            self.label_text.set(self.message)
            self.label.grid(row=0, column=0, columnspan=2, sticky=(N, E, W), pady=5, padx=5)
            """
            
            
            return True

        try: 
            
           
            
            FileExt = path[ (len(path)-4):len(path) ]
            
            if path == "" or FileExt != ".npy":   
                """
                self.message = "Wrong Path!"
                self.label_text.set(self.message)
                self.label.grid(row=0, column=0, columnspan=2, sticky=(N, E, W), pady=5, padx=5)
                self.data_path = ""
                """
                #self.entry.delete(0, tk.END)
            
            else: 
                
                """
                self.data_path = path 
                self.message = "Correct Path!"
                self.label_text.set(self.message) 
                self.label.grid(row=0, column=0, columnspan=2, sticky=(N, E, W), pady=5, padx=5)
                """

                self.__ititFrames__()
                self.nodule = preprocess(path)
                self.view = self.nodule[0,:,:,self.z,0]
                fig = Figure(figsize=(4, 4), dpi=100)
                fig.add_subplot(111).imshow(self.view,cmap='gray')
                self.draw_figure(self.frame1,fig)
                



                

                
                """
                ImageM1 = resize(view, (385, 500))
                ImageM2 = resize(view, (385, 500))
                ImageM3 = resize(view, (385, 500))
                
                pyplot.figure(1,figsize=(3, 3)), pyplot.subplot(3,1,2), pyplot.xticks([]), pyplot.yticks([])
                pyplot.imshow( ImageM1 , cmap = 'gray' )
    
                pyplot.figure(1,figsize=(3, 3)), pyplot.subplot(3,1,3), pyplot.xticks([]), pyplot.yticks([])
                pyplot.imshow( ImageM2, cmap = 'gray' )
    
                pyplot.figure(1,figsize=(3, 3)), pyplot.subplot(3,1,1), pyplot.xticks([]), pyplot.yticks([])
                pyplot.imshow( ImageM3, cmap = 'gray' )
                
                pyplot.savefig( ResultsPath + 'Figures.png', format='png', dpi=300 ) 
                pyplot.close() 
                """
                """ 
                Timage = Image.open(ResultsPath + 'Figures.png')
                Timage = Timage.resize([385, 500], Image.NEAREST)
                self.image1 = ImageTk.PhotoImage(Timage)
                self.label6 = tk.Label(self.frame1, image=self.image1)   
                self.label6.grid(row=7, column=0, sticky=(N, E, W)) 
                """
                
            return True
            
        except ValueError:
            
            return False
        
        
    def OpenFile(self):
            print('Inside OpenFile')
            name = askopenfilename(
                           filetypes =(("Python File", "*.npy"),("All Files","*.npy")),
                           title = "Choose a file."
                           )    
    
            #Using try in case user types in unknown file or closes without choosing a file.
            try:
                
                with open(name,'rb') as UseFile:                    
                    self.message = UseFile.read()
                    #self.label_text.set(self.message)                    
                    #self.label.grid(row=0, column=0, columnspan=2, sticky=(tk.N, tk.E, tk.W), pady=5, padx=5)
                #self.entry.insert(0, str(name))
            
                self.validate(name)
                    
            except:
                """
                self.message = "Wrong Path!"
                self.label_text.set(self.message)
                self.label.grid(row=0, column=0, columnspan=2, sticky=(N, E, W), pady=5, padx=5)
                self.data_path = ""  
                """
                #self.entry.delete(0, tk.END)
        


    def browse(self):
        
        #self.message = "Enter the path of the nodule to be segmented:"
        #self.label_text.set(self.message)
        #self.label.grid(row=0, column=0, columnspan=2, sticky=(N, E, W), pady=5, padx=5)
        
        #self.data_path = ""
        
        #self.entry.delete(0, tk.END)
        
        """
        self.content1 = tk.Frame(self.master, padding=(3,3,12,12))
        self.frame1 = tk.Frame(self.content1, borderwidth=10, relief="sunken", width=410, height=535)  
        self.content1.grid(row=7, column=0, sticky=(tk.N,tk.E, tk.W))
        self.frame1.grid(row=7, column=0, columnspan=3, rowspan=2, sticky=(tk.N, tk.E, tk.W)) 
        """
        print('Inside browse')
        
        self.OpenFile()
        
        self.browse_button.configure(state=tk.NORMAL)
        self.segmentation_button.configure(state=tk.NORMAL)
        

    def segmentation(self):
        
        pred = self.model.predict([self.nodule,np.zeros(self.nodule.shape)])
        
        self.initial_segmentation = pred[-1] >= 0.5
        
        self.initial_segmentation = filterSegmentation(self.initial_segmentation)
        self.segmentation_button.configure(state=tk.DISABLED)
        
        
        self.initial_overlay = createOverlay(self.view,self.initial_segmentation[0,:,:,self.z,0],color=self.colors[0])
        fig = Figure(figsize=(4, 4), dpi=100)
        fig.add_subplot(111).imshow(self.initial_overlay)
        self.draw_figure(self.frame2,fig)  
        self.correction_button.configure(state=tk.NORMAL)
        

        self.figc = Figure(figsize=(4, 4), dpi=100)
        self.ax3 = self.figc.add_subplot(111)
        self.imshow3 = self.ax3.imshow(self.view,cmap='gray')
        self.canvas3 = FigureCanvasTkAgg(self.figc, master=self.frame3)  # A tk.DrawingArea.
        self.canvas3.draw()
        self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)       
        
        

        
    def select_points(self):
        

        self.imshow3.set_data(self.view)
        self.canvas3.draw()
        self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                
        self.clicked_points = []
        
        
        
        
        

        self.point_mask = np.zeros(self.nodule[0,:,:,self.z,0].shape)
        
        
        self.point_marks = np.copy(self.view)
        
        
        
        self.clicked = False
        

        self.cid = self.figc.canvas.callbacks.connect('button_press_event', self.on_click)
           
        
    def correct_segmentation(self):
        
        
        
        #p1 = [int(self.p1c.get()),int(self.p1r.get()),self.z]
        #p2 = [int(self.p2c.get()),int(self.p2r.get()),self.z]
        
        _,_,_,gravmap = attractionField3D([self.clicked_points[0]],[self.clicked_points[1]],weight_mode='constant',p=0.4376)
        gravmap = np.expand_dims(gravmap,axis=0)
        gravmap = np.expand_dims(gravmap,axis=-1)

        
        pred = self.model.predict([self.nodule,gravmap])
        self.corrected_segmentation = pred[0] >= 0.5
        self.corrected_segmentation = filterSegmentation(self.corrected_segmentation)
        self.corrected_overlay = createOverlay(self.view,self.corrected_segmentation[0,:,:,self.z,0],color=self.colors[1])
        self.corrected_overlay = createOverlay(self.corrected_overlay,self.point_mask,color=(171/255,221/255,164/255),contour=False)

        
        self.imshow3.set_data(self.corrected_overlay)
        self.canvas3.draw()
        self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        
        self.correction_button.configure(state=tk.NORMAL)
        
    def __ititFrames__(self):
        self.frame1 = tk.Frame(self.master, borderwidth=10, relief="sunken", width=self.w, height=self.h) 
        self.frame1.grid(row=7, column=0, columnspan=3, rowspan=2, sticky=(tk.N, tk.E, tk.W)) 
        self.frame2 = tk.Frame(self.master, borderwidth=10, relief="sunken", width=self.w, height=self.h) 
        self.frame2.grid(row=7, column=3, columnspan=3, rowspan=2, sticky=(tk.N, tk.E, tk.W)) 
        self.frame3 = tk.Frame(self.master, borderwidth=10, relief="sunken", width=self.w, height=self.h) 
        self.frame3.grid(row=7, column=6, columnspan=3, rowspan=2, sticky=(tk.N, tk.E, tk.W))  
    
    def draw_figure(self,master,fig,click=None):

        
        canvas = FigureCanvasTkAgg(fig, master=master)  # A tk.DrawingArea.
        

        canvas.draw()
                
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
        
        #
        #toolbar = NavigationToolbar2Tk(canvas, master)
        #toolbar.update()
        #canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
    def exit(self):
        
        self.master.destroy()            
        sys.exit()
    #https://stackoverflow.com/questions/27565939/getting-the-location-of-a-mouse-click-in-matplotlib-using-tkinter 
    def on_click(self,event):
        if event.inaxes is not None:
            
            self.clicked_points.append([event.xdata,event.ydata,self.z])
            
            print(self.clicked_points)
            
            self.point_mask[int(self.clicked_points[-1][1])-1:int(self.clicked_points[-1][1])+1,
                            int(self.clicked_points[-1][0])-1:int(self.clicked_points[-1][0]+1)] = 1
            self.point_marks = createOverlay(self.point_marks,self.point_mask,color=(171/255,221/255,164/255),contour=False)
            
            
    
            #self.figc.add_subplot(111).imshow(self.point_marks)
            self.imshow3.set_data(self.point_marks)
            self.canvas3.draw()
            self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
            if len(self.clicked_points) == 2:
                self.figc.canvas.callbacks.disconnect(self.cid)
                self.correction_button.configure(state=tk.NORMAL)
                self.correct_segmentation()

                return
            
            
            
        else:
            print('Clicked ouside axes bounds but inside plot window' )
        self.clicked=True
        
# ########################### End of Main Class ###############################
   
     
root = tk.Tk()
# The main program starts here by instantiating the NoduleSegmentation class.
my_gui = NoduleSegmentation(root)
# Starts the application's main loop, waiting for mouse and keyboard events.
root.mainloop()


