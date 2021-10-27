from tkinter import *
from tkinter import filedialog
from PIL import ImageTk , Image          # We need pillow to visualize image in tkinter in an easy way 
import cv2

from ttkbootstrap import Style
from tkinter import ttk
import numpy as np

# 1- Color Tracking

def lower_upper(color_no):
    # define range of blue color in HSV
    lower_blue = np.array([105,50,50])
    upper_blue = np.array([130,255,255])

    # define range of green color in HSV
    lower_green = np.array([45,50,50])
    upper_green = np.array([75,255,255])

    # define range of red color in HSV
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    
    # Color Testing
    if color_no == 1:
        lower = lower_red
        upper = upper_red
    elif color_no == 2:
        lower = lower_green
        upper = upper_green
    else:
        lower = lower_blue
        upper = upper_blue
    
    return lower, upper

def track_method(track_var, img_in):
    color_no = track_var.get()
    lower, upper = lower_upper(color_no)
    
    hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only blue colors
    if color_no == 1:
        lower_2 = np.array([170,50,50])
        upper_2 = np.array([179,255,255])
        mask_1 = cv2.inRange(hsv, lower, upper)
        mask_2 = cv2.inRange(hsv, lower_2, upper_2)
        mask = mask_1 + mask_2
    else:
        mask = cv2.inRange(hsv, lower, upper)
        
    res = cv2.bitwise_and(img_in,img_in, mask= mask)
    
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    
    return res

# 2- Thresholding

def threshold_method(threshold_var, img_in):
    threshold_type = threshold_var.get()
    if len(img_in.shape) == 3:
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    
    if threshold_type == 1:
        # Binary/Global Thresholding
        ret, img_res = cv2.threshold(img_in,127,255,cv2.THRESH_BINARY)
        #return img_res
    elif threshold_type == 2:
        # Adaptive Gaussian Thresholding
        img_res = cv2.adaptiveThreshold(img_in,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        #return img_res
    else:
        # Otsu's Thresholding
        ret, img_res = cv2.threshold(img_in,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    return img_res
        
# 3- Bluring

def blur_method(blur_var, img_in):
    blur_type = blur_var.get()
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    
    if blur_type == 1:
        # Averaging Blurring
        img_res = cv2.blur(img_in,(5,5))
    elif blur_type == 2:
        # Gaussain Blurring
        img_res = cv2.GaussianBlur(img_in,(21,21),0) # img , kernel size ,  Sigma ( how fat your kernel is?)
    else:
        # Median Blurring
        img_res = cv2.medianBlur(img_in,5)
        
    return img_res
    
# 4- Morphology

def morph_method(morph_var, img_in):
    morph_type = morph_var.get()
    if len(img_in.shape) == 3:
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    #ret, img_in = cv2.threshold(img_in,127,255,cv.THRESH_BINARY_INV)
    
    if morph_type == 1:
        # Erosion
        img_res = cv2.erode(img_in,kernel,iterations = 1)  #  img , kernel , iterations
    elif morph_type == 2:
        # Dilation
        img_res = cv2.dilate(img_in,kernel,iterations = 1)
    elif morph_type == 3:
        # Openning
        img_res = cv2.morphologyEx(img_in, cv2.MORPH_OPEN, kernel)
    else:
        # Closing
        img_res = cv2.morphologyEx(img_in, cv2.MORPH_CLOSE, kernel)
        
    return img_res

# 5- Edge detection

def edge_method(edge_var, img_in):
    edge_type = edge_var.get()
    if len(img_in.shape) == 3:
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    
    if edge_type == 1:
        # Sobel-X Edge
        img_res = cv2.Sobel(img_in,-1,1,0,ksize=5) # Image , DDepth = -1 (uint8) would be the result
                                        # Caution : you may need to try cv2.CV_64F for negative values to be considered
                                        # Black to white and White to Black Both would appear.
    elif edge_type == 2:
        # Sobel-Y Edge
        img_res = cv2.Sobel(img_in,-1,0,1,ksize=5) # 1 , 0  for dx   and  0 , 1 for dy
    elif edge_type == 3:
        # Scharr-X Edge
        img_res = cv2.Scharr(img_in,cv2.CV_64F,1,0,5)  # img ,ddepth, dx , dy, kernel
    elif edge_type == 4:
        # Scharr-Y Edge
        img_res = cv2.Scharr(img_in,cv2.CV_64F,0,1,5)
    elif edge_type == 5:
        # Laplacian Edge
        img_res = cv2.Laplacian(img_in,cv2.CV_64F,ksize=3)  # img , ddepth , kernel
    else:
        # Canny Edge
        img_res = cv2.Canny(img_in,100,200) # image , min threshold , max threshold
        
    return img_res
    


        


def scale_h(Wm, Hm, W, H):
    X = Hm / H   # Scale Factor for Height
    Hf = int(X * H)   # Height Final
    Wf = int(X * W)   # Width Final
    return Wf, Hf

def scale_w(Wm, Hm, W, H):
    X = Wm / W   # Scale Factor for Width
    Hf = int(X * H)   # Height Final
    Wf = int(X * W)   # Width Final
    return Wf, Hf

def new_shape(Wm, Hm, W, H):
    # This function resizes image to be as program window size and saving its aspect ratio where (H1/W1 = H2/W2)
    if (H <= Hm) and (W <= Wm):
        Hf = H
        Wf = W
    elif (H <= Hm) or (W <= Wm):
        if H <= Hm:
            Wf, Hf = scale_w(Wm, Hm, W, H)
        else:
            Wf, Hf = scale_h(Wm, Hm, W, H)
    elif H <= W:
        Wf, Hf = scale_w(Wm, Hm, W, H)
    else:
        Wf, Hf = scale_h(Wm, Hm, W, H)
    return Wf, Hf
    

def select_image(panelA, panelB):

    path = filedialog.askopenfilename()     # open a file chooser dialog and allow the user to select an input image
    
    global img, img_show                    # global reference to the image panels
    img = cv2.imread(path)
    
    # Program Window Size (max size)
    width_m = 500
    height_m = 400
    
    # Image Size
    width = img.shape[1]
    height = img.shape[0]
    
    # Image Final Size
    width_f, height_f = new_shape(width_m, height_m, width, height)

    # dsize
    #dsize = (width, height)
    global dsize
    dsize = (width_f, height_f)

    # resize image
    img_show = cv2.resize(img, dsize)
        
    img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    
    img_show = Image.fromarray(img_show)       # convert the images to PIL format...
    
    img_show = ImageTk.PhotoImage(img_show)    # ...and then to tkinter format
    
    panelA.configure(image=img_show)
    panelB.configure(text="Output Image", image=out_empty)
    panelA.image = img_show
    panelB.image = out_empty
            
def apply_method(panelA, panelB, track_var, threshold_var, blur_var, morph_var, edge_var,
                track_entry, threshold_entry, blur_entry, morph_entry, edge_entry):
    img_check = False
    img_res = img.copy()
    for filter_index in range(1, 6, 1):
        if (int(track_entry.get()) == filter_index) and (track_var.get() != 4):
            img_check = True
            img_res = track_method(track_var, img_res)
        if (int(threshold_entry.get()) == filter_index) and (threshold_var.get() != 4):
            img_check = True
            img_res = threshold_method(threshold_var, img_res)
        if (int(blur_entry.get()) == filter_index) and (blur_var.get() != 4):
            img_check = True
            img_res = blur_method(blur_var, img_res)
        if (int(morph_entry.get()) == filter_index) and (morph_var.get() != 5):
            img_check = True
            img_res = morph_method(morph_var, img_res)
        if (int(edge_entry.get()) == filter_index) and (edge_var.get() != 7):
            img_check = True
            img_res = edge_method(edge_var, img_res)
        

    
    #######
    # Check if any of above filters are applyed, or not
    if img_check:
        img_res = cv2.resize(img_res, dsize)

        img_res = Image.fromarray(img_res)       # convert the images to PIL format...
        img_res = ImageTk.PhotoImage(img_res)    # ...and then to tkinter format

        panelB.configure(image=img_res)
        panelB.image = img_res
    else:
        panelB.configure(text="Output Image", image=out_empty)
        panelB.image = out_empty
        
    
    
# initialize the window toolkit along with the two image panels
style = Style(theme='cyborg')
#root = Tk()
root = style.master

# Program Size & Show Location
# root.geometry('widthxheight+left+top')
#root.geometry('1030x600+350+100')
root.title('Image Control Panel')
root.iconbitmap('Icon-01.ico')

# Main 3 Frames of Program
#image_frame = ttk.Frame(root, width=1010, height=1000)
image_frame = ttk.Frame(root)
option_frame = ttk.Frame(root)
control_frame = ttk.Frame(root)

image_frame.pack(side="top", fill="both", expand="yes", padx="10", pady="10")
option_frame.pack(fill="both", expand="yes", padx="10", pady="10")
control_frame.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

panelA = ttk.Label(root, text="Original Image")
panelB = ttk.Label(root, text="Output Image")
panelA.pack(in_=image_frame, side="left", fill="both", expand="yes", padx="5", pady="5") #  Localization
panelB.pack(in_=image_frame, side="right", fill="both", expand="yes", padx="5", pady="5") #  Localization

global in_empty, out_empty
in_empty = cv2.imread('InputImage.jpg')
out_empty = cv2.imread('OutputImage.jpg')
in_empty = cv2.cvtColor(in_empty, cv2.COLOR_BGR2RGB)
out_empty = cv2.cvtColor(out_empty, cv2.COLOR_BGR2RGB)

in_empty = Image.fromarray(in_empty)       # convert the images to PIL format...
out_empty = Image.fromarray(out_empty)
    
in_empty = ImageTk.PhotoImage(in_empty)    # ...and then to tkinter format
out_empty = ImageTk.PhotoImage(out_empty)

panelA.configure(image=in_empty)
panelB.configure(image=out_empty)
panelA.image = in_empty
panelB.image = out_empty





# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = ttk.Button(root, text="Select an image", command=lambda: select_image(panelA, panelB)) # yourapp , text ,  binded function
btn.pack(in_=control_frame, side="left", fill="both", expand="yes", padx="5", pady="5") #  Localization

btn2 = ttk.Button(root, text="Applying Transformation", 
                  command=lambda: apply_method(panelA, panelB, track_var, threshold_var, blur_var, morph_var, edge_var, 
                                              track_entry, threshold_entry, blur_entry, morph_entry, edge_entry)) # yourapp , text ,  binded function
btn2.pack(in_=control_frame, side="left", fill="both", expand="yes", padx="5", pady="5") #  Localization

# Sub 5 Frames of Options Frame
# ( track_frame - threshold_frame - blur_frame - morph_frame - edge_frame )

# 1- Color Tracking
track_var = IntVar()
track_var.set(4) # need to use track_var.set and track_var.get to
# set and get the value of this variable
track_label = ttk.Label(root, text="Select color to track it:")
track_radio1 = ttk.Radiobutton(root, text="Red", variable=track_var, value=1)
track_radio2 = ttk.Radiobutton(root, text="Green", variable=track_var, value=2)
track_radio3 = ttk.Radiobutton(root, text="Blue", variable=track_var, value=3)
track_radio4 = ttk.Radiobutton(root, text="None", variable=track_var, value=4)
track_label2 = ttk.Label(root, text="Index:")
track_entry = Entry(root, justify="center", width=2)
track_entry.insert(0, 1)

track_label.grid(in_=option_frame, row=0, column=0, padx="2", pady="2", sticky="w")
track_radio1.grid(in_=option_frame, row=0, column=1, padx="2", pady="2", sticky="w")
track_radio2.grid(in_=option_frame, row=0, column=2, padx="2", pady="2", sticky="w")
track_radio3.grid(in_=option_frame, row=0, column=3, padx="2", pady="2", sticky="w")
track_radio4.grid(in_=option_frame, row=0, column=4, padx="2", pady="2", sticky="w")
track_label2.grid(in_=option_frame, row=0, column=6, padx="2", pady="2", sticky="w")
track_entry.grid(in_=option_frame, row=0, column=7, padx="2", pady="2", sticky="w")

# 2- Thresholding
threshold_var = IntVar()
threshold_var.set(4) # need to use threshold_var.set and threshold_var.get to
# set and get the value of this variable
threshold_label = ttk.Label(root, text="Select thresholding filter:")
threshold_radio1 = ttk.Radiobutton(root, text="Binary", variable=threshold_var, value=1)
threshold_radio2 = ttk.Radiobutton(root, text="Adaptive", variable=threshold_var, value=2)
threshold_radio3 = ttk.Radiobutton(root, text="Otsu", variable=threshold_var, value=3)
threshold_radio4 = ttk.Radiobutton(root, text="None", variable=threshold_var, value=4)
threshold_label2 = ttk.Label(root, text="Index:")
threshold_entry = Entry(root, justify="center", width=2)

threshold_label.grid(in_=option_frame, row=1, column=0, padx="2", pady="2", sticky="w")
threshold_radio1.grid(in_=option_frame, row=1, column=1, padx="2", pady="2", sticky="w")
threshold_radio2.grid(in_=option_frame, row=1, column=2, padx="2", pady="2", sticky="w")
threshold_radio3.grid(in_=option_frame, row=1, column=3, padx="2", pady="2", sticky="w")
threshold_radio4.grid(in_=option_frame, row=1, column=4, padx="2", pady="2", sticky="w")
threshold_label2.grid(in_=option_frame, row=1, column=6, padx="2", pady="2", sticky="w")
threshold_entry.grid(in_=option_frame, row=1, column=7, padx="2", pady="2", sticky="w")
threshold_entry.insert(0, 2)

# 3- Bluring
blur_var = IntVar()
blur_var.set(4) # need to use blur_var.set and blur_var.get to
# set and get the value of this variable
blur_label = ttk.Label(root, text="Select bluring filter:")
blur_radio1 = ttk.Radiobutton(root, text="Averaging", variable=blur_var, value=1)
blur_radio2 = ttk.Radiobutton(root, text="Gaussain", variable=blur_var, value=2)
blur_radio3 = ttk.Radiobutton(root, text="Median", variable=blur_var, value=3)
blur_radio4 = ttk.Radiobutton(root, text="None", variable=blur_var, value=4)
blur_label2 = ttk.Label(root, text="Index:")
blur_entry = Entry(root, justify="center", width=2)
blur_entry.insert(0, 3)

blur_label.grid(in_=option_frame, row=2, column=0, padx="2", pady="2", sticky="w")
blur_radio1.grid(in_=option_frame, row=2, column=1, padx="2", pady="2", sticky="w")
blur_radio2.grid(in_=option_frame, row=2, column=2, padx="2", pady="2", sticky="w")
blur_radio3.grid(in_=option_frame, row=2, column=3, padx="2", pady="2", sticky="w")
blur_radio4.grid(in_=option_frame, row=2, column=4, padx="2", pady="2", sticky="w")
blur_label2.grid(in_=option_frame, row=2, column=6, padx="2", pady="2", sticky="w")
blur_entry.grid(in_=option_frame, row=2, column=7, padx="2", pady="2", sticky="w")

# 4- Morphology
morph_var = IntVar()
morph_var.set(5) # need to use morph_var.set and morph_var.get to
# set and get the value of this variable
morph_label = ttk.Label(root, text="Select morphology:")
morph_radio1 = ttk.Radiobutton(root, text="Erosion", variable=morph_var, value=1)
morph_radio2 = ttk.Radiobutton(root, text="Dilation", variable=morph_var, value=2)
morph_radio3 = ttk.Radiobutton(root, text="Openning", variable=morph_var, value=3)
morph_radio4 = ttk.Radiobutton(root, text="Closing", variable=morph_var, value=4)
morph_radio5 = ttk.Radiobutton(root, text="None", variable=morph_var, value=5)
morph_label2 = ttk.Label(root, text="Index:")
morph_entry = Entry(root, justify="center", width=2)
morph_entry.insert(0, 4)

morph_label.grid(in_=option_frame, row=3, column=0, padx="2", sticky="w")
morph_radio1.grid(in_=option_frame, row=3, column=1, padx="2", pady="2", sticky="w")
morph_radio2.grid(in_=option_frame, row=3, column=2, padx="2", pady="2", sticky="w")
morph_radio3.grid(in_=option_frame, row=3, column=3, padx="2", pady="2", sticky="w")
morph_radio4.grid(in_=option_frame, row=3, column=4, padx="2", pady="2", sticky="w")
morph_radio5.grid(in_=option_frame, row=3, column=5, padx="2", pady="2", sticky="w")
morph_label2.grid(in_=option_frame, row=3, column=6, padx="2", pady="2", sticky="w")
morph_entry.grid(in_=option_frame, row=3, column=7, padx="2", pady="2", sticky="w")

# 5- Edge detection
edge_var = IntVar()
edge_var.set(7) # need to use edge_var.set and edge_var.get to
# set and get the value of this variable
edge_label = ttk.Label(root, text="Select edge detection:")
edge_radio1 = ttk.Radiobutton(root, text="Sobel-X", variable=edge_var, value=1)
edge_radio2 = ttk.Radiobutton(root, text="Sobel-Y", variable=edge_var, value=2)
edge_radio3 = ttk.Radiobutton(root, text="Scharr-X", variable=edge_var, value=3)
edge_radio4 = ttk.Radiobutton(root, text="Scharr-Y", variable=edge_var, value=4)
edge_radio5 = ttk.Radiobutton(root, text="Laplacian", variable=edge_var, value=5)
edge_radio6 = ttk.Radiobutton(root, text="Canny", variable=edge_var, value=6)
edge_radio7 = ttk.Radiobutton(root, text="None", variable=edge_var, value=7)
edge_label2 = ttk.Label(root, text="Index:")
edge_entry = Entry(root, justify="center", width=2)
edge_entry.insert(0, 5)

edge_label.grid(in_=option_frame, row=4, column=0, padx="2", pady="2", sticky="w")
edge_radio1.grid(in_=option_frame, row=4, column=1, padx="2", pady="2", sticky="w")
edge_radio2.grid(in_=option_frame, row=5, column=1, padx="2", pady="2", sticky="w")
edge_radio3.grid(in_=option_frame, row=4, column=2, padx="2", pady="2", sticky="w")
edge_radio4.grid(in_=option_frame, row=5, column=2, padx="2", pady="2", sticky="w")
edge_radio5.grid(in_=option_frame, row=4, column=3, padx="2", pady="2", sticky="w")
edge_radio6.grid(in_=option_frame, row=4, column=4, padx="2", pady="2", sticky="w")
edge_radio7.grid(in_=option_frame, row=4, column=5, padx="2", pady="2", sticky="w")
edge_label2.grid(in_=option_frame, row=4, column=6, padx="2", pady="2", sticky="w")
edge_entry.grid(in_=option_frame, row=4, column=7, padx="2", pady="2", sticky="w")


option_frame.columnconfigure(tuple(range(7)), weight=1)
option_frame.rowconfigure(tuple(range(4)), weight=1)


# kick off the GUI
root.mainloop()
