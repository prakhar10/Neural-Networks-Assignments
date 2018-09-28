# Sapre, Prakhar
# 1001-514-586
# 2018-09-23
# Assignment-02-01

from tkinter import *
import numpy as np
import matplotlib as mpl
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import Sapre_02_02


class MainWindow:

    def __init__(self,master):

        # set default values
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.first_weight = 1.0
        self.second_weight = 1.0
        self.bias = 0.0
        self.transfer_function_type = "Symmetrical Hard Limit"
        self.input_points = []
        self.target_value = [1,1,-1,-1]
        self.error = 0
        self.actual_value = 0
        self.x_inputs = 0
        self.y_inputs = 0

        #Set up top frame where we will have the graph
        self.topFrame = Frame(master,borderwidth =2,relief="sunken", width="1000",height="300")
        self.topFrame.grid(row=0,columnspan=3,sticky="nesw")
        self.topFrame.rowconfigure(0,weight=1)
        self.topFrame.columnconfigure(0,weight=1)

        #Set up bottom from where we will have the buttons, sliders and dropdown
        self.bottomFrame = Frame(master, borderwidth = 2,relief="sunken",width="1000",height="800")
        self.bottomFrame.grid(row=1,columnspan=5,sticky="ns")
        self.bottomFrame.rowconfigure(0,weight=1)
        self.bottomFrame.columnconfigure(0,weight=1)

        #Set up for creating the graph where we will plot the points and line
        #Referred from Professor's code for Assignment 1
        self.figure = plt.figure(figsize=(13, 5))
        self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.8])
        self.axes = self.figure.gca()
        self.axes.set_xlabel("x axis")
        self.axes.set_ylabel("y axis")
        self.axes.set_title(self.transfer_function_type)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.topFrame)
        self.plot_figure = self.canvas.get_tk_widget()
        self.plot_figure.grid(row=0, column=1, sticky="nesw")

        #create buttons for training the perceptron and creatinf random data on the graph
        self.train_perceptron_button = Button(self.bottomFrame,text="Train",bg="yellow", width="16")
        self.train_perceptron_button.bind("<Button-1>", lambda event: self.train())
        self.create_data_button = Button(self.bottomFrame,text="Create Random Data",bg="yellow")
        self.create_data_button.bind("<Button-1>",lambda event: self.create_random_data())
        self.train_perceptron_button.grid(row=0,column=3,columnspan=2,padx=10,pady=10)
        self.create_data_button.grid(row=1,column=3,columnspan=2,padx=10,pady=10)

        #create the weight 1 slider
        self.first_weight_label = Label(self.bottomFrame,text="Weight W1")
        self.first_weight_label.grid(row=0,column=0,padx=10)
        self.first_weight_slider = Scale(self.bottomFrame,variable=DoubleVar(),from_=-10.0,to_=10.0,resolution=0.01, orient= HORIZONTAL,command= lambda event: self.get_first_weight())
        self.first_weight_slider.set(self.first_weight)
        self.first_weight_slider.bind("<ButtonRelease-1>", lambda event: self.get_first_weight())
        self.first_weight_slider.grid(row=0,column=1,padx=20)

        #create the weight 2 slider
        self.second_weight_label = Label(self.bottomFrame, text="Weight W2")
        self.second_weight_label.grid(row=1, column=0, padx=10)
        self.second_weight_slider = Scale(self.bottomFrame,variable=DoubleVar(), from_=-10.0, to_=10.0,resolution=0.01, orient=HORIZONTAL)
        self.second_weight_slider.grid(row=1, column=1,padx=20)
        self.second_weight_slider.bind("<ButtonRelease-1>", lambda event: self.get_second_weight())
        self.second_weight_slider.set(self.second_weight)

        #create the bias slider
        self.bias_label = Label(self.bottomFrame, text="Bias")
        self.bias_label.grid(row=2, column=0, padx=10)
        self.bias_slider = Scale(self.bottomFrame, variable=DoubleVar(),from_=-10.0, to_=10.0,resolution=0.01, orient=HORIZONTAL)
        self.bias_slider.grid(row=2, column=1,padx=20)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.get_bias())
        self.bias_slider.set(self.bias)

        #create the activation function dropdown
        self.transfer_function_list = ["Symmetrical Hard Limit","Hyperbolic Tangent","Linear"]
        self.transfer_function_variable = StringVar()
        self.transfer_function_variable.set(self.transfer_function_list[0])
        self.transfer_function_dropdown = OptionMenu(self.bottomFrame,self.transfer_function_variable,*self.transfer_function_list,command=lambda event: self.get_transfer_function())
        self.transfer_function_dropdown.grid(row=2,column=3)

    #This method will fetch the activation function selected from dropdown
    def get_transfer_function(self):
        self.transfer_function_type = self.transfer_function_variable.get()
        self.display_line()

    #This method will fetch the weight 1 selected from the slider
    def get_first_weight(self):
        self.first_weight = self.first_weight_slider.get()
        self.display_line()

    # This method will fetch the weight 2 selected from the slider
    def get_second_weight(self):
        self.second_weight = self.second_weight_slider.get()
        self.display_line()

    # This method will fetch the bias selected from the slider
    def get_bias(self):
        self.bias = self.bias_slider.get()
        self.display_line()

    #This function will calculate the weights and bias according to the input points and activation function type and
    #gives us the values to generate a line
    def train(self):
        for i in range(100):
            self.first_weight,self.second_weight,self.bias=Sapre_02_02.train_perceptron(self.transfer_function_type,self.first_weight,self.second_weight,self.bias,self.input_points,self.target_value)
            self.first_weight_slider.set(self.first_weight)
            self.second_weight_slider.set(self.second_weight)
            self.bias_slider.set(self.bias)
            self.display_line()

    #This function will plot the line on the graph
    def display_line(self):
        self.x_inputs = np.linspace(-10,10,250,endpoint=True)
        self.y_inputs = (-self.bias - (self.first_weight * self.x_inputs))/self.second_weight
        self.axes.cla()
        self.axes.set_xlabel('X Axis')
        self.axes.set_ylabel('Y Axis')

        activation_val = 0
        #Referred from Professor's ipython code : http://ranger.uta.edu/~kamangar/CSE-5368-FA18/LinkedDocuments/example_of_how_to_draw_decision_boundary.ipynb
        xs = np.linspace(-10,10,250)
        ys = np.linspace(-10,10,250)
        xx, yy = np.meshgrid(xs,ys)
        zz = self.first_weight*xx + self.second_weight*yy + self.bias
        if self.transfer_function_type == "Symmetrical Hard Limit":
            zz[zz<0] = -1
            zz[zz>0] = 1
        elif self.transfer_function_type == "Hyperbolic Tangent":
            activation_val = np.tanh(zz)
        elif self.transfer_function_type == "Linear":
            activation_val = zz

        colors = mpl.colors.ListedColormap(['r','g'])
        self.axes.pcolormesh(xs,ys,zz,cmap=colors)

        #this code will generate random points on the graph with different colors for each class
        if len(self.input_points) > 0:
            for i in range(0,2):
                pp = plt.scatter(self.input_points[i][0],self.input_points[i][1], c="blue")
            for j in range(2,4):
                qq = plt.scatter(self.input_points[j][0],self.input_points[j][1], c="yellow")
        self.axes.plot(self.x_inputs, self.y_inputs)
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()

    #This function will create random integer points and plot them on the graph
    def create_random_data(self):
        self.input_points = [[np.random.uniform(-10,10),np.random.uniform(-10,10)] for i in range(4)]
        self.axes.cla()
        plt.xlim(self.xmin,self.xmax)
        plt.ylim(self.ymin,self.ymax)
        for i in range(0,2):
            xx = plt.scatter(self.input_points[i][0],self.input_points[i][1], c="blue")
        for j in range(2,4):
            yy = plt.scatter(self.input_points[j][0],self.input_points[j][1], c="red")
        self.canvas.draw()
        self.display_line()


root = Tk()
main = MainWindow(root)
root.mainloop()
