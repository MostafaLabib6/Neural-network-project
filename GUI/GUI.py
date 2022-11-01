import tkinter as tk
from tkinter import ttk
from tkinter import *
import GUI.DependencyInjector as di

from dependency_injector.wiring import Provide, inject
from GUI.Controllers.MainController import MainController
from click import command 
class GUI:
    def __init__(self  ):
        container = di.Container()
        container.wire(modules=[__name__])
        # self._controller:MainController= Provide[di.Container.controlls]
    # # this is a function to get the user input from the text input box
    def getLearningRate(self):
        userInput = self.tInput.get()
        return userInput


    # this is a function to get the user input from the text input box
    def getEpochs(self):
        userInput = self.secondtextbox.get()
        return userInput


    # # this is a function which returns the selected combo box item
    # def getSelectedComboItem(self):
    #     return comboOneTwoPunch.get()


    # # this is a function which returns the selected combo box item
    # def getSelectedComboItem(self):
    #     return classes.get()


    # this is a function to check the status of the checkbox (1 means checked, and 0 means unchecked)
    def getBais(self):
        checkedOrNot = self.cbVariable.get()
        return checkedOrNot

        
    def feature1change(self,event):
        self.ModifiedFeatureList=self.FeatureList.copy()

        self.ModifiedFeatureList.remove(self.selected_feature1.get())
        self.Features2['values']=self.ModifiedFeatureList
    def feature2change(self,event):
        self.ModifiedFeatureList=self.FeatureList.copy()

        self.ModifiedFeatureList.remove(self.selected_feature2.get())
        self.Features1['values']=self.ModifiedFeatureList

        
    def class1change(self,event):
        self.ModifiedclassList=self.classList.copy()

        self.ModifiedclassList.remove(self.selected_class1.get())
        self.class2['values']=self.ModifiedclassList
    def class2change(self,event):
        self.ModifiedclassList=self.classList.copy()

        self.ModifiedclassList.remove(self.selected_class2.get())
        self.class1['values']=self.ModifiedclassList
    def train(self):
        if self.selected_feature1.get() and self.selected_class1.get() and self.selected_class2.get() and self.selected_feature2.get():
            self.controller.classFillter(class1=self.selected_class1.get(),class2=self.selected_class2.get())
            self.controller.FeatureFillter(feat1=self.selected_feature1.get(),feat2=self.selected_feature2.get())
            self.controller.train(learning_rate=float(self.getLearningRate()),bais=self.getBais(),epochs=int(self.getEpochs()))
    def test(self):
        if self.selected_feature1.get() and self.selected_class1.get() and self.selected_class2.get() and self.selected_feature2.get():
            self.controller.classFillter(class1=self.selected_class1.get(),class2=self.selected_class2.get())
            self.controller.FeatureFillter(feat1=self.selected_feature1.get(),feat2=self.selected_feature2.get())
            self.controller.test(learning_rate=self.getLearningRate(),bais=self.getBais(),epochs=self.getEpochs())
    @inject  
    def run(self,_controller:MainController= Provide[di.Container.controlls]):
        root = Tk()
        self.FeatureList=["bill_length_mm","bill_depth_mm","flipper_length_mm","gender","body_mass_g"]
        self.ModifiedFeatureList=["bill_length_mm","bill_depth_mm","flipper_length_mm","gender","body_mass_g"]
        self.classList=["Chinstrap","Gentoo","Adelie"]
        self.ModifiedclassList=["Chinstrap","Gentoo","Adelie"]
        self.controller=_controller
        #this is the declaration of the variable associated with the checkbox
        self.cbVariable = tk.IntVar()



        # This is the section of code which creates the main window
        root.geometry('681x454')
        root.configure(background='#F0F8FF')
        root.title('Deep Learning')


        # This is the section of code which creates a text input box
        self.tInput=Entry(root)
        self.tInput.place(x=120, y=48)


        # This is the section of code which creates a text input box
        self.secondtextbox=Entry(root)
        self.secondtextbox.place(x=120, y=95)


        # This is the section of code which creates the a label
        Label(root, text='Learning rate', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=10, y=48)


        # This is the section of code which creates the a label
        Label(root, text='Epochs', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=10, y=95)


        # This is the section of code which creates the a label
        Label(root, text='Features 1', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=10, y=138)


        # This is the section of code which creates a combo box
        self.selected_feature1=tk.StringVar()
        self.Features1= ttk.Combobox(root, values= self.ModifiedFeatureList, textvariable=self.selected_feature1,font=('arial', 12, 'normal'), width=10)
        self.Features1.place(x=120, y=138)
        self.Features1.bind('<<ComboboxSelected>>', self.feature1change)
        # comboOneTwoPunch.current(0)
        Label(root, text='Features 2', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=10, y=181)


        # This is the section of code which creates a combo box
        self.selected_feature2=tk.StringVar()
        self.Features2= ttk.Combobox(root, values= self.ModifiedFeatureList,textvariable=self.selected_feature2, font=('arial', 12, 'normal'), width=10)
        self.Features2.place(x=120, y=181)
        self.Features2.bind('<<ComboboxSelected>>', self.feature2change)

        # comboOneTwoPunch.current(0)

        # This is the section of code which creates a combo box
        self.selected_class1=tk.StringVar()

        self.class1= ttk.Combobox(root, values=self.ModifiedclassList,textvariable=self.selected_class1, font=('arial', 12, 'normal'), width=10)
        self.class1.place(x=120, y=236)
        self.class1.bind('<<ComboboxSelected>>', self.class1change)

        # classes.current(0)
        # This is the section of code which creates a combo box
        self.selected_class2=tk.StringVar()

        self.class2= ttk.Combobox(root, values=self.ModifiedclassList,textvariable=self.selected_class2, font=('arial', 12, 'normal'), width=10)
        self.class2.place(x=120, y=279)
        self.class2.bind('<<ComboboxSelected>>', self.class2change)

        # classes.current(0)


        # This is the section of code which creates a checkbox
        CheckBoxOne=Checkbutton(root, text='Bias', variable=self.cbVariable, bg='#F0F8FF', font=('arial', 12, 'normal'))
        CheckBoxOne.place(x=120, y=322)


        # This is the section of code which creates the a label
        Label(root, text='class 1', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=10, y=236)
         # This is the section of code which creates the a label
        Label(root, text='class 2', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=10, y=279)


        # This is the section of code which creates the a label
        # Label(root, text='this is a label5', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=353, y=206)


        # This is the section of code which creates a button
        # command=btnClickFunction
        Button(root, text='train', bg='#F0F8FF', font=('arial', 12, 'normal'), width=6,height=2,command=self.train ).place(x=480, y=286)


        # This is the section of code which creates a button
        #  command=btnClickFunction
        Button(root, text='test', bg='#F0F8FF', font=('arial', 12, 'normal'), width=6,height=2,command=self.test).place(x=480, y=364)
        Button(root, text='graphs', bg='#F0F8FF', font=('arial', 12, 'normal'), width=6,height=2).place(x=480, y=210)

        root.mainloop()
