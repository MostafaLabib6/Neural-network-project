import tkinter as tk
from tkinter import ttk
from tkinter import * 
class GUI:
    # this is a function to get the user input from the text input box
    def getInputBoxValue(self):
        userInput = tInput.get()
        return userInput


    # this is a function to get the user input from the text input box
    def getInputBoxValue(self):
        userInput = secondtextbox.get()
        return userInput


    # this is a function which returns the selected combo box item
    def getSelectedComboItem(self):
        return comboOneTwoPunch.get()


    # this is a function which returns the selected combo box item
    def getSelectedComboItem(self):
        return classes.get()


    # this is a function to check the status of the checkbox (1 means checked, and 0 means unchecked)
    def getCheckboxValue(self):
        checkedOrNot = cbVariable.get()
        return checkedOrNot


    # this is the function called when the button is clicked
    def btnClickFunction(self):
        print('clicked')


    # this is the function called when the button is clicked
    def btnClickFunction(self):
        print('clicked')
        
        
        
        
    def run(self):
        root = Tk()
        #this is the declaration of the variable associated with the checkbox
        cbVariable = tk.IntVar()



        # This is the section of code which creates the main window
        root.geometry('681x454')
        root.configure(background='#F0F8FF')
        root.title('Hello, I\'m the main window')


        # This is the section of code which creates a text input box
        tInput=Entry(root)
        tInput.place(x=91, y=48)


        # This is the section of code which creates a text input box
        secondtextbox=Entry(root)
        secondtextbox.place(x=89, y=95)


        # This is the section of code which creates the a label
        Label(root, text='this is a label', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=260, y=51)


        # This is the section of code which creates the a label
        Label(root, text='this is a label2', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=260, y=99)


        # This is the section of code which creates the a label
        Label(root, text='this is a label3', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=428, y=70)


        # This is the section of code which creates a combo box
        comboOneTwoPunch= ttk.Combobox(root, values=[], font=('arial', 12, 'normal'), width=10)
        comboOneTwoPunch.place(x=127, y=155)
        # comboOneTwoPunch.current(0)


        # This is the section of code which creates a combo box
        classes= ttk.Combobox(root, values=[], font=('arial', 12, 'normal'), width=10)
        classes.place(x=128, y=193)
        # classes.current(0)


        # This is the section of code which creates a checkbox
        CheckBoxOne=Checkbutton(root, text='Check me, I\'m a box!', variable=cbVariable, bg='#F0F8FF', font=('arial', 12, 'normal'))
        CheckBoxOne.place(x=92, y=279)


        # This is the section of code which creates the a label
        Label(root, text='this is a label4', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=337, y=160)


        # This is the section of code which creates the a label
        Label(root, text='this is a label5', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=353, y=206)


        # This is the section of code which creates a button
        # command=btnClickFunction
        Button(root, text='Button text!', bg='#F0F8FF', font=('arial', 12, 'normal') ).place(x=480, y=286)


        # This is the section of code which creates a button
        #  command=btnClickFunction
        Button(root, text='Button text!2', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=470, y=364)


        root.mainloop()
