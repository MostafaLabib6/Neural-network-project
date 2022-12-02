import tkinter as tk
from tkinter import ttk
from tkinter import *
from Controllers.MainController import MainController


class GUI:
    def getLearningRate(self):
        try:
            userInput = float(self.step_size.get())
        except:
            userInput = 0.01
        return userInput

    def setAccuracy(self, acc):
        self.accuarcyEntry.configure(state='normal')
        self.accuarcyEntry.delete(0, END)
        self.accuarcyEntry.insert(0, str(acc))
        self.accuarcyEntry.configure(state='disable')

    def getEpochs(self):
        try:
            userInput = int(self.epochs.get())
        except:
            userInput = 1000
        return userInput

    def getBais(self):
        checkedOrNot = self.cbVariable.get()
        return checkedOrNot

    def get_neurons(self):
        neurons_list = self.neurons.get().split(',')
        neurons_list = [int(i) for i in neurons_list]
        return neurons_list

    def train(self):
        self.controller.reset()
        if self.select_activationFn.get() and self.neurons.get():
            acc = self.controller.train_model(dims=self.get_neurons(), learning_rate=self.getLearningRate(),
                                              bias=self.getBais(), activation=self.select_activationFn.get())
            self.controller.test_model(self.select_activationFn.get())
            self.setAccuracy(acc)

    def drowing_plots(self):
        self.controller.showGraphs()

    def run(self, ):
        self.root = Tk()
        self.FeatureList = ['Sigmoid', 'Tanh']
        self.controller = MainController()
        self.cbVariable = tk.IntVar()

        self.root.geometry('630x380')
        self.root.resizable(width=0, height=0)
        self.root.configure(background='#F0F8FF')
        self.root.title('MNIST  :)')

        # step size
        self.step_size = Entry(self.root, width=25)
        self.step_size.place(x=150, y=50)

        # neurons
        self.neurons = Entry(self.root, width=25)
        self.neurons.place(x=450, y=100)

        # Epocs
        self.epochs = Entry(self.root, width=25)
        self.epochs.place(x=150, y=100)

        # hidden layers entry
        self.hidden_layers = Entry(self.root, width=25)
        self.hidden_layers.place(x=450, y=50)

        self.accuarcyEntry = Entry(self.root, width=5, state='disable', font=('arial', 16, 'bold'))
        self.accuarcyEntry.place(x=510, y=270)

        Label(self.root, text='Enter Step Size', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=10, y=50)

        Label(self.root, text='Accuracy', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=430, y=270)

        Label(self.root, text='Enter Epochs', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=10, y=100)

        Label(self.root, text='Activation Fn', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=10, y=150)

        Label(self.root, text='List of Neurons', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=325, y=100)

        Label(self.root, text='Hidden Layers', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=325, y=50)

        self.select_activationFn = tk.StringVar()
        self.activationFn = ttk.Combobox(self.root, values=self.FeatureList, textvariable=self.select_activationFn,
                                         font=('arial', 12, 'normal'), width=15)
        self.activationFn.place(x=150, y=150)
        self.activationFn.bind('<<ComboboxSelected>>')

        CheckBoxOne = Checkbutton(self.root, text='Bias', variable=self.cbVariable, bg='#F0F8FF',
                                  font=('arial', 12, 'normal'))
        CheckBoxOne.place(x=100, y=270)

        Button(self.root, text='RUN', bg='#F0F8FF', font=('arial', 12, 'normal'), width=14, command=self.train).place(
            x=450, y=330)

        Button(self.root, text='Drawing', bg='#F0F8FF', font=('arial', 12, 'normal'), width=14,
               command=self.drowing_plots).place(x=280, y=330)

        self.root.mainloop()
