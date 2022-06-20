import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from dataset import Churn_ModellingDataset
import model as mdl
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

# HyperParameter
NumOfEpoch = 100
NumOfFeature = 10
NumOfHiddenLayer1Nodes = 100
NumOfHiddenLayer2Nodes = 100
NumOfHiddenLayer3Nodes = 100
NumOfLabel = 2
LearningRate = 0.01

# Create 1 Layer Model
MlpModel1 = mdl.MlpNet1(NumOfFeature, NumOfHiddenLayer1Nodes, NumOfLabel)

# Create 2 Layer Model
MlpModel2 = mdl.MlpNet2(NumOfFeature, NumOfHiddenLayer1Nodes, NumOfHiddenLayer2Nodes, NumOfLabel)

# Create 3 Layer Model
MlpModel3 = mdl.MlpNet3(NumOfFeature, NumOfHiddenLayer1Nodes, NumOfHiddenLayer2Nodes, NumOfHiddenLayer3Nodes, NumOfLabel)

# Create Dataset
ChurnDt = Churn_ModellingDataset("./Datasets/Churn_Modelling.xlsx")
NumOfTrainData = int(0.7 * len(ChurnDt)).__int__()
NumOfTestData = int(0.2 * len(ChurnDt)).__int__()
NumOfValidData = len(ChurnDt).__int__() - NumOfTrainData - NumOfTestData

# Split data
TrainData, ValidData, TestData = random_split(ChurnDt, [NumOfTrainData, NumOfValidData, NumOfTestData])

TrainDataLoader = DataLoader(TrainData, batch_size=10, shuffle=True)
ValidDataLoader = DataLoader(ValidData, batch_size=NumOfValidData, shuffle=True)
TestDataLoader = DataLoader(TestData, batch_size=NumOfTestData, shuffle=True)

# Select Loss Function and Optimizer
LossFunc = nn.CrossEntropyLoss()
#Optimizer = torch.optim.Adam(MlpModel1.parameters(), lr=LearningRate)
Optimizer = torch.optim.Adam(MlpModel2.parameters(), lr=LearningRate)
#Optimizer = torch.optim.Adam(MlpModel3.parameters(), lr=LearningRate)

# Train Model
for epoch in range(NumOfEpoch):
    for Data_Batch, Label_Batch in TrainDataLoader:
        Optimizer.zero_grad()
        #Train_Out = MlpModel1(Data_Batch)
        Train_Out = MlpModel2(Data_Batch)
        #Train_Out = MlpModel3(Data_Batch)
        Train_Loss = LossFunc(Train_Out, Label_Batch)
        Train_Loss.backward()
        Optimizer.step()

    # Show Acuuracy
    if epoch % 10 == 0:
        for Valid_Data_Batch, Valid_Label_Batch in ValidDataLoader:
            #Valid_Out = MlpModel1(Valid_Data_Batch)
            Valid_Out = MlpModel2(Valid_Data_Batch)
            #Valid_Out = MlpModel3(Valid_Data_Batch)
            _, Predicted = torch.max(Valid_Out.data, 1)
            Validation_Loss = LossFunc(Valid_Out, Valid_Label_Batch)
            print('Epoch [%d/%d] Train Loss: %.4f' % (epoch + 1, NumOfEpoch, Train_Loss.item()))
            print('Epoch [%d/%d] Validation Loss: %.4f' % (epoch + 1, NumOfEpoch, Validation_Loss.item()))
            Accuracy = (100 * torch.sum(Valid_Label_Batch == Predicted) / NumOfValidData)
            print('Accuracy of the network in Validation %.4f %%' % Accuracy)

# Test Model and Claculate Accuracy And Creat Confusion Matrix
for Test_Data_Batch, Test_Label_Batch in TestDataLoader:
    #Test_Out = MlpModel1(Test_Data_Batch)
    Test_Out = MlpModel2(Test_Data_Batch)
    #Test_Out = MlpModel3(Test_Data_Batch)

    # Claculate Accuracy
    _, Predicted = torch.max(Test_Out.data, 1)
    Test_Loss = LossFunc(Test_Out, Test_Label_Batch)
    Accuracy = (100 * torch.sum(Test_Label_Batch == Predicted) / NumOfTestData)

    print('---------------------------------------------')

    print('Final Accuracy of the network %.4f %%' % Accuracy)

    # Create Confusion Matrix
    y_pred = []
    y_true = []

    output = (torch.max(torch.exp(Test_Out), 1)[1]).data.cpu().numpy()
    y_pred.extend(output)  # Save Prediction

    labels = Test_Label_Batch.data.cpu().numpy()
    y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = ('1', '0')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    TP = cf_matrix[0][0]
    FP = cf_matrix[1][0]
    FN = cf_matrix[0][1]
    TN = cf_matrix[1][1]

    # Calculate The Accuracy Using The Confusion Matrix
    Accuracy = 100 * (TP + TN)/(TP+TN+FN+FP)
    print('Calculate The Accuracy : %.4f %%' % Accuracy)
    # Calculate The Error rate Using The Confusion Matrix
    ErrorRate = 100 * (FP + FN)/(TP+TN+FN+FP)
    print('Calculate The Error Rate : %.4f %%' % ErrorRate)
    # Calculate The Precision Using The Confusion Matrix
    Precision = 100 * TP/(TP+FP)
    print('Calculate The Precision : %.4f %%' % Precision)
    # Calculate The Recall Using The Confusion Matrix
    Recall = 100 * TP/(TP+FN)
    print('Calculate The Recall : %.4f %%' % Recall)
    # Calculate The FPR Using The Confusion Matrix
    FPR = 100 * FP/(TN+FP)
    print('Calculate The FPR : %.4f %%' % FPR)
    # Calculate The FPR Using The Confusion Matrix
    Specificity = 100 * TN/(TN+FP)
    print('Calculate The Specificity(True Negative Rate) : %.4f %%' % Specificity)

    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 100, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(6, 3))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    #plt.savefig('output.png')



