import itertools
import os
import sklearn.manifold as s
import numpy
from numpy import interp
from opt_einsum.backends import tensorflow
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import skimage.io
from torchvision import datasets
import cv2
from torch.utils.data import DataLoader, Dataset
from math import floor, ceil
import torch.utils
import torch.distributions
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import random_split
import torch.nn as nn
from torchvision import transforms

# hyper parameters
lr = 0.001
trainEpochs = 15
batchSize = 8

classes = ['adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma', 'normal']
class_to_idx = {"adenocarcinoma": 0, "large.cell.carcinoma": 1, "squamous.cell.carcinoma": 2, "normal": 3}

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1),
            nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3),
            nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7),
            nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        #self.fc1 = nn.Linear(97344, 20000)
        #self.fc2 = nn.Linear(20000, 4)
        self.fc1 = nn.Linear(6400, 80)
        self.fc2 = nn.Linear(80, 4)
        #self.fc1 = nn.Linear(4096, 400)
        #self.fc2 = nn.Linear(400, 4)
        #self.fc1 = nn.Linear(7680, 400)
        #self.fc2 = nn.Linear(400, 30)
        #self.fc1 = nn.Linear(20736, 830)
        #self.fc2 = nn.Linear(830, 4)
        #self.fc1 = nn.Linear(25600, 1600)
        #self.fc2 = nn.Linear(1600, 4)

    def forward(self, x):
        x = self.layer0(x)
        #print(x.shape)
        x = self.layer1(x)
        ##print(x.shape)
        x = self.layer2(x)
        ##print(x.shape)
        x = self.layer3(x)
        ##print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.nn.functional.log_softmax(x, -1)

def train(model, dataset):
    model.train()
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
    img_num = 0
    correct_tr = 0
    epoch_loss_sum = 0    
    for sample, label in trainLoader:
        img_num += 1
        optimizer.zero_grad()
        output = model(sample)
        loss = F.nll_loss(output, label)
        epoch_loss_sum += loss
        loss.backward()
        optimizer.step()
        
        prediction_tr = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_tr += prediction_tr.eq(label.view_as(prediction_tr)).cpu().sum().item()

    accuracy_tr = correct_tr / (len(trainLoader) * batchSize)
    print("epoch num:", i, "train accuracy:", accuracy_tr)
    epochls = epoch_loss_sum/(len(trainLoader) * batchSize)
    return accuracy_tr ,epochls.item()


##Prints pretty confusion metric with normalization option
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    # Add Normalization Option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color ="white" if cm[
            i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def test(model, data):
    model.eval()
    correct_List = []
    FP_List = [0, 0, 0, 0] #### FOR FALSE POSITIVE
    #sum correct per class:
    correct_List.append(0)
    correct_List.append(0)
    correct_List.append(0)
    correct_List.append(0)
    # for wrong predictions:
    correct_List.append(0)
    # predict to outputList
    testLoader = torch.utils.data.DataLoader(data)

    test_labels=[]
    preds=[]
    #predsTuples =[]
   # dataTuples = []
    with torch.no_grad():
        for sample, label in testLoader:
            test_labels.append(int(label[0]))
            #dataTuples.append((sample,label))
            output = model(sample)
            #print(output)
            pred = output.max(1, keepdim=True)[1]

           # predsTuples.append((output,pred[0][0]))
            preds.append(pred[0][0])
            predClass = classes[pred]

            # class_to_idx = {"adenocarcinoma": 0, "large.cell.carcinoma": 1, "squamous.cell.carcinoma": 2, "normal": 3}
            #print("label",label, "predClass", predClass)
            #print("label[0]",label[0])
            if label[0] == 0 and predClass == "adenocarcinoma":
                #print("check1")
                correct_List[0] += 1
         #   elif predClass == "adenocarcinoma" and label[0] != 0:
           #     FP_List[0] += 1

            elif label[0] == 1 and predClass == "large.cell.carcinoma":
                correct_List[1] += 1
                #print("check2")
           # elif predClass == "large.cell.carcinoma" and label[0] != 1:
            #    FP_List[1] += 1

            elif label[0] == 2 and predClass == "squamous.cell.carcinoma":
                correct_List[2]+=1
                #print("check3")
          #  elif predClass == "squamous.cell.carcinoma" and label[0] != 2:
            #    FP_List[2] += 1

            elif label[0] == 3 and predClass == "normal":
                correct_List[3] += 1
                #print("check4")
            #elif predClass == "normal" and label[0] != 3:
            #    FP_List[3] += 1

            else:
                correct_List[4] += 1 #wrong pred
    #print(correct_List)

    #print("test_labels", test_labels)
    #print("preds", preds)
    #print("classes", classes)
    classification_matrix = metrics.classification_report(test_labels,preds,target_names=classes)
    print(classification_matrix)
    confusion_mat = confusion_matrix(test_labels, preds)
    plot_confusion_matrix(confusion_mat, classes, normalize=True)

    '''
    # roc curve:
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(numpy.array(test_labels), numpy.array(preds))
    roc_auc = roc_auc_score(numpy.array(test_labels), numpy.array(preds))
    # print(fpr, tpr, roc_auc)
    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k–')
    plt.plot(fpr, tpr, label='CNN(area={: .3f})'.format(roc_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    '''



def validate(model, validSet, i):
    model.eval()
    correct = 0
    val_loss_sum = 0
    validation_loader = torch.utils.data.DataLoader(validSet, batch_size=batchSize, shuffle=True)
    with torch.no_grad():
        for sample, label in validation_loader:
            output = model(sample)
            val_loss = F.nll_loss(output, label)
            val_loss_sum += val_loss
            prediction = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += prediction.eq(label.view_as(prediction)).cpu().sum().item()
    val_accuracy = correct / (len(validation_loader)*batchSize)
    print("epoch num:" ,i, "val accuracy:", val_accuracy)
    return val_accuracy, (val_loss_sum/(len(validation_loader)*batchSize)).item()

def crop_black_borders(path):
    dirs = ["adenocarcinoma","large.cell.carcinoma","normal","squamous.cell.carcinoma"]
    file_count = 0
    for dir in dirs:
        full_path =path+"/"+dir
        file_count = len(os.listdir(full_path))-1
        print(file_count)
        #exit()
        for image_name in os.listdir(full_path):
            image_path = full_path+"/"+image_name

            file_count = -1

            im = cv2.imread(image_path,0)
            if im is None:
            #if file_count == 0 or im.any()==None:# or im.all()==0 or im.all() ==None:
                break
            """print("1",im)
            print("2",im.size)
            print("3",im.shape)"""



            im = im[:,:, np.newaxis]
            #plt.imshow(im,cmap='gray')
            #plt.show()

            h, w, d = im.shape

            left = -1
            right = -1
            height = -1
            bottom = -1

            # upper limit
            for g in range(h):
                if np.sum(im[g, :, :]) > 10000:
                    print(np.sum(im[g, :, :]))
                    height = g
                    break
            # bottom limit
            for h in range(h-1,0,-1):
                if np.sum(im[h-1, :, :]) > 8000:
                    bottom = h
                    break
            # left limit
            for i in range(w):
                if np.sum(im[:, i, :]) > 10000:
                    left = i
                    break
            # right limit
            for j in range(w - 1, 0, -1):
                if np.sum(im[:, j, :]) > 10000:
                    right = j
                    break

            #print("bottom",bottom, "height", height, "right", right, "left", left)
            #print(im.shape)
            cropped = im[height:bottom, left:right, :].copy()
            #print(cropped.shape)
            ####plt.imshow(cropped,cmap='gray')
            ####plt.show()
            new_image_name = image_name[:-4] + "cropped" +".png"
            new_full_path = full_path + "/Cropped/" + new_image_name
            cv2.imwrite(new_full_path, cropped)

# Graphing our training and validation accuracy and loss
def train_val_loss_accuracy_graphs(tr_accuracy,val_accuracy,tr_loss, val_loss,epoch_num):
    #accuracy ploting:
    plt.plot(epoch_num, tr_accuracy, "r", label ="Training accuracy")
    plt.plot(epoch_num, val_accuracy, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.figure()

    #loss ploting:
    plt.plot(epoch_num, tr_loss, "r", label="Training loss") #tensor.detach().numpy()
    plt.plot(epoch_num, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

if __name__ == '__main__':

    """
    crop_black_borders("./test")
    crop_black_borders("./train")
    crop_black_borders("./valid")"""

    """
    crop_black_borders("./original_imgs/test")
    crop_black_borders("./original_imgs/train")
    crop_black_borders("./original_imgs/valid")"""
    #exit(1)


    #import os
    #cwd = os.getcwd()
    #exit(1)

    # prior processing the images:
    #transform = transforms.Compose([transforms.CenterCrop((350, 350)), transforms.Grayscale(1), transforms.ToTensor()])
    #transform = transforms.Compose([transforms.Resize((350,350)), transforms.Grayscale(1), transforms.ToTensor()])
    #transform = transforms.Compose([transforms.Resize((50, 50)), transforms.Grayscale(1), transforms.ToTensor()])
    #transform = transforms.Compose([transforms.Resize((200, 200)), transforms.Grayscale(1), transforms.ToTensor()])
    transform = transforms.Compose([transforms.Resize((120, 120)), transforms.Grayscale(1), transforms.ToTensor()])

    #root = "./cropped_data" #2 agumentation than cropped added
    #root = "./only_cropped" #only cropped data
    root = "./CroppedFirstThenAugmentation" #crop + hirizintal flip augmentation

    # load image with labels - which image came from each folders
    dataset = datasets.ImageFolder(root=root, transform=transform)

    # load data
    trainData = datasets.ImageFolder(root=root+"/train", transform=transform)
    valData = datasets.ImageFolder(root=root+"/valid", transform=transform)
    testData = datasets.ImageFolder(root=root+"/test", transform=transform)

    lab_lst = []
    vec_lst = []
    for i in trainData:
        z = torch.reshape(i[0], [1, -1])
        vec_lst.append(z)
        # vec_lst.append(autoencoder.encoder(i[0].to(device)))
        lab_lst.append(i[1])
    #vec_lst, lab_lst

    # precessing encoder output to t-sne algorithm
    vec_lst_cat = torch.cat(vec_lst)
    vec_lst_cat_np = np.array(vec_lst_cat)

    # T-sne algorithm:
    X_embedded = s.TSNE(n_components=2).fit_transform(vec_lst_cat_np)  # make vec input as 2d data axis



    x = X_embedded[:, 0]  # x axis
    y = X_embedded[:, 1]  # y axis
    plt.figure()
    scatter = plt.scatter(x, y, c=lab_lst)
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, classes)
    plt.scatter(x, y, c=lab_lst)  # plot with labels of each class
    plt.show()

    """
    trainD = DataLoader(dataset=trainData, shuffle = True,batch_size=batchSize)
    valD = DataLoader(dataset=trainData, shuffle=False, batch_size=batchSize)
    testD = DataLoader(dataset=trainData, shuffle=False, batch_size=batchSize)
    """
    model = myModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    epoch_lst=[]

    # train and test
    for i in range(trainEpochs):
        res_accuracy_train, epoch_ls_train = train(model, trainData) #get epoch train accuracy
        train_accuracy.append(res_accuracy_train) #list of accuracy of trains epoch
        train_loss.append(epoch_ls_train)  # list of loss of each epoch
        res_accuracy_val, epoch_ls_val = validate(model, valData, i) #get epoch validation accuracy
        val_accuracy.append(res_accuracy_val) #list of accuracy of trains epoch
        val_loss.append(epoch_ls_val)  # list of loss of each epoch
        epoch_lst.append(i)
        #test(model, testData)

    # do print graphs results, with:
    train_val_loss_accuracy_graphs(train_accuracy, val_accuracy, train_loss, val_loss, epoch_lst)



    test(model, testData)
    
