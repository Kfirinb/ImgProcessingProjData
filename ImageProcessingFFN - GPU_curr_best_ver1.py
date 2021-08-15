import itertools

from torchvision import datasets
import math
import sklearn.manifold as s
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from math import floor, ceil
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import torch.utils
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
seed = 1
torch.manual_seed(seed)

# hyper params:
learning_rate = 0.6
epochs_num = 2
BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Graphing our training and validation accuracy and loss
def train_val_loss_accuracy_graphs(tr_accuracy,val_accuracy,tr_loss, val_loss,epoch_num,algo):
    if algo != "encoder-decoder":
        #accuracy ploting:
        plt.plot(epoch_num, tr_accuracy, "r", label ="Training accuracy")
        plt.plot(epoch_num, val_accuracy, "b", label="Validation acc")
        conc = "Training and validation accuracy in " + algo
        plt.title(conc)
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend()
        plt.figure()

    #loss ploting:
    plt.plot(epoch_num, tr_loss, "r", label="Training loss")
    plt.plot(epoch_num, val_loss, "b", label="Validation loss")
    conc2 = "Training and validation loss in " + algo
    plt.title(conc2)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()


# Build encoder/decoder classes, and construct the AE from them
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        #self.linear1 = nn.Linear(122500, 950)
        #self.linear1 = nn.Linear(32400, 5000)
        self.linear1 = nn.Linear(32400, 950)
        #self.linear2 = nn.Linear(5000, 1000)
        self.linear2 = nn.Linear(950, 920)
        #self.linear3 = nn.Linear(1000, latent_dims)
        self.linear3 = nn.Linear(920, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu_(self.linear1(x))
        x = F.leaky_relu_(self.linear2(x))
        return self.linear3(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 920)
        self.linear2 = nn.Linear(920, 950)
        #self.linear3 = nn.Linear(950, 122500)
        self.linear3 = nn.Linear(950, 32400)
        #self.linear3 = nn.Linear(1125, 40000)


    def forward(self, z):
        z = F.leaky_relu_(self.linear1(z))
        z = F.leaky_relu_(self.linear2(z))
        z = torch.sigmoid(self.linear3(z))
        #return z.reshape((-1, 1, 350, 350))
        return z.reshape((-1, 1, 180, 180))
        #return z.reshape((-1, 1, 200, 200))


class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


criterion = nn.MSELoss()  # comparison between pixel of output nn imgae and input

# training:
def train_data(autoencoder, training_data, val_loader, epochs=epochs_num):
    #opt = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    opt = torch.optim.SGD(autoencoder.parameters(), lr=learning_rate)

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    epoch_num = []
    autoencoder.train()  # added, hope its right
    for epoch in range(epochs):
        epoch_loss = 0
        for x in training_data:
            x = x.to(device)  # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = criterion(x_hat, x)
            epoch_loss += loss.item()
            loss.backward()
            opt.step()
        epoch_loss = epoch_loss / len(training_data)
        train_loss.append(epoch_loss)

        # validation:
        autoencoder.eval()  # don't change model while using it...
        with torch.no_grad():
            epoch_loss_val = 0
            for val in val_loader:
                val_hat = autoencoder(val)
                loss_val = criterion(val_hat, val)
                epoch_loss_val += loss_val.item()
        epoch_loss_val = epoch_loss_val / len(val_loader)
        val_loss.append(epoch_loss_val)
        epoch_num.append(epoch)

        print("In training: epoch : {}/{}, train loss = {:.8f}".format(epoch + 1, epochs, epoch_loss))
        print("In training: epoch : {}/{}, valid loss = {:.8f}".format(epoch + 1, epochs, epoch_loss_val))

    train_val_loss_accuracy_graphs(train_accuracy, val_accuracy, train_loss, val_loss, epoch_num, "encoder-decoder")
    return autoencoder

def encoder_output(autoencoder, valdata, samples):
    autoencoder.eval()
    with torch.no_grad():
        lab_lst = []
        vec_lst = []
        for i in valdata:
            if samples == 0:
                break
            samples -= 1
            z = autoencoder.encoder(i[0].to(device))
            z = z.to('cpu').detach()
            vec_lst.append(z)
            # vec_lst.append(autoencoder.encoder(i[0].to(device)))
            lab_lst.append(i[1])
    return vec_lst, lab_lst


def train(autoencoder, data, val_data, epochs=epochs_num):
    #opt = torch.optim.Adam(autoencoder.parameters())
    opt = torch.optim.SGD(autoencoder.parameters(), lr = learning_rate)
    autoencoder.train()
    img_sample_lst = []
    imgs_counter = 0

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    epoch_num = []

    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in data:
            x = x.to(device)  # GPU
            #x.to(device)
            x_hat = autoencoder(x)
            loss = criterion(x_hat, x)

            if epochs-1 == epoch:
                if imgs_counter < 16:

                    x_hat.reshape((-1, 1, 180, 180))
                    #x_hattt = x_hat
                    x.reshape((-1, 1, 180, 180))

                    x_result = x[0, :, :]
                    x_hat_result = x_hat.data[0, :, :]
                    img_sample_lst.append(x_result)
                    img_sample_lst.append(x_hat_result)

                    #plt.imshow(x_hat_result.numpy()[0], cmap='gray')
                    #plt.imshow(x_result.numpy()[0], cmap='gray')
                    imgs_counter += 2
            opt.zero_grad()
            loss.backward()

            opt.step()

            epoch_loss += loss.item()
            torch.cuda.empty_cache()
        epoch_loss = epoch_loss / len(data)
        train_loss.append(epoch_loss)

        # validation:
        autoencoder.eval()  # don't change model while using it...
        with torch.no_grad():

            epoch_loss_val = 0
            for val, zzzZZZ in val_data:
                val = val.to(device)  # GPU
                val_hat = autoencoder(val)
                loss_val = criterion(val_hat, val)
                epoch_loss_val += loss_val.item()
                torch.cuda.empty_cache()
        epoch_loss_val = epoch_loss_val / len(val_data)
        val_loss.append(epoch_loss_val)
        epoch_num.append(epoch)
        print("In training: epoch : {}/{}, train loss = {:.8f}".format(epoch + 1, epochs, epoch_loss))
        print("In training: epoch : {}/{}, valid loss = {:.8f}".format(epoch + 1, epochs, epoch_loss_val))
    train_val_loss_accuracy_graphs(train_accuracy, val_accuracy, train_loss, val_loss, epoch_num, "encoder-decoder")
    return autoencoder, img_sample_lst

class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class valData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 900.
        self.layer_1 = nn.Linear(900, 200)
        self.layer_2 = nn.Linear(200, 20)
        self.layer_out = nn.Linear(20, 1)

        # self.relu = F.leaky_relu_()
        # self.dropout = nn.Dropout(p=0.1)
        # self.batchnorm1 = nn.BatchNorm1d(4)
        # self.batchnorm2 = nn.BatchNorm1d(4)

    def forward(self, inputs):
        x = F.leaky_relu_(self.layer_1(inputs))
        x = F.leaky_relu_(self.layer_2(x))
        x = F.leaky_relu_(self.layer_out(x))
        return x


# accuracy calculation
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

unloader = transforms.ToPILImage()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


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

def main():
    # prior processing the images:
    #transform = transforms.Compose([transforms.CenterCrop((350, 350)), transforms.Grayscale(1), transforms.ToTensor()])
    #transform = transforms.Compose([transforms.Resize((180, 180)), transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),transforms.Grayscale(1)])
    transform = transforms.Compose([transforms.Resize((180, 180)), transforms.Grayscale(1), transforms.ToTensor()])
    root = "./binary_cropped_miroring_augmentation"

    # load image with labels - which image came from each folders
    dataset = datasets.ImageFolder(root = root, transform = transform)

    # split data, 99% goes to train, 0.1% to validation:
    traindata, valdata, = random_split(dataset, [ceil(0.99 * len(dataset)), floor(0.01 * len(dataset))])  # Ori

    """
    sum = len(dataset)
    a = ceil(0.8 * len(dataset))
    b = floor(0.2 * len(dataset))
    print ("a+b = ",(a+b),"a=",a,"b=",b,"sum = ", sum)
    exit(0)
    
    traindata, valdata, = random_split(dataset, [floor(0.2 * len(dataset)), ceil(0.8 * len(dataset))])  # checking
    """
    # Load data in batches, and shuffling the data before the training:
    trainloader = DataLoader(traindata, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(valdata, batch_size = BATCH_SIZE, shuffle = True)

    latent_dims = 900
    autoencoder = Autoencoder(latent_dims)  # initializing nn
    autoencoder.to(device)
    #autoencoder = autoencoder.to(device)
    autoencoder, comp_imgs_lst = train(autoencoder, trainloader, val_loader, epochs_num)  # trainig data

    #showing encoder decoder before and after output
    files = []
    rows = 4
    for i in range(len(comp_imgs_lst)):
        files.append(tensor_to_PIL(comp_imgs_lst[i]))
    for num, x in enumerate(files):
        #img = PIL.Image.open(x)
        img = x

        plt.subplot(rows, 4, num + 1)
        # plt.title(x.split('.')[0])
        plt.axis('off')
        plt.imshow(img,aspect='auto',cmap='gray')

    # get encoder output with 200 pictures from traindata:
    ecn_samp_output, labels = encoder_output(autoencoder, traindata, 200)

    # get encoder output with all training data pictures:
    #########3#ecn_samp_output_sec_nn, labels_sec_nn = encoder_output(autoencoder, traindata, 3500)
    ecn_samp_output_sec_nn, labels_sec_nn = encoder_output(autoencoder, traindata, 1700) ##########@@@@@@@@@@@@@@@@@22

    # precessing encoder output to classification algorithm
    ecn_samp_output_sec_nn = torch.cat(ecn_samp_output_sec_nn)
    ecn_samp_output_sec_nn = np.array(ecn_samp_output_sec_nn)
    data_classifi, labels_classifi = ecn_samp_output_sec_nn, labels_sec_nn

    # precessing encoder output to t-sne algorithm
    ecn_samp_output = torch.cat(ecn_samp_output)
    ecn_samp_output = np.array(ecn_samp_output)

    # T-sne algorithm:
    X_embedded = s.TSNE(n_components=2).fit_transform(ecn_samp_output)  # make vec input as 2d data axis
    x = X_embedded[:, 0]  # x axis
    y = X_embedded[:, 1]  # y axis
    plt.figure()
    plt.scatter(x, y, c=labels)  # plot with labels of each class
    plt.show()

    # Training Binary Classification:
    # train & test data split to train data and labels
    tr_data = []
    val_data = []
    tst_data = []
    y_train = []
    y_val = []
    y_test = []
    for i in range(math.floor(len(data_classifi) * 0.8)):
        tr_data.append(data_classifi[i])  # train data
        y_train.append(labels_classifi[i])  # train labels
    for k in range(math.floor(len(data_classifi) * 0.1)):
        val_data.append(data_classifi[(math.floor(len(data_classifi) * 0.8)+k)])  # val data
        y_val.append(labels_classifi[(math.floor(len(data_classifi) * 0.8)+k)])  # val labels
    for j in range(math.floor(len(data_classifi) * 0.1)):
        tst_data.append(data_classifi[len(data_classifi) - j - 1])  # test data
        y_test.append(labels_classifi[len(labels_classifi) - j - 1])  # test labels

    # we didnt use that at the end, optional:
    # standartization input:
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(tr_data)
    # X_test = scaler.transform(tst_data)
    X_train = tr_data
    X_test = tst_data

    # transform data to tensors
    train_data = trainData(torch.FloatTensor(X_train),
                           torch.FloatTensor(y_train))
    val_data = valData(torch.FloatTensor(val_data),
                           torch.FloatTensor(y_val))
    test_data = testData(torch.FloatTensor(X_test))

    # set batch size to the data. shuffle isn't needed at that stage, as it was shuffled already
    train_loader = DataLoader(dataset=train_data, batch_size=int(BATCH_SIZE), shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=int(BATCH_SIZE), shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    # Initialization of the model:
    model = binaryClassification()
    #model.to(device)
    criterion3 = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss = One Sigmoid Layer + BCELoss (solved numerically unstable problem)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    bc_train_loss = []
    bc_train_accuracy = []
    bc_val_loss = []
    bc_val_accuracy = []
    bc_epoch_num = []

    # train data:
    for e in range(1, 15):  # epochs_num = 15
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            model.to(device)
            optimizer.zero_grad()
            # get prediction
            y_pred = model(X_batch)
            # calc loss:
            loss = criterion3(y_pred, y_batch.unsqueeze(1))  # unsqueeze - get batch relevant data from specifix index
            # calc accuracy
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
        bc_train_accuracy.append(epoch_acc / len(train_loader))
        bc_train_loss.append(epoch_loss / len(train_loader))
        bc_epoch_num.append(e)


        the_val_loss = 0
        the_val_acc = 0
        model.eval()
        with torch.no_grad():
            for a_batch, b_batch in val_loader:
                a_batch, b_batch = a_batch.to(device), b_batch.to(device)
                model.to(device)
                optimizer.zero_grad()
                # get prediction
                b_pred = model(a_batch)
                # calc loss:
                loss = criterion3(b_pred, b_batch.unsqueeze(1))  # unsqueeze - get batch relevant data from specifix index
                # calc accuracy
                acc = binary_acc(b_pred, b_batch.unsqueeze(1))
                torch.cuda.empty_cache()
                the_val_loss += loss.item()
                the_val_acc += acc.item()
            print(
                f'Epoch {e + 0:03}: | Loss: {the_val_loss / len(val_loader):.5f} | Acc: {the_val_acc / len(val_loader):.3f}')
            bc_val_loss.append(the_val_loss / len(val_loader))
            bc_val_accuracy.append(the_val_acc / len(val_loader))

    train_val_loss_accuracy_graphs(bc_train_accuracy, bc_val_accuracy, bc_train_loss, bc_val_loss, bc_epoch_num, "binary classification")

    # test data:
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())  # change to cpu

    # compare predictions of the images with their real labels:
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    print(classification_report(y_test, y_pred_list))  # print measurement results
    plot_confusion_matrix(confusion_matrix(y_test, y_pred_list), ['cancer','normal'], normalize=True)


if __name__ == '__main__':
    main()
