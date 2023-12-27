import torch
from torch import nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import random
import cv2
import os
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataloader related hyper parameters.., please note for (most) data related changes to take place you need to create new dataset. This can be done by simply deleting the previous.
CLASS_DICT = {0: "Cat", 1: "Dog"}

DATA_PATH = "V:\\PATH\\TO\\DATA\\"
SAVE_PATH = "V:\\PATH\\TO\\SAVE\\"

COLOR_CHANNELS = 1
HEIGHT = 224
WIDTH = 224

BATCH = 32

# Deep learning hyper parameters...
EPOCH = 5
LR = 0.0001
KERNEL_SIZE = (3, 3)
STRIDE = 1
PADDING = 0

INPUT_COUNT = int(HEIGHT * WIDTH * COLOR_CHANNELS)
HIDDEN_COUNT = 128
OUTPUT_COUNT = len(CLASS_DICT)


def plot_loss(train_loss, test_loss, epoch):
    plt.subplot(1, 2, 1)
    plt.title("Train loss by epoch.")
    for idx in range(epoch):
        loss = train_loss[idx]
        loss = loss.cpu().detach().numpy()
        plt.stem(str(idx), loss, linefmt=None, markerfmt="Blue", basefmt=None)

    plt.subplot(1, 2, 2)
    plt.title("Test loss by epoch.")
    for idx in range(epoch):
        loss = test_loss[idx]
        loss = loss.cpu().detach().numpy()
        plt.stem(str(idx), loss, linefmt=None, markerfmt="Blue", basefmt=None)

    plt.show()


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def load_data(
    data_path=DATA_PATH, save_path=SAVE_PATH, height=HEIGHT, width=WIDTH, batch=BATCH
):
    dataset = []
    e_count, s_count, dog_count, cat_count = 0, 0, 0, 0

    for subdir, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                try:
                    # OpenCV natively uses numpy arrays. type(img) = <class 'numpy.ndarray'>
                    img = cv2.imread(filename=os.path.join(subdir, filename))
                    img = cv2.resize(img, (width, height))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                    if "Cat_Dog\\Cat\\" in os.path.join(subdir, filename):
                        target = int(0)
                        cat_count += 1

                    else:
                        target = int(1)
                        dog_count += 1

                    my_x = torch.as_tensor(img, dtype=torch.float32)
                    d = my_x, target
                    dataset.append(d)
                    s_count += 1

                except Exception as e:
                    # print(e)
                    # print(f"Found error with file: {os.path.join(subdir, filename)}")
                    e_count += 1
                    if e_count > 200:
                        print("over 200 errors")
                        break

    print(f"Success count! {s_count}")
    print(f"Error count! {e_count}")
    print(f"Images of dogs: {dog_count}")
    print(f"Images of cats: {cat_count}")

    random.shuffle(dataset)

    split = int(0.8 * len(dataset))
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]

    #    full_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    #    torch.save(full_dataloader, save_path + "full_dataloader.pth")
    #    print("full_dataloader has been saved at: ", save_path + "full_dataloader.pth")

    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
    torch.save(test_dataloader, save_path + "test_dataloader.pth")
    print("test_dataloader has been saved at: ", save_path + "test_dataloader.pth")

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    torch.save(train_dataloader, save_path + "train_dataloader.pth")
    print("train_dataloader has been saved at: ", save_path + "train_dataloader.pth")

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")


class catVSdog_CNN(nn.Module):
    def __init__(self, input_channels, hidden_units, output_units):
        super().__init__()

        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_units,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_units,
                hidden_units,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE),
        )

        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(
                hidden_units,
                hidden_units,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_units,
                hidden_units,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE),
        )

        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(
                hidden_units,
                hidden_units,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_units,
                hidden_units,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 6 * 6, out_features=output_units),
        )

    def flipper(self, x):
        """
        Flip tensor (image) vertically. Will help with generalization of the data.
        """
        x = torch.flip(x, dims=(1,))
        return x

    def forward(self, x):
        if random.randint(0, 1) == 1:
            x = self.flipper(x)

        x = self.cnn_block_1(x)
        x = self.cnn_block_2(x)
        x = self.cnn_block_3(x)
        #        print(x.shape[-1], x.shape[-2]) # - USE THIS TO CALCULATE in_features FOR self.classifier !!!
        x = self.classifier(x)
        return x


def train_step(model, loss_fn, optimizer, dataloader):
    train_loss, train_acc = 0, 0

    model.to(device)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        X = X.unsqueeze(1)
        X.type(torch.float32)

        preds = model(X)

        loss = loss_fn(preds, y)
        train_loss += loss
        train_acc += accuracy_fn(y, preds.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(model, loss_fn, dataloader):
    test_loss, test_acc = 0, 0

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.unsqueeze(1)
            X.type(torch.float32)
            test_preds = model(X)
            test_loss += loss_fn(test_preds, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_preds.argmax(dim=1))

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


def main(data_path=DATA_PATH, save_path=SAVE_PATH):
    train_loss_list = []
    test_loss_list = []

    if (
        os.path.exists(f"{save_path}train_dataloader.pth") == False
        or os.path.exists(f"{save_path}test_dataloader.pth") == False
    ):
        print(
            f"Couldn't locate dataloaders in: '{save_path}' ...Creating new dataloaders!"
        )
        load_data(data_path=data_path, save_path=save_path)

    train = torch.load(f"{save_path}train_dataloader.pth")
    test = torch.load(f"{save_path}test_dataloader.pth")

    model_0 = catVSdog_CNN(
        input_channels=COLOR_CHANNELS,
        hidden_units=HIDDEN_COUNT,
        output_units=OUTPUT_COUNT,
    )
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model_0.parameters(), lr=LR)

    best_score = 99999
    best_acc = 0

    for epoch in range(EPOCH):
        train_loss, train_acc = train_step(
            model=model_0, loss_fn=loss_fn, optimizer=optimizer, dataloader=train
        )
        test_loss, test_acc = test_step(model=model_0, loss_fn=loss_fn, dataloader=test)

        print(
            f"Epoch: {epoch+1} | Train loss (avg): {train_loss:.4f} | Train accuracy (avg): {train_acc:.4f}% | Test loss (avg): {test_loss:.4f} | Test accuracy (avg): {test_acc:.4f}%"
        )
        if test_loss < best_score:
            best_score = test_loss
            best_acc = test_acc

            torch.save(model_0, save_path + "best_model.pth")
            torch.save(model_0.state_dict(), save_path + "best_model_state_dict.pth")

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    plot_loss(train_loss=train_loss_list, test_loss=test_loss_list, epoch=EPOCH)

    print(f"Best saved model | loss (avg): {best_score} | accuracy (avg): {best_acc}")
    print(f"Saved at path: {save_path}best_model.pth")


if __name__ == "__main__":
    main()
