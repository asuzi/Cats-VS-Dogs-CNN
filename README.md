Solving the Cats vs Dogs AI problem with <b>87% accuracy</b> using PyTorch, implementing a Convolutional Neural Network (CNN) model.

Correct guesses:  4387 Out of:  4991 images. |  Accuracy: 87.8%

Images of cats: 12476 <br>
Images of dogs: 12477 <br>

Training data: 19962 <br>
Testing data: 4991 <br> <br>
With torchinfo we can get a summary of the model. When running with currently set settings (Batch size 32, hidden inputs 128 and image size 224x224)<br>
![image](https://github.com/asuzi/Cats-VS-Dogs-CNN/assets/61744031/f5cd5d2b-44b9-413b-9693-5c19b9c4285c)
<br>
Here is a simple plot to show the progress for training and testing data.
![image](https://github.com/asuzi/Cats-VS-Dogs-CNN/assets/61744031/5335d6f4-b736-49db-ba81-e76ace39766e)
Using [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) and [torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam) the training loss keeps gradually going down. How ever it is a little bit different for testing loss. This can be caused by many reasons. Over fitting probably being the most common and the case here as well. <br>
