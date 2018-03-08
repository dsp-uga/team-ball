# Neuron Recognition with fully convolutional neural network (FCN)    

In this project we were challenged to identify neurons in time-series image data in the testing set. The training sets contain 19 samples and the testing sets contain 9 samples. Each sample includes a variable number of time-varying TIFF images. Each sample has unique numbers and positions of the neurons. In the end, the model gives us averages of recall, prescision, inclusion, exclusion and combined to be  respectively on the test set.

## Data    
![image](https://camo.githubusercontent.com/8b0a462a43fcab3e83992d7b4aed5a92feda0dc7/687474703a2f2f6e6575726f66696e6465722e636f64656e6575726f2e6f72672f636f6d706f6e656e74732f6173736574732f6d6f7669652e676966)
![image](https://camo.githubusercontent.com/21fcbc0a48052b77af30d741b71a736dbf9ed4b0/687474703a2f2f6e6575726f66696e6465722e636f64656e6575726f2e6f72672f636f6d706f6e656e74732f6173736574732f7a6f6f6d696e672e676966)    

The image on the left represents the time-varying images in the training and testing data. The image on the right is the extract "regions of interest" (the pink regions on the right) that correspond to individual neurons, which is the goal to draw circles around the regions that contain neurons.

## Preprocessing    
1.
2. Using NMF to reduce the dimensions.

## Model Training and Testing    

## Collect recognition results    
Save results for the 9 samples in the test set by .json form. Each consists of all the coordinates surrounding predicted neurons in the sample.    
    
## Running

