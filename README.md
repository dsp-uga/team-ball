# team-ball

# Neuron Recognition  

In this project we were challenged to identify neurons in time-series image data in the testing set. The training sets contain 19 samples and the testing sets contain 9 samples. Each sample includes a variable number of time-varying TIFF images. Each sample has unique numbers and positions of the neurons. In the end, the model gives us averages of recall, prescision, inclusion, exclusion and combined to be  respectively on the test set.

## Data    
![image](https://camo.githubusercontent.com/8b0a462a43fcab3e83992d7b4aed5a92feda0dc7/687474703a2f2f6e6575726f66696e6465722e636f64656e6575726f2e6f72672f636f6d706f6e656e74732f6173736574732f6d6f7669652e676966)
![image](https://camo.githubusercontent.com/21fcbc0a48052b77af30d741b71a736dbf9ed4b0/687474703a2f2f6e6575726f66696e6465722e636f64656e6575726f2e6f72672f636f6d706f6e656e74732f6173736574732f7a6f6f6d696e672e676966)    

The image on the left represents the time-varying images in the training and testing data. The image on the right is the extract "regions of interest" (the pink regions on the right) that correspond to individual neurons, which is the goal to draw circles around the regions that contain neurons.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes.

## Dependencies
-[keras](https://keras.io/#installation)

-[tensorflow](https://www.tensorflow.org/install/)

-[Fully Convolutional Networks with Keras](https://github.com/JihongJu/keras-fcn)

## How to run
To get it running
### Main PRogram 
    src/main.py is the main file to run the project.
     
Following keys are settable through command arguments :
     
* --epoch : sets number of epochs for which training will go ( this is applicable to UNET and FCN models )
* --dataset : this sets the path to the training files. target folder should contain one folder per sample and they have to comply to the original dataset format
* --testset : this is the path test samples, this folder should contain one folder per each sample and they should follow the original dataset's format
* --model : this sets the model to either of UNET/FCN/NMF 
* --train : if supplied training will be done 
* --exportpath : set the path to which numpy arrays for train and test file as well as model will be saved. note that this same path will be used to load them 
* --predict : if supplied, prediction will also be done on the data set
* --logfile : sets the path to the logging file 
* --preprocessor : selects the preprocessor to be applied to the dataset 
* --bach : sets the batch size for the training models, this only applies to UNET and FCN

A sample running command : 
    
    $ python main.py --train --predict --exportpath="../output" --dataset="../data/train" --testset="../data/test"

### Downloader 

A shell script has been design to download the datasets and decompresses them. This shell code is  under data/download.sh

## Contributing

There are no specific guidelines for contributing.  Feel free to send a pull request if you have an improvement.


## Authors

See the [contributors](./CONTRIBUTORS.md) file for details

## License

This project is licensed under the MIT License - see the [License](./LICENSE) file for details

## Acknowledgments

* This project was completed as a part of the Data Science Practicum 2018 course at the University of Georgia
* [Dr. Shannon Quinn](https://github.com/magsol)
 is responsible for the problem formulation and initial guidance towards solution methods.
* We partly used provided code from [This Repository](https://github.com/JihongJu/keras-fcn).
It provides implementation of Fully Convolutional Networks with Keras
* Other resources used have been cited in their corresponding wiki page. 

