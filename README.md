# CS310 Project

Welcome to the project. This document will explain each subfolder within this project folder. This file is an extension of the final project.pdf. Knowledge on models there is assumed when elaborating here.

1. Game
This folder contains the CardClash.zip file which contains the playable game and a READMEGame.md file. Please do read the READMEGame.md file before attempting to play the game as it extensively details the steps to play the game. The game is only playable on macOS, and the file aslo details steps to install the game. This game is developed purely as an experimental platform to showcase the power of the AI which represents the bulf of the project.

2. AIModelsAndTraining
This folder contains all the finalized models used to generate the data for the cards which is used in the game. It also contains code for the training process of each model. Here, we will breakdown each model and some key functions used within each file. 

## Pre requisites
* Python 3.10
* pip3 for Mac or pip for Windows

Initially, ensure appropriate executable permissions are available for the following files. If not executable, please appropriately change permissions using the chmod command.:
- build.sh (for Mac users)
- build.ps1 (for Windows users)
- mynltk.py
- tor.sh (for Mac users)
- tor.ps1 (for Windows users)
- main.py

Before running any of the files, make sure to run either the :

    ./build.sh 
if you're using a Mac, or the :

    ./build.ps1 
if you're using Windows. These scripts will install all the modules you need to run the files. If there are any errors, use pip and install the following modules:
- torch
- numpy
- pandas
- nltk (after installing nltk, run the file mynltk.py using "python mynltk.py" in the terminal.)
- itertools
- utils
- openpyxl

## Running the code
After that, running one of the following commands would run the generative AI model (a forward pass):
### Mac Users
    ./tor.sh
or

    python3 main.py
You would then be able to see the generated output on the terminal. 

    python3 [filename].py

### Windows Users (on Powershell terminal)
    ./tor.ps1
or

    python main.py
    
You would then be able to see the generated output on the terminal. 

As for running the **training data**, comment and uncomment sections within the code as necessary, i.e., either comparison or performance training, and run:
### Mac Users
    python3 [filename].py

### Windows Users
    python [filename].py

## Files
* desc_generator.py
This file contains the finalized model class (as discussed in the final report) for the description generator.
The key functions :
  - generate_sent(self,word,max_len) generates a sentence for a given max_len, starting from word.
                    
  - forward(self, x) starts a forward pass through the neural network and outputs a vector of probabilities of words to follow x.
                    
  - predict(self,o) takes in the probabilites from forward() and returns the three most probable words to follow the word,x.

* desc_classifier.py
This file contains the finalized model class (as discussed in the final report) for the description classifier.
The key functions :
  - forward(self, x) starts a forward pass through the neural network with the input sentence. and activates one of the 5 neuron outputs representing a power scale.
                    
  - give_scale(self,x) takes in the output from forward() and returns the classified scale of the sentence.

* skill_text_generator.py
This file contains the finalized model class (as discussed in the final report) for the card name generator.
The key functions :
  - forward(self, x) starts a forward pass through the neural network and outputs a vector of probabilities of words to follow x.
                    
  - predict(self,o) takes in the probabilites from forward() and returns the most probable word to follow the word,x.

* attribute_generator.py
This file contains the finalized model class (as discussed in the final report) for the integer attributes generator.
The key functions :
  - read(data) reads from the datafile and categorizes each power scale into a pandas dataframe.
                    
  - sample(ps) creates a mean and covariance matrix before creating a multinomial gaussian distribution. It then samples from the distribution, rounds the values and returns the integer attributes.

* main.py
The main file which contains the flow of execution of initializing one model and piping its output into another. At the end,
all outputs are printed onto the stdout which is then used by the game.

* tor.sh
The bash script that is ran from the game which runs main.py. The output on the stdout is then used in game.

* desc_gen_train.py
This file contains the training code used to train the description generator, oulined in desc_generator.py. The training has two main parts. One is the comparison training, and the other is pure performance training. 

  Comparison training : The model is reinitialised on every fold of the training where the model's performance is reviewed at the end of every fold. The final epoch's performance   over each fold is then averaged. Different models and variations were trained this way to extract the performance of each model for comparison purposes.

  Pure performance training: Each model here is trained over 20 epochs with no reinitialisation. This is to highlight the actual potential of the model for a longer training        period, and highlight shortcomings within the model in of itself.

* desc_class_train.py
This file contains the training code used to train the description classifier, outlined in desc_classifier.py. Unlike previously, this training only constists of the pure performance training as the only model used showed outstanding results. It also includes a pre-trained embedding matrix initialisation which is used in the neural network training.

* cardname-gen_train.py
This file contains the training code used to train the card name generator, outlined in skill_text_generator.py. The training has two main parts. One is the comparison training, and the other is pure performance training. 

  Comparison training : The model is reinitialised on every fold of the training where the model's performance is reviewed at the end of every fold. The final epoch's performance over each fold is then averaged. Different models and variations were trained this way to extract the performance of each model for comparison purposes.

  Pure performance training: Each model here is trained over 20 epochs with no reinitialisation. This is to highlight the actual potential of the model for a longer training period, and highlight shortcomings within the model in of itself.

* attr_gen_train.py
This file contaions the training code for the integer attribute generator, outlined in attribute_generator.py. This tarining extracts each power scale as a multinomial gaussian distribution (MGD). It first prepares a DataFrame for each power scale, the proceeds to create the mean and covariance matrices for each dataset. It then computes the MGD and returns the mean and covariance matrices. The model is trained well if the mean vector appropriately shows a relationship between the power scales.
