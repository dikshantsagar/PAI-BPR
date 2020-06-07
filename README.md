# MCA-Project


The whole project was implemented on Google Colab, hence it is suggested to run it there itself!!

## folder structure for Code files
Group_9
	- src
	  - Attribute_keras.ipynb - Attribute Representation Model Training 
	  - GetAttribute.ipynb - Getting Visual representations from the Attribute Model
	  - GPBPR2.py - The Previous State of the Art Model File.
	  - train.py - Training file for our state of the art model.
	  - test.py - Testing our model
	  - Testing_Model.pynb - Notebook visualising our SOTA model and its predictions and outputs
	  - main.ipynb - To run train.py with the desired requirements

You don't need to run "Attribute_keras.ipynb" and "GetAttribute.ipynb". They are just to extract visual features from an image.
Also, no need to run "train.py". Model is already trained.

The flow of code is "main.ipynb" -->  "test.py" --> "GPBPR2.py".

Following things can be done with these set of files:

1) The model is already trained.
IF YOU WANT TO TEST THE MODEL, Then you require these files "main.ipynb" -->  "test.py" --> "GPBPR2.py".
 CHANGE THE VARIABLE NAMED "PATH" in file "test.py" with the parent directory of "Group_9" folder- 
 Example: PATH = r"/content/drive/My Drive/GPBPR" where GPBPR is the parent directory of Group_9.
 Also when you run "main.ipynb", change the value to the path of the parent directory of Group_9 folder in the following cell --> 
#!python 'Enter the path of parent directory of Group_9 folder/Group_9/src/test.py'

2) IF YOU WANT TO TEST THE PERSONALIZED RETRIEVAL PART, the you require "Testing_Model.pynb" --> "GPBPR2.py".
 CHANGE THE VARIABLE NAMED "PATH" in file "Testing_Model.pynb" with the parent directory of "Group_9" folder- 
 Example: PATH = r"/content/drive/My Drive/GPBPR" where GPBPR is the parent directory of Group_9.
 Simply run all the cells of the notebook. You will get Personalized Retrieval of one user(random user in the test set).
 Also the attributes of each of highest ranked bottom and lowest ranked bottom will be shown.  


