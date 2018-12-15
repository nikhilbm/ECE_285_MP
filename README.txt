Description
===========

The aim of this project is to classify human protein into 28 different categories (multi object classification) using GAPnet type architecture on CNNs.

Requirements
============

Download the data from demo.7z file and upload to Jupyter notebook before running the demo. 

Code organization
=================
Data Folder - Contains images necessary to run the demo folder.

demo.ipynb          --   Download the files from the Data folder and upload onto your Jupyter notebook, run our code to get F1 score on test and train data. ****NOTE ALSO DISPLAYS THE SUCCESSFULLY CLASSIFIED IMAGES AND FAILURES AS WELL ***** The images have also been uploaded to the repository.


final_code.ipynb   --    Run the code to train, validate and test data of our model. Uses data which is locally uploaded to our                          jupyter notebook. It will not run with data from UCSD DSMLP. Note that full data cannot be provided due to size constraints.


optimal_model       --    model giving the highest F1 score for training and test data. This is the same model which is loaded into the demo file.


visu.py             --    Displays random images from the dataset.
