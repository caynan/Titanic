This is the README file for the RandomForest.py code.
The first thing to check before running this code is to make sure you have the proper Python libraries installed.
The requirements.txt file in the main folder shows all the libraries that need to be installed.

One thing to point out: if for any reason you put the train.csv file in any other location, you will need to change the path
to train.csv to match where you put it. I commented the first line stating this.

There are several points in the code where I have commented out code. I put the code there as a way for you to see the actual
action of some of the algorithms and the output (such as grid search), but did not want it to be in there by default. If you
want to see the details, you can just simply uncomment the code that I point out in the code. One thing to note is the code
takes around 60-90 seconds to execute since we are using KFold and grid search with cross-validation and this takes a while.
At about 60-70 seconds, the average prediction score will be displayed, and at about 90 seconds, the learning curve will
be displayed. I have put in a piece of code that will print out the most important features for each fold (most important
features are defined here as being higher than average). This is to show progress within the program as well as I believe it is
pretty interesting to see what is being considered as the important features for each fold.

Run the code by navigating to the code folder from a command prompt and simple enter:
python RandomForest.py
python support_vector_machine.py

If the libraries are installed, the programs should execute without problem.
