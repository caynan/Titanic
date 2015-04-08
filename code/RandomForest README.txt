This is the README file for the RandomForest.py code.
The first thing to check before running this code is to make sure you have the proper Python libraries installed.
The libraries are listed below:

pandas
numpy
sklearn
scipy
matplotlib

One change needs to be made to the code: you need to change the path to train.csv to match where you put it on your computer.
I commented the first line stating this. If you simply put the csv file in the same directory as the RandomForest code,
you can run the code without changing the path (this is the easiest option).

There are several points in the code where I have commented out code. I put the code there as a way for you to see the actual
action of some of the algorithms and the output (such as grid search), but did not want it to be in there by default. If you
want to see the details, you can just simply uncomment the code that I point out in the code. One thing to note is the code
takes around 60-90 seconds to execute since we are using KFold and grid search with cross-validation and this takes a while.
At about 60-70 seconds, the average prediction score will be displayed, and at about 90 seconds, the learning curve will
be displayed.
