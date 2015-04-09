This is to explain where everything is. If you look into requirements.txt, it will show you the libraries that need to be
installed to run the code (open using Word, it is all run together in notepad). The easiest way to install the libraries
is to use pip. This tool can be downloaded quite easily. Once downloaded, it's as easy to install all of the packages at
once if you have an Apple or Linux machine by using the requirements.txt file. Once in the Titanic folder, all that is
needed to install all of the packages is to run 'sudo pip install -r requirements.txt'. This will get all of the libraries 
in one go. The data folder holds the training data (the test.csv file is not used for our programs). 
Our algorithms are located in the code folder. There is a separate README file in the code folder explaining the code further.
In a command prompt, navigate to the code folder. Once there,

run the code: python RandomForest.py    
              python support_vector_machine.py
  
If the libraries are installed, the code should execute without a problem.
