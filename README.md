This is a comparative study of different Machine Learning approaches.
We have investigated various neural network design choices and compared them on
the Kaggle Titanic Dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset.
We have compared linear classification, logistic regression, MLP (1 and 2D),
4D MLP with a residual network, and have researched and implemented the tabular autoencoder.
We have concluded that some of these designs were great at classifying our data (Linear, Logistic, MLP),
and some are not meant to be used for this type of data (Tabular autoencoder).
After a thourough investigation, we have determined that MLP is the best
at classifying the Titanic data, and have deployed it into a simple geussing game.
In the game a users geuss is compared with the ML model to see if the user made the correct choice.

To run, clone this repository. 
pip install -r requirements.txt
Run every cell in main.ipynb to walk through our research, train, and store a copy of each model. 
Then, play Titanic Geusser in deployment_game.py to see the MLP model in action.

Authors: Olly Love, Nathan Singer, David Kelly
