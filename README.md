# bcdnet
BCDnet: Parallel heterogeneous eight-category automatic diagnosis model of breast pathology

Deep learning environment: Keras + python3.6 + Tensorflow 2.0


Create a new myconfig folder, and then put the file configt.py in this folder.

Create the directory where the data set is located according to configt.py, randomly assign the data samples to the directories where the train set, validation set, and test set are located at a ratio of 8:1:1, and execute python bcdnet.py or python bcdnet_co.py on the terminal.

Model fusion generally has Average method and concatenate method. If you run the bcdnet_co.py file, the model fusion adopts the concatenate method. If you run the bcdnet.py file, the model fusion adopts the average method.
