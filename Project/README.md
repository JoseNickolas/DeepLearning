Medical Concept Embedding with Variable Temporal Scopes for Patient Similarity - Paper Implementation

The project implements 6 files, and train.py implements all the modules required to perform the training component.

The following files are modules that run train.py

1. mimic.py is a class that has as a parameter the dataset and contains a set of functions that manage the data preprocessing, the extraction of patients’ cohorts, the building of the medical concepts, to be utilized as natural text in another module.

2. word2vec.py; This module implements two classes, word2vec and SkimGramDataset. The former handles the medical concept embedding for the patients’ representation learning, a key element for evaluating patient similarity. In this class, an embedding linear layer is implemented in which it takes an embedding dimension of medical codes (or medical concepts), and the number of codes for each patient. The latter, implements the skip-gram model, and takes as parameters data, a list of patients’ medical concepts, alpha, the variable temporal scope of the sliding window, and beta, the maximum size of the context window across patients. Such module basically builds a pair of “centers” medical concepts and their corresponding “context” pair, which is how close (or far) are medical concepts to their “centers.”

3. siamese_cnn.py is a module that manages the Siamese CNN with the SPP (spatial pyramid pooling), and therefore, it contains two classes. The former, SiameseCNN, has as parameters the feature maps, which are used as an output parameter in the Convolutional layer, kernel size, spp levels, and output dimension, a parameter not mentioned in the paper. All these parameters have been described in the hyperparameters section. The latter, SPP class manipulates the spatial pyramid pooling, which is built in 3 levels (4,2,1). Like the former class, SPP also implements a forward function, but in this case with an adaptive max pooling to handle the 3 levels already stated. 

4.  utils.py module basically performs an intersection between sets to check whether there are common elements across patients, or in other words, between x and y (a pairwise intersect between 2 data points). 

5. model.py performs the main idea of the proposed framework, which is the Patient Similarity Evaluation (PSE), It performs the word2vec module along with the Siamese CNN to score for patient similarity. 

6. train.py implements all the modules stated above and performs the training component of the original paper implementation. For the purposes of draft testing, and making sure the data preprocessing pipeline is working, the cross-entropy loss has been utilized to measure performance, and the Adam optimization is also used to handle SGD. Within this module, a collate function is implemented along the computation of the contrastive loss function.
