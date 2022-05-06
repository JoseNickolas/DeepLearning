# Medical Concept Embedding with Variable Temporal Scopes for Patient Similarity - Paper Implementation

## Jose Nicolas Noriega Portilla 
## May 7, 2022

The project implements 6 files, and train.py implements all the modules required to perform the training component.

The following files are modules that run train.py

1. `mimic.py` is a class that has as a parameter the dataset and contains a set of functions that manage the data preprocessing, the extraction of patients’ cohorts, the building of the medical concepts, to be utilized as natural text in another module.

2. `word2vec.py` This module implements two classes, word2vec and SkimGramDataset. The former handles the medical concept embedding for the patients’ representation learning, a key element for evaluating patient similarity. In this class, an embedding linear layer is implemented in which it takes an embedding dimension of medical codes (or medical concepts), and the number of codes for each patient. The latter, implements a torch dataset for the skip-gram model, and takes as parameters a mimic dataset object (a list of patients’ medical concepts), alpha, the variable temporal scope of the sliding window, and beta, the maximum size of the context window across patients. Such module basically builds a pair of “centers” medical concepts and their corresponding “context” pair, which is how close (or far) are medical concepts to their “centers.”

3. `siamese_cnn.py` is a module that manages the Siamese CNN with the SPP (spatial pyramid pooling), and therefore, it contains two classes. The former, SiameseCNN, has as parameters the feature maps, which are used as an output parameter in the Convolutional layer, kernel size, spp levels, and output dimension, a parameter not mentioned in the paper. All these parameters have been described in the hyperparameters section of the final report. The latter, SPP class manipulates the spatial pyramid pooling, which is built in 3 levels (4,2,1). Like the former class, SPP also implements a forward function, but in this case with an adaptive max pooling to handle the 3 levels already stated. 

4.  `utils.py` includes some utility functions; pairwise_intersect() performs an intersection between two lists to check whether there are common elements. It is used to check if two patients have a common cohort. find_nearest() takes patient representations and finds the closest patients based on the given metric.

5. `model.py` includes the main model of the proposed framework, which is the Patient Similarity Evaluation (PSE). It stacks a (trained) word2vec module to a Siamese CNN to calculate patient representations and score for patient similarity. 

6. `train.py` includes functions for training component of the original paper implementation. First word2vec with variable temporal scope is trained, then the PSE model is trained using the word2vec from previous training and a Siamese CNN.

7. `evaluations.ipynb` a Jupyter Notebook to implement the basline models along with the avaluation metrics.
