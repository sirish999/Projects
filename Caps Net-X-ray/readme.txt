2nd dataset is the Dataset which is solely used for comparison of Capsule vs CNN performance.Since this dataset gives high accuracy we figured this will be better for comparing the capsule with CNN(and transfer learning).


Main dataset is the dataset proposed in the paper which is of our primary interest.

The submission also contains revised project proposal. During the course of completion we made few small changes to our approach and architecture. All such changes are reflected in this version of project proposal.


Main dataset has four parts:
1.clean data and preprocessing for pre-training: Here we add data from different sources to form 112k images, later this data is cleaned and duplicate values and images with multiple labels are removed to form 94323 images which are stored in numpy arrays files(.npy) of size 2500 images each. These will then be uploaded to google drive to concatenate on google colab because this required very high computational capability.

2.combining datasets for training data: Here we take covid, non covid and normal cases from multiple datasets.
stack them together according to their labels and split them into training and testing. finally the numpy arrays of training and testing with balanced classes is generated.

3.generating test numpy: similiarly test numpy is generated.

4.Finally the concatenated numpy arrays in google drive are loaded into numpy arrays and deployed into the capsule network. This notebook also contains the results and comparison


2nd dataset:
Here the numpy arrays of another data set are formed in similiar fashion. These are plugged into transfer learning model using resnetv2 and the same data is used in our developed capsule network. The performance comparison was the main idead behind this for capsule vs cnn(with TF).