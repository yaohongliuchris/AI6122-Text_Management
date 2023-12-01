# Dataset Analysis

In this task,we have chosen three distinct domain-specific datasets, to effectively simulate diverse text environments. And perform **tokenizaion** of the documents and **analyst** the difference between before preprocess(e.g stemming, stopwords remove etc) and after. Hence, implemented the **sentences segmentation** and **analyst** the behaviour of common number of words in one sentence. Lastly, we utilized the POS-tag to difference sentence and plotting the result.


Three tiny tasks codes (Tokenization and stemming AND Sentence segmentation AND POS tagging) are in dataset_analysis.py.

(1) Please download  all three documents dataset under the same folder and set the correct path for the dataset and results figures saving path in the dataset_analysis.py file just like below:

    dataset_path = '/your/dataset/folder/path/' AND results_save_path = '/your/results/figures/saving/path/'. 
    
    #The specific Tokenization and stemming AND Sentence segmentation results will be shown in figures form under the results_save_path folder. The POS Tagging results will be printed in your command window.

(2) You can try all three different datasets to test our analysis functions by changing the dataset index, see comments part in the "Dataset selection" code in dataset_analysis.py for details.

(3) The specific three datasets we used could be downloaded from google drive with the link:https://drive.google.com/drive/folders/1Mq5Gbo-DnywedCzsFMhLocrNTB3ILg9B?usp=drive_link or the original link is attatched:
    
    a. **British Airways Reviews**: https://www.kaggle.com/datasets/lapodini/british-airway-reviews
    
    b. **Healthcare NLP: LLMs, Transformers, Datasets**: https://www.kaggle.com/datasets/jpmiller/layoutlm/data
    
    c. **ARXIV Paper Abstruct**: https://www.kaggle.com/datasets/neelshah18/arxivdataset
