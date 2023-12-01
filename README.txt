# 6122_text_group_project

Code for 3.1 Domain Specific Dataset Analysis is in Dataset Analysis File. 
Code for 3.2 Development of a Simple Search Engine is in SearchEngine File. 
Code for 3.3 Development of a Research Trend Explorer is in ResearchTrendExplorer File. 
Code for 3.4 Application is in SearchEngine File. 
You may also read the readme.md under those files.


# Dataset Analysis

In this task,we have chosen three distinct domain-specific datasets, to effectively simulate diverse text environments. And perform **tokenizaion** of the documents and **analyst** the difference between before preprocess(e.g stemming, stopwords remove etc) and after. Hence, implemented the **sentences segmentation** and **analyst** the behaviour of common number of words in one sentence. Lastly, we utilized the POS-tag to difference sentence and plotting the result.


Three tiny task codes (Tokenization and stemming AND Sentence segmentation AND POS tagging) are in dataset_analysis.py.

(1) Please download  all three documents dataset under the same folder and set the correct path for the dataset and results figures saving path in the dataset_analysis.py file just like below:

    dataset_path = '/your/dataset/folder/path/' AND results_save_path = '/your/results/figures/saving/path/'. 
    
    #The specific Tokenization and stemming AND Sentence segmentation results will be shown in figures form under the results_save_path folder. The POS Tagging results will be printed in your command window.

(2) You can try all three different datasets to test our analysis functions by changing the dataset index, see the comments part in the "Dataset selection" code in dataset_analysis.py for details.

(3) The specific three datasets we used could be downloaded from google drive with the link:https://drive.google.com/drive/folders/1Mq5Gbo-DnywedCzsFMhLocrNTB3ILg9B?usp=drive_link or the original link is attatched:
    

    a. **British Airways Reviews**: https://www.kaggle.com/datasets/lapodini/british-airway-reviews
    
    b. **Healthcare NLP: LLMs, Transformers, Datasets**: https://www.kaggle.com/datasets/jpmiller/layoutlm/data
    
    c. **ARXIV Paper Abstruct**: https://www.kaggle.com/datasets/neelshah18/arxivdataset



# SearchEngine

This folder contains three packages, the indexDB, researchTrendExplorer and searchApplication package. 

### indexDB package

ParseDB.java is mainly used to parse xml and build index data. 
SpecificAnalyzer.java is mainly used to process the content in different xml tags.
SearchEngine.java is used for user input and show search results.

### researchTrendExplorer package

ResearchTrendExplorer.java is used to get the certain type of conference data, and save the filtered data into result.csv files for further processing.

### searchApplication package

ApplicationSearchEngine.java is a Java GUI application shows different search interface and results.


### Search Function

This search engine is capable of accommodating term-based searches, encompassing both single terms and phrases. It supports Strict Term Search, Flexible Term Search, Specific Field Search, and Advanced Field Search. It will return the top 10 similar snippets with search queries highlighted in descending ranking scores. 

**Strict Term Search**: Treat the search query as a unified entity without segmentation and search in all fields. 

**Flexible Term Search**: Segment the search query and match the words with a distance of less than 10 characters in all fields. 

**Specific Field Search**: Segment the search query and match the words with a distance of less than 10 characters in a single field. 

**Advanced Field Search**: Users need to specify all fields. The entered phrase will be segmented and matched to the words with a distance of less than 10 characters in a single field.



Please download all files in the folder SearchEngine and set the correct xmpath for your dataset in SearchEngine/src/indexDB/SearchEngine.java: 
    

    xmlPath = '/your/dataset/folder/path/';
    indexDir = '/your/index/files/saving/path/';

The dataset we use is "https://dblp.org/faq/How+can+I+download+the+whole+dblp+dataset.html". You can download other datasets to build your custom implementation. 


You can find Lucene at https://lucene.apache.org/
You can find Apache Commons CLI at https://commons.apache.org/proper/commons-cli/

### Application

The application of the search engine with a simple interface is in the folder SearchEngine/src/searchApplication. 
Users can choose four types of searching by clicking the corresponding button and typing in the search bar. The search results will be shown in the text area in descending order containing the ranking number, the document ID, the score, and the snippets with the search query highlighted in red.


Please download all files in the folder SearchEngine and set the correct xmpath for your dataset in SearchEngine/src/searchApplication/ApplicationSearchEngine.java: 
    

    xmlPath = '/your/dataset/folder/path/';
    indexDir = '/your/index/files/saving/path/';

The dataset we use is "https://dblp.org/faq/How+can+I+download+the+whole+dblp+dataset.html". You can download other datasets to build your custom implementation.



# ResearchTrendExplorer

The codes are in research_trend.py, we support four types of parsing sentences. We need to install some python libs.

```bash
pip install rake-nltk
pip install yake
pip install -U spacy
pip install pytextrank
python -m spacy download en_core_web_sm
```

In this task, we choose SIGIR conference data, the filtered data saved in results.csv are produced by the SearchEngine Project.

We can use command to get the certain year trend.

```bash
python research_trend.py spacy 1990
```

the first arg is parse type, the second arg is year.



