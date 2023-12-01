## library
import nltk
import json
import tarfile
import numpy as np
import pandas as pd
from pandas import json_normalize
from nltk import word_tokenize,pos_tag
from nltk.corpus import stopwords
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from matplotlib import pyplot as plt
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

## Dataset Build Function
def orignal_flight_text_dataset(txt_path):
    count=0
    reviewerID = []
    reviewHeader = []
    detailedReview = []
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            dicts= json.loads(line.strip())
            #print(dicts)
            for i in range(0,20):
                if 'id' in dicts[i]:
                    reviewerID.append(dicts[i]['id'])
                else:
                    reviewerID.append('NoID')

                if 'review_header' in dicts[i]:
                    reviewHeader.append(dicts[i]['review_header'])
                else:
                    reviewHeader.append('review_header')

                if 'detailed_review' in dicts[i]:
                    detailedReview.append(dicts[i]['detailed_review'])
                else:
                    detailedReview.append('detailed_review')
    return detailedReview

def orignal_healthcare_text_dataset(txt_path):
    count=0
    reviewerID = []
    reviewHeader = []
    detailedReview = []
    with open(txt_path, encoding='utf-8') as f:
        line_no = 0

        for line in f :
            if (line_no<=20):
                line_no = line_no +1
                dicts= json.loads(line.strip())
                print(dicts.keys())
                if 'abstract' in dicts.keys():
                    print("1")
                    detailedReview.append(dicts['abstract'])
                else:
                    print("0")
                    detailedReview.append('abstract')
            else:
                break

    return detailedReview

## Collect the frequency of each token
def Collection_Frequency(detailedReview):
    # Collection Frequency
    word_token = {}
    eT_join = []
    token = []

    for i in range(len(detailedReview)):
        eT_join = detailedReview[i]
        # eT_join the text in the ith reviewText
        # print(eT_join)
        token = eT_join.split(' ')

        for tk in token:
            #print('============')
            #print(tk)
            if tk not in '.':
                #print(tk)
                if tk not in word_token:
                    word_token[tk] = 1
                else:
                    word_token[tk] += 1

    print(word_token)
    return word_token

## Combine the stop words form each library
def combined_stop_words():
    from nltk.corpus import stopwords
    from spacy.lang.en.stop_words import STOP_WORDS
    from nltk.stem import PorterStemmer,SnowballStemmer

    stopwords_nltk = stopwords.words('english')
    stopwords_spacy = list(STOP_WORDS)
    stopwords_spacy.append('\n')
    stopwords = list(set(stopwords_spacy+stopwords_nltk))
    porterstem=PorterStemmer()
    snowballstem=SnowballStemmer('english')
    porterstem_tk = [
        [porterstem.stem(token) for token in stopwords]
    ]
    porterstem_tk = [item for sublist in porterstem_tk for item in sublist]
    porterstem_tk = list(set(stopwords+porterstem_tk))

    snowball_tk = [
        [snowballstem.stem(token) for token in stopwords]
    ]
    snowball_tk = [item for sublist in snowball_tk for item in sublist]
    snowball_tk = list(set(stopwords+snowball_tk))
    # print("sw nltk: ", len(stopwords_nltk))
    # print("sw spacy: ", len(stopwords_spacy))
    # print("combined: ", len(stopwords))
    return stopwords,porterstem_tk,snowball_tk

## Remove the punctions 
def get_cleantokens(tokens,stopwords):
    clean_tokens = []
    punctuations = '!"#$%&\'’()*+,-/:;<=>?@[\]^_`{|}~©.'
    for token in tokens:
        if token.lower() not in stopwords and token.lower() not in punctuations:
            #and (token not in string.punctuation):
            clean_tokens.append(token)
    return clean_tokens

## Collect the non-stop words tokens 
def clean_word_token(word_token,stopwords):
    FINAL_clearn_token=get_cleantokens(word_token,stopwords)
    selected_word_token = {word: freq for word, freq in word_token.items() if word in FINAL_clearn_token}
    # print the words frequency
    for word, freq in selected_word_token.items():
        print(f"Word: {word}, Frequency: {freq}")

    df = pd.DataFrame.from_dict(selected_word_token,orient='index').reset_index().rename(columns={'index':'token',0:'count'}).sort_values(by='count',ascending=False)
    #plt.bar(x=df['token'],height=df['count'])
    df[df['count']!=1]
    return selected_word_token,df

## Plot the token frequency
def token_frequency_plot(df_or, figure_name,results_save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create a new figure
    plt.close()
    plt.figure()

    # Set a larger default font size
    sns.set_context("talk")  # 'talk' context will generally have larger fonts

    # Barplot
    fig1 = sns.barplot(df_or.head(20), x="token", y="count")
    sns.set(rc={'figure.figsize': (11.7, 10)})

    # Increase the rotation and font size of x-tick labels
    fig1.set_xticklabels(fig1.get_xticklabels(), rotation=45, fontsize=16)

    # Increase the font size of y-tick labels
    fig1.set_yticklabels(fig1.get_yticks(), size=16)

    # Annotate bars with larger font
    for p in fig1.patches:
        fig1.annotate(f'{int(p.get_height())}',
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center',
                      fontsize=16,  # Increased fontsize
                      color='black',
                      xytext=(0, 5),
                      textcoords='offset points')

    # Set title and y-label with larger font sizes
    fig1.set_title(figure_name, fontsize=18)
    fig1.set_ylabel('token frequency', fontsize=15)
    
    final_path = results_save_path + figure_name + '.png'
    # Save the figure to the desired location
    plt.tight_layout()
    plt.savefig(final_path)  # Specify your path and filename

## Get the token and its frequency without steeming function
def tokens_wo_steeming(detailedReview,stepwords):
    word_token = Collection_Frequency(detailedReview)
    selected_word_token = clean_word_token(word_token,stepwords)
    return selected_word_token

## Get the texts daya after orignal steeming function (PorterStemmer)  
def orignal_stemming(detailedReview):
    import nltk
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    processed_texts = []
    for text in detailedReview:
        words = nltk.word_tokenize(text)
        stemmed_words = [stemmer.stem(word) for word in words]
        processed_texts.append(" ".join(stemmed_words))
    for i, processed_text in enumerate(processed_texts):
        print(f"Processed Text {i + 1}:", processed_text)
    return processed_texts

## Get the texts data after the new steeming function (SnowballStemmer)
def stemming_Snowball(detailedReview):
    from nltk.stem.snowball import EnglishStemmer
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("english")
    processed_texts = []
    for text in detailedReview:
        words = nltk.word_tokenize(text)
        stemmed_words = [stemmer.stem(word) for word in words]
        processed_texts.append(" ".join(stemmed_words))
    for i, processed_text in enumerate(processed_texts):
        print(f"Processed Text {i + 1}:", processed_text)
    return processed_texts

## Steeming the tokens
def stemming_token(processed_texts):
    word_token_1 = {}
    eT_join_1 = []
    token_1 = []
    for i, processed_text in enumerate(processed_texts):
        eT_join_1 = processed_text
        token_1 = eT_join_1.split(' ')
        print(token_1)
        for tk_1 in token_1:
            print(tk_1)
            if tk_1 not in ['.','']:
                if tk_1 not in word_token_1 :
                    word_token_1[tk_1] = 1
                else:
                    word_token_1[tk_1] += 1
    return word_token_1

## Get the token and its frequency after the two steeming operations
def tokens_after_steeming(detailedReview,stopwords,stopwordsSb):
    stemming_texts = stemming_Snowball(detailedReview)
    stemming_tokens = stemming_token(stemming_texts)
    selected_word_token,_ = clean_word_token(stemming_tokens,stopwords)

    stemming_texts_1 = stemming_Snowball(detailedReview)
    stemming_tokens_1 = stemming_token(stemming_texts_1)
    selected_word_token_1,_ = clean_word_token(stemming_tokens_1,stopwordsSb)
    return selected_word_token,selected_word_token_1

## Get the sentence length distribution information
def sentence_length(detailedReview, titleName,results_save_path):
    import nltk
    from nltk.tokenize import sent_tokenize
    import matplotlib.pyplot as plt

    domain1_sentence_lengths = []
    paper_sentence = []
    plt.figure()
    for i in range(len(detailedReview)):
        text = detailedReview[i]
        sentences = sent_tokenize(text)
        print(sentences)
        sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
        paper_sentence.extend(sentences)
        domain1_sentence_lengths.extend(sentence_lengths)

    plt.hist(domain1_sentence_lengths, bins=range(1, max(domain1_sentence_lengths) + 2), alpha=0.5, label=titleName)
    plt.xlabel('Sentence Length (Number of Words)', fontsize=16)
    plt.ylabel('Number of Sentences', fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.title(titleName + ' DataSet Sentence Length Distribution', fontsize=16)

    # Specify the path and filename where you want to save the image
    final_path =  results_save_path + titleName + ' DataSet Sentence Length Distribution' + '.png'
    # Save the figure to the desired location
    plt.savefig(final_path)

    # Display the plot
    plt.show()

    return domain1_sentence_lengths, paper_sentence

## Do POS tagging
def pos(review_sentence):
    import nltk
    nltk.download('averaged_perceptron_tagger')
    from nltk import sent_tokenize
    from nltk import pos_tag
    # Apply POS tagging to the selected sentences
    domain1_pos = [pos_tag(nltk.word_tokenize(review_sentence)) ]
    return domain1_pos


if __name__ == '__main__':
    stopwords,porterstem_tk,snowball_tk = combined_stop_words()

    ## Make sure you have set the correct path to the dataset (dataset_path) and the results save path (results_save_path)
    file1 = 'british_airways_reviews.json'
    file2 = 'medquad.csv'
    file3 = 'arxiv-metadata-oai-snapshot.json'
    dataset_path = '/content/drive/MyDrive/textproject/TextProject/'
    results_save_path = '/content/drive/MyDrive/textproject/TextProject/test_result/'

    ################################################### Dataset selection  ###################################################
    ## First Test the file 1, the flight reviews dataset,we mainly choose the first 20 documents to build our dataset
    text_data_path = dataset_path + file1
    detailedReview = orignal_flight_text_dataset(text_data_path)
    
    # ##  You can also change file1 to file2 to test the results on healthcare datesets as below:
    # df_text=pd.read_csv(f'/content/drive/MyDrive/textproject/TextProject/{file2}')
    # df_text=df_text.drop_duplicates(subset=['answer']).reset_index(drop=True)
    # # We could also do random selection (rather than choose the first 20 documents )to build this tiny dataset and the example index_lst is one time randon chosen result
    # np.random.randint(0, high=15817, size=20, dtype='l')
    # index_lst = [ 6292,  8157,  9774, 14087,  4131,  4492,  1326, 12976,  6125,
    #         2210, 13942,   264,  6555, 11537,  7963, 11596,  5461, 15390,
    #         1171,  5009]
    # df_text1 = df_text.loc[df_text.index.isin(index_lst)]
    # detailedReview=df_text1['answer'].tolist()

    # ## or change file1 to file3 to test the results on paper abstarct datesets as below:
    # text_data_path = f'/content/drive/MyDrive/textproject/TextProject/{file3}'
    # detailedReview = orignal_healthcare_text_dataset(text_data_path)


    ################################################### Tokenization and Stemming  ###################################################
    clean_word_token_wo_steeming,_ = tokens_wo_steeming(detailedReview,stopwords)
    df_or = pd.DataFrame.from_dict(clean_word_token_wo_steeming,orient='index').reset_index().rename(columns={'index':'token',0:'count'}).sort_values(by='count',ascending=False)
    df_or[df_or['count']!=1]
    orignal_steeming_clean_tokens,Snowball_steeming_clean_tokens =  tokens_after_steeming(detailedReview,stopwords,snowball_tk)
    df = pd.DataFrame.from_dict(orignal_steeming_clean_tokens,orient='index').reset_index().rename(columns={'index':'token',0:'count'}).sort_values(by='count',ascending=False)
    df[df['count']!=1]

    df_1 = pd.DataFrame.from_dict(Snowball_steeming_clean_tokens,orient='index').reset_index().rename(columns={'index':'token',0:'count'}).sort_values(by='count',ascending=False)
    df_1[df_1['count']!=1]
 
    
    token_frequency_plot(df_or,"Function Test",results_save_path)
    token_frequency_plot(df_or,"Airways Review Dataset without stemming",results_save_path)
    token_frequency_plot(df,"Airways Review Dataset Token stemming",results_save_path)
    token_frequency_plot(df_1,"Airways Review Dataset Stopwords stemming",results_save_path)

    ################################################### Sentence Length Distribution ###################################################
    domain1_sentence_lengths,paper_sentence = sentence_length(detailedReview,'Airways Review',results_save_path)


    # token_frequency_plot(df_or,"Function Test",results_save_path)
    # token_frequency_plot(df_or,"Healthcare Dataset without stemming",results_save_path)
    # token_frequency_plot(df,"Healthcare Dataset Token stemming",results_save_path)
    # token_frequency_plot(df_1,"Healthcare Dataset Stopwords stemming",results_save_path)
    # domain1_sentence_lengths,paper_sentence = sentence_length(detailedReview,'Healthcare',results_save_path)


    # token_frequency_plot(df_or,"Function Test",results_save_path)
    # token_frequency_plot(df_or,"Arxiv Paper Abstract Dataset without stemming",results_save_path)
    # token_frequency_plot(df,"Arxiv Paper Abstract Dataset Token stemming",results_save_path)
    # token_frequency_plot(df_1,"Arxiv Paper Abstract Dataset Stopwords stemming",results_save_path)
    # domain1_sentence_lengths,paper_sentence = sentence_length(detailedReview,'Arxiv Paper Abstract',results_save_path)


    ################################################### POS Tagging ###################################################
    randomNumber=np.random.randint(0, high=len(paper_sentence)-1, size=3, dtype='l')
    for i in randomNumber:
        domain1_pos=pos(paper_sentence[i])
        print(f'{str(i)}th Sentence:',domain1_pos)


    


