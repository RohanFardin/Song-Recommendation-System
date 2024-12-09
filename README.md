# Song-Recommendation-System
**Description:** This project builds a song recommendation system using Python and machine learning techniques, specifically leveraging TF-IDF and cosine similarity for song recommendations. The code includes comprehensive data preprocessing and feature engineering steps to prepare the dataset for analysis. It focuses on enhancing the data quality and preparing it to be used in recommendation models.


**Feature:**
**Data Preprocessing: Cleaning and preparing the dataset by removing unnecessary columns, handling missing values, and formatting features.**
1. Cleaning & Handling unnecessary values: Unnecessary columns ('views', 'language_cld3', 'language_ft') are dropped to streamline the dataset. Rows with missing data are removed using the .drop() function, specifying the axis=0 to operate on rows and inplace=True to modify the DataFrame directly.
2. clean_feature(value): i) Uses re.sub(r'[{}()"]', '', value) to remove extra characters like {}, (), and quotation marks ("), which are not useful for analysis.
                         ii) Applies .encode('utf-8').decode('utf-8', errors='ignore') to clean up any text with unusual or invalid characters by encoding and decoding it to UTF-8.
                         iii) This ensures that only valid text data is retained. Splits the cleaned string into a list using commas, strips spaces from each word, and removes internal spaces for uniform representation.
This function is specifically applied to the features and artist. droppping non-English rows.
3. modify(text):<br> i) Expands contractions (e.g., "don't" to "do not") using the contractions library.<br>
                 ii) Noise Removal: Replaces newline characters and removes text within square brackets (e.g., [intro]).<br>
                 iii) Tokenization: Splits the text into individual words (tokens).<br>
                 iv) Stopword Removal: Removes common English stopwords (e.g., "the," "and", "or") to focus on meaningful words.<br>
                 v) Punctuation Removal: Filters out punctuation to retain only words.<br>
                 vi) Stemming: Reduce words to their base or root form using the Porter Stemmer (e.g., "running" to "run").
This function is used for lyrics column.
5. convert_to_list(value):<br> i) Uses ast.literal_eval to evaluate the string and convert it into a list.<br>
                           ii) If the conversion fails (e.g., due to malformed input), it safely returns an empty list instead of raising an exception.
This function is used for feature, artist & modify_lyrics

**Feature Engineering: Transformming raw data into features that can be used to train machine learning models.**
 1. Clean the 'lyrics' column by tokenizing, stemming, and removing stop words. This created a new, clean column called "modify_lyrics". [3(iii,iv,v,vi)]
 2. Created a new column, "combined" by concatenating the following text column: "artist" and "features". 
 3. Created a new column, "combined_all" by concatenating the following text columns: "tag", "lyrics" and "combined". 

 **Recommendation System:**
 1. Vectorization:<br> TfidfVectorizer is used to convert text data into numerical representations.<br>
                  max_features=7000 limits the vocabulary size to the top 7000 most frequent words.<br>
                  stop_words='english' removes common English stop words.<br>
                  The vector variable stores the TF-IDF matrix.
2. Recommendation:
                  **song_index = df[df['title'] == song].index[0]:
                  Identifies the index of the input song from the df DataFrame by matching the song title.
                  Calculate Cosine Similarity: 
                  distances = cosine_similarity(vector[song_index].reshape(1, -1), vector)[0]:
                  Computes the cosine similarity between the feature vector of the input song (vector[song_index]) and all other songs in the dataset.
                  The result is a list of similarity scores where each score indicates how similar a song is to the input song.
                  Sort and Select the Top 10 Recommendations:
                  songs_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[:11]:
                  Enumerates the distances list and sorts it in descending order based on the similarity scores.
                  Selects the top 10 songs (along with their indices) that are most similar to the input song, including the input song itself.



 **Dataset Details**
The main dataset is collected from kaggle. This dataset contain information as recent as 2022 scraped from Genius, a place where people can upload and annotate songs, poems and even books (but mostly songs). It builds upon the 5 Million Song Lyrics Dataset by using models to identify the native language of each entry. here is the given link -
https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information

I chunked the files into 600K+ data. the dataset file and pickle file is given below -
https://drive.google.com/drive/folders/1F2ohOJ8C9gn9ajW2CrG8pfEm1N-9_U9l?usp=sharing


**Project Workflow**
Provide a step-by-step breakdown of the project's workflow:
1. Load the dataset.
2. Preprocess the data (column removal, cleaning, handling missing values).
3. Apply feature extraction techniques.
4. Build the recommendation system using similarity measures.
If you want to run it on Streamlit. Please follow these steps<br>
**i) Download the pickle file provided in the "Dataset Details" section above.**<br>
**ii) Open a terminal and navigate to the directory containing the pickle file and app.py. Then, type ``streamlit run app.py.``**<br>
**iii) Once the app launches, you will be able to select a song from the interface.**


**Liabraries Used**<br>
Pandas: Data manipulation and cleaning<br>
NumPy: Numerical operations and array manipulation<br>
NLTK: Natural language processing tasks (tokenization, stemming, stop word removal)<br>
Scikit-learn: Machine learning algorithms, including TF-IDF vectorization and cosine similarity<br>
AST: Safe evaluation of Python expressions<br>
Contractions: Expanding contractions in text<br>
re: Regular expressions for text pattern matching and replacement







   
