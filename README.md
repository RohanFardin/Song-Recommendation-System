# Song-Recommendation-System
**Description:** This project builds a song recommendation system using Python and machine learning techniques, specifically leveraging TF-IDF and cosine similarity for song recommendations. The code includes comprehensive data preprocessing and feature engineering steps to prepare the dataset for analysis. It focuses on enhancing the data quality and preparing it to be used in recommendation models.

**Feature:**

**Data Preprocessing: Cleaning and preparing the dataset by removing unnecessary columns, handling missing values, and formatting features.**
1. Cleaning & Handling unnecessary values: Unnecessary columns ('views', 'language_cld3', 'language_ft') are dropped to streamline the dataset. Rows with missing data are removed using the .drop() function, specifying the axis=0 to operate on rows and inplace=True to modify the DataFrame directly.
2. clean_feature(value): i) Uses re.sub(r'[{}()"]', '', value) to remove extra characters like {}, (), and quotation marks ("), which are not useful for analysis.
                         ii) Applies .encode('utf-8').decode('utf-8', errors='ignore') to clean up any text with unusual or invalid characters by encoding and decoding it to UTF-8.
                         iii) This ensures that only valid text data is retained. Splits the cleaned string into a list using commas, strips spaces from each word, and removes internal spaces for uniform representation.
This function is specifically applied to the features and artist. droppping non-English rows.
3. modify(text): i) Expands contractions (e.g., "don't" to "do not") using the contractions library.
                 ii) Noise Removal: Replaces newline characters and removes text within square brackets (e.g., [intro]).
                 iii) Tokenization: Splits the text into individual words (tokens).
                 iv) Stopword Removal: Removes common English stopwords (e.g., "the," "and", "or") to focus on meaningful words
                 v) Punctuation Removal: Filters out punctuation to retain only words.
                 vi) Stemming: Reduces words to their base or root form using the Porter Stemmer (e.g., "running" to "run").
This function is used for lyrics column.
5. convert_to_list(value): i) Uses ast.literal_eval to evaluate the string and convert it into a list.
                           ii) If the conversion fails (e.g., due to malformed input), it safely returns an empty list instead of raising an exception.
This function is used for feature, artist & modify_lyrics

**Feature Engineering: Transformming raw data into features that can be used to train machine learning models.**
 1. clean the 'lyrics' column by tokenizing, stemming, and removing stop words. This created a new, clean column called "modify_lyrics". [3(iii,iv,v,vi)]
 2. created a new column, "combined" by concatenating the following text columns: "artist" and "features". 
 3. created a new column, "combined_all" by concatenating the following text columns: "tag", "lyrics" and "combined". 

 **Recommendation System:**
 1. Vectorization: TfidfVectorizer is used to convert text data into numerical representations.
                  max_features=7000 limits the vocabulary size to the top 7000 most frequent words.
                  stop_words='english' removes common English stop words.
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


   
