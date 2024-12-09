import streamlit as st
import pickle
import pandas as pd

songs_dict = pickle.load(open('songs_dict.pkl', 'rb'))

df = pd.DataFrame(songs_dict)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


vectorizer = TfidfVectorizer(max_features= 7000,stop_words='english')
vector = vectorizer.fit_transform(df['combined_all'])

def recommend(song):
    if song not in df['title'].values:
        print(f"'{song}' is not found in the dataset.")
        return

    try:
        song_index = df[df['title'] == song].index[0]
        distances = cosine_similarity(vector[song_index].reshape(1, -1), vector)[0]
        songs_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[:21]
        print(f"Recommendations for '{song}':")

        for i in songs_list:
            if i[0] != song_index:
                print(df.iloc[i[0]].title)
    except Exception as e:
        print("An error occurred:", e)



st.title('Movie Recommender System')

option = st.selectbox('Choose a Song', df['title'].values)

if st.button("Recommend"):
    if option:
        recommendations = recommend(option)
        st.write("### Recommended Songs:")
        for song in recommendations:
            st.write(f"- {song}")
