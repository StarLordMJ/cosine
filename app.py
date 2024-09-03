from flask import Flask, request, jsonify
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def model_load(place_selec: str):
    df = pd.read_csv('Reviews.csv', encoding='ISO-8859-1')
    df1 = df.drop_duplicates(subset='Location_Name')

    features = ['Location_Name','Location','Location_Type','Text']
    def combine_features(row):
        return row['Location_Name']+" "+row['Location']+" "+row['Location_Type']+" "+row['Text']
    for feature in features:
        df1[feature] = df1[feature].fillna('')
    df1["combined_features"] = df1.apply(combine_features, axis=1)

    columns_to_keep = ['Location_Name', 'Location', 'Location_Type', 'Text', 'combined_features']
    df1 = df1[columns_to_keep]

    nltk.download('stopwords')
    from nltk.corpus import stopwords

    def clean(text):
        text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)
        punctuations = ['@','#','!','?','+','&','*','[',']','-','%','.',':','/','(',')',';','$','=','>','<','|','{','}','^']
        for p in punctuations:
            text = text.replace(p, f' {p} ')
        return text

    df1.iloc[4] = df1.iloc[4].apply(lambda s: clean(s))

    df1["combined_features"] = df1["combined_features"].str.lower().str.split()
    stop = stopwords.words('english')
    df1['combined_features'] = df1['combined_features'].apply(lambda x: [item for item in x if item not in stop])
    df1["combined_features"] = df1["combined_features"].str.join(" ")

    ind = []
    for i in range(76):
        ind.append(i)

    df1.insert(0, "index", ind)
    df1.set_index(['index'])

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df1['Location'].values.astype('U'))
    cosine_sim = cosine_similarity(count_matrix)

    features2 = ['Location_Name', 'Location']
    def combine_features2(row):
        return row['Location_Name']+" "+row['Location']
    for feature in features2:
        df1[feature] = df1[feature].fillna('')
    df1["place_names"] = df1.apply(combine_features2, axis=1)
    df1["place_names"] = df1["place_names"].str.lower()

    def get_index_from_title(place):
        for ind in df1['index']:
            mond = df1.iloc[ind]['place_names']
            if re.search(place, mond):
                return ind
            
    place_index = get_index_from_title(place_selec)

    similar_places = list(enumerate(cosine_sim[place_index]))

    sorted_similar_places = sorted(similar_places, key=lambda x: x[1], reverse=True)[1:]

    top_places = []
    i = 0
    for element in sorted_similar_places:
        top_places.append(df1.loc[df1['index'] == element[0], 'Location_Name'].values[0])
        i += 1
        if i > 2:
            break
    return top_places

@app.route('/handle_get', methods=['GET'])
def handle_get():
    if request.method == 'GET':
        place_selec = request.args.get('name')
        top_places = model_load(place_selec)
        
        # Print the results in the terminal
        print(f"Top 3 similar travel packages like {place_selec} are:")
        for place in top_places:
            print(f"Place: {place}")
        
        return jsonify({"similar_places": top_places})

@app.route('/handle_post', methods=['POST'])
def handle_post():
    if request.method == 'POST':
        place_selec = request.form['place']
        top_places = model_load(place_selec)
        
        # Print the results in the terminal
        print(f"Top 3 similar travel packages like {place_selec} are:")
        for place in top_places:
            print(f"Place: {place}")
        
        return jsonify({"similar_places": top_places})

if __name__ == '__main__':
    app.run()
