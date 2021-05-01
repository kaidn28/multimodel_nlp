from functions import *
query_vectors, inferences, vectorizer = pickle.load(open('./saves/tfidf_query_data.pkl', 'rb'))
predict('ai cứ ở cạnh cái laptop thế nhỉ', query_vectors, inferences, vectorizer)

