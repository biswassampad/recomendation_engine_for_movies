from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
text = ["London Paris London","Paris Paris London"]
cv = CountVectorizer()

#to count the matrix 
count_matrix = cv.fit_transform(text)
#printing to array
#print (count_matrix.toarray())

similarity_score = cosine_similarity(count_matrix)

print(similarity_score)