import pickle
import numpy as np
import scipy

def similarity(s1, s2, criterial='cosine'):
  assert len(s1) == len(s2)
  s1_a = np.array(s1)
  s2_a = np.array(s2)
  if criterial =='norm':
    return 1/(np.linalg.norm(s1_a - s2_a, ord=2))
  if criterial == 'cosine':
    return 1-scipy.spatial.distance.cosine(s1_a, s2_a)

def predict(sentence, query_vectors, inferences, vectorizer, criterial = 'cosine'):
  s = vectorizer.transform([sentence]).toarray()[0]
  max_sim = 0
  closest = 0
  for (i,dp) in enumerate(query_vectors.toarray()):
    #print(i, dp)
    sim = similarity(s, dp, criterial)
    if max_sim < sim:
      max_sim = sim
      closest = inferences[i]
  return closest, max_sim