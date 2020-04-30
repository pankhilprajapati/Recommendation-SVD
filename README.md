# Recommender package

This package performs the recommendation using user-item matrix and predicts the output
three main tye of recommendation 
- Knowlegde based
- Collaborative based 
- Content based
# Files
- Recommender
- Recommender_functions

# Installation 

```
cd Recommendation-SVD
python setup.py sdist
pip install dist/Recommender-1.0.tar.gz


>>> from Recommender import Recommender
>>> rec = Recommender()
>>> rec.fit(review_pth,movie_pth, latent_features=12,learning_rate,iters) // fit the model  provide the data files the 
>>> rec.predict(user,item) // prediction
>>> rec.make_recs(_id,_id_type,rec_num)//Recommendations
```

# About
```
This project was part of the Udacity's Data Science Nanodegree Program

```