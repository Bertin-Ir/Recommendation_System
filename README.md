# Recommendation_System

# Movie Recommendation Engine: From Matrix Factorization to Deep Learning

##  Project Overview
This project explores different approaches to building recommendation systems using the [MovieLens 100k](https://grouplens.org/datasets/movielens/) dataset. The goal is to move beyond generic "top-rated" lists to deliver truly personalized movie suggestions by predicting unobserved user ratings.

Such systems are central to modern platforms—powering content discovery on streaming services, improving user engagement, and reducing information overload by surfacing relevant items from large catalogs. By modeling user behavior and item similarity, this project demonstrates how recommendation algorithms can be applied to real-world scenarios where personalization directly impacts user experience and retention.


## Recommendation Strategies

### Collaborative Filtering
This approach is based on the assumption that **users who agreed in the past will likely agree again in the future.** By analyzing the sparse interaction matrix of users and items, the model identifies taste clusters to fill in the blanks of user preferences.

### Matrix Factorization
This technique deconstructs a user's taste into core "latent factors" (embeddings). By representing both users and movies in a shared mathematical space, we can estimate ratings based on the hidden connections between a user’s profile and a movie’s characteristics.

### Deep Hybrid Recommender
While linear models are efficient, they often miss the subtle, non-linear nuances of human behavior. Our **Deep Learning** architecture feeds embeddings into multiple Dense layers to:
* **Capture Complexity:** Learn non-linear interactions between users and films.
* **Integrate Metadata:** Seamlessly blend collaborative data with item attributes like release year and popularity.
* **Enhance Generalization:** Utilize Dropout layers to prevent overfitting on smaller datasets.

##  Key Findings & Insights
* **The Scalability Trade-off:** While Deep Learning offers more "reasoning" power, simple Matrix Factorization remains highly competitive on smaller datasets (like 100k rows) due to its lower risk of overfitting.
* **The Role of Metadata:** Adding content-based features helps bridge the "Cold Start" problem, though it requires careful regularization to ensure the model doesn't focus on noisy features.
* **Metric Analysis:** We utilize **MAE (Mean Absolute Error)** and **MRSE (Mean Root Squared Error)** to evaluate performance.

## Tech Stack
* **Core:** Python, NumPy, Pandas
* **Deep Learning:** Keras / TensorFlow
* **Preprocessing:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

---

