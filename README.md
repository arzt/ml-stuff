# Movie Recommendation using ALS algorithm
Example code for movie recommendation using the MovieLens (http://grouplens.org/datasets/movielens/) dataset. 

The program loads the publicly available movie rating data from MovieLens and creates a rating matrix containing a movie rating 1 < r < 5 for each user-movie-combination for which a rating exists. This matrix is subsequently decomposed into two lower-rank (rank=10) matrices containing user and move feature vectors, respectively. These features are learned from the existing ratings allowing for predicting the rating of previously unrated movies for each user. 

This implementation uses an iterative variant of the ALS algorithm performing a simple gradient descent to minimize the model loss function. (Although there is an explicit solution) Additionally, Apache Spark (http://spark.apache.org/) is used for data preprocessing. Alternatively, the sparks implementation of ALS can be used, by slightly modifying the source code.

The focus of this example is not efficiency but demonstration of simple numerical computation using the linear algebra Scala library Breeze as well as Apache Spark.

To run the code install SBT from http://www.scala-sbt.org/, checkout this git repository and change into the project folder. Then run the following command:

    sbt run

The execution will take a while, presenting some output of the optimization followed by user specific movie recommendations.

The code might be constantly modified over the next days and weeks.

If you have questions or experience problems running the code feel free to contact me.

The code is tested/developed under Ubuntu 15.10 with installed Open JDK 8.
