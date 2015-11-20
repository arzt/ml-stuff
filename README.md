# Movie recommendation using collaborative filtering
Example code for movie recommendation using the MovieLens (http://grouplens.org/datasets/movielens/) dataset. 

The program loads the publicly available movie rating data from MovieLens and creates a rating matrix containing a movie rating 1 < r < 5 for each user-movie-combination for which a rating exists. This matrix is subsequently decomposed into two lower-rank (rank=10) matrices containing user and move feature vectors, respectively. These features are learned from the existing ratings allowing for predicting the rating of previously unrated movies for each user. 

This implementation uses an iterative variant of a matrix factorization and the classic ALS algorithm. The former performs a gradient descent to minimize the model loss function. Alternativ to the own ALS implementation the Spark (http://spark.apache.org/) implementation can be used, as it is much faster. Spark is also used for data preprocessing.

The focus of this example is not speed but demonstration of simple numerical computation using the linear algebra Scala library Breeze as well as Apache Spark.

To run the code install SBT from http://www.scala-sbt.org/ and a Java 8 SDK, locally clone this git repository, change into the project folder and run:

    sbt run

And wait for command prompt. Run the Scala tests with:

    sbt test

The execution will take a while, presenting some output of the optimization followed by user specific movie recommendations.

The code might be constantly modified over the next days and weeks.

If you have questions or experience problems running the code feel free to contact me.

The code is tested/developed under Ubuntu 15.10 with installed Open JDK 8.

#TODOs
* add more Scala tests and comments
* more user interactivity (e.g. set lambda, and rank for factorization) 
* try to speed um own ALS implementation
