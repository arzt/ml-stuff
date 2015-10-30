package de.arzt.berlin.ml

import java.nio.file.Paths

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

case class MovieModel(
  userFeatures: DenseMatrix[Double],
  movieFeatures: DenseMatrix[Double],
  seenMovies: Map[Int, Set[Int]],
  idx2Name: Map[Int, String]
)

case class Result(
  user: Int,
  recommendations: Seq[RankedMovie]
)

case class RankedMovie(
  name: String,
  rank: Double
)

case class SeenMovies(
  user: Int,
  movies: Set[Int]
)

case class Movie(
  idx: Int,
  name: String
)

/*
case class Rating(
  user: Int,
  product: Int,
  rating: Double
)
*/

object MovieLensExample {

  val folder = "/Users/sarzt/Downloads/ml-100k"
  val modelPath = Paths.get(s"$folder/model.bin")

  def main(args: Array[String]) = {
    /*
    */
    implicit val context = sparkContext()
    val (ratings, seen) = movieLensRatings(folder)
    val idxToName = movieNames(folder)
    val modelToSave = trainMovieLensModel(ratings, seen, idxToName)
    saveModel(modelPath, modelToSave)

    val model = loadModel(modelPath)
    val users = List(1, 4, 5, 9)
    val limit = 10
    val results = predictMovieLens(model, users, limit)

    printResults(results)
  }

  def printResults(results: Seq[Result]) {
    results.foreach {
      case Result(user, recommendations) =>
        println(s"Recommendation for user $user:")
        recommendations.foreach {
          case RankedMovie(name, rank) =>
            println(f"    Rank: $rank%.3f   Movie: $name")
        }
        println()
    }
  }

  def trainMovieLensModel(
    ratings: RDD[Rating],
    seen: RDD[SeenMovies],
    movie: RDD[Movie]
  )(implicit context: SparkContext) = {

    println("Starting training.")

    val users = ratings.map(_.user).max
    val movies = ratings.map(_.product).max

    val rank = 10

    val als = new ALS()
      .setSeed(8)
      .setRank(rank)
      .setIterations(10)
      .setLambda(0.1)
      .setNonnegative(true)

    val model = als.run(ratings)

    val movieFeat = model.productFeatures.collect()
    val userFeat = model.userFeatures.collect()

    val finalSeen = seen.map {
      case SeenMovies(user, seen) =>
        user -> seen
    }

    val idxToName = movie.map {
      case Movie(idx, name) =>
        (idx, name)
    }

    MovieModel(
      userFeatures = featuresToMatrix(rank, users, userFeat).t,
      movieFeatures = featuresToMatrix(rank, movies, movieFeat),
      seenMovies = finalSeen.collectAsMap().toMap,
      idx2Name = idxToName.collectAsMap().toMap
    )
  }

  def predictMovieLens(model: MovieModel, users: Seq[Int], limit: Int) = {
    println("Predicting Recommendations.")
    model match {
      case MovieModel(userFeats, movieFeats, seenMovies, idx2Name) =>
        users.map {
          case user =>
            val seen = seenMovies.getOrElse(user, Set())
            val userFeat = userFeats(user, ::)
            val ratings: DenseVector[Double] = movieFeats.t * userFeat.t
            val idAndScore: Map[Int, Double] = ratings.data.zipWithIndex.map(_.swap)(collection.breakOut)
            val unseen = idAndScore -- seen
            val sortedUnseen = unseen.toSeq.sortBy(_._2).reverse.take(limit)
            val recommendations = sortedUnseen.map {
              case (idx, score) =>
                RankedMovie(idx2Name(idx), score)
            }
            Result(user, recommendations)
        }
    }
  }

  def movieNames(folder: String)(implicit context: SparkContext) = {
    println("Reading movie names.")
    val pairs = context.textFile(folder + "/u.item").map {
      case line =>
        line.split("[|]").take(2)
    }
    pairs.collect {
      case Array(id, name) =>
        Movie(
          idx = id.toInt,
          name
        )
    }
  }

  def movieLensRatings(folder: String)(implicit context: SparkContext) = {
    println("Reading movie ratings.")
    val lines = context.textFile(folder + "/u.data").map {
      case line =>
        line.split( """\s""").take(3)
    }
    val ratings = lines.collect {
      case Array(user, movie, rating) =>
        Rating(
          user = user.toInt,
          product = movie.toInt,
          rating = rating.toDouble
        )
    }
    val rated = ratings.map {
      case Rating(user, movie, _) =>
        user -> movie
    }.groupByKey().map {
      case (id, movies) =>
        SeenMovies(
          user = id,
          movies = movies.toSet
        )
    }
    (ratings, rated)
  }

  def sparkContext() = {
    println("Setting up spark context.")
    val conf = new SparkConf()
      .setAppName("MovieLens Recommendation Example")
      .setMaster("local")
    new SparkContext(conf)
  }
}