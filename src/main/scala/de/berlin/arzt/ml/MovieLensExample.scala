package de.berlin.arzt.ml

import java.nio.file.Paths

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Codec
import scala.io.Source.fromURL
import scala.util.Try

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

object MovieLensExample {

  val pre = "http://files.grouplens.org/datasets/movielens/ml-100k"
  val uItem = s"$pre/u.item"
  val uData = s"$pre/u.data"
  val modelPath = Paths.get(s"./model.bin")

  def main(args: Array[String]) = {
    val master = Try(args(0)).toOption
    implicit val context = sparkContext(master)
    implicit val codec = Codec.ISO8859
    val lines = getLinesFromUrl(uData)
    val ratings = movieLensRatings(lines)
    //the data size can be reduced here
    //.filter { rating => rating.product <= 200 && rating.user <= 200 }

    val seen = ratedMovies(ratings)
    val uItemLines = getLinesFromUrl(uItem)
    val idxToName = movieNames(uItemLines)

    val n = ratings.map(_.user).max + 1
    val m = ratings.map(_.product).max + 1

    val rank = 10
    val λ = 0.01
    val ε = 0.0005
    val (y, rated, unrated) = createMatrices(n, m, ratings)

    //use ALS implementation from apache spark (much faster)
    //val (row, col) = trainSparkAlsModel(n, m, ratings, rank, λ)

    val (row, col) = BreezeAls.trainModel(y, rated, unrated, rank, ε, λ)

    /* String representations of original and predicted rating matrices
    val r2 = DenseMatrix.zeros[Boolean](rated.rows, rated.cols)
    val original  = ratingMatToString(y, unrated)
    val predicted = ratingMatToString(row*col, r2)
    * */

    val modelToSave = MovieModel(
      userFeatures = row,
      movieFeatures = col,
      seenMovies = seen.collectAsMap().toMap,
      idx2Name = idxToName.collectAsMap().toMap
    )

    //just to simulate production
    saveModel(modelPath, modelToSave)
    context.stop()
    val model = loadModel(modelPath)
    val results = recommendMovies(
      model,
      users = 20 to 30,
      limit = 20
    )
    printResults(results)
  }

  def printResults(results: Iterable[Result]) {
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

  def trainSparkAlsModel(
    users: Int,
    movies: Int,
    ratings: RDD[Rating],
    rank: Int,
    λ: Double
  )(implicit context: SparkContext) = {
    println("Starting training.")

    val als = new ALS()
      .setSeed(8)
      .setRank(rank)
      .setIterations(20)
      .setLambda(λ)
      .setNonnegative(true)

    val model = als.run(ratings)
    val movieFeat = model.productFeatures
    val userFeat = model.userFeatures
    val userFeatures = featuresToMatrix(rank, users, userFeat).t
    val movieFeatures = featuresToMatrix(rank, movies, movieFeat)

    (userFeatures, movieFeatures)
  }


  def createMatrices(
    rows: Int,
    cols: Int,
    ratings: RDD[Rating]
  )(implicit context: SparkContext) = {

    import DenseMatrix.{zeros, ones}
    val R = zeros[Boolean](rows, cols)
    val Y = zeros[Double](rows, cols)
    val rats: Array[Rating] = ratings.collect()
    rats.foreach {
      case Rating(i, j, y) =>
        R(i, j) = true
        Y(i, j) = y
    }
    val N = R.map(!_)
    (Y, R, N)
  }

  def recommendMovies(model: MovieModel, users: Iterable[Int], limit: Int) = {
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
                val name = idx2Name.getOrElse(idx, "")
                RankedMovie(name, score)
            }
            Result(user, recommendations)
        }
    }
  }

  def movieNames(lines: RDD[String])(implicit context: SparkContext) = {
    println("Reading movie names.")
    lines.map {
      case line =>
        line.split("[|]").take(2)
    }.collect {
      case Array(id, name) => (id.toInt, name)
    }
  }

  def getLinesFromUrl(url: String)(implicit context: SparkContext, codec: Codec) =
    context.parallelize(fromURL(url)(codec).getLines().toVector)

  def movieLensRatings(lines: RDD[String])(implicit context: SparkContext) = {
    println("Reading movie ratings.")
    lines.map(_.split( """\s""").take(3)).collect {
      case Array(user, movie, rating) =>
        new Rating(
          user = user.toInt - 1,
          product = movie.toInt - 1,
          rating = rating.toDouble
        )
    }
  }

  def ratedMovies(ratings: RDD[Rating]) =
    ratings.map {
      case Rating(user, movie, _) =>
        user -> movie
    }.groupByKey().map {
      case (id, movies) =>
        (id, movies.toSet)
    }


  def sparkContext(url: Option[String]) = {
    println("Setting up spark context.")
    val conf = new SparkConf()
      .setAppName("MovieLens Recommendation Example")
    url.foreach(conf.setMaster)
    if (Try(conf.get("spark.master")).isFailure) {
      conf.setMaster("local")
    }
    for (master <- Try(conf.get("spark.master"))) {
      println(s"Master: $master")
    }
    val context = new SparkContext(conf)
    context.setCheckpointDir("/tmp")
    context
  }
}