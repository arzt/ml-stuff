package de.berlin.arzt.ml

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD

import scala.io.Codec
import scala.io.Source.fromURL

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

object MovieLens {

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
    val userFeatures = featuresToMatrix(rank, users, userFeat)
    val movieFeatures = featuresToMatrix(rank, movies, movieFeat)

    (userFeatures, movieFeatures)
  }


  def createMatrices(
    rows: Int,
    cols: Int,
    ratings: RDD[Rating]
  )(implicit context: SparkContext) = {

    import DenseMatrix.zeros
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
            val userFeat = userFeats(::, user)
            val ratings: DenseVector[Double] = movieFeats.t * userFeat
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


}