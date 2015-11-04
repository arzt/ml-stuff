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

/*
case class Rating(
  user: Int,
  product: Int,
  rating: Double
)
*/

object MovieLensExample {

  val modelPath = Paths.get(s"./model.bin")

  def main(args: Array[String]) = {
    /*
    */
    val master = Try(args(0)).toOption
    implicit val context = sparkContext(master)
    val pre = "http://files.grouplens.org/datasets/movielens/ml-100k"
    val udata = s"$pre/u.data"
    val uitem = s"$pre/u.item"
    implicit val codec = Codec.ISO8859
    val lines = getLinesFromUrl(udata)
    val ratings = movieLensRatings(lines)
    val seen = ratedMovies(ratings)
    val uitemLines = getLinesFromUrl(uitem)
    val idxToName = movieNames(uitemLines)
    val modelToSave = trainMovieLensModel(ratings, seen, idxToName)
    saveModel(modelPath, modelToSave)

    val model = loadModel(modelPath)
    val results = predictMovieLens(
      model,
      users = 50 to 60,
      limit = 10
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

  def predictMovieLens(model: MovieModel, users: Iterable[Int], limit: Int) = {
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

  def movieNames(items: RDD[String])(implicit context: SparkContext) = {
    println("Reading movie names.")
    val pairs = items.map {
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

  def getLinesFromUrl(url: String)(implicit context: SparkContext, codec: Codec) = context.parallelize(fromURL(url)(codec).getLines().toVector)

  def movieLensRatings(lines: RDD[String])(implicit context: SparkContext) = {
    println("Reading movie ratings.")
    lines.map(_.split( """\s""").take(3)).collect {
      case Array(user, movie, rating) =>
        new Rating(
          user = user.toInt,
          product = movie.toInt,
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
        SeenMovies(
          user = id,
          movies = movies.toSet
        )
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
    new SparkContext(conf)
  }
}