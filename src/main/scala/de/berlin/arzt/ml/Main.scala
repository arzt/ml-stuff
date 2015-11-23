package de.berlin.arzt.ml

import java.io.IOException
import java.nio.file.Paths

import org.apache.spark.{SparkContext, SparkConf}

import scala.io.Codec
import scala.util.{Try, Success}
import MovieLens._

/**
  * Created by realdocx on 18.11.15.
  */
object Main {
  val pre = "http://files.grouplens.org/datasets/movielens/ml-100k"
  val uItem = s"$pre/u.item"
  val uData = s"$pre/u.data"
  val modelPath = Paths.get(s"./model.bin")

  def computeModel()(implicit context: SparkContext) = {
    implicit val codec = Codec.ISO8859
    val lines = getLinesFromUrl(uData)
    val ratings = movieLensRatings(lines)
    val seen = ratedMovies(ratings)
    //the data size can be reduced here
    //.filter { rating => rating.product <= 400 && rating.user <= 400 }

    val uItemLines = getLinesFromUrl(uItem)
    val idxToName = movieNames(uItemLines)
    val n = ratings.map(_.user).max + 1

    val m = ratings.map(_.product).max + 1
    val normalizer = ratings.count().toDouble
    val rank = 10

    val λ = 0.01
    val ε = 0.001
    val (y, rated, unrated) = createMatrices(n, m, ratings)
    println(
      """Enter a number to choose the matrix factorization implementation used for collaborative filtering:
        |1: Gradient Descent (own implementation)
        |2: Alternating Least Squares (Spark implementation)
        |3: Alternating Least Squares (own implementation, slow)
      """.stripMargin
    )
    val (row, col) =
      readInt match {
        case 1 =>
          Factorization.trainModel(y, rated, unrated, rank, ε, λ, normalizer)
        case 2 =>
          trainSparkAlsModel(n, m, ratings, rank, λ)
        case 3 =>
          Factorization.runAls(y, rated, rank, ε, λ)
        case i => throw new IOException(s"Unsupported Option: $i")
      }
    MovieModel(
      userFeatures = row,
      movieFeatures = col,
      seenMovies = seen.collectAsMap().toMap,
      idx2Name = idxToName.collectAsMap().toMap
    )
  }

  def main(args: Array[String]) = {
    val master = Try(args(0)).toOption
    implicit val context = sparkContext(master)
    val model =
      if (modelPath.toFile.exists()) {
        println("Found an existing model. Do you want to reuse it (y/n)?")
        readLine() match {
          case "y" | "yes" =>
            loadModel(modelPath)
          case "n" | "no" =>
            saveModel(
              modelPath,
              model = computeModel()
            )
          case i =>
            throw new IOException(s"Unsupported input: '$i'")
        }
      } else {
        println("No existing model found. Creating new one.")
        saveModel(
          modelPath,
          model = computeModel()
        )
      }


    /* String representations of original and predicted rating matrices
    val r2 = DenseMatrix.zeros[Boolean](rated.rows, rated.cols)
    val original  = ratingMatToString(y, unrated)
    val predicted = ratingMatToString(row*col, r2)
    * */

    context.stop()
    recommendDialog(model)
  }


  def recommendDialog(model: MovieModel): Unit = {
    val maxId = model.userFeatures.cols - 1
    println(s"Enter a list of user id numbers ∊ [0, $maxId] to get recommendations (CTRL-C to quit):")
    Try {
      val a = readLine().split("[^0-9]+").map(_.toInt)
      val result = recommendMovies(model, a, limit = 20)
      printResults(result)
    }
    recommendDialog(model)
  }

  def sparkContext(url: Option[String]) = {
    println("Setting up spark context.")
    val conf = new SparkConf()
      .setAppName("MovieLens Recommendation Example")
    url.foreach(conf.setMaster)
    if (Try(conf.get("spark.master")).isFailure) {
      conf.setMaster("local[2]")
    }
    for (master <- Try(conf.get("spark.master"))) {
      println(s"Master: $master")
    }
    val context = new SparkContext(conf)
    context.setCheckpointDir("/tmp")
    context
  }
}
