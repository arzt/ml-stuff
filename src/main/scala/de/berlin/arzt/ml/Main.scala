package de.berlin.arzt.ml

import java.nio.file.Paths

import org.apache.spark.{SparkContext, SparkConf}

import scala.io.Codec
import scala.util.Try
import MovieLens._

/**
  * Created by realdocx on 18.11.15.
  */
object Main {
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
    //.filter { rating => rating.product <= 400 && rating.user <= 400 }

    val seen = ratedMovies(ratings)
    val uItemLines = getLinesFromUrl(uItem)
    val idxToName = movieNames(uItemLines)

    val n = ratings.map(_.user).max + 1
    val m = ratings.map(_.product).max + 1

    val rank = 10
    val λ = 0.01
    val ε = 0.00075
    val (y, rated, unrated) = createMatrices(n, m, ratings)

    //use ALS implementation from apache spark (much faster)
    //val (row, col) = trainSparkAlsModel(n, m, ratings, rank, λ)

    val (row, col) = Factorization.trainModel(y, rated, unrated, rank, ε, λ)
    //val (row, col) = Factorization.runAls(y, rated, rank, ε, λ)

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
