package de.arzt.berlin.ml

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.nio.file.{Files, Path, Paths}

import breeze.linalg.DenseMatrix.zeros
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random
import scala.util.Random.shuffle



object RecommendationExample {

  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName("Simple Application")
    conf.setMaster("local")
    conf.setSparkHome("/Users/sarzt/PredictionIO/vendors/spark-1.4.1")
    val sc = new SparkContext(conf)

    val seed = 2
    val ran = new Random(seed)
    val nUser = 7
    val nProduct = 7
    val rank = 3
    val iterations = 10
    val ratings = randomRatings(nUser, nProduct, 0.2, ran)
    val mat = ratingsToMatrix(nUser, nProduct, ratings)
    val lambda = 0.33
    val alpha = 0.01
    val als = new ALS()
    als.setSeed(seed)
    als.setImplicitPrefs(true)
    als.setRank(rank)
    als.setIterations(iterations)
    als.setLambda(lambda)
    als.setNonnegative(true)
    als.setAlpha(alpha)
    val model = als.run(sc.parallelize(ratings))

    val prodFeat = model.productFeatures.collect
    val userFeat = model.userFeatures.collect

    val cprod = prodFeat.length
    val cuser = userFeat.length
    printMat(mat, "u", "i")
    println("Counts: " +(cprod, cuser))
    val userMat = featuresToMatrix(rank, nUser, userFeat).t
    val prodMat = featuresToMatrix(rank, nProduct, prodFeat)
    val mat2: DenseMatrix[Double] = userMat * prodMat
    println("User:")
    printMat(userMat, "u", "r")
    println("Item:")
    printMat(prodMat, "r", "i")

    println()
    printMat(mat2, "u", "i")
    println()
    printMat(mat, "u", "i")

  }

  /*
  def normalizeArray(m: Array[Double]) {
    var length = 0.0
    var i = 0
    while (i < m.length) {
      length += m(i) * m(i)
      i += 1
    }
    length = math.sqrt(length)
    i = 0
    while (i < m.length) {
      m(i) /= length
      i += 1
    }
  }
  * */

  def printMat(m: DenseMatrix[Double], row: String, col: String): Unit = {
    val lines = for (i <- 0 until m.rows) yield {
      val line = (0 until m.cols).flatMap { j =>
        val x = (m(i, j) * 100).toInt
        f"$x%03d "
      }(collection.breakOut)
      s"$row$i: $line\n"
    }
    val header = (0 until m.cols).flatMap {
      case a =>
        f"$col$a%1d: "
    }(collection.breakOut)
    println("    " + header)
    println
    lines.foreach(println)
  }




  def randomRatings(nUser: Int, nItem: Int, p: Double, r: Random): IndexedSeq[Rating] = {
    val rat = Vector(1)
    val rows = new Array[Boolean](nUser)
    val cols = new Array[Boolean](nItem)
    val ratings = for (
      i <- 0 until nUser;
      j <- 0 until nItem
      if r.nextDouble() < p
    ) yield {
        val r = shuffle(rat).head
        rows(i) = true
        cols(j) = true
        Rating(i, j, r)
      }
    val allRows = rows.reduce(_ && _)
    val allCols = cols.reduce(_ && _)
    if (allRows && allCols)
      ratings
    else {
      randomRatings(nUser, nItem, p, r)
    }
  }

  def ratingsToMatrix(nUser: Int, nItems: Int, ratings: Seq[Rating]) = {
    val m = zeros[Double](nUser, nItems)
    ratings.foreach {
      case Rating(u, i, r) =>
        m(u, i) = r
    }
    m
  }

}
