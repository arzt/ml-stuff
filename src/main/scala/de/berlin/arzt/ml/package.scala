package de.berlin.arzt

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.nio.file.{Files, Path}
import java.util.zip.{GZIPInputStream, GZIPOutputStream}

import breeze.linalg.DenseMatrix._
import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions.Gaussian
import org.apache.spark.rdd.RDD

package object ml {

  type Matrix = DenseMatrix[Double]
  type Vec = DenseVector[Double]

  def normal(rows: Int, cols: Int, variance: Double = 1) = rand[Double](rows, cols, Gaussian(0, variance))

  def featuresToMatrix(
    rows: Int, cols: Int,
    features: RDD[(Int, Array[Double])]
  ) = {
    val zeros = new Array[Double](rows)
    val indices = features.map(_._1).distinct().collect.toSet
    val allIndices = (1 to cols).toSet
    val missing = allIndices -- indices
    val missingFeat = features.sparkContext.parallelize(missing.toSeq).map {
      case i => (i, zeros)
    }
    val data =
      features.union(missingFeat).sortBy {
        case (i, _) => i
      } flatMap {
        case (_, feat) =>
          assert(feat.length == rows)
          feat
      }
    new DenseMatrix[Double](rows, cols, data.collect)
  }

  def loadModel(path: Path) = {
    println("Loading model.")
    val in = new ObjectInputStream(new GZIPInputStream(Files.newInputStream(path)))
    val model = in.readObject().asInstanceOf[MovieModel]
    in.close()
    model
  }

  def saveModel(path: Path, model: MovieModel) = {
    println("Storing model.")
    val out = new ObjectOutputStream(new GZIPOutputStream(Files.newOutputStream(path)))
    out.writeObject(model)
    out.close()
  }

  def printMat(m: DenseMatrix[Double], row: String = "", col: String = ""): Unit = {
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

  def ratingMatToString(m: DenseMatrix[Double], r: DenseMatrix[Boolean]) = {
    val intMat = m.map(v => ('0' + Math.rint(v).toInt).toChar)
    intMat(r) := ' '
    (0 until m.rows).flatMap( x => new String(intMat(x, ::).t.toArray) + "\n")(collection.breakOut)
  }


  def printDot(i: Int, n: Int, d: Double) = {
    val a = 1.0*i/n % d
    val b = 1.0*(i-1)/n % d
    if (a != b) print(".")
  }
}
