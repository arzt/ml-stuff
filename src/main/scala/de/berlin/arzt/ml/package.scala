package de.berlin.arzt

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.nio.file.{Files, Path}

import breeze.linalg.DenseMatrix

package object ml {

  def featuresToMatrix(
    rows: Int, cols: Int,
    features: Array[(Int, Array[Double])]
  ) = {
    assert(features.length == cols)
    val data =
      features.sortBy {
        case (i, _) => i
      } flatMap {
        case (_, feat) =>
          assert(feat.length == rows)
          feat
      }
    new DenseMatrix[Double](rows, cols, data)
  }

  def loadModel(path: Path) = {
    println("Loading model.")
    new ObjectInputStream(Files.newInputStream(path)).readObject().asInstanceOf[MovieModel]
  }

  def saveModel(path: Path, model: MovieModel) = {
    println("Storing model.")
    new ObjectOutputStream(Files.newOutputStream(path)).writeObject(model)
  }
}
