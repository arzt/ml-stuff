package de.berlin.arzt.ml

import breeze.linalg.DenseMatrix.{ones, zeros}
import breeze.linalg.DenseVector.vertcat
import breeze.linalg.View.Require
import breeze.linalg._
import System.{currentTimeMillis => time}
import Optimization._
import org.apache.spark.SparkContext


object Factorization {

  def loss(y: Matrix, r: DenseMatrix[Boolean], λ: Double, normalizer: Double = 1)(x: Vec) = {
    val (α, β) = fromFlat(y.rows, y.cols, x)
    lossUnflattened(y, r, λ, normalizer)(α, β)
  }

  def lossUnflattened(y: Matrix, r: DenseMatrix[Boolean], λ: Double, normalizer: Double = 1)(α: Matrix, β: Matrix) = {
    val μ = α.t * β
    val δ = y(r) :- μ(r)
    0.5 * (sum(δ :* δ) + λ * sum(α :* α) + λ * sum(β :* β)) / normalizer
  }

  def gradientFlat(y: Matrix, unrated: DenseMatrix[Boolean], λ: Double, normalizer: Double = 1)(x: Vec): Vec = {
    val (α, β) = fromFlat(y.rows, y.cols, x)
    val (dα, dβ) = gradient(y, unrated, λ, normalizer)(α, β)
    toFlat(dα, dβ)
  }

  def gradient(y: Matrix, unrated: DenseMatrix[Boolean], λ: Double, normalizer: Double = 1)(α: Matrix, β: Matrix) = {
    val μ = α.t * β
    val δ = μ - y
    δ(unrated) := 0.0
    val dα = β * δ.t + λ * α
    val dβ = α * δ + λ * β
    (dα :/ normalizer, dβ :/ normalizer)
  }

  def ridgeRegression(y: Vec, x: Matrix, λ: Double): Vec = {
    val I = DenseMatrix.eye[Double](x.cols)
    pinv(x.t * x + λ * I) * x.t * y
  }

  def filteredRidgeRegression(y: Vec, X: Matrix, λ: Double, c: Vec): Vec = {
    val C = diag(c)
    val I = DenseMatrix.eye[Double](X.cols)
    pinv(X.t * C * X + λ * I) * X.t * C * y
  }

  def alsStep(y: Matrix, r: DenseMatrix[Boolean], λ: Double)(in: Matrix) = {
    print("Performing ALS step")
    val c = r.map(x => if (x) 1.0 else 0.0)
    val out = zeros[Double](in.rows, y.cols)
    for (j <- 0 until y.cols) {
      val yvec = y(::, j)
      val cvec = c(::, j)
      val βvec = filteredRidgeRegression(yvec, in.t, λ, cvec)
      out(::, j) := βvec
      if (j % 50 == 0) print(".")
    }
    println
    out
  }

  def alsStepPar(y: Matrix, r: DenseMatrix[Boolean], λ: Double)(in: Matrix)(implicit context: SparkContext): Matrix = {
    val c = r.map(x => if (x) 1.0 else 0.0)
    val vecs = for (j <- 0 until y.cols) yield {
      val yvec = y(::, j)
      val cvec = c(::, j)
      (yvec, cvec)
    }
    val hui = context.parallelize(vecs).flatMap {
      case (y , c) =>
        filteredRidgeRegression(y, in.t, λ, c).toArray
    }
    val data = hui.collect()
    new DenseMatrix[Double](in.rows, y.cols, data)
  }

  def toFlat(α: Matrix, β: Matrix) = vertcat[Double](α.flatten(Require), β.flatten(Require))

  def fromFlat(n: Int, m: Int, x: Vec) = {
    val rank = x.length / (n + m)
    val split = n * rank
    val α = x(0 until split).toDenseMatrix.reshape(rank, n)
    val β = x(split to -1).toDenseMatrix.reshape(rank, m)
    (α, β)
  }

  def trainModel(y: Matrix, rated: DenseMatrix[Boolean], unrated: DenseMatrix[Boolean], rank: Int, ε: Double, λ: Double, normalizer: Double) = {
    println("train ALS model.")
    val rows = y.rows
    val cols = y.cols
    val α = normal(rank, rows)
    val β = normal(rank, cols)
    val x = toFlat(α, β)
    gradientDescent4(0.001, ε)(loss(y, rated, λ, normalizer), gradientFlat(y, unrated, λ, normalizer))(x)
    fromFlat(rows, cols, x)
  }

  def runAls(y: Matrix, rated: DenseMatrix[Boolean], rank: Int, ε: Double, λ: Double) = {
    val α = normal(rank, y.rows)
    val β = normal(rank, y.cols)
    runAlsRec(y, rated, ε, λ)(α, β, Nil)
  }

  def runAlsRec(y: Matrix, rated: DenseMatrix[Boolean], ε: Double, λ: Double)(α: Matrix, β: Matrix, steps: List[Double]): (Matrix, Matrix) = {
    β := alsStep(y, rated, λ)(α)
    α := alsStep(y.t, rated.t, λ)(β)
    val l = lossUnflattened(y, rated, λ)(α, β)
    println(f"loss:$l%10.6f")
    steps match {
      case Nil =>
        runAlsRec(y, rated, ε, λ)(α, β, l :: Nil)
      case head :: tail =>
        if (l + ε < head) {
          runAlsRec(y, rated, ε, λ)(α, β, l :: steps)
        } else {
          (α, β)
        }
    }
  }

  def runAlsRecPar(y: Matrix, rated: DenseMatrix[Boolean], ε: Double, λ: Double)(α: Matrix, β: Matrix, steps: List[Double])(implicit context: SparkContext): (Matrix, Matrix) = {
    β := alsStepPar(y, rated, λ)(α)
    α := alsStepPar(y.t, rated.t, λ)(β)
    val yj = α.t * β
    val l = lossUnflattened(y, rated, λ)(α, β)
    steps match {
      case Nil =>
        runAlsRecPar(y, rated, ε, λ)(α, β, l :: Nil)
      case head :: tail =>
        println(f"loss:$l%10.6f")
        if (l + ε < head) {
          runAlsRecPar(y, rated, ε, λ)(α, β, l :: steps)
        } else {
          (α, β)
        }
    }
  }


  def main(args: Array[String]) {
    val n = 300
    val m = 400
    val r = 10

    val row = normal(r, n)
    val col = normal(r, m)

    val y = row.t * col
    //val y = normal(n, m)
    val unrated = DenseMatrix.zeros[Boolean](n, m).map(x => Math.random() > 1)

    val rated = unrated.map(!_)
    val normalizer = 1.0

    val α = normal(r, n)
    val β = normal(r, m)


    val x0n = toFlat(α, β)
    val x1 = x0n.copy
    val a = 0.01
    var t0 = System.currentTimeMillis() / 10d
    val ε = 0.001
    val λ = 0.01
    implicit val context = Main.sparkContext(None)
    val (a1, b1) = runAlsRec(y, rated, ε, λ)(α, β, Nil)
    val result1 = gradientDescent4(a, ε, adjust = 10)(loss(y, rated, λ, normalizer), gradientFlat(y, unrated, λ, normalizer))(x1)

    val (rowr, colr) = fromFlat(n, m, x1)

    val l1 = lossUnflattened(y, rated, λ, normalizer)(rowr, colr)
    val l2 = lossUnflattened(y, rated, λ, normalizer)(a1, b1)
    println(l1)
    println(l2)
  }
}