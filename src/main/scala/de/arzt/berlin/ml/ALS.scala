package de.arzt.berlin.ml

import breeze.linalg.DenseMatrix.{ones, rand, zeros}
import breeze.linalg.{DenseMatrix, View, sum}
import breeze.stats.distributions.Gaussian

import scala.math.abs


object BreezeAls {

  type Mat = DenseMatrix[Double]

  val eps = 0.00000001

  def train(mat: Mat): BreezeAlsModel = {
    BreezeAlsModel(
      1,
      zeros[Double](1, 1),
      zeros[Double](1, 1)
    )
  }

  def initModel(mat: Mat, rank: Int, seed: Long) = {
    val g = Gaussian(0, 1)
    val row = rand[Double](mat.rows, rank)
    val col = rand[Double](rank, mat.cols)
    BreezeAlsModel(
      rank,
      row,
      col
    )
  }

  def rowGradient(mat: Mat, row: Mat, col: Mat) = {

  }

  def colGradient(mat: Mat, row: Mat, col: Mat) = {

  }

  def empGradient(fun: Mat => Double)(x: Mat): Mat = {
    val grad: Mat = zeros[Double](x.rows, x.cols)
    val size = grad.size
    val fgrad = grad.flatten(View.Require)
    val a = x.copy
    val b = x.copy
    val fa = a.flatten(View.Require)
    val fb = b.flatten(View.Require)
    for (i <- 0 until size) {
      fa(i) += eps
      fb(i) -= eps
      val l1 = fun(b)
      val l2 = fun(a)
      fa(i) -= eps
      fb(i) += eps
      val gr = (l2 - l1)/(eps+eps)
      fgrad(i) = gr
    }
    grad
  }

  def empRowGradient(mat: Mat, row: Mat, col: Mat) = {
    val grad: Mat = zeros[Double](row.rows, row.cols)
    val size = grad.size
    val fgrad = grad.flatten(View.Require)
    val a = row.copy
    val b = row.copy
    val fa = a.flatten(View.Require)
    val fb = b.flatten(View.Require)
    for (i <- 0 until size) {
      fa(i) += eps
      fb(i) -= eps
      val l1 = loss(mat, b, col)
      val l2 = loss(mat, a, col)
      fa(i) -= eps
      fb(i) += eps
      val gr = (l2 - l1)/(eps+eps)
      fgrad(i) = gr
    }
    grad
  }

  def empColGradient(mat: Mat, row: Mat, col: Mat) = {

  }

  def loss(mat: Mat, row: Mat, col: Mat
  ) = {
    val mat2: Mat = row * col
    val loss: Mat = mat - mat2
    val loss2: Mat = loss :* loss
    sum(loss2)
  }

  def minStep(f: Mat => Double, grad: Mat => Mat, x: Mat) = {
    val y = f(x)
    val gr = grad(x)
    x := x - gr
    y
  }

  def minF(a: Double)(fun: Mat => Double, grad: Mat => Mat)(x: Mat) = {
    val x2 = x.copy
    var y = 0.0
    while (abs(y - fun(x2)) > eps) {
      y = fun(x2)
      println("y:" + y)
      //println("x: " + x2)
      val gr = grad(x2) * 0.05
      x2 := x2 - gr
    }
    x2
  }

  case class Step(x: Mat, y: Double, grad: Mat)

  def minimize(a: Double)(fun: Mat => Double, grad: Mat => Mat)(x: Mat, acc: List[Step] = Nil): List[Step] = {
    val y0 = fun(x)
    val gr: Mat = grad(x)
    x -= gr * a
    val y1 = fun(x)
    val diff = abs(y0 - y1)
    if (diff > eps) {
      minimize(a)(fun, grad)(x, Step(x.copy, y1, gr) :: acc)
    } else {
      acc
    }
  }

  lazy val one = ones[Double](1, 1)

  def main(args: Array[String]) {
    def fun(m: Mat): Double = {
      val p = m.copy
      p(0, 0) += 5
      val a = p.toDenseVector
      val d: Double = a.dot(a)
      d
    }
    def grad(m: Mat): Mat = {
      m.copy*2.0 + new Mat(1, 1, Array(10.0))
    }

    val x0: Mat = ones[Double](1,1) * 0.5
    val a = 0.1
    val gradient = empGradient(fun)(_)
    val min1 = minF(a)(fun, gradient)(_)
    val result = minimize(a)(fun, gradient)(x0.copy)
    val result2 = minimize(a)(fun, grad)(x0.copy)
    result.foreach(println)
  }
}

case class BreezeAlsModel(
  rank: Int,
  row: DenseMatrix[Double],
  col: DenseMatrix[Double]
)