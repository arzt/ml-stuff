package de.berlin.arzt.ml

import breeze.linalg.{ max, sum, DenseVector, DenseMatrix }
import org.scalatest.{ FlatSpec, Matchers }
import Optimization._
import Factorization._

class OptimizationSpec extends FlatSpec with Matchers {

  "Empirical gradient" should "correctly approximate the numerical gradient" in {
    def f(x: Vec) = 3 * x(0) * x(0) * x(0) - 4 * x(1) * x(0) + 6 * x(2)
    def δf(x: Vec) =
      DenseVector(
        9 * x(0) * x(0) - 4 * x(1),
        -4.0 * x(0),
        6.0)
    val ε = 0.001
    for (i <- 0 to 100) {
      val x = DenseVector.rand[Double](3)
      val δfx = δf(x)
      val δfx2 = empiricalGradient(f, ε)(x)
      val diff = δfx - δfx2
      sum(diff :* diff) should be < ε
    }
  }

  "Gradient verification" should "test whether a numerical gradient of a function is correct" in {
    def f(x: Vec) = 3 * x(0) * x(0) * x(0) - 4 * x(1) * x(0) + 6 * x(2)
    val ε = 0.001
    val N = 100
    def δf(x: Vec) =
      DenseVector(
        9 * x(0) * x(0) - 4 * x(1),
        -4.0 * x(0),
        6.0)
    def negative(x: Vec) =
      DenseVector(
        -4.0 * x(0),
        6.0,
        9 * x(0) * x(0) - 4 * x(1))
    val p = new DenseVector[Double](N)
    val n = new DenseVector[Double](N)
    for (i <- 0 until 100) {
      val x = DenseVector.rand[Double](3)
      p(i) = verifyGradient(f, δf, ε)(x)
      n(i) = verifyGradient(f, negative, ε)(x)
    }
    max(p) should be < ε
    max(n) should be > ε

  }

  "Both new optimization strategies" should " behave equally" in {
    val n = 200
    val m = 300
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
    val ε = 0.001
    val λ = 0.01
    val f = loss(y, rated, λ, normalizer) _
    val df = gradientFlat(y, unrated, λ, normalizer) _
    val result0 = fmincg(f, df, x0n.copy)
    val result1 = fmincg2(f, df, x0n.copy)
    val diff = result0 :- result1
    sum(diff :* diff) should be < ε
  }
}
