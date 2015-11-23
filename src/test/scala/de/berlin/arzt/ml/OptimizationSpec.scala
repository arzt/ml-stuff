package de.berlin.arzt.ml

import breeze.linalg.{max, sum, DenseVector}
import org.scalatest.{FlatSpec, Matchers}
import Optimization._

class OptimizationSpec extends FlatSpec with Matchers {

  "Empirical gradient" should "correctly approximate the numerical gradient" in {
    def f(x: Vec) = 3*x(0)*x(0)*x(0) - 4*x(1)*x(0) + 6*x(2)
    def δf(x: Vec) =
      DenseVector(
        9*x(0)*x(0) - 4*x(1),
        -4.0*x(0),
        6.0
      )
    val ε = 0.001
    for (i <- 0 to 100) {
      val x = DenseVector.rand[Double](3)
      val δfx = δf(x)
      val δfx2 = empiricalGradient(f, ε)(x)
      val diff = δfx - δfx2
      sum(diff :* diff)  should be < ε
    }
  }

  "Gradient verification" should "test whether a numerical gradient of a function is correct" in {
    def f(x: Vec) = 3*x(0)*x(0)*x(0) - 4*x(1)*x(0) + 6*x(2)
    val ε = 0.001
    val N = 100
    def δf(x: Vec) =
      DenseVector(
        9*x(0)*x(0) - 4*x(1),
        -4.0*x(0),
        6.0
      )
    def negative(x: Vec) =
      DenseVector(
        -4.0*x(0),
        6.0,
        9*x(0)*x(0) - 4*x(1)
    )
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
}
