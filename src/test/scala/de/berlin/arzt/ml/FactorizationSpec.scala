package de.berlin.arzt.ml

import breeze.linalg._
import org.scalatest.{Matchers, FlatSpec}
import Factorization._

import scala.util.Random

class FactorizationSpec extends FlatSpec with Matchers {

  "The factorization loss function" should "be close to zero" in {
    val ε = 0.001
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val rat = DenseMatrix.ones[Boolean](n, m)
    val y = α0.t * β0
    val λ0 = 0
    val loss = lossUnflattened(y, rat, λ0)(α0, β0)
    println(loss)
    loss should be < ε
  }

  it should "roughly equal 6" in {
    val ε = 0.001
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val rat = DenseMatrix.ones[Boolean](n, m)
    val expected = 6
    val λ0 = 0
    val y = α0.t * β0
    val y2 = y :+ 1.d
    val loss = lossUnflattened(y2, rat, λ0)(α0, β0)
    loss should (be < expected + ε and be > expected - ε)
  }

  "The flatten function" should "compute a vector of correct length" in {
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val flat = toFlat(α0, β0)
    flat should have length 14
  }

  it should "produce the correctly flattened vector" in {
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val flat = toFlat(α0, β0)
    flat(0) should be(α0(0, 0))
    flat(1) should be(α0(1, 0))
    flat(2) should be(α0(0, 1))
    flat(3) should be(α0(1, 1))
    flat(4) should be(α0(0, 2))
    flat(5) should be(α0(1, 2))
    flat(6) should be(β0(0, 0))
    flat(7) should be(β0(1, 0))
    flat(8) should be(β0(0, 1))
    flat(9) should be(β0(1, 1))
    flat(10) should be(β0(0, 2))
    flat(11) should be(β0(1, 2))
    flat(12) should be(β0(0, 3))
    flat(13) should be(β0(1, 3))
  }

  it should "be inverse to unflatten" in {
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val flat = toFlat(α0, β0)
    val (α1, β1) = fromFlat(n, m, flat)
    val flat2 = toFlat(α1, β1)
    flat should equal(flat2)
  }

  "An unflattened vector" should "be represented by two matrices of correct sizes" in {
    val (α, β) = fromFlat(2, 2, DenseVector.zeros[Double](12))
    α.rows should be(3)
    α.cols should be(2)
    β.rows should be(3)
    β.cols should be(2)
  }

  it should "be correctly represented by to matrices" in {
    val flat = DenseVector(Array[Double](1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    val (α, β) = fromFlat(2, 2, flat)
    α(0, 0) should be(flat(0))
    α(1, 0) should be(flat(1))
    α(2, 0) should be(flat(2))
    α(0, 1) should be(flat(3))
    α(1, 1) should be(flat(4))
    α(2, 1) should be(flat(5))
    β(0, 0) should be(flat(6))
    β(1, 0) should be(flat(7))
    β(2, 0) should be(flat(8))
    β(0, 1) should be(flat(9))
    β(1, 1) should be(flat(10))
    β(2, 1) should be(flat(11))
  }

  "Flattening an unflattened vector" should "yield the same vector" in {
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val flat = toFlat(α0, β0)
    val (α1, β1) = fromFlat(n, m, flat)
    α1 should equal(α0)
    β1 should equal(β0)
  }

  "The gradient of the loss" should "be roughly the zero vector at the optimum" in {
    val ε = 0.001
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val rat = DenseMatrix.ones[Boolean](n, m)
    val unrat = rat.map(!_)
    val xOpt = toFlat(α0, β0)
    val y = α0.t * β0
    val λ0 = 0
    val grad = gradientFlat(y, unrat, λ0)(xOpt)
    (sum(grad)) should be < ε
  }

  it should "be roughly equal to the empirical loss λ = 0" in {
    val ε = 0.001
    val seed = 0
    val rand = new Random(seed)
    val λ = 0
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val xOpt = toFlat(α0, β0)
    xOpt := xOpt :* rand.nextGaussian()
    val y = α0.t * β0
    val rat = DenseMatrix.ones[Boolean](n, m)
    val unrat = rat.map(!_)
    val grad = gradientFlat(y, unrat, λ)(xOpt)
    val x2 = Optimization.empiricalGradient(loss(y, rat, λ), ε)(xOpt)
    val diff = grad :- x2
    sum(diff :* diff) should be < ε
  }

  it should "be roughly equal to the empirical loss for λ > zero" in {
    val ε = 0.001
    val seed = 0
    val rand = new Random(seed)
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val λ = 0.5
    val xOpt = toFlat(α0, β0)
    xOpt := xOpt :* rand.nextGaussian()
    val y = α0.t * β0
    val rat = DenseMatrix.ones[Boolean](n, m)
    val unrat = rat.map(!_)
    val grad = gradientFlat(y, unrat, λ)(xOpt)
    val x2 = Optimization.empiricalGradient(loss(y, rat, λ), ε)(xOpt)
    val diff = grad :- x2
    sum(diff :* diff) should be < ε
  }

  "Ridge regression" should "solve a linear regression" in {
    val ε = 0.001
    val n = 10
    val m = 10
    val x = normal(n, m)
    val β = DenseVector.rand[Double](10)
    val y = x * β
    val β2 = ridgeRegression(y, x, 0)
    val diff = β :- β2
    sum(diff :* diff) should be < ε
  }

  "Filtered ridge regression" should "be equal to ridge regression for c=ones" in {
    val ε = 0.001
    val tn = 300
    val tm = 400
    val c = DenseVector.ones[Double](tn)
    val x = normal(tn, tm)

    val β = DenseVector.rand[Double](tm)
    val y = x * β

    val β1 = ridgeRegression(y, x, 0.1)
    val β2 = filteredRidgeRegression(y, x, 0.1, c)

    val diff = β1 :- β2
    sum(diff :* diff) should be < ε
  }

  it should "solve a linear regression" in {
    val ε = 0.001
    val n = 30
    val m = 20
    val c = DenseVector.ones[Double](n)
    val x = normal(n, m)
    val β = DenseVector.rand[Double](m)
    val y = x * β
    val β2 = filteredRidgeRegression(y, x, 0, c)
    val y2 = x * β2
    val diff = y :- y2
    sum(diff :* diff) should be < ε
  }


  "A alternating least square step" should "reduce the loss" in {
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val α1 = normal(α0.rows, α0.cols)
    val y = α0.t * β0
    val rat = DenseMatrix.ones[Boolean](n, m)
    val unrat = rat.map(!_)
    val λ0 = 0
    val β1 = alsStep(y, rat, λ0)(α1)
    val y1 = α1.t * β1
    val α2 = alsStep(y.t, rat.t, λ0)(β1)
    val β2 = alsStep(y, rat, λ0)(α2)
    val y2 = α2.t * β2
    val diff1 = y :- y1
    val diff2 = y :- y2
    sum(diff2 :* diff2) should be < (sum(diff1 :* diff1))
  }

  it should "reach the optimum for one parameter matrix" in {
    val ε = 0.001
    val n = 3
    val r = 2
    val m = 4
    val λ0 = 0
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val y = α0.t * β0
    val α1 = normal(α0.rows, α0.cols)
    val rat = DenseMatrix.ones[Boolean](n, m)
    val unrat = rat.map(!_)
    val β1 = alsStep(y, rat, λ0)(α1)
    val α2 = alsStep(y.t, rat.t, λ0)(β1)
    val (_, dβ) = gradient(y, unrat, λ0)(α1, β1)
    val (dα, _) = gradient(y, unrat, λ0)(α2, β1)
    sum(dβ :* dβ) should be < ε
    sum(dα :* dα) should be < ε
  }

  "Alternating least squares" should "solve matrix factorization" in {
    val ε = 0.001
    val n = 3
    val r = 2
    val m = 4
    val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
    val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
    val y = α0.t * β0
    val rat = DenseMatrix.ones[Boolean](n, m)
    val unrat = rat.map(!_)
    val (α, β) = runAls(y, rat, r, ε = 0.0001, λ = 0)
    val diff = y :- α.t * β
    sum(diff :* diff) should be < ε
  }

}
