package de.berlin.arzt.ml

import breeze.linalg.{sum, DenseVector, DenseMatrix}
import org.scalatest.{Matchers, FlatSpec}
import BreezeAls._

import scala.util.Random

class BreezeAlsSpec extends FlatSpec with Matchers {

  val ε = 0.001
  val n = 3
  val r = 2
  val m = 4
  val α0 = new DenseMatrix(r, n, Array[Double](1, 2, 3, 4, 5, 6))
  val β0 = new DenseMatrix(r, m, Array[Double](8, 7, 6, 5, 4, 3, 2, 1))
  val rat = DenseMatrix.ones[Boolean](n, m)
  val unrat = rat.map(!_)
  val y = α0.t * β0
  val λ0 = 0
  val y2 = y :+ 1.d
  val seed = 0
  val rand = new Random(seed)

  "The unflattened ALS loss function" should "be close to zero" in {
    val loss = lossUnflattened(y, rat, λ0)(α0, β0)
    println(loss)
    loss should be < ε
  }

  it should "roughly equal 6" in {
    val expected = 6
    val loss = lossUnflattened(y2, rat, λ0)(α0, β0)
    loss should (be < expected + ε and be > expected - ε)
  }

  "The flatten function" should "compute a vector of correct length" in {
    val flat = flatten(α0, β0)
    flat should have length 14
  }

  it should "produce the correctly flattened vector" in {
    val flat = flatten(α0, β0)
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

  it should "be inverse to unflatten other" in {
    val flat = flatten(α0, β0)
    val (α1, β1) = unflatten(n, m, flat)
    val flat2 = flatten(α1, β1)
    flat should equal(flat2)
  }

  "An unflattened vector" should "be represented by two matrices of correct sizes" in {
    val (α, β) = unflatten(2, 2, DenseVector.zeros[Double](12))
    α.rows should be(3)
    α.cols should be(2)
    β.rows should be(3)
    β.cols should be(2)
  }

  it should "be correctly represented by to matrices" in {
    val flat = DenseVector(Array[Double](1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    val (α, β) = unflatten(2, 2, flat)
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
    val flat = flatten(α0, β0)
    val (α1, β1) = unflatten(n, m, flat)
    α1 should equal(α0)
    β1 should equal(β0)
  }

  "The gradient of the loss" should "be roughly the zero vector at the optimum" in {
    val xOpt = flatten(α0, β0)
    val grad = gradient(y, unrat, λ0)(xOpt)
    (sum(grad)) should be < ε
  }

  it should "be roughly equal to the empirical loss lambda equal to zero" in {
    val λ = 0
    val xOpt = flatten(α0, β0)
    xOpt := xOpt :* rand.nextGaussian()
    val grad = gradient(y, unrat, λ)(xOpt)
    val x2 = Optimization.empiricalGradient(loss(y, rat, λ), ε)(xOpt)
    val diff = grad :- x2
    sum(diff :* diff) should be < ε
  }

  it should "be roughly equal to the empirical loss lambda greater than zero" in {
    val λ = 0.5
    val xOpt = flatten(α0, β0)
    xOpt := xOpt :* rand.nextGaussian()
    val grad = gradient(y, unrat, λ)(xOpt)
    val x2 = Optimization.empiricalGradient(loss(y, rat, λ), ε)(xOpt)
    val diff = grad :- x2
    sum(diff :* diff) should be < ε
  }
}
