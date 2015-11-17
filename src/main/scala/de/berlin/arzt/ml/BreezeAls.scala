package de.berlin.arzt.ml

import breeze.linalg.DenseMatrix.{ones, zeros}
import breeze.linalg.DenseVector.vertcat
import breeze.linalg.View.Require
import breeze.linalg.{sum, DenseMatrix}
import System.{currentTimeMillis => time}
import Optimization.{gradientDescent2, gradientDescent3, gradientDescent4}


object BreezeAls {

  def loss(y: Matrix, r: DenseMatrix[Boolean], λ: Double)(x: Vec) = {
    val (α, β) = unflatten(y.rows, y.cols, x)
    lossUnflattened(y, r, λ)(α, β)
  }

  def lossUnflattened(y: Matrix, r: DenseMatrix[Boolean], λ: Double)(α: Matrix, β: Matrix) = {
    val μ = α.t * β
    val δ = y(r) :- μ(r)
    0.5*(sum(δ:*δ) + λ*sum(α:*α) + λ*sum(β:*β))
  }

  def gradient(y: Matrix, unrated: DenseMatrix[Boolean], λ: Double)(x: Vec): Vec = {
    val (α, β) = unflatten(y.rows, y.cols, x)
    val μ = α.t * β
    val δ = μ - y
    δ(unrated) := 0.0
    val dα = β * δ.t + λ*α
    val dβ = α * δ + λ*β
    flatten(dα, dβ)
  }

  /*
  def alphaStar(y: Matrix, r: DenseMatrix[Boolean], λ: Double, β: Matrix) = {
  β.t ** β
}
  * */

  def flatten(α: Matrix, β: Matrix) = vertcat[Double](α.flatten(Require), β.flatten(Require))

  def unflatten(n: Int, m: Int, x: Vec) = {
    val rank = x.length / (n + m)
    val split = n * rank
    val α = x(0 until split).toDenseMatrix.reshape(rank, n)
    val β = x(split to -1).toDenseMatrix.reshape(rank, m)
    (α, β)
  }

  def trainModel(y: Matrix, rated: DenseMatrix[Boolean], unrated: DenseMatrix[Boolean], rank: Int, ε: Double, λ: Double) = {
    println("train ALS model.")
    val rows = y.rows
    val cols = y.cols
    val α = normal(rank, rows)
    val β = normal(rank, cols)
    val x = flatten(α, β)
    gradientDescent4(0.001, ε)(loss(y, rated, λ), gradient(y, unrated, λ))(x)
    unflatten(rows, cols, x)
  }

  def main(args: Array[String]) {
    val n = 300
    val m = 300
    val r = 10

    val row = normal(r, n)
    val col = normal(r, m)

    val flat = flatten(row, col)
    val (a1, a2) = unflatten(n, m, flat)

    val y = row * col
    //val Y = normal(rows, cols)
    val unrated = DenseMatrix.zeros[Boolean](n, m).map(x => Math.random() > 0.1)

    val rated = unrated.map(!_)

    val α = normal(r, n)
    val β = normal(r, m)

    Optimization.verifyGradient(loss(y, rated, 0.1), gradient(y, unrated, 0.1), 0.001)(flatten(α, β))

    val x0n = flatten(α, β)
    val x1 = x0n.copy
    val x2 = x0n.copy
    val a = 0.001
    var t0 = System.currentTimeMillis() / 10d
    val ε = 0.001
    val λ = 0.00
    val result1 = gradientDescent3(1, ε, minIterations = 20)(loss(y, rated, λ), gradient(y, unrated, λ))(x1)
    println("min 1 done")
    val t1 = System.currentTimeMillis() / 10d - t0
    t0 = System.currentTimeMillis() / 10d
    val result2 = gradientDescent2(a)(loss(y, rated, λ), gradient(y, unrated, λ))(x2)
    println("min 2 done")
    val t2 = System.currentTimeMillis() / 10d - t0

    val (rowr, colr) = unflatten(n, m, x1)
    val Yr = rowr * colr
    val Y2 = normal(n, m)
    val s1 = Yr(rated) - y(rated)
    val s2 = Yr(unrated) - y(unrated)
    val s3 = Y2(unrated) - y(unrated)

    val sum1 = sum(s1 :* s1)
    val sum2 = sum(s2 :* s2)
    val sum3 = sum(s3 :* s3)
    println(sum1 + " " + sum2 + " " + sum3)
  }
}