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
    val μ = α * β
    val δ = y(r) :- μ(r)
    0.5*(sum(δ:*δ) + λ*sum(α:*α) + λ*sum(β:*β))
  }

  def gradient(y: Matrix, unrated: DenseMatrix[Boolean], λ: Double)(x: Vec): Vec = {
    val (α, β) = unflatten(y.rows, y.cols, x)
    val μ = α * β
    val δ = μ - y
    δ(unrated) := 0.0
    val dα = δ * β.t + λ*sum(α)
    val dβ = α.t * δ + λ*sum(β)
    flatten(dα, dβ)
  }

  def flatten(α: Matrix, β: Matrix) = vertcat[Double](α.flatten(Require), β.flatten(Require))

  def unflatten(m: Int, n: Int, x: Vec) = {
    val rank = x.length / (m + n)
    val split = m * rank
    val α = x(0 until split).toDenseMatrix.reshape(m, rank)
    val β = x(split to -1).toDenseMatrix.reshape(rank, n)
    (α, β)
  }

  def trainModel(Y: Matrix, rated: DenseMatrix[Boolean], unrated: DenseMatrix[Boolean], rank: Int, ε: Double, λ: Double) = {
    println("train ALS model.")
    val rows = Y.rows
    val cols = Y.cols
    val row0 = normal(rows, rank)
    val col0 = normal(rank, cols)
    val x = flatten(row0, col0)
    gradientDescent4(0.001, ε)(loss(Y, rated, λ), gradient(Y, unrated, λ))(x)
    unflatten(rows, cols, x)
  }

  def main(args: Array[String]) {
    val rows = 300
    val cols = 300
    val rank = 10

    val row = normal(rows, rank)
    val col = normal(rank, cols)

    val flat = flatten(row, col)
    val (a1, a2) = unflatten(rows, cols, flat)

    val Y = row * col
    //val Y = normal(rows, cols)
    val R = DenseMatrix.zeros[Boolean](rows, cols).map(x => Math.random() > 0.1)

    val R2 = R.map(!_)

    val row0 = normal(rows, rank, 0.01)
    val col0 = normal(rank, cols, 0.01)
    val x0n = flatten(row0, col0)
    val x1 = x0n.copy
    val x2 = x0n.copy
    val a = 0.001
    var t0 = System.currentTimeMillis() / 10d
    val ε = 0.001
    val λ = 0.00
    val result1 = gradientDescent3(1, ε, minIterations = 20)(loss(Y, R2, λ), gradient(Y, R, λ))(x1)
    println("min 1 done")
    val t1 = System.currentTimeMillis() / 10d - t0
    t0 = System.currentTimeMillis() / 10d
    val result2 = gradientDescent2(a)(loss(Y, R2, λ), gradient(Y, R, λ))(x2)
    println("min 2 done")
    val t2 = System.currentTimeMillis() / 10d - t0

    val (rowr, colr) = unflatten(rows, cols, x1)
    val Yr = rowr * colr
    val Y2 = normal(rows, cols)
    val s1 = Yr(R2) - Y(R2)
    val s2 = Yr(R) - Y(R)
    val s3 = Y2(R) - Y(R)

    val sum1 = sum(s1 :* s1)
    val sum2 = sum(s2 :* s2)
    val sum3 = sum(s3 :* s3)
    println(sum1 + " " + sum2 + " " + sum3)
  }
}