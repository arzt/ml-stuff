package de.berlin.arzt.ml

import breeze.linalg.DenseMatrix._
import breeze.linalg.{sum, DenseVector, norm}
import breeze.linalg.View.Require

import scala.math._

object Optimization {

  def empiricalGradient(f: Vec => Double, ε: Double)(x: Vec): Vec = {
    val grad = new DenseVector[Double](x.length)
    for (i <- 0 until x.length) {
      val org = x(i)
      x(i) = org + ε
      val y1 = f(x)
      x(i) = org - ε
      val y2 = f(x)
      x(i) = org
      val gr = (y1 - y2) / (ε + ε)
      grad(i) = gr
    }
    grad
  }

  def verifyGradient(f: Vec => Double, gradient: Vec => Vec, ε: Double)(x: Vec) = {
    val y1 = empiricalGradient(f, ε)(x)
    val y2 = gradient(x)
    val δ = y1 :- y2
    sum(δ :* δ)
  }

  case class StepInfo(
    iter: Int,
    y: Double,
    red: Double
  )

  def gradientDescent(a: Double, ε: Double)(f: Vec => Double, δf: Vec => Vec)(x: Vec, acc: List[StepInfo] = Nil): List[StepInfo] = {
    val δ = δf(x)
    val step = δ * a
    x -= step
    val y = f(x)
    acc match {
      case Nil =>
        val init = StepInfo(0, y, 0.5)
        gradientDescent(a, ε)(f, δf)(x, List(init))
      case head :: tail =>
        val diff = abs(head.y - y)
        val next = StepInfo(
          iter = head.iter + 1,
          y,
          red = y / head.y
        )
        println(f"Step: ${head.iter + 1}%03d, y= $y%.10f")
        if (diff > ε)
          gradientDescent(a, ε)(f, δf)(x, next :: acc)
        else
          acc
    }
  }


  def gradientDescent3(a: Double, ε: Double)(f: Vec => Double, δf: Vec => Vec)(x: Vec, acc: List[StepInfo] = Nil): List[StepInfo] = {
    acc match {
      case Nil =>
        val y = f(x)
        val δ = δf(x)
        val init = StepInfo(0, y, 0.5)
        gradientDescent3(a, ε)(f, δf)(x, List(init))
      case head :: tail =>
        val shr = shrink(acc, 10)
        val δ = δf(x)
        val step = δ * a
        val y = f(x - step)
        if (shr < ε) {
          acc
        } else if (y < head.y) {
          val red = y / head.y
          x -= step
          val next = StepInfo(head.iter + 1, y, y / head.y) :: acc
          println(f"Step: ${head.iter + 1}%3d, y=$y%12.5f, a=$a%8.5f, |δ|=${norm(δ)}%10.5f, shrink=$shr%10.8f")

          gradientDescent3(a * 1.1, ε)(f, δf)(x, next)
        } else {
          gradientDescent3(a * 0.8, ε)(f, δf)(x, acc)
        }
    }
  }

  def gradientDescent4(a: Double, ε: Double, adjust: Int = 10)(f: Vec => Double, δf: Vec => Vec)(x: Vec, acc: List[StepInfo] = Nil): List[StepInfo] = {
    val δ = δf(x)
    acc match {
      case Nil =>
        val y = f(x)
        val (an, yn) = minFGrad(f, δ, x, a)
        val init = StepInfo(0, yn, yn / y)
        gradientDescent4(an, ε, adjust = 10)(f, δf)(x, List(init))
      case head :: tail =>
        val shr = shrink(acc, 10)
        val step = δ * a
        val y = f(x - step)
        if (y < head.y) {
          x -= step
        }
        if (shr < ε && y < head.y/* && y + 0.001 < head.y*/) {
          acc
        } else if (adjust == 0) {
          gradientDescent4(a * 1.5, ε, adjust = 10)(f, δf)(x, acc)
        } else if (y < head.y) {
          val red = y / head.y
          val next = StepInfo(head.iter + 1, y, red) :: acc
          println(f"Iteration: ${head.iter + 1}%3d, loss=${head.y}%.5f, step=$a%.5f, stop if $shr%.5f < $ε%6.5f")
          gradientDescent4(a, ε, adjust - 1)(f, δf)(x, next)
        } else {
          val (an, yn) = minFGrad(f, δ, x, a)
          gradientDescent4(an, ε, 10)(f, δf)(x, acc)
        }
    }
  }

  def shrink(steps: Seq[StepInfo], n: Int = 10) = {
    val b = steps.view.take(n).foldLeft((0.0, 0.0)) {
      case ((x1, x2), z) => (x1 + 1, x2 + z.red)
    }
    1 - b._2/b._1
  }

  def minFGrad(f: Vec => Double, grad: Vec, x: Vec, a: Double = 1) = {
    var an = a
    var astar = a
    val y0 = f(x)
    val x0 = x.copy
    val xn = x - grad * an
    val xstar = x.copy
    var ystar = f(xstar)
    var yn = f(xn)
    while (yn > y0) {
      an *= 0.1
      xn := x0 - grad * an
      yn = f(xn)
    }
    while (yn <= ystar) {
      xstar := xn
      ystar = yn
      astar = an
      an *= 10
      xn := x0 - grad * an
      yn = f(xn)
    }
    an = astar
    yn = ystar
    xn := xstar
    while (yn <= ystar) {
      xstar := xn
      ystar = yn
      astar = an
      an *= 1.25
      xn := x0 - grad * an
      yn = f(xn)
    }
    an = astar
    yn = ystar
    xn := xstar
    while (yn <= ystar) {
      xstar := xn
      ystar = yn
      astar = an
      an *= 1.023
      xn := x0 - grad * an
      yn = f(xn)
    }
    x := xstar
    (astar, ystar)
  }
}
