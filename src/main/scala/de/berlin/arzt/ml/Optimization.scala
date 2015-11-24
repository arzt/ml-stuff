package de.berlin.arzt.ml

import breeze.linalg.DenseMatrix._
import breeze.linalg.{ sum, DenseVector, norm }
import breeze.linalg.View.Require

import scala.util.control.Breaks._
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
    red: Double)

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
          red = y / head.y)
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
        if (shr < ε && y < head.y /* && y + 0.001 < head.y*/ ) {
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
    1 - b._2 / b._1
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

  def fmincg(f: Vec => Double, δf: Vec => Vec, X: Vec): Vec = {

    /*
  function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
  % Minimize a continuous differentialble multivariate function. Starting point
  % is given by "X" (D by 1), and the function named in the string "f", must
  % return a function value and a vector of partial derivatives. The Polack-
  % Ribiere flavour of conjugate gradients is used to compute search directions,
  % and a line search using quadratic and cubic polynomial approximations and the
  % Wolfe-Powell stopping criteria is used together with the slope ratio method
  % for guessing initial step sizes. Additionally a bunch of checks are made to
  % make sure that exploration is taking place and that extrapolation will not
  % be unboundedly large. The "length" gives the length of the run: if it is
  % positive, it gives the maximum number of line searches, if negative its
  % absolute gives the maximum allowed number of function evaluations. You can
  % (optionally) give "length" a second component, which will indicate the
  % reduction in function value to be expected in the first line-search (defaults
  % to 1.0). The function returns when either its length is up, or if no further
  % progress can be made (ie, we are at a minimum, or so close that due to
  % numerical problems, we cannot get any closer). If the function terminates
  % within a few iterations, it could be an indication that the function value
  % and derivatives are not consistent (ie, there may be a bug in the
  % implementation of your "f" function). The function returns the found
  % solution "X", a vector of function values "fX" indicating the progress made
  % and "i" the number of iterations (line searches or function evaluations,
  % depending on the sign of "length") used.
  %
  % Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
  %
  % See also: checkgrad
  %
  % Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
  %
  %
  % (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
  %
  % Permission is granted for anyone to copy, use, or modify these
  % programs and accompanying documents for purposes of research or
  % education, provided this copyright notice is retained, and note is
  % made of any changes that have been made.
  %
  % These programs and documents are distributed without any warranty,
  % express or implied.  As the programs were written for research
  % purposes only, they have not been tested to the degree that would be
  % advisable in any important application.  All use of these programs is
  % entirely at the user's own risk.
  %

  % Read options
  if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
      length = options.MaxIter;
  else
      length = 100;
  end

  */
    val length = 100
    val RHO = 0.01 // bunch of constants for line searches
    val SIG = 0.5 // RHO and SIG are the constants in the Wolfe-Powell conditions
    val INT = 0.1 // don't reevaluate within 0.1 of the limit of the current bracket
    val EXT = 3 // extrapolate maximum 3 times the current bracket
    val MAX = 20 // max 20 function evaluations per line search
    val RATIO = 100.0 // maximum allowed slope ratio

    /*
 if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
 */
    val red = 1

    // S=['Iteration '];
    var i = 0 //                                            // zero the run length counte
    var ls_failed = false //                            // no previous line search has failed
    var fX: List[Double] = List();
    //[f1 df1] = eval(argstr);                      // get function value and gradient
    var f1 = f(X)
    val df1 = δf(X)
    i = i + 1 //                                            // count epochs?!
    val s = -df1 //                                        // search direction is steepest
    var d1 = -s.t * s //                                               // this is the slope
    var z1 = red / (1 - d1); // initial step is red/(|s|+1)

    while (i < length) { // while not finished
      i = i + 1 // count iterations?!

      val X0 = X.copy
      val f0 = f1
      val df0 = df1.copy // make a copy of current values
      X := X + s * z1; // begin line search
      //[f2 df2] = eval(argstr);
      var f2 = f(X)
      val df2 = δf(X)
      i = i + 1 // count epochs?!
      var d2 = df2.t * s;
      // initialize point 3 equal to point 1
      var f3 = f1
      var d3 = d1;
      var z3 = -z1;

      //if length>0, M = MAX; else M = min(MAX, -length-i); end
      var M = if (length > 0) MAX else Math.min(MAX, -length - i)
      var success = 0
      var limit = -1.0 // initialize quanteties
      //while 1
      breakable {

        while (true) {
          //while ((f2 > f1+z1*RHO*d1) | (d2 > -SIG*d1)) & (M > 0)

          while ((f2 > f1 + z1 * RHO * d1 | d2 > -SIG * d1) & M > 0) {

            limit = z1 // tighten the bracket

            var z2 = if (f2 > f1) {
              z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3) // quadratic fit
            } else {
              val A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3) // cubic fit
              val B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)

              //if isnan(z2) | isinf(z2)
              //z2 = z3/2;                  // if we had a numerical problem then bisect
              //end
              val C = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A // numerical error possible - ok!
              if (C.isNaN() || C.isInfinity)
                z3 / 2
              else
                C
            }

            z2 = max(min(z2, INT * z3), (1 - INT) * z3); // don't accept too close to limits
            z1 = z1 + z2; // update the step
            X := X + z2 * s;
            //[f2 df2] = eval(argstr);
            f2 = f(X)
            df2 := δf(X)
            M = M - 1
            i = i + 1 // count epochs?!
            d2 = df2.t * s;
            z3 = z3 - z2; // z3 is now relative to the location of z2
          }

          if (f2 > f1 + z1 * RHO * d1 | d2 > -SIG * d1) {
            break // this is a failure
          } else if (d2 > SIG * d1) {
            success = 1
            break // success
          } else if (M == 0) {
            break // failure
          }
          val A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3); // make cubic extrapolation
          val B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
          var z2 = -d2 * z3 * z3 / (B + sqrt(B * B - A * d2 * z3 * z3)); // num. error possible - ok!

          if (z2.isNaN() | z2.isInfinite() | z2 < 0) {
            if (limit < -0.5) {
              z2 = z1 * (EXT - 1)
            } else {
              z2 = (limit - z1) / 2
            }

          } else if (limit > -0.5 && z2 + z1 > limit) {
            z2 = (limit - z1) / 2
          } else if (limit < -0.5 && z2 + z1 > z1 * EXT) {
            z2 = z1 * (EXT - 1.0)
          } else if (z2 < -z3 * INT) {
            z2 = -z3 * INT
          } else if (limit > -0.5 && z2 < (limit - z1) * (1.0 - INT)) {
            z2 = (limit - z1) * (1.0 - INT)
          }
          f3 = f2 // set point 3 equal to point 2
          d3 = d2
          z3 = -z2
          z1 = z1 + z2
          X := X + z2 * s // update current estimates
          //[f2 df2] = eval(argstr);
          f2 = f(X)
          df2 := δf(X)
          M = M - 1
          i = i + 1; // count epochs?!
          d2 = df2.t * s;
        }
      }
      if (success == 1) {
        f1 = f2
        fX = f1 :: fX;
        println(f"Iteration $i%d | Cost: $f1%4.6f");
        s := (df2.t * df2 - df1.t * df2) / (df1.t * df1) * s - df2; // Polack-Ribiere direction
        val tmp = df1
        df1 := df2
        df2 := tmp; // swap derivatives
        d2 = df1.t * s;
        // new slope must be negative
        if (d2 > 0) {
          s := -df1; // otherwise use steepest direction
          d2 = -s.t * s;
        }

        z1 = z1 * min(RATIO, d1 / d2) // slope ratio but max RATIO
        d1 = d2
        ls_failed = false; // this line search did not fail

      } else {
        X := X0
        f1 = f0
        df1 := df0 // restore point from before failed line search
        if (ls_failed | i > length) {
          // line search failed twice in a row
          break // or we ran out of time, so we give up
        }
        val tmp = df1
        df1 := df2
        df2 := tmp // swap derivatives
        s := -df1; // try steepest
        d1 = -s.t * s;
        z1 = 1 / (1 - d1);
        ls_failed = true; // this line search failed        
      }
    }
    X
  }

  def innerWhile(RHO: Double, SIG: Double, INT: Double)(f: Vec => Double, δf: Vec => Vec, X: Vec, df2: Vec, s: Vec, f1: Double, f2: Double, f3: Double, d1: Double, d2: Double, d3: Double, M: Int, z1: Double, z3: Double): Int = {
    if ((f2 < f1 + z1 * RHO * d1 && d2 < -SIG * d1) || M <= 0) {
      5
    } else {
      val limit = z1 // tighten the bracket

      var z2 = if (f2 > f1) {
        z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3) // quadratic fit
      } else {
        val A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3) // cubic fit
        val B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
        val C = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A // numerical error possible - ok!
        if (C.isNaN() || C.isInfinity)
          z3 / 2
        else
          C
      }

      z2 = max(min(z2, INT * z3), (1 - INT) * z3); // don't accept too close to limits
      val Xn = X + z2 * s
      innerWhile(RHO, SIG, INT)(f, δf, Xn, δf(Xn), s, f1, f(Xn), f3, d1, df2.t * s, d3, M - 1, z1 + z2, z3 - z2)
    }
  }
  
  def fmincg2(f: Vec => Double, δf: Vec => Vec, X: Vec): Vec = {

    /*
  function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
  % Minimize a continuous differentialble multivariate function. Starting point
  % is given by "X" (D by 1), and the function named in the string "f", must
  % return a function value and a vector of partial derivatives. The Polack-
  % Ribiere flavour of conjugate gradients is used to compute search directions,
  % and a line search using quadratic and cubic polynomial approximations and the
  % Wolfe-Powell stopping criteria is used together with the slope ratio method
  % for guessing initial step sizes. Additionally a bunch of checks are made to
  % make sure that exploration is taking place and that extrapolation will not
  % be unboundedly large. The "length" gives the length of the run: if it is
  % positive, it gives the maximum number of line searches, if negative its
  % absolute gives the maximum allowed number of function evaluations. You can
  % (optionally) give "length" a second component, which will indicate the
  % reduction in function value to be expected in the first line-search (defaults
  % to 1.0). The function returns when either its length is up, or if no further
  % progress can be made (ie, we are at a minimum, or so close that due to
  % numerical problems, we cannot get any closer). If the function terminates
  % within a few iterations, it could be an indication that the function value
  % and derivatives are not consistent (ie, there may be a bug in the
  % implementation of your "f" function). The function returns the found
  % solution "X", a vector of function values "fX" indicating the progress made
  % and "i" the number of iterations (line searches or function evaluations,
  % depending on the sign of "length") used.
  %
  % Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
  %
  % See also: checkgrad
  %
  % Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
  %
  %
  % (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
  %
  % Permission is granted for anyone to copy, use, or modify these
  % programs and accompanying documents for purposes of research or
  % education, provided this copyright notice is retained, and note is
  % made of any changes that have been made.
  %
  % These programs and documents are distributed without any warranty,
  % express or implied.  As the programs were written for research
  % purposes only, they have not been tested to the degree that would be
  % advisable in any important application.  All use of these programs is
  % entirely at the user's own risk.
  %

  % Read options
  if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
      length = options.MaxIter;
  else
      length = 100;
  end

  */
    val length = 100
    val RHO = 0.01 // bunch of constants for line searches
    val SIG = 0.5 // RHO and SIG are the constants in the Wolfe-Powell conditions
    val INT = 0.1 // don't reevaluate within 0.1 of the limit of the current bracket
    val EXT = 3 // extrapolate maximum 3 times the current bracket
    val MAX = 20 // max 20 function evaluations per line search
    val RATIO = 100.0 // maximum allowed slope ratio

    /*
 if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
 */
    val red = 1

    // S=['Iteration '];
    var i = 0 //                                            // zero the run length counte
    var ls_failed = false //                            // no previous line search has failed
    var fX: List[Double] = List();
    //[f1 df1] = eval(argstr);                      // get function value and gradient
    var f1 = f(X)
    val df1 = δf(X)
    i = i + 1 //                                            // count epochs?!
    val s = -df1 //                                        // search direction is steepest
    var d1 = -s.t * s //                                               // this is the slope
    var z1 = red / (1 - d1); // initial step is red/(|s|+1)

    while (i < length) { // while not finished
      i = i + 1 // count iterations?!

      val X0 = X.copy
      val f0 = f1
      val df0 = df1.copy // make a copy of current values
      X := X + s * z1; // begin line search
      //[f2 df2] = eval(argstr);
      var f2 = f(X)
      val df2 = δf(X)
      i = i + 1 // count epochs?!
      var d2 = df2.t * s;
      // initialize point 3 equal to point 1
      var f3 = f1
      var d3 = d1;
      var z3 = -z1;

      //if length>0, M = MAX; else M = min(MAX, -length-i); end
      var M = if (length > 0) MAX else Math.min(MAX, -length - i)
      var success = 0
      var limit = -1.0 // initialize quanteties
      //while 1
      breakable {

        while (true) {

          //while ((f2 > f1+z1*RHO*d1) | (d2 > -SIG*d1)) & (M > 0)
          while ((f2 > f1 + z1 * RHO * d1 | d2 > -SIG * d1) & M > 0) {

            limit = z1 // tighten the bracket

            var z2 = if (f2 > f1) {
              z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3) // quadratic fit
            } else {
              val A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3) // cubic fit
              val B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)

              //if isnan(z2) | isinf(z2)
              //z2 = z3/2;                  // if we had a numerical problem then bisect
              //end
              val C = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A // numerical error possible - ok!
              if (C.isNaN() || C.isInfinity)
                z3 / 2
              else
                C
            }

            z2 = max(min(z2, INT * z3), (1 - INT) * z3); // don't accept too close to limits
            z1 = z1 + z2; // update the step
            X := X + z2 * s;
            //[f2 df2] = eval(argstr);
            f2 = f(X)
            df2 := δf(X)
            M = M - 1
            i = i + 1 // count epochs?!
            d2 = df2.t * s;
            z3 = z3 - z2; // z3 is now relative to the location of z2
          }

          if (f2 > f1 + z1 * RHO * d1 | d2 > -SIG * d1) {
            break // this is a failure
          } else if (d2 > SIG * d1) {
            success = 1
            break // success
          } else if (M == 0) {
            break // failure
          }
          val A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3); // make cubic extrapolation
          val B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
          var z2 = -d2 * z3 * z3 / (B + sqrt(B * B - A * d2 * z3 * z3)); // num. error possible - ok!

          if (z2.isNaN() | z2.isInfinite() | z2 < 0) {
            if (limit < -0.5) {
              z2 = z1 * (EXT - 1)
            } else {
              z2 = (limit - z1) / 2
            }

          } else if (limit > -0.5 && z2 + z1 > limit) {
            z2 = (limit - z1) / 2
          } else if (limit < -0.5 && z2 + z1 > z1 * EXT) {
            z2 = z1 * (EXT - 1.0)
          } else if (z2 < -z3 * INT) {
            z2 = -z3 * INT
          } else if (limit > -0.5 && z2 < (limit - z1) * (1.0 - INT)) {
            z2 = (limit - z1) * (1.0 - INT)
          }
          f3 = f2 // set point 3 equal to point 2
          d3 = d2
          z3 = -z2
          z1 = z1 + z2
          X := X + z2 * s // update current estimates
          //[f2 df2] = eval(argstr);
          f2 = f(X)
          df2 := δf(X)
          M = M - 1
          i = i + 1; // count epochs?!
          d2 = df2.t * s;
        }
      }
      if (success == 1) {
        f1 = f2
        fX = f1 :: fX;
        println(f"Iteration $i%d | Cost: $f1%4.6f");
        s := (df2.t * df2 - df1.t * df2) / (df1.t * df1) * s - df2; // Polack-Ribiere direction
        val tmp = df1
        df1 := df2
        df2 := tmp; // swap derivatives
        d2 = df1.t * s;
        // new slope must be negative
        if (d2 > 0) {
          s := -df1; // otherwise use steepest direction
          d2 = -s.t * s;
        }

        z1 = z1 * min(RATIO, d1 / d2) // slope ratio but max RATIO
        d1 = d2
        ls_failed = false; // this line search did not fail

      } else {
        X := X0
        f1 = f0
        df1 := df0 // restore point from before failed line search
        if (ls_failed | i > length) {
          // line search failed twice in a row
          break // or we ran out of time, so we give up
        }
        val tmp = df1
        df1 := df2
        df2 := tmp // swap derivatives
        s := -df1; // try steepest
        d1 = -s.t * s;
        z1 = 1 / (1 - d1);
        ls_failed = true; // this line search failed        
      }
    }
    X
  }

}
