package de.berlin.arzt.ml

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import breeze.linalg._

/**
  * Created by realdocx on 06.11.15.
  */
object PicTest {

  def maxColor(i: Int) = {
    val r = red(i)
    val g = green(i)
    val b = blue(i)
    val (ma, mi) = maxmin(r, b, g)
    val diff = ma - mi
    if (diff == 0) {
      i
    } else {
      val nr = (255*(r - mi)/diff.toDouble).toInt
      val ng = (255*(g - mi)/diff.toDouble).toInt
      val nb = (255*(b - mi)/diff.toDouble).toInt
      rgba(nr, ng, nb)
    }
  }

  def red(i: Int) = i & 255
  def green(i: Int) = (i >> 8) & 255
  def blue(i: Int) = (i >> 16) & 255
  def alpha(i: Int) = (i >> 24) & 255
  def rgba(r: Int, g: Int, b: Int) = r + (g << 8) + (b << 16) + (255 << 24)

  def maxmin(a: Int, b: Int, c: Int) =
    if (a > b) {
      if (b > c)
        (a, c)
      else
        (a, b)
    } else {
      if (a > c)
        (b, c)
      else
        (b, a)
    }

  def intToRgba(p: Int) =
    new DenseVector[Int](
      Array(
        p & 255,
        (p >> 8) & 255,
        (p >> 16) & 255,
        (p >> 24) & 255
      )
    )

  def rgbaToThreeD(rgba: DenseVector[Int]) =
    rgba(0 to 2).map {
      x => x / 255.0
    }

  def threeDtoRgba(threeD: DenseVector[Double]) = {
    threeD *= 255.0
    val o = threeD.map(_.toInt)
    DenseVector(Array(o(0), o(1), o(2), 255))
  }


  def rgbaToInt(v2: DenseVector[Int]) =
    v2(0) + (v2(1) << 8) + (v2(2) << 16) + (v2(3) << 24)


  def vecToInt(v: DenseVector[Double]) = {
    val v2 = v.map(x => (x * 255).toInt)
    v2(0) + (v2(1) << 8) + (v2(2) << 16) + (v2(3) << 24)
  }

  //val file = "/home/realdocx/Dropbox/musik-buffer/Grimes - 2015 - Art Angels/cover.jpg"
  val file = "/home/realdocx/Bilder/Wallpaper/Interpol Marquee.jpg"
  println("loading image")
  val img = ImageIO.read(new File(file))
  val ocols = img.getWidth
  val orows = img.getHeight
  println(s"Pixelcount=${ocols * orows}")
  val cols = ocols
  val rows = orows
  val offcols = 0
  val offrows = 0
  println("creating array")
  val data = new Array[Int](cols * rows)
  img.getRGB(offcols, offrows, cols, rows, data, 0, cols)
  /*println("creating matrix")
  val image = new DenseMatrix[Int](cols, rows, data)
  println("map1")
  val test = data.map { i =>
    //println(i)
    rgbaToThreeD(intToRgba(i))
  }
  println("map2")
  val test3 = test.map { i =>
    val mi = min(i)
    val ma = max(i) - mi
    if (ma == 0) {
      i
    } else {
      i -= mi
      i /= ma
    }
  }
  println("m3p1")*/

  val resu = new Array[Int](data.length)

  import System.currentTimeMillis
  var time0 = currentTimeMillis
  println("run1")
  //val test2 = data.map(maxColor)
  val test2 = new Array[Int](data.length)
  for (i <- 0 until data.length) {
    if (i % 100000 == 0) println(i)
    test2(i) = maxColor(data(i))
  }
  println("done")

  val out = new BufferedImage(cols, rows, img.getType)
  out.setRGB(0, 0, cols, rows, test2.toArray, 0, cols)
  val res = ImageIO.write(out, "png", new File(file + ".png"))

  def main(args: Array[String]): Unit = {}
}
