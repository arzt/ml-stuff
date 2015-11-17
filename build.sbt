name := "ml-stuff2"

version := "1.0"

scalaVersion := "2.10.6"

sbtVersion := "0.13"

fork := false

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "1.5.1",
  "org.apache.spark" % "spark-mllib_2.10" % "1.5.1",
  "org.scalacheck" %% "scalacheck" % "1.12.2" % "test",
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "com.github.fommil.netlib" % "all" % "1.1.2",
  "org.scalatest" % "scalatest_2.10" % "2.2.5" % "test"
)

mainClass in(Compile, run) := Some("de.berlin.arzt.ml.MovieLensExample")

