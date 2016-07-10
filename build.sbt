name := "ml-stuff2"

version := "1.0"

scalaVersion := "2.11.7"

sbtVersion := "0.13.+"

fork := false

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.+",
  "org.apache.spark" %% "spark-mllib" % "1.6.+",
  "org.scalacheck" %% "scalacheck" % "1.12.2" % "test",
  "com.github.fommil.netlib" % "all" % "1.1.2+",
  "org.scalatest" %% "scalatest" % "2.2.+" % "test",
  "org.scalaz" %% "scalaz-core" % "7.2.+"
)

mainClass in(Compile, run) := Some("de.berlin.arzt.ml.Main")

