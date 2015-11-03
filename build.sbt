lazy val root = (project in file(".")).
  settings(
    name := "ml-stuff",
    version := "1.0",
    scalaVersion := "2.11.4",
    sbtVersion := "0.13",
    libraryDependencies ++= Seq(
      "org.apache.spark" % "spark-core_2.11" % "1.5.1",
      "org.apache.spark" % "spark-mllib_2.11" % "1.5.0",
      "org.scalacheck" %% "scalacheck" % "1.12.2" % "test",
      "org.scalanlp" %% "breeze" % "0.11.2",
      "org.scalanlp" %% "breeze-natives" % "0.11.2",
      "com.github.fommil.netlib" % "all" % "1.1.2"
    ),
    mainClass in(Compile, run) := Some("de.arzt.berlin.ml.MovieLensExample")
  )
