lazy val root = (project in file(".")).
  settings(
    name := "hello",
    version := "1.0",
    scalaVersion := "2.10.4",
    sbtVersion := "0.13",
    libraryDependencies ++= Seq(
      "org.apache.spark" % "spark-core_2.10" % "1.5.0",
      "org.apache.spark" % "spark-mllib_2.10" % "1.5.0",
      "com.github.fommil.netlib" % "all" % "1.1.2"
    ),
    mainClass in Compile := Some("de.arzt.berlin.ml.MovieLensExample")
  )
