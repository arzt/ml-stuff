name := "ml-stuff"

scalaVersion := "2.11.4"

libraryDependencies ++= Seq(
  "org.scalacheck" %% "scalacheck" % "1.12.2" % "test",
  "org.apache.spark" % "spark-core_2.11" % "1.5.1",
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "com.github.fommil.netlib" % "all" % "1.1.2"
)

mainClass in(Compile, run) := Some(
  "de.berlin.arzt.wordlist.Wordlist"
)
