import sbt._
import Process._
import sbt.Keys._


lazy val commonSettings = Seq(
  organization := "com.untrackedfiles",
  version := "0.1.0",
  scalaVersion := "2.11.8"
)


lazy val untrackedfiles = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "untrackedfiles",
    libraryDependencies ++= Seq(
      "org.specs2" % "specs2-core_2.11" % "3.6.5",
      "com.typesafe.scala-logging" %% "scala-logging" % "3.1.0",
      "org.slf4j" % "slf4j-api" % "1.7.12",
      "org.slf4j" % "slf4j-simple" % "1.7.12"
    )
  )