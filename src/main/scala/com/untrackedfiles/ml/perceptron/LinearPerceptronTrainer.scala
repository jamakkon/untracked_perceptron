package com.untrackedfiles.ml.perceptron

import scala.annotation.tailrec
import com.typesafe.scalalogging.LazyLogging

import scala.io.Source

case class LabeledSample(x: Array[Double], label: Int) {
  val length: Double = Weights.length(x)
}

case class Weights(w: Array[Double], b: Double) {
  val length: Double = Weights.length(this)
}

object Weights {

  def dontNormalize(weights: Weights): Weights = weights

  def normalize(weights: Weights): Weights = {
    val newWeights: Array[Double] = weights.w.map(_ / weights.length)
    val newBias: Double = weights.b / weights.length

    Weights(newWeights, newBias)
  }

  def length(weights: Weights): Double = {
    math.sqrt((weights.w :+ weights.b).foldLeft(0.0d)((z, i) => z + i * i))
  }

  def length(x: Array[Double]): Double = {
    math.sqrt(x.foldLeft(0.0d)((z, i) => z + i * i))
  }

}

class LinearPerceptronTrainer(eta: Double,
                              epochs: Int,
                              normalization: (Weights => Weights) = Weights.normalize) extends LazyLogging {

  def train(data: List[LabeledSample]): (Weights, Int) = {
    val stamp: Long = System.currentTimeMillis()
    val dim = data.head.x.size
    val radius = resolveRadius(data)
    val (weights, errors) = iterate(epochs, data, Weights(Array.ofDim[Double](dim), 0.0d), radius,
      (Weights(Array.ofDim[Double](dim), 0.0d), data.size), normalization)
    val elapsed: Long = System.currentTimeMillis() - stamp

    logger.info(s"Finished in ${elapsed} ms (weights: [${weights.w.mkString(",")}, ${weights.b}], errors: $errors)")
    (weights, errors)
  }

  @tailrec
  final def iterate(epoch: Int,
                    data: List[LabeledSample],
                    weights: Weights,
                    radius: Double,
                    pocket: (Weights, Int),
                    normalization: (Weights => Weights)): (Weights, Int) = {
    val (adjWeights, errors) = {
      val trialResults = trial(data, weights, radius)

      (normalization(trialResults._1), trialResults._2)
    }
    val newPocket = if (errors < pocket._2) {
      (adjWeights, errors)
    } else {
      pocket
    }

    logger.info(s"epoch ${epoch} : (weights: [${weights.w.mkString(",")}, ${weights.b}], errors: $errors)")
    if (errors == 0) {
      (adjWeights, errors)
    } else if (epoch == 0) {
      pocket
    } else {
      iterate(epoch - 1, data, adjWeights, radius, newPocket, normalization)
    }
  }

  def trial(data: List[LabeledSample], weights: Weights, radius: Double): (Weights, Int) = {
    data.foldLeft(weights, 0) { (acc, s) =>
      val p = innerProduct(s.x, weights.w) + weights.b
      logger.trace(s"trial : (sample: ${s.x.mkString(",")}, weights: [${weights.w.mkString(",")}, ${weights.b}] ) => $p")

      if (p * s.label <= 0) {
        (Weights(
          acc._1.w.zip(s.x).map(w => w._1 + (eta * s.label * w._2)),
          acc._1.b + (eta * s.label)
        ),
          acc._2 + 1)
      } else {
        acc
      }
    }
  }

  def innerProduct(w: Array[Double], v: Array[Double]): Double = {
    assert(w.size == v.size, s"Mismatching dimensions. (${w.size} vs. ${v.size})")
    w.zip(v).foldLeft(0.0d)((z, i) => z + i._1 * i._2)
  }

  def resolveRadius(data: List[LabeledSample]): Double = {
    data.map(x => math.sqrt(innerProduct(x.x, x.x))).max
  }

}

object LinearPerceptronTrainer {

  def readLabeled2DData(src: Source, separator: String = "\t"): List[LabeledSample] = {
    src.getLines().map(_.split(separator)).map {
      case Array(x1, x2, y) => LabeledSample(Array[Double](x1.toDouble, x2.toDouble), y.toInt)
    }.toList
  }

  def translateLabels(samples: List[LabeledSample]): List[LabeledSample] = {
    samples.map { sample =>
      val label = if (sample.label > 0) {
        1
      } else {
        -1
      }

      LabeledSample(sample.x, label)
    }
  }


}