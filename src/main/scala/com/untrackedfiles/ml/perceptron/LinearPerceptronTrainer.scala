package com.untrackedfiles.ml.perceptron

import scala.annotation.tailrec
import com.typesafe.scalalogging.LazyLogging

case class LabeledSample(x: Array[Double], label: Int)

case class Weights(w: Array[Double], b: Double)

class LinearPerceptronTrainer(eta: Double, epochs: Int) extends LazyLogging {

  def train(data: List[LabeledSample]): (Weights, Int) = {
    val stamp: Long = System.currentTimeMillis()
    val dim = data.head.x.size
    val radius = resolveRadius(data)
    val (weights, errors) = iterate(epochs, data, Weights(Array.ofDim[Double](dim), 0.0d), radius)
    val elapsed: Long = System.currentTimeMillis() - stamp

    logger.info(s"Finished in ${elapsed} ms (weights: [${weights.w.mkString(",")}, ${weights.b}], errors: $errors)")
    (weights, errors)
  }

  @tailrec
  final def iterate(epoch: Int, data: List[LabeledSample], weights: Weights, r: Double): (Weights, Int) = {
    val (adjWeights, errors) = trial(data, weights, r)
    logger.info(s"epoch ${epochs - epoch + 1} : (weights: [${weights.w.mkString(",")}, ${weights.b}], errors: $errors)")
    if (epoch == 0 || errors == 0) {
      (adjWeights, errors)
    } else {
      iterate(epoch - 1, data, adjWeights, r)
    }
  }

  def trial(data: List[LabeledSample], weights: Weights, r: Double): (Weights, Int) = {
    data.foldLeft(weights, 0) { (acc, s) =>
      val p = innerProduct(s.x, weights.w) + weights.b
      logger.trace(s"trial : (sample: ${s.x.mkString(",")}, " +
        s"weights: [${weights.w.mkString(",")}, ${weights.b}] ) => $p")

      if (p * s.label <= 0) {
        (Weights(acc._1.w.zip(s.x).map(w => w._1 + eta * s.label * w._2),
          acc._1.b + (eta * s.label * r * r)), acc._2 + 1)
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
