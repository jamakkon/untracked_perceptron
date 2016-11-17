package com.untrackedfiles.ml.perceptron

import org.specs2.mutable.Specification

class LinearPerceptronSpec extends Specification {

  "LinearPerceptron" should {

    val data = List(
      LabeledSample(Array[Double](0.0d, 2.0d), -1),
      LabeledSample(Array[Double](1.0d, 2.0d), -1),
      LabeledSample(Array[Double](2.0d, 1.0d), -1),
      LabeledSample(Array[Double](2.0d, 5.0d), 1),
      LabeledSample(Array[Double](3.0d, 3.0d), 1),
      LabeledSample(Array[Double](4.0d, 3.0d), 1)
    )

    def testInnerProduct(w: Array[Double], v: Array[Double], expected: Double) = {
      val perceptron = new LinearPerceptronTrainer(0.01d, 10)

      perceptron.innerProduct(w, v) must beCloseTo(expected, 0.001d)
    }

    def testTrial(data: List[LabeledSample], weights: Weights, expectedErrors: Int) = {
      val perceptron = new LinearPerceptronTrainer(0.01d, 10)
      val radius = perceptron.resolveRadius(data)

      perceptron.trial(data, weights, radius)._2 mustEqual expectedErrors
    }

    def testTrain(data: List[LabeledSample], expectedErrors: Int) = {
      val perceptron = new LinearPerceptronTrainer(0.001d, 1000)

      perceptron.train(data)._2 mustEqual expectedErrors
    }

    "compute inner-product" in {
      List(
        (Array[Double](), Array[Double](), 0.0d),
        (Array[Double](1.0d), Array[Double](1.0d), 1.0d),
        (Array[Double](1.0d, 1.0d), Array[Double](1.0d, 1.0d), 2.0d),
        (Array[Double](1.0d, 0.0d), Array[Double](0.0d, 1.0d), 0.0d)
      ).map(p => testInnerProduct(p._1, p._2, p._3))
    }

    "resolve radius" in {
      val perceptron = new LinearPerceptronTrainer(0.01d, 10)
      perceptron.resolveRadius(data) must beCloseTo(math.sqrt(4.0d + 25.0d), 0.001d)
    }

    "trial given weights" in {
      List(
        (Array[Double](0.0d, 0.0d), 0.0d, 6),
        (Array[Double](0.0d, 0.0d), 1.0d, 3),
        (Array[Double](0.0d, 0.0d), -1.0d, 3)
      ).map(p => testTrial(data, Weights(p._1, p._2), p._3))
    }

    "learn classifier" in {
      testTrain(data, 0)
    }

  }


}
