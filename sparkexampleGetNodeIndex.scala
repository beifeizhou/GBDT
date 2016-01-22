package GBDT
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
//import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.model.Node

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.Logging
import org.apache.spark.mllib.tree.configuration.FeatureType._
import org.apache.spark.mllib.linalg.Vector


import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import scala.collection._
import breeze.linalg._
import scala.math._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import scala.collection.immutable.Map

/**
 * @author nali
 */
object sparkexampleGetNodeIndex {
  def main(args: Array[String]){
    val Conf = new SparkConf().setAppName("tag").setMaster("local")
    val sc = new SparkContext(Conf)
    val data = MLUtils.loadLibSVMFile(sc, "/Users/nali/Softwares/spark/data/mllib/sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    
    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 9
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
   
    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)   

    def predictIndex(aNode: Node, features: Vector) : Int = {
      if (aNode.isLeaf) {  
        aNode.id
      } else{
        if (aNode.split.get.featureType == Continuous) {
          if (features(aNode.split.get.feature) <= aNode.split.get.threshold) {
            predictIndex(aNode.leftNode.get, features)
          } else {
            predictIndex(aNode.rightNode.get, features)
          }
        } else {
          if (aNode.split.get.categories.contains(features(aNode.split.get.feature))) {
            predictIndex(aNode.leftNode.get, features)
          } else {
            predictIndex(aNode.rightNode.get, features)
          }
        }
      }
    }  
   
    val nodes = model.trees.map{s => s.topNode}

    val labelAndPreds = testData.map { point =>
      val feature = point.features
      nodes.map(s => predictIndex(s, feature)).toList
    }
    labelAndPreds.take(10).foreach(println)
  }
}