package GBDT
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import scala.collection._
import breeze.linalg._
import scala.math._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils

object featurescore {
  def main(args: Array[String]){
    val Conf = new SparkConf().setAppName("tag").setMaster("local")
    val sc = new SparkContext(Conf)

    val path = "/Users/nali/Beifei/ximalaya2015/code_ximalaya/code_GBDT/input/"
    val play = sc.textFile(path+"uidAlbum1229_1230_1.csv")
    .map(s => s.split(","))
    .map(s => (s(0), s(1)))
    val users = play.map(s => s._1).distinct.collect
    val user2tag = sc.textFile(path+"part-00000")
    .map(s => s.split("\\s+"))
    .filter(s => users.contains(s(0)))
    .map(s => (s(0), s(1).dropRight(1).drop(1).split(",")))
    .map(s => (s._1, s._2.map(ss => ss.split(":")(0))))
    .collect.toMap
    
    val album2tag = sc.textFile(path+"albumTag.txt")
    .map(s => s.split(","))
    .map(s => (s(0), s.tail))
    .collect.toMap
    
    val tag3000 = sc.textFile(path+"tags3000.txt")
    .map(s => s.split(",")(0))
    .collect
    
    def vec(user: String, item: String) = {
      var vector = Array.empty[Int]
      if (user2tag.contains(user) & album2tag.contains(item)){
        val usertags = user2tag(user)
        val albumtags = album2tag(item)
        vector = tag3000.map{s=>
          if (usertags.contains(s) & albumtags.contains(s)){
            3
          }else if (!usertags.contains(s) & albumtags.contains(s)){
            2
          }else if (usertags.contains(s) & !albumtags.contains(s)){
            1
          }else{
            0
          }
        }
      }else{
        vector = Array.fill(3000)(0)
      }
      vector
    }

    val featurescore = sc.textFile(path+"sample2")
    .map(s => s.split(","))
    .map(s => (s(0), vec(s(0), s(1)), s(2)))
    .filter(s => s._2.sum != 0)
    .map(s => s._1+","+s._2.mkString(",")+","+s._3)
    
    featurescore.saveAsTextFile(path+"featurescore")
  }
}