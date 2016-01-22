package GBDT
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import scala.collection._
import breeze.linalg._
import scala.math._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils

object sample {
  def main(args: Array[String]){
    val Conf = new SparkConf().setAppName("tag")
    Conf.set("spark.hadoop.fs.default.name", "hdfs://master1.hh:8020")
    val sc = new SparkContext(Conf)  
    val path1 = "hdfs://master1.hh:8020/tmp/beifei.zhou/logit/"
    val path2 = "hdfs://master1.hh:8020/data/recsys/2015/12/29/relativeAlbumJob_1451324400017/"
    val play = sc.textFile(path1+"uidAlbum1229_1230_1.csv")
    .map(s => s.split(","))
    .map(s => (s(0), s(1)))
    val users = play.map(s => s._1).distinct.collect
    val albums = play.map(s => s._2).distinct.collect
    val playUsrItems = play.groupBy(s => s._1)
    .map(s => (s._1, s._2.toList.map(ss => ss._2)))
    
    
    val fileList = List("cfResult_cat_1-4-8", "cfResult_cat_11-21-23-24", 
        " cfResult_cat_2-10-17-20", "cfResult_cat_3-12-15-16", 
        "cfResult_cat_5-6", "cfResult_cat_7-13-18", "  cfResult_cat_9-14-22")
    var reco = sc.textFile(path2+fileList(0))
    .map(s => s.split("\\s+"))
    .filter(s => users.contains(s(0)))
    .map(s => (s(0), s(1).dropRight(1).drop(1).split(",")))
    .map(s => (s._1, s._2.map(ss => ss.split(":")(0))))
    
    for (i <- 1 to fileList.size-1){
      val each = sc.textFile(path2+fileList(i))
      .map(s => s.split("\\s+"))
      .filter(s => users.contains(s(0)))
      .map(s => (s(0), s(1).dropRight(1).drop(1).split(",")))
      .map(s => (s._1, s._2.map(ss => ss.split(":")(0))))
      reco = reco ++ each
    }
    
    val sample = playUsrItems.cogroup(reco)
    .map(s => (s._1, s._2._1.toList, s._2._2.toList))
    .filter(s => s._2.size !=0 & s._1.size != 0)
    .map(s => (s._1, s._2(0).toList, s._3(0).toList))
    
    def getposneg(x: List[String], y: List[String]) = {
      //if x intersects y
      val inter = x.intersect(y)
      var neg = List.empty[String]
      var pos = List.empty[String]
      if (inter.size == 0){
        neg = List(y(0))
        pos = List(x(x.size -1))
      }else{
        val indexes = inter.map(s => y.indexOf(s))
        val shift = List(0) ++ indexes.take(inter.size-1)
        val diff = indexes.zip(shift).map(s => s._1-s._2)
        val nonone = diff.filter(s => s != 1)
        if (nonone.size != 0){
          val indexpos = nonone.map(s => diff(s))
          pos = indexpos.map(s => y(s))
          neg = indexpos.map(s => y(s-1))
        }
      }
      (pos, neg)
    }
    val result = sample.map(s => (s._1, getposneg(s._2, s._3)))
    .filter(s => s._2._1.size != 0 || s._2._2.size != 0)
    .map(s => s._2._1.map(ss => (s._1, ss, 1)) ++ s._2._2.map(ss => (s._1, ss, -1)))
    .flatMap(s => s)
    .map(s => s._1+","+s._2+","+s._3)
    
    result.saveAsTextFile(path1+"sample2")
  }
}