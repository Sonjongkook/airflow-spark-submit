# pyspark
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, BooleanType, ArrayType, MapType
from pyspark.sql.functions import *


def spark_yelp_analysis(input_loc, output_loc):
    spark = SparkSession.builder.appName("yelp data analysis").getOrCreate()

    business_schema = StructType([
        StructField("business_id", StringType(), False),
        StructField("name", StringType(), False),
        StructField("address", StringType(), False),
        StructField("city", StringType(), False),
        StructField("state", StringType(), True),
        StructField("postal_code", StringType(), False),
        StructField("latitude", FloatType(), False),
        StructField("longitude", FloatType(), False),
        StructField("stars", FloatType(), False),
        StructField("review_count", IntegerType(), False),
        StructField("is_open", IntegerType(), False),
        StructField("attributes", MapType(StringType(), StringType(), False), False),
        StructField("categories", StringType(), False),
        StructField("hours", MapType(StringType(), StringType(), False), False)
    ])

    business_df_origin = spark.read.json(input_loc + "/yelp_academic_dataset_business.json", schema=business_schema)

    business_df_origin.createOrReplaceTempView("business_df_origin")
    spark.sql("SELECT COUNT(*) AS total_rows FROM business_df_origin").show()
    spark.sql("SELECT COUNT(DISTINCT business_id) AS distinct_business_id_count FROM business_df_origin").show()

    business_df_origin.printSchema()

    business_df_origin = business_df_origin.filter(col("categories").isNotNull() & col("business_id").isNotNull())

    business_df = business_df_origin.select(col("*"),
                                            explode(business_df_origin.attributes).alias("attribute_key",
                                                                                         "attribute_value"))

    @udf
    def extract_map_values(attribute_value):
        if attribute_value[0] == "{" and attribute_value[-1] == "}":
            dict = ast.literal_eval(attribute_value)
            attributes = ""
            for key, value in dict.items():
                if value == True:
                    attributes += (key + ",")
            return attributes[:-1]
        else:
            return

    business_df = business_df.select(
        col("*"),
        col("attribute_key").alias("outer_attribute"),
        col("attribute_value").alias("inner_attribute")
    ).withColumn(
        "inner_attribute_value",
        extract_map_values(col("inner_attribute"))
    )

    business_df = business_df.withColumn("inner_attribute", regexp_replace(col("inner_attribute"), "^u'(.*)'$", "$1"))
    business_df = business_df.withColumn("inner_attribute", regexp_replace(col("inner_attribute"), "^'(.*)'$", "$1"))

    business_df = business_df.filter(col("inner_attribute") != "False")

    business_df = business_df.withColumn("overall_attribute",
                                         when((col("inner_attribute_value").isNotNull()) & (
                                                     col("inner_attribute_value") != ""),
                                              concat(col("outer_attribute"), lit(":"), col("inner_attribute_value")))
                                         .otherwise(col("outer_attribute")))

    business_df = business_df.withColumn("overall_attribute",
                                         lower(col("overall_attribute")))

    business_df = business_df.drop("attributes", "attribute_key", "attribute_value", "outer_attribute",
                                   "inner_attribute", "inner_attribute_value")

    attribute_grouped_df = business_df.groupBy("business_id").agg(
        collect_list("overall_attribute").alias("overall_attributes"))

    hour_df = business_df_origin.select(business_df_origin.business_id, explode(business_df_origin.hours))
    hour_df = hour_df.withColumn("start_time", split(col("value"), "-").getItem(0)).withColumn("end_time",
                                                                                               split(col("value"),
                                                                                                     "-").getItem(1))
    hour_df = hour_df.withColumn("start_time_timestamp", to_timestamp(col("start_time"), "H:m")).withColumn(
        "end_time_timestamp", to_timestamp(col("end_time"), "H:m"))
    hour_df = hour_df.withColumn("duration_hours", (
                unix_timestamp(col("end_time_timestamp")) - unix_timestamp(col("start_time_timestamp"))) / 3600)
    hour_df = hour_df.withColumn("start_time", date_format(col("start_time_timestamp"), "HH:mm")).withColumn("end_time",
                                                                                                             date_format(
                                                                                                                 col("end_time_timestamp"),
                                                                                                                 "HH:mm"))
    hour_df = hour_df.drop("value", "start_time_timestamp", "end_time_timestamp")
    hour_df = hour_df.withColumn("duration_hours", abs(col("duration_hours")))
    hour_grouped_df = hour_df.groupBy("business_id").agg(sum("duration_hours").alias("total_business_hours"))

    business_df = business_df_origin.join(attribute_grouped_df, on="business_id", how="inner").join(hour_grouped_df,
                                                                                                    on="business_id",
                                                                                                    how="inner")

    business_df = business_df.withColumn("categories", lower(col("categories")))
    business_df = business_df.withColumn("categories", split("categories", ", "))

    business_df = business_df.drop("attributes", "hours")

    checkin_df = spark.read.json(input_loc + "/yelp_academic_dataset_checkin.json")

    checkin_df = checkin_df.withColumn("date_array", split(checkin_df["date"], ",\s*"))
    checkin_df = checkin_df.withColumn("date_array", split(checkin_df["date"], ",\s*"))
    checkin_df = checkin_df.select("business_id", explode(col("date_array")).alias("check_in_date"))

    check_in_df = checkin_df.groupBy("business_id") \
        .agg(count(col("check_in_date")).alias("check_in_count"),
             max(col("check_in_date")).alias("latest_checkout"),
             min(col("check_in_date")).alias("first_checkout")) \
        .orderBy("check_in_count", ascending=False)

    df_review = spark.read.json(input_loc + "/yelp_academic_dataset_review.json")

    df_tip = spark.read.json(input_loc + "/yelp_academic_dataset_tip.json")

    user_schema = StructType([
        StructField("user_id", StringType(), False),
        StructField("name", StringType(), True),
        StructField("review_count", IntegerType(), True),
        StructField("yelping_since", StringType(), True),
        StructField("friends", StringType(), True),
        StructField("useful", IntegerType(), True),
        StructField("funny", IntegerType(), True),
        StructField("cool", IntegerType(), True),
        StructField("fans", IntegerType(), True),
        StructField("elite", StringType(), True),
        StructField("average_stars", FloatType(), True),
        StructField("compliment_hot", IntegerType(), True),
        StructField("compliment_more", IntegerType(), True),
        StructField("compliment_profile", IntegerType(), True),
        StructField("compliment_cute", IntegerType(), True),
        StructField("compliment_list", IntegerType(), True),
        StructField("compliment_note", IntegerType(), True),
        StructField("compliment_plain", IntegerType(), True),
        StructField("compliment_cool", IntegerType(), True),
        StructField("compliment_funny", IntegerType(), True),
        StructField("compliment_writer", IntegerType(), True),
        StructField("compliment_photos", IntegerType(), True)
    ])

    df_user = spark.read.json(input_loc + "/yelp_academic_dataset_user.json", schema=user_schema)

    # Replace "20,20" with "2020" in the DataFrame column
    df_user = df_user.withColumn("elite", regexp_replace(col("elite"), "20,20", "2020"))
    # Split the string into an array using the delimiter ","
    df_user = df_user.withColumn("elite", split(col("elite"), ","))
    df_user = df_user.withColumn("friends", split(col("friends"), ", "))

    grouped_df_review = df_review.groupBy("user_id").agg(sum("cool").alias("total_cool"),
                                                         sum("stars").alias("total_stars"),
                                                         sum("useful").alias("total_useful"),
                                                         sum("funny").alias("total_funny"),
                                                         count("review_id").alias("total_review_counts"))

    grouped_df_review = grouped_df_review.withColumn("total_rate_value_received",
                                                     col("total_cool") + col("total_stars") + col("total_useful") + col(
                                                         "total_funny"))

    grouped_df_review = grouped_df_review.drop("total_cool", "total_stars", "total_useful", "total_funny")

    df_user = df_user.withColumn("total_rate_value_sent",
                                 col("useful") + col("funny") + col("cool"))

    df_user = df_user.withColumn("total_compliment_received",
                                 col("compliment_hot") + col("compliment_more") + col("compliment_profile")
                                 + col("compliment_cute") + col("compliment_list") + col("compliment_note")
                                 + col("compliment_plain") + col("compliment_cool") + col("compliment_funny")
                                 + col("compliment_writer") + col("compliment_photos"))

    df_user = df_user.withColumn("friends_number", size(col("friends")))

    df_user = df_user.select("user_id", "name", "yelping_since", "friends_number", "fans", "total_rate_value_sent",
                             "total_compliment_received")

    joined_df = df_user.join(grouped_df_review, on="user_id", how="inner")

    top_50_friends_user_df = joined_df.orderBy("friends_number", ascending=False).limit(50)

    top_50_friends_user_df.write.mode("overwrite").csv(output_loc + "/top50_users_friends_number.csv")


    joined_df_business_checkin = business_df.join(check_in_df, on="business_id", how="inner")

    joined_df_business_checkin = joined_df_business_checkin.select("business_id", "name", "state", "city", "is_open",
                                                                   "total_business_hours", "check_in_count",
                                                                   "latest_checkout", "first_checkout")

    top_20_best_checkin_business_df = joined_df_business_checkin.where(
        (col("is_open") == 1) & (col("total_business_hours") >= 40)).orderBy(col("check_in_count"),
                                                                             ascending=False).limit(20)

    top_20_best_checkin_business_df.write.mode("overwrite").csv(output_loc + "/top20_best_checkin_business.csv")


    businessid_grouped_df_review = df_review.groupBy("business_id").agg(sum("cool").alias("total_cool"),
                                                                        sum("stars").alias("total_stars"),
                                                                        sum("useful").alias("total_useful"),
                                                                        sum("funny").alias("total_funny"),
                                                                        countDistinct("review_id").alias(
                                                                            "total_users_reviewed"))

    businessid_grouped_df_review = businessid_grouped_df_review.withColumn("total_rate_value_received",
                                                                           col("total_cool") + col("total_stars") + col(
                                                                               "total_useful") + col("total_funny"))

    businessid_grouped_df_review = businessid_grouped_df_review.drop("total_cool", "total_stars", "total_useful",
                                                                     "total_funny")

    joined_df_review_business = businessid_grouped_df_review.join(business_df, on="business_id", how="inner")

    top_20_business_good_reviewed = joined_df_review_business \
        .select("business_id", "name", "state", "city", "is_open", "total_business_hours", "total_users_reviewed",
                "total_rate_value_received") \
        .orderBy(col("total_users_reviewed") + col("total_rate_value_received"), ascending=False) \
        .limit(20)

    top_20_business_good_reviewed.write.mode("overwrite").csv(output_loc + "/top20_business_good_reviewed.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="HDFS input", default="/yelp-data")
    parser.add_argument("--output", type=str, help="HDFS output", default="/output")
    args = parser.parse_args()
    spark_yelp_analysis(input_loc=args.input, output_loc=args.output)
