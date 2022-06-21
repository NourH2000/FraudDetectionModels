#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cassandra.cluster import Cluster
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

from pyspark.sql import SQLContext
import numpy as np
from pyspark.sql.functions import split, col


# In[3]:


#new spark session 
spark = SparkSession.builder.appName('PPA detection').getOrCreate()


# In[4]:


#connection to cassandra database and cnas keyspace
cluster = Cluster(['127.0.0.1'])
session = cluster.connect('cnas')


# In[6]:


#get parameters (start_date and end_date)
import sys

date_debut=sys.argv[1]
date_fin = sys.argv[2]

query = "SELECT *  FROM cnas  WHERE date_paiement >= '{}' AND date_paiement <= '{}' LIMIT 10  ALLOW FILTERING;".format(date_debut,date_fin)
rows = session.execute(query)
#print the data : print(rows)


# In[ ]:


#transform the cassandra.cluster ( rows) to pandas dataframe to make some changes
dftable = pd.DataFrame(list(rows))
# print the data : print (df)


# In[ ]:


# transformation :

#remplacer None avec -1 ( aucune affection) dans la coloumn affection
dftable.affection.fillna(value=-1, inplace=True)

#remplacer None avec 0 ( aucune quantitée rejetée ) dans la coloumn qte_rejet
dftable.qte_rejet.fillna(value=0, inplace=True)


# delete rows where the quantite_med == 0
dftable.drop(dftable[dftable['quantite_med'] == 0].index, inplace = True)

#delete rejected quantity , but before this , we need to save the rows having quantity rejected > 0
#df_rejected = dftable[dftable['qte_rejet'] > 0]
#dftable.drop(dftable[dftable['qte_rejet'] > 0].index, inplace = True)


#remplacer None avec 0 ( aucune durée spécifiée ) dans la coloumn duree_traitement
dftable.duree_traitement.fillna(value=0, inplace=True)

#change the type of some lines 
dftable = dftable.astype({"affection": str})
dftable = dftable.astype({"fk": float})
dftable = dftable.astype({"age": int})


# In[ ]:


# garder les coloumns qu'on est besoin 
dftable=dftable[['id','fk','codeps','affection','age','applic_tarif','date_paiement','num_enr','sexe','ts','quantite_med','qte_rejet']]
# print the columns that we need : dftable.info()


# In[ ]:


# split the table into two table : rejected one and accepted one
rejected = dftable[dftable['qte_rejet'] > 0]
accepted = dftable[dftable['qte_rejet'] == 0]


# In[ ]:


#Create spark dataframe for the two pandas table (accepted and rejected)
sparkdf = spark.createDataFrame(accepted)
rejected_sparkdf = spark.createDataFrame(rejected)


# In[ ]:


#transform the affection column to array of int ( splited by ',')
sparkdf = sparkdf.withColumn("affection", split(col("affection"), ",").cast("array<int>"))


# In[ ]:


#sort the affection array 
import pyspark.sql.functions as F
sparkdf = sparkdf.withColumn('affection', F.array_sort('affection'))


# In[ ]:


## put the age in ranges
from pyspark.sql.functions import udf
@udf("String")
def age_range(age):
    if age >= 0 and age <= 5:
        return '0-5'
    elif age > 5 and age <= 10:
        return '6-10'
    elif age > 10 and age <= 16:
        return '11-16' 
    elif age > 16 and age <= 24:
        return '17-24' 
    elif age > 24 and age <= 60:
        return '25-60' 
    elif age > 60 and age <= 76:
        return '61-76' 
    else:
        return '75+'
    


sparkdf = sparkdf.withColumn("age", age_range(col("age")))


# In[ ]:


# transform the affection column to a string again ( so we can index it)
from pyspark.sql.functions import col, concat_ws
sparkdf = sparkdf.withColumn("affection",
   concat_ws(",",col("affection")))


# In[ ]:


### Handling Categorical Features
from pyspark.ml.feature import StringIndexer
indexer=StringIndexer(inputCols=["sexe","applic_tarif","ts","affection","age"],outputCols=["sex_indexed","applic_tarif_indexed",
                                                                         "ts_indexes","affection_indexes","age_indexes"])
df_r=indexer.setHandleInvalid("keep").fit(sparkdf).transform(sparkdf)


# In[ ]:


# put the data into one vector 
from pyspark.ml.feature import VectorAssembler
featureassembler=VectorAssembler(inputCols=['id','fk','age_indexes','sex_indexed','affection_indexes',
                          'ts_indexes','quantite_med',],outputCol="Independent Features")
output=featureassembler.transform(df_r)


# In[ ]:


#prepare the data to fit it to the model 
finalized_data=output.select("Independent Features","quantite_med")


# In[ ]:


# call and fit the model 
from pyspark.ml.regression import LinearRegression
##train test split
train_data,test_data=finalized_data.randomSplit([0.75,0.25])
regressor=LinearRegression(featuresCol='Independent Features', labelCol='quantite_med', maxIter=10, regParam=0.3, elasticNetParam=0.8)
regressor=regressor.fit(train_data)


# In[ ]:


# result
print("Coefficients: %s" % str(regressor.coefficients))
print("Intercept: %s" % str(regressor.intercept))


# In[ ]:


### Predictions
pred_results=regressor.evaluate(test_data)
## Final comparison
# print the prediction vs the vector : pred_results.predictions.show()


# In[ ]:


### Performance Metrics
pred_results.r2,pred_results.meanAbsoluteError,pred_results.meanSquaredError


# In[ ]:


# print the final predicted quantity ( rounded ) vs the real quantity
Final_result = pred_results.predictions.where("quantite_med > prediction ")

from pyspark.sql.functions import round, col
Final_result = Final_result.select("Independent Features"  , "quantite_med", round(col('prediction')))
Final_result.show(50)

