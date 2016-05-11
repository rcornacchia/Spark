"""
run.py
Michael Lin (mbl2109@columbia.edu)
Philippe-Guillaume Losembe (pvl2109@columbia.edu)
Robert Cornacchia (rlc2160@columbia.edu)
Oriana Fuentes (oriana.i.fuentes@columbia.edu)

Recommender to be used with an AWS EMR Spark cluster
Outputs 10 recommendations for user id: 2093760

Note: hw3part1.ipynb has the same functionality, and was what we used for development
"""

from pyspark.mllib.recommendation import *
from pyspark import SparkContext

sc = SparkContext()

rawUserArtistData = sc.textFile("s3://aws-logs-340211587520-us-east-1/data/user_artist_data.txt")
rawArtistData = sc.textFile("s3://aws-logs-340211587520-us-east-1/data/artist_data.txt")
rawArtistAlias = sc.textFile("s3://aws-logs-340211587520-us-east-1/data/artist_alias.txt")

def getArtistByID(line):
    try:
        (artistID, name) = line.split('\t',1)
        artistID = int(artistID)
    except:
        return []
    if not name:
        return []
    else:
        return [(artistID, name.strip())]

artistByID = rawArtistData.flatMap(getArtistByID)

def tokenCheck(line):
    (id1, id2) = line.split('\t',1)
    if not id1:
        return []
    else:
        return [(int(id1), int(id2))]
    
artistAlias = rawArtistAlias.flatMap(tokenCheck).collectAsMap()

bArtistAlias = sc.broadcast(artistAlias)

def createRating(line):
    (userID, artistID, count) = line.split(' ')
    finalArtistID = bArtistAlias.value.get(artistID)
    if not finalArtistID:
        finalArtistID = artistID
    return Rating(int(userID), int(finalArtistID), int(count))
    
trainData = rawUserArtistData.map(createRating).cache()

model = ALS.trainImplicit(trainData, rank=10, iterations=5, lambda_=0.01, alpha=1.0)

def userFilter(line):
    (user, artist, playcount) = line
    return (int(user) == 2093760)
rawArtistsForUser = rawUserArtistData.map(lambda l: l.split(' ')).filter(userFilter)

def artistToInt(line):
    (user, artist, playcount) = line
    return int(artist)
existingProducts = rawArtistsForUser.map(artistToInt).collect()

def filterForArtistID(line):
    (id_, name) = line
    return id_ in existingProducts

# function recommendProducts copied from 
# https://spark.apache.org/docs/1.5.1/api/python/_modules/pyspark/mllib/recommendation.html 
def recommendProducts(self, user, num):
        """
        Recommends the top "num" number of products for a given user and returns a list
        of Rating objects sorted by the predicted rating in descending order.
        """
        return list(self.call("recommendProducts", user, num))
recommendations = recommendProducts(model, 2093760, 10)
recommendedProductIDs = map(lambda (userID, productID, rating): productID, recommendations)

def filterForRecommendedIDs(line):
    (artistID, name) = line
    return artistID in recommendedProductIDs
recommendedProducts = artistByID.filter(filterForRecommendedIDs).values().collect()
for p in recommendedProducts: print p



