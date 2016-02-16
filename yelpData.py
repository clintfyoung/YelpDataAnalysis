import json
import math
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from random import randint
import numpy as np
from scipy.misc import imread
import matplotlib.cbook as cbook
import re
from matplotlib import collections as mc

LVFile = cbook.get_sample_data('/home/youngc/dataIncubator/LasVegasMap.png') #I hope this is fair use. I got it from http://www.openstreetmaps.org
img = imread(LVFile)

#The ranges of latitudes and longitudes for the map of Las Vegas:
MinLVLat = 35.8774
MaxLVLat = 36.3793
MinLVLong = -115.6091
MaxLVLong = -114.7028

REarth = 6371. #Earth radius in kilometers

LVBusinessDict = {}

#Ranges of latitudes and longitudes of Las Vegas, unfortunately a city I know little about:
minLong = -115.55
maxLong = -114.9
minLat = 35.95
maxLat = 36.4

NLong = 80
NLat  = 80
#NLong = 40
#NLat  = 20

dLat = (maxLat-minLat)/NLat
dLong = (maxLong-minLong)/NLong

LVOpenBusinessesLat = []
LVOpenBusinessesLong = []
LVClosedBusinessesLat = []
LVClosedBusinessesLong = []
starsOfOpenDistribution = [0. for i in range(0, 11)]
starsOfClosedDistribution = [0. for i in range(0, 11)]

#This will make a heat map of the reviews of restaurants:
starsFood = [[0. for i in range(0, NLong)] for j in range(0, NLat)]
numberOfReviewsFood = [[0. for i in range(0, NLong)] for j in range(0, NLat)]
reviewsDistribution = [[0. for i in range(0, NLong)] for j in range(0, NLat)]
averageReviewDistribution = [[0. for i in range(0, NLong)] for j in range(0, NLat)]
goodReviewThreshold = 2.5 #The midpoint for the range seems like a good guess to classify the good and bad reviews.
badReviewDistribution = [[0. for i in range(0, NLong)] for j in range(0, NLat)] #A distribution of restaurants with bad reviews

fractionClosed = [[0. for i in range(0, NLong)] for j in range(0, NLat)]
totalDist = [[0. for i in range(0, NLong)] for j in range(0, NLat)]

latitudeList = []
longitudeList = []

#This function does a number of tasks: first, it adds restaurants in Las Vegas to a dictionary for 
#future reference. Then, it bins the latitudes and longitudes of these restaurants into a list, for 
#some of the simple analyses.
def binRestaurant(lat, longitude, stars, reviews, Open, businessID):
    latBin = int(math.floor((lat-minLat)/dLat))
    longBin = int(math.floor((longitude-minLong)/dLong))

    if (latBin>=0 and longBin>=0 and latBin<NLat and longBin<NLong):
        LVBusinessDict[businessID] = [lat, longitude]

        starsFood[latBin][longBin] += stars*reviews
        numberOfReviewsFood[latBin][longBin] += reviews
        starsBin = int(math.floor(2.*stars))
        if Open:
            LVOpenBusinessesLat.append(lat)
            LVOpenBusinessesLong.append(longitude)
            starsOfOpenDistribution[starsBin] += 1.
            totalDist[latBin][longBin] += 1.
        else:
            LVClosedBusinessesLat.append(lat)
            LVClosedBusinessesLong.append(longitude)
            starsOfClosedDistribution[starsBin] += 1.
            fractionClosed[latBin][longBin] += 1.
            totalDist[latBin][longBin] += 1.

    return


#This function simply bins the reviews of restaurants in Las Vegas:
def binReviews(lat, longitude, stars):
    latBin = int(math.floor((lat-minLat)/dLat))
    longBin = int(math.floor((longitude-minLong)/dLong))

    if (latBin>=0 and longBin>=0 and latBin<NLat and longBin<NLong):
        reviewsDistribution[latBin][longBin] += 1.
        averageReviewDistribution[latBin][longBin] += stars
        if stars<goodReviewThreshold:
            badReviewDistribution[latBin][longBin] += 1.

    return


counter = 0
businesses = open("yelp_academic_dataset_business.json", 'rb')
for line in businesses:
    counter += 1
    jsonLine = json.loads(line)
    Open = bool(jsonLine['open'])

    #if "Fast Food" in jsonLine['categories']:
    if "Restaurants" in jsonLine['categories']:
        latitude = float(jsonLine['latitude'])
        longitude = float(jsonLine['longitude'])
        latitudeList.append(latitude)
        longitudeList.append(longitude)
        stars = float(jsonLine['stars'])
        reviews = float(jsonLine['review_count'])
        businessID = str(jsonLine['business_id'])
        binRestaurant(latitude, longitude, stars, reviews, Open, businessID)

print "The total number of businesses =", counter
weightedAverageOfReviewsFood = [['null' for j in range(NLong)] for i in range(NLat)]
for i in range(NLat):
    for j in range(NLong):
        if numberOfReviewsFood[i][j]>0.:
            weightedAverageOfReviewsFood[i][j] = starsFood[i][j]/numberOfReviewsFood[i][j]

nonzeroReviews = []

#For sorting:
def lastElement(s):
    return s[-1]

for i in range(NLat):
    for j in range(NLong):
        if weightedAverageOfReviewsFood[i][j] != 'null':
            nonzeroReviews.append([minLat+(i+0.5)*dLat, minLong+(j+0.5)*dLong, numberOfReviewsFood[i][j], weightedAverageOfReviewsFood[i][j]])

fig, ax = plt.subplots()
# add some text for labels, title and axes ticks                                                                                                                                               
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
ax.set_title('Open and closed restaurants')

plt.imshow(img, zorder=0, extent=[MinLVLong, MaxLVLong, MinLVLat, MaxLVLat])
plt.scatter(LVOpenBusinessesLong, LVOpenBusinessesLat, c='blue', label='In business')
plt.scatter(LVClosedBusinessesLong, LVClosedBusinessesLat, c='red', label='Out of business')
ax.legend()

plt.show()

totalOpen = 0.
totalClosed = 0.
averageStarsOpen = 0.
averageStarsClosed = 0.
for i in range(0, 11):
    totalOpen += starsOfOpenDistribution[i]
    totalClosed += starsOfClosedDistribution[i]
    averageStarsOpen += (1.*i)*starsOfOpenDistribution[i]/2.
    averageStarsClosed += (1.*i)*starsOfClosedDistribution[i]/2.
    
print "The average number of stars for businesses still open =", averageStarsOpen/totalOpen
print "The average number of stars for businesses which are out of business =", averageStarsClosed/totalClosed

for i in range(0, NLat):
    for j in range(0, NLong):
        if totalDist[i][j]>5.: #There need to be a few businesses to have reasonable statistics.
            fractionClosed[i][j] /= totalDist[i][j]
        else:
            fractionClosed[i][j] = 0.

unsortedFraction = []
for i in range(0, NLat):
    for j in range(0, NLong):
        if(fractionClosed[i][j] != 0.):
            unsortedFraction.append([NLat, NLong, fractionClosed[i][j]])

sortedFraction = sorted(unsortedFraction, key=lastElement)
print "The location with the largest fraction of closed businesses =", MinLVLat+dLat*sortedFraction[-1][0], ",", MinLVLong+dLong*sortedFraction[-1][1], "with a fraction =", sortedFraction[-1][2]
print "The location with the second largest fraction of closed businesses =", MinLVLat+dLat*sortedFraction[-2][0], ",", MinLVLong+dLong*sortedFraction[-2][1], "with a fraction =", sortedFraction[-2][2]
print "The location with the third largest fraction of closed businesses =", MinLVLat+dLat*sortedFraction[-3][0], ",", MinLVLong+dLong*sortedFraction[-3][1], "with a fraction =", sortedFraction[-3][2]

counter = 0.
bestCounter = 0.
worstCounter = 0.
reviews = open("yelp_academic_dataset_review.json", 'rb')
for line in reviews:
    counter += 1.
    jsonLine = json.loads(line)

    businessID = str(jsonLine['business_id'])
    if businessID in LVBusinessDict:
        stars = jsonLine['stars']
        [lat, longitude] = LVBusinessDict[businessID]
        binReviews(lat, longitude, stars)

#print badReviewDistribution

#We now will cluster the restaurants with bad reviews, to find good locations for a new business:
#With the data binned, we now can try K-clustering of the trips in this 4-dimensional space. This may be challenging. To start, 10 clusters are randomly initialized:
NClusters = 15
KClusters = []
for i in range(0, NClusters):
    KClusters.append([randint(0, NLat), randint(0, NLong)])

NewKClusters = []
for i in range(0, NClusters):
    NewKClusters.append([0., 0.])

WhichCluster = [[0 for i in range(0, NLong)] for j in range(0, NLat)]

def ClustersUnchanged():
    metric = 0.
    for i in range(0, NClusters):
        for j in range(0, 2):
            metric += abs(NewKClusters[i][j]-KClusters[i][j])
    if metric<1e-5:
        return True

unchanged = False
while(not unchanged):
    #First, determine to which cluster each chunk of the distribution belongs:
    for i in range(0, NLat):
        for j in range(0, NLong):
            minDistance2 = (i-KClusters[0][0])**2 + (j-KClusters[0][1])**2
            WhichCluster[i][j] = 0
            for clusterI in range(1, NClusters):
                distance2 = (i-KClusters[clusterI][0])**2 + (j-KClusters[clusterI][1])**2
                if distance2 < minDistance2:
                    WhichCluster[i][j] = clusterI
                    minDistance2 = distance2
    #Next, re-center the means based on a weighted average of the clusters' members:
    meanNorm = [0. for i in range(0, NClusters)]
    for i in range(0, NClusters):
        for j in range(0, 2):
            NewKClusters[i][j] = 0.
    for i in range(0, NLat):
        for j in range(0, NLong):
            meanNorm[WhichCluster[i][j]] += badReviewDistribution[i][j]
            spot = [i, j]
            for m in range(0, 2):
                #NewKClusters[WhichCluster[i][j]] += badReviewDistribution[i][j]*spot[m]
                NewKClusters[WhichCluster[i][j]][m] += badReviewDistribution[i][j]*spot[m]
    for i in range(0, NClusters):
        for j in range(0, 2):
            if meanNorm[i] > 0.:
                NewKClusters[i][j] = NewKClusters[i][j]/meanNorm[i]

    unchanged = ClustersUnchanged()

    for i in range(0, NClusters):
        for j in range(0, 2):
            KClusters[i][j] = NewKClusters[i][j]

KClusterSize = [0. for i in range(0, NClusters)]
for i in range(0, NLat):
    for j in range(0, NLong):
        KClusterSize[WhichCluster[i][j]] += badReviewDistribution[i][j]

print 'KClusterSize =', KClusterSize

fig, ax = plt.subplots()
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
ax.set_title('Clusters of poorly reviewed businesses')

plt.imshow(img, zorder=0, extent=[MinLVLong, MaxLVLong, MinLVLat, MaxLVLat])
for i in range(0, NClusters):
    plt.scatter(MinLVLong+dLong*(KClusters[i][1]+0.5), MinLVLat+dLat*(KClusters[i][0]+0.5), c='blue', s=KClusterSize[i]/20.)

plt.show()

#Finally, we determine the average number of reviews with which open and closed businesses share a grid point:
averageNearbyReviewsOpen = 0.
totalBusinessesOpen = 0.
for i in range(0, len(LVOpenBusinessesLat)):
    totalBusinessesOpen += 1.
    for i in range(0, NLat):
        deltaLat = MinLVLat + dLat*(i+0.5) - LVOpenBusinessesLat[i]
        meanLat = 0.5*(MinLVLat + dLat*(i+0.5) + LVOpenBusinessesLat[i])
        #print deltaLat
        b = REarth*(deltaLat*(math.pi/180.))
        #print b
        for j in range(0, NLong):
            deltaLong = MinLVLong + dLong*(j+0.5) - LVOpenBusinessesLong[i]
            a = REarth*math.cos(meanLat*(math.pi/180.))*(deltaLong*(math.pi/180.))
            #print a
            #print a**2 + b**2
            if (a**2 + b**2 < 10.):
                averageNearbyReviewsOpen += reviewsDistribution[i][j]

averageNearbyReviewsOpen /= totalBusinessesOpen

print "The average number of nearby reviews of open businesses =", averageNearbyReviewsOpen, "and there are", totalBusinessesOpen, "total open businesses."

averageNearbyReviewsClosed = 0.
totalBusinessesClosed = 0.
for i in range(0, len(LVClosedBusinessesLat)):
    totalBusinessesClosed += 1.
    for i in range(0, NLat):
        deltaLat = MinLVLat + dLat*(i+0.5) - LVClosedBusinessesLat[i]
        meanLat = 0.5*(MinLVLat + dLat*(i+0.5) + LVClosedBusinessesLat[i])
        #print deltaLat
        b = REarth*(deltaLat*(math.pi/180.))
        #print b
        for j in range(0, NLong):
            deltaLong = MinLVLong + dLong*(j+0.5) - LVClosedBusinessesLong[i]
            a = REarth*math.cos(meanLat*(math.pi/180.))*(deltaLong*(math.pi/180.))
            #print a
            #print a**2 + b**2
            if (a**2 + b**2 < 10.):
                averageNearbyReviewsClosed += reviewsDistribution[i][j]

averageNearbyReviewsClosed /= totalBusinessesClosed

print "The average number of nearby reviews of closed businesses = ", averageNearbyReviewsClosed, "and there are", totalBusinessesClosed, "total closed businesses."
