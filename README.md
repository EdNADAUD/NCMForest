# NCM Forest Algorithm

Implementation of the NCM Forest algorithm and incremental learning methods.

## Contributors :

* CARLU Ludovic
* CATRISSE Arthur
* FAIVRE Maxime
* VERDY Sylvain
* Laboratory ESIEA : LDR 


*Resulting of 3 weeks work in LDR Laboratory as part of research project at ESIEA.*
<br>
<br>
<br>

## **NCM Classifier**
* Calculating centroids
* Predicting with differents distances : Euclidean, Mahalanobis
* Selecting subfeatures

## **NCM Node**
* Selecting subset centroids
* Splitting decision for child nodes :
	*  **Random** : Random class selection
	*  **Maj_Class** : Separation of the majority class
	*  **Eq_Samples** : Balancing the number of samples in child nodes
	*  **Farthest_max** : Separating the centroids from the most distant classes then group the rest by the farthest centroid
	*  **Farthest_min** : Separating the centroids from the most distant classes then group the rest by the nearest centroid

## **NCM Tree**
* Building tree recursively with usual parameters (max\_depth, min\_sample\_split, min\_sample\_leaf)


## **NCM Forest**
* Bootstrapping samples
* Prediction with trees probabilities (Voting)


## **Incremental methods**
* Update leaf statistics (ULS)
* Incrementally grow tree (IGT)
* Re-train subtree (RTST)


<br>

## References :
[[1]](https://www.researchgate.net/publication/282546052_Incremental_Learning_of_Random_Forests_for_Large-Scale_Image_Classification)
Ristin, Marko & Guillaumin, Matthieu & Gall, Juergen & Van Gool, Luc. (2015). Incremental Learning of Random Forests for Large-Scale Image Classification. IEEE Transactions on Pattern Analysis and Machine Intelligence. 38. 1-1. 10.1109/TPAMI.2015.2459678.
