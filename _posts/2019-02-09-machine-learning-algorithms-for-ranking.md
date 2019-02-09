---
layout: post
title: "Machine Learning Algorithms for Ranking"
author: "Abraham Theodorus"
categories: journal
tags: [journal, machine learning]
image: posts/2019/feb/ranking.jpg
---

In the last three months I have done two machine learning projects with two friends of mine ([Akul](https://github.com/akul08) & [Max](https://github.com/maxxiefjv)) for my university courses. We deliberately chose unfamiliar topics because of curiosity reason. This blog post will uncover one of our projects which deals with a ranking problem.

Ranking algorithms' main task is to optimize the order of given data-sets, in a way that retrieved results are sorted in most relevant manner. Ranking is a commonly found task in our daily life and it is extremely useful for the society. Typical problems which are solved by ranking algorithms, e.g., ranking web pages in [Google](https://www.google.com), personalized product feeds for particular customers in [Amazon](https://www.amazon.com), or even top playlists to listen in [Spotify](https://www.spotify.com).

From mentioned examples above, most of ranking problems deal with information retrieval and personalization domain. In information retrieval point of view, documents indeed need to be indexed in order to match given keywords with relevant documents. If we add personalization into the puzzle, the retrieved results also need to satisfy users' preference as well.

## Learning-To-Rank Algorithms

Ranking problems can be solved by specific learning algorithms, namely *Learning-To-Rank*. Citing from [a paper written by Yahoo!](http://proceedings.mlr.press/v14/chapelle11a/chapelle11a.pdf), *Learning-To-Rank* algorithms can be classified into three types based on their optimization objectives:

1. **Pointwise** <br>
   In this algorithm's perspective, data points are seen independently and every data point has its corresponding score. The learning algorithms will typically minimize mean squared error loss function, hence it becomes a regression problem.
2. **Pairwise** <br>
   Several queries are tagged with a list of relevant data-sets. Then, this algorithm will optimize the loss function of paired data-sets belonging to the same list. The result of this optimization is the correct swapping of pairs, thus generating a list of documents with reasonable order.
3. **Listwise** <br>
   Similar like Pairwise, but the optimization is performed at whole list of documents level instead of paired data-points.

## Kaggle Challenge?!

We participated in a Kaggle challenge entitled [*PUBG Finish Placement Prediction*](https://www.kaggle.com/c/pubg-finish-placement-prediction). [PUBG](https://en.wikipedia.org/wiki/PlayerUnknown%27s_Battlegrounds) is a popular online multiplayer game where there could be 100 players playing in one match and the goal is to be the last man standing. 

In this Kaggle challenge, player attributes such as number of kills, walk distance, boost pickups, and damage dealt are provided as features. Player placements per match are given as continuous values and the objective is to predict the placements of all players in one particular match. The data-set can be illustrated further below:

![alt text](/assets/img/posts/2019/feb/dataset.png "PUBG Data-set Illustration")

Here are some statistics that could describe our data-set:

- 4.4 million records
- 48k matches
- 4.4 million players
- 2 million teams

I am actually a quite avid gamer myself and I found this data-set to be pretty intriguing. Furthermore, this was my first time dealing with ranking problem, so it was a pretty exciting experience. The goal of our project is to evaluate several *Learning-To-Rank* algorithms and find out which one is better solving this ranking problem.

## Preprocessing & Feature Engineering

Based on our observation, it was found out that for each match, rankings are assigned to teams instead of per individual players. Therefore we aggregated player-level features to team-level features. This can be illustrated by the graphic below.

![alt text](/assets/img/posts/2019/feb/agg_group_feature.png "Team-Level Features")

We also inspected features which have the most correlations with ranking placement and below is the top 5 highest correlated features.

![alt text](/assets/img/posts/2019/feb/feature_corr.png "High correlated features")

Feature                 | Correlation               
:----------------------| :----------: 
Walking distance        | 0.81     
Ranking based on kills  | -0.72     
Number of boost pickups | 0.63    
Number of weapon pickups| 0.58     
Damage dealt to enemies | 0.44    


We then generated some more features based on these high correlated ones.

## Implementation

In this project we only evaluated pointwise and pairwise algorithms due to feasibility factor.
Pointwise algorithm is literally a regression algorithm, therefore we used a quite popular regression algorithm namely gradient boosted regression tree, with [*LightGBM*](https://github.com/Microsoft/LightGBM) library as the implementation.

The second approach is a pairwise algorithm. A pairwise algorithm needs to compare pairs of data-points belonging to one list. In the PUBG data-set, one list is represented by one match. Therefore, a different training mechanism for this approach is required. Training and test data is split based on matches, then for each match, ranking predictions are assigned by the trained model to each of the groups.

[*LambdaMART*](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) is the current state-of-the-art pairwise algorithms. Thus, in this category *LambdaMART* is used with [*XGBoost*](https://github.com/dmlc/xgboost) library as the implementation. Both pointwise and pairwise approaches use boosting as their core learning algorithm and boosting is one of the ensemble techniques that has been proven to be powerful.

## NDCG as Evaluation Metric

In order to view this problem from information retrieval perspective, where the order and relevance of documents are evaluated per query ID, NDCG is chosen as the evaluation metric. **matchId** represents the query ID and **winPlacePerc** represents the relevance score. NDCG (Normalized Discounted Cumulative Gain) is a quite popular metric for evaluating such information retrieval model performances. This metric evaluates a collection of retrieved documents based on the order and also the relevant scores of each document. It is particularly useful when the documents have a non-binary relevance score as weights. NDCG is a normalized version of DCG. DCG is denoted by:

$$
    DCG = \frac{relevanceScore_i}{\log_2 (i + 1)}
$$

Here, $$i$$ signifies the actual order of a document among a collection of retrieved documents. This formula penalizes retrieved documents with a high relevance score that get lower rank. This calculation is further illustrated with a table below. Here, the right-most column represents DCG scores for each retrieved document and even though the first and third row have the same relevant score, the third row has less score because it got lower rank than it should.

**$$i$$**|  **$$rel_i$$** | **$$log_2(i+1)$$** | **$$\frac{rel_i}{log_2(i+1)}$$**               
|:---- | :-------  | :--------| :------------
1      | 3        |   1  |  3
2      | 2        |   1.585  |  1.262
3      | 3        |   2  |  1.5
4      | 0        |   2.322  |  0
5      | 1        |   2.585  |  0.387
6      | 2        |   2.807  |  0.712

In order to have a normalized version, DCG is divided by iDCG (Ideal Discounted Cumulative Gain which is the DCG score for an ideal ordering of the documents given a query ID.

$$
    NDCG = \frac{DCG}{iDCG}
$$

NDCG always ranges from 0 to 1. In this project, the output of the pointwise algorithm's ranking was converted to NDCG to get a NDCG score comparison between all algorithms. However, Mean Absolute Error (MAE) score was also computed for Kaggle submission purpose.

## Results

We used 80:20 train test split ratio and 5-folds cross validation for hyper parameter tuning. The following is our achieved result. 


Model               |  MAE          | NDCG      | Kaggle Ranking                   
:-------------------| :----------:  | :--------:| :------------:
GBRT                | 0.0347        |   0.9939  |  565th (top 38%)
LambdaMART          | 0.045         |   0.9939  |  650th (top 44%)

Based on [Microsoft's paper](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/), *LambdaMART* as a pairwise algorithm should perform better than pointwise algorithm. However in usual ranking problem, multiple records are possible to have identical relevance scores and this problem relevance scores are unique for every group in one match. This might cause the issue. Furthermore, training *LambdaMART* model using *XGBoost* is too slow when we specified number of boosting rounds parameter to be greater than 200. As the NDCG scores in cross validation and test evaluation haven't reached plateau, it is possible to keep increasing this with larger machines (we used free machine provided in kaggle kernel).

![alt text](/assets/img/posts/2019/feb/ndcg_cv.png "NDCG cross validation and test scores")

*Learning-To-Rank* algorithm is renowned for solving ranking problems in text retrieval, however it is also possible to apply the algorithm into non-text data-sets such as player leaderboard.

This is my first Kaggle challenge experience and I was quite delighted with this result. I would definitely participate in more challenges. Thanks for reading, I hope this blog can be useful for you. See you in my next data science journey!