### Predicting Fully Funded Projects at Time of Posting

#### Summary

This report explains the results of setting up a machine learning pipeline to predict whether a Donors Choose project will be fully funded at the time it is posted. Using the 2011-2013 project data, we trained several models using outcomes from the 2011-2012 data and then tested using the 2012-2013 data. After comparing several metrics, arrive at an optimal model for the client. This model can be used to identify and rank projects at risk of not being fully funded. In turn, the client may create interventions to help bolster the proposals of projects that are at risk of not getting fully funded. 

#### Exploratory Analysis

Of the 2011-2012 projects, about 72 percent were fully funded within a year of being posted. For the 2012-2013 data, the percentage is slightly lower at 70 percent being fully funded. The project data include many school characteristics, including but not limited to

- whether a school is a magnet or charter school
- whether the poster is a Teach for America teacher
- school state/zip code
- grade level
- primary/second focus area
- total pricing including and excluding optional support
- students reached

After exploring and cleaning the data, we use these characterisitcs to make predictions. However, there is opportunity to augment the characteristics used for predictions by adding neighborhood demographic information based on a school's latitude and longitude. This could help control for neighbordhood effects and identify whether schools in less affluent areas have more projects get posted or fully funded. 

#### Methods

The pipeline is capable of applying several methods to train models using different combinations of parameters to optimize different evaluation metrics. This analysis implements decision trees, k-nearest neighbors, logisitc regression, as well as random forests and other ensemble boosting methods. Each of these methods attempts to understand which variables or features of the training data are useful in predicting the outcome, whether a project is fully funded. Some methods suffer from overfitting on the training data, which is why ensemble methods often improve on the results of normal methods. 

#### Evaluation

We evaluate the models based on a combination of metrics including precision, recall, and area under the roc curve (AUC). Precision is the rate of positive predicted values, while recall is the ratio of true positives to all positives. AUC gives us an understanding of the tradeoffs between precision and recall at various thresholds of the population, but the score itself is based on the entire population of interest. 

Working under the assumption that the client has infinite resources is often tenuous, so we assume the client would like to target the top 5 percent of posts with the highest risk scores of not being fully funded. Based on this, we can choose the best performing model using precision at 5 percent of the population. 

Using precision at 5 percent, we find that logistic regression models performed the best. That being said, KNN and decision trees performed the best based on AUC and recall, respectively. In addition, the non-ensemble methods saw decreases in performance over time but only slightly. The models trained using methods did not perform as well for this temporal data, but there is room for more model tweaking by expanding the set of parameter combinations used. This is another opportunity to improve the pipeline. 

Another way to improve the pipeline is to obtain the most important features of the top performing models and then re-training the models to check whether the results improve. Obtaining the important features using a decision tree with a small number of levels could also help confirm expert heuristics that have traditionally been used to identify at-risk subjects. 

#### Recommendations & Policy Implications

Using our top performing model based on precision at 5 percent of the population, we recommend the client apply this model to incoming projects and then use the risk scores to provide extra resources to these posts in terms of promotion on the front page of the website. In addition, these insights could help inform a guidebook on how to create a successful project post to offer new posters. This could be offered as a premium service provided by the client. Granted, this may fall within a gray area in terms of ethics since the risk scores could be perceived as a proxy for "valuable projects" when in reality, almost all of the projects on the site have some sort of value proposition with social impact in mind. 

