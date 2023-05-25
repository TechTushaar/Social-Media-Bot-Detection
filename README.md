# Social Media Bot Detection

Introduction
--------------------------------
Social Media has completely changed the way the world communicates with each other. They have become our primary source of social interaction, getting information, advertising businesses as well as purchasing products. Different platforms - such as Twitter, Instagram, LinkedIn exist to serve some or all of the above mentioned purposes. However, bots are increasingly being used to spread fake news(Shao et al. 2018), influence opinions during elections (Bessi and Ferrara 2016), leave fake reviews (Salminen et al, 2021) etc, especially considering the rise of AI-generated human-like responses by technologies such as chatGPT. A great deal of research remains to done for identifying and removing spam accounts. Through our study, we aim to use different ML techniques on the largest Twitter Dataset yet and analyze their performances against each other.

Literature Review: The literature surrounding this topic most frequently deals with the methods used to detect bots (typically social bots) and the classification between social bots and spam bots. Social bots are the bots that mimic human profiles and interact with users, whereas spambots are more traditional and easily detectable as bots (Aljabri et al., 2023). Many methods were used; the methods with the highest accuracy, however, are those such as Deep Forest and Convolutional Neural Network. Deep Forest had an accuracy of 97.55%, whereas the CNN achieved 98.71% accuracy when looking at a single post (Mohammad et al., 2019).


Dataset Features:

In our study, we use the dataset TwiBot-20*, a comprehensive Twitter bot detection benchmark that presents one of the largest Twitter datasets to date. It provides diversified entities and relations on the Twitter network, and has considerably better annotation quality than most existing datasets (Feng et al, 2018). TwiBot-20 contains the user data, tweet data, labels to classify as genuine or bot, as well as edges - which contain data about relation type between users and tweets. 

*While we originally planned to use TwiBot-22, given the size of the dataset and power of our machines and the time we had, we opted to scale down and use TwiBot-20 for now. 

The Gantt Chart schedule for this group's project can be found [here](https://gtvault-my.sharepoint.com/:x:/g/personal/vfang8_gatech_edu/EXefW5TOnPNAk_qdoP_TbqAB_6jij6N_rqb24fetc3OhiA?e=s8jlTu).

Contribution chart based on the Gantt Chart:

| Team Member | Responsibilities |
| --- | ----------- |
| Vincent | Time/Schedule Chart, Github creation, M1 feature reduction, M2 Data Visualization |
| Anthony | Proposal Video, M2 feature reduction, M3 Data Visualization |
| Tushaar | Proposal Intro/background, M2 Data Cleaning, M3 Feature extraction, Final Visualizations|
| Jontae| Proposal Problem def, Potential results and discussion, M3 Data Cleaning|
| Alex | Proposal methods, potential datasets, M1-M3 Comparison|
|All | M1-3 Design, Implementation, Result Evaluation Midterm Report, Final Report, Final Video|


Problem Definition
--------------------------------
In public discourse, the prevalence of bots in social media is regularly talked about. By showing that bots can be detected and have key patterns, it may help highlight these patterns to reduce their effect.

Methods
--------------------------------

### Data Preprocessing: 
The first step is to preprocess the dataset by cleaning and transforming the data into a format that can be used by machine learning algorithms. For this, we had to split the node.json file in the dataset into multiple smaller files - since we weren't able to load the dataset at once because of its size. Node.json had user information and tweet information combined into one dataset, while edge.csv had information linking each tweet to a user. After creating smaller pickle files, we divided the dataset into a user dataset and a tweets dataset, dividing work among ourselves to extract features from each category separately before combining for our final features. In the end, we recombined tweet features differentiated by each user back to the user_dataframe. We also dropped all unlabeled or incomplete records from the dataset. In the end, this gave us 11746 users, out of which 5237 are Humans and 6589 are Bots.

### Feature Extraction: 
Once the data had been preprocessed, we extracted features from the data that would be used to train the Machine Learning Model. Features of a spam detection bot can be broadly classfied into 3 categories:

1. User-level Data
This includes information obtained from a user's Profile- such as name, username, description, verified, following count, listed count etc. We used 'followers_count', 'following_count','tweet_count', 'listed_count', 'verified' as our profile level features in our model. This decision was taken in consideration of the most accessible profile information, as well as processing power limitations.  
  Our first model run included only user-features.

2. Tweet-level Data
This includes characteristic information about a user's tweets. This includes basic features like number of tweets, retweets, average tweet word count etc. We also calculated more specific stuff such as average number of hashtags, average number of mentions, average number of links, Number of retweets etc. We used regular expression to find patterns to find such features in tweets. The reasoning for this is the argument that Bots try to use a lot of hashtags and mentions to increase their coverage. They are also likely to have a high retweet count.  
  Our second model collected the Mean and Standard Deviation of number of Hashtags/ Links/ Mentions/ Words for a maximum of 50 user's tweets. We also had a metric of number of Retweets out of those 50. These 9 features were used as features, combined with user features. Given computational limitations, we had to abandon some features such as Sentiment Analysis, Measure of Tweet Similarity etc. In future, we hope to increase this number and include other features that we wanted to implement. /n For our final Model run, we decided to increase our 50 Tweet Limit per user. Since storing data linking a tweet data to user for 3 million plus tweets was timing out our resources, we employed a running mean and std calculation based on Welford's Online Algorithm (Wilford), which allows for one pass update of mean and std when storing all values is not an option. This proved to be extremely beneficial and allowed us to process the whole tweet dataset with 1/10th of the RAM Usage over our previous limited tweet run. The Retweet feature was also changed to be a ratio of Retweets/ Total number of Tweets.   
  We also added another feature - Tweet Similarity. To calculate this, we use Cosine Similarity Test. Our implementation draws heavily from GloVe - Global Vectors for Word Representation model. Basically, it provides a way to convert sentences to vectors, such that similar sentences would point in a similar direction. To convert to vectors, the words in a tweet are compared to a set of pre-trained word Vectors in the Twitter Dataset provided by GloVe authors. Word vectors are then added to form one composite vector, and the cosine angle between two composite vector is a score of similarity between two tweets.  
  For our model, we maintain a set of the user's 5 most recent tweets before the previous tweet, and compare the new tweet to each of the previous ones and calculate an average. Hence, every user tweet is compared to 5 tweets before that tweet to check for similarity. These values are averaged over the whole tweet set of a user using the same running mean and std technique. While ideally we wanted to do more stuff with tweet similarity, this archiecture alone took more than 5 hours, limitng us to 5 tweets for now. This cosine_mean and cosine_std were added as features to our model


3. Graph-level Data
While Graph-level Data such as who follows who, relationship between accounts is important - we choose to ignore it in this phase because it is generally hard to obtain and very computationally expensive to process. Once we reach an upper limit on model accuracy after optimizing User and Tweet-Level data, we plan to employ graph level data and use a Graph CNN classifier for our next project.


### Choosing a Model
Our task is a simple binary classficiation - Bot or not. Since one of the objectives of this project is to compare different models - we split our dataset into a train_set and test_set in a 70:30 split, and run it through Random Forest, AdaBoost, Decision Tree, MLP, and K-neighbors Classification Algorithms.

### Performance Evaluation
We will use classic and well known metrics such as Accuracy, Precision, F1 and Recall. These are defined below:  

* Accuracy is the proportion of correct predictions (both true positives and true negatives) out of the total number of predictions. It is defined as (TP + TN) / (TP + TN + FP + FN), where TP is true positives, TN is true negatives, FP is false positives, and FN is false negatives.

* Precision is the proportion of true positives (correctly classified positives) out of all positive predictions, which is defined as TP / (TP + FP).

* Recall, also known as sensitivity or true positive rate, is the proportion of true positives (correctly classified positives) out of all actual positives, which is defined as TP / (TP + FN).

* F1-score is the harmonic mean of precision and recall, which is defined as 2 * (precision * recall) / (precision + recall). It provides a balance between precision and recall and is often used in situations where there is an imbalance between the classes.

In summary, accuracy measures the overall performance of the model, while precision and recall provide information about the model's performance on positive and negative cases. F1-score combines precision and recall and is useful in cases where there is a tradeoff between precision and recall.  
We also employ Feature Importances in our random Forest to find out the most important features in our classification. We also analyzed best features using backward and forward feature selection for KNeighbours and RandomForest classifiers. 

Results and Discussion
--------------------------------
* Model 1: We used only profile features. These were ['public_metrics.followers_count', 'public_metrics.following_count',
       'public_metrics.tweet_count', 'public_metrics.listed_count', 'verified']  

Our results are visualized below:  

![image](https://user-images.githubusercontent.com/80716118/229247169-99df0930-9ca4-4fab-b9e3-0015f5483683.png)

* Model 2: We added tweet features as described above. We also added more comparison metrics - such as precision, recall and F1. From the graphs, we can see that we increased our model's accuracy and other metrics by 4-5% throughout the models.

![image](https://user-images.githubusercontent.com/80716118/229245298-b81554c0-16bc-43ca-bdfb-790a89b64910.png)

We also calculated Feature Importances to get insights into the underlying relationships between the input features and the target variable. By identifying the most important features, we hoped to gain a better understanding of which factors driving the predictions our model. Since Random Forest Classifier was on average our best performing model, we calculated feature importances for it - visualized below:

![image](https://user-images.githubusercontent.com/80716118/229246067-cbe16a64-a3c2-41d1-9be3-44cad1d2fdea.png)

As expected, verified field has a large impact on determining whether or not an account is a bot. Number of Tweets had the least importance. Hence, in our next run of model 2, we decided to omit the verified and number field and test the difference in Model Performances.

* Model 2 - Without Verified Feature:

![image](https://user-images.githubusercontent.com/80716118/229246361-a6df9411-3838-4810-afa0-527bdcb0124e.png)

### Feature Importance:

![image](https://user-images.githubusercontent.com/80716118/229246510-16b8cc3c-c870-4243-83b6-37ff7739296b.png)

Overall, we can see a dip in performaces by a few %, which aligns with what we expected. However, the model is still pretty good, and adding more features like discussed above should get us closer to our target 90% accuracy.

Final Update: 

* Model 3: This model included all of user features as well as Tweet Features calculated over all of user's tweets available in the dataset. We also updated our model split into train and test dataset to be more random in nature. Our Features were also Z-Normalized as compared to previous models.  

Our final set of features were: ['public_metrics.followers_count', 'public_metrics.following_count',
       'public_metrics.tweet_count', 'public_metrics.listed_count', 'verified',
       'Number', 'Links_mean', 'Links_std', 'Words_std', 'Words_mean',
       'Mentions_std', 'Mentions_mean', 'Hashtags_std', 'Hashtags_mean', 'RT']

Results are visualized below:

<img width="382" alt="image" src="https://user-images.githubusercontent.com/80716118/233766194-ca949d82-a60e-4e22-9365-04f59f10e77a.png">

![image](https://user-images.githubusercontent.com/80716118/233766212-ca4af2e5-c545-4079-a272-eda7596ca826.png)

### Feature Importance:

![image](https://user-images.githubusercontent.com/80716118/233766262-73f550bb-510b-4c31-8ed8-c7a0c6faf25a.png)

### Backward and Forward Feature Selection:

<img width="817" alt="image" src="https://user-images.githubusercontent.com/80716118/233766279-0733575e-d06c-45c9-b41b-d6e663fbe75c.png">

The pattern with verified being most important feature was visible again, hence we removed it for another run. A visualization of the decrease in metrics is shown as the difference between the metrics for two runs. Color Red signifies the model with verified, and we can see that it consistently fairs better than Blue - unverified by 4-5%. This leads to the inference that verified tags do make a difference and bots are not immune to them yet.

![image](https://user-images.githubusercontent.com/80716118/233766361-ed8cc244-c99b-45d6-8856-b36710bb9187.png)

* Model 4: Model 3 + Tweet Similarity Metrics.  
Final Feature List: ['public_metrics.followers_count', 'public_metrics.following_count',
       'public_metrics.tweet_count', 'public_metrics.listed_count', 'verified',
       'label', 'Number', 'Links_mean', 'Links_std', 'Words_std', 'Words_mean',
       'Mentions_std', 'Mentions_mean', 'Hashtags_std', 'Hashtags_mean', 'RT',
       'Cosine_mean', 'Cosine_std']

Results: 

<img width="457" alt="image" src="https://user-images.githubusercontent.com/80716118/233766588-261a9323-0bd7-4a1d-8257-8aaf9a2096d1.png">

![image](https://user-images.githubusercontent.com/80716118/233766552-d0afee3d-bb1c-483f-bff0-3eb3e094e384.png)

![image](https://user-images.githubusercontent.com/80716118/233766557-cc2c607a-4c4b-496b-9907-cc2aa6773314.png)

Removing Verified field resulted in a similar change as before, decreasing metrics by 3-4% across board.  

### Feature Importance

![image](https://user-images.githubusercontent.com/80716118/233766640-aa915ce4-3adb-45dc-909e-8f74b29a02d0.png)

### Backward and Forward Feature Selection

<img width="761" alt="image" src="https://user-images.githubusercontent.com/80716118/233767535-653ae092-3a44-47cf-a381-9e35113a266b.png">

### Comparison

<img width="473" alt="image" src="https://user-images.githubusercontent.com/80716118/233766766-301b25c7-4fe4-4b80-a4d7-166b678c08aa.png">


## Final Results

Best Models: As we can see from metric comparison graphs, AdaBoost and Random Forest are our best classifiers models. They achieve a high accuracy, and also a High F1 Score - which means they balance recall and precision.  

Best Features: Adding Tweet Features to our model increased our accuracy by an average of 4-5%, helping us near the 80% mark. The increase in increasing tweet analysis from a hard limit of 50 to as many tweets (200 max per user in dataset) did not increase or decrease the metrics by a considerable margin. This leads us to the conclusion that habits such as word count, retweet ratio, use of links/mentions in post can be meaningfully estimated from a small subset of tweets rather than all of a users tweets.  
To our surprise, our GloVe Cosine Similarity Test for Tweet similarity did not increase our accuracy either. Potential reasons for this include:
* We were only able to compare one tweet to 5 precious tweets of a person due to computational limitations.
* On analyzing tweets, many tweets make use of Internet Slang words, which were not present in our Tweet Pre-Trained Word Vectors Dictionary. Stuff like links/hashtags etc were also ignored. Tweet words that did not conform to standard English were also dropped.
Due to such potential weaknesses in our current implementation, it is hard to judge if Tweet Similarity is a useful metric or not.

Final Thoughts: Over the course of multiple runs, our accuracy seems to have stagnated at 80% using basic classifier models. This supports the hypothesis that higher accuracy levels will be achieved after incorporating Graph based information into our model.


-----------------------------------------------------------------------------------------------------------------------------

Additional Figures:

**Accuracy of Random Forest from Number of Estimators**

![image](https://user-images.githubusercontent.com/11355355/228995210-85e5e0c5-eae7-43f1-b4e0-fb17c83d988f.png)

**AdaBoostClassifier Accuracy from Number of Estimators**

![image](https://user-images.githubusercontent.com/11355355/228995277-33299fd0-0069-46a6-8bcc-6c74f981d3e3.png)

**Line graph for RandomForest accuracy from number of estimators:**

![image](https://user-images.githubusercontent.com/11355355/228995354-4ebe800a-ffad-46f7-990c-9601c3b2f6c9.png)

**Line graph for AdaBoostClassifier Accuracy from Number of Estimators:**

![image](https://user-images.githubusercontent.com/11355355/228995331-e3721b9f-26fb-49d8-92f9-6eeae9b4046e.png)

Video Link
--------------------------------
https://youtu.be/2IWhReNnz94 - M1

Midterm Report Contribution Chart
--------------------------------

| Team Member | Responsibilities |
| --- | ----------- |
| Vincent | Tweet-level feature extraction and Model Run 1 |
| Anthony | User-level feature extraction, Model Run 1|
| Tushaar | Tweet-level feature dataset integration, Model 2/3, Midterm Report Writeup |
| Jontae| User-level feature extraction, Midterm Report Writeup |
| Alex | User-level feature extraction, Model Run 1 and Visualizations |


Video Link
--------------------------------
https://www.youtube.com/watch?v=7qiwDZDuj0w 

References
--------------------------------
Aldayel, A., & Magdy, W. (2022). Characterizing the role of bots’ in polarized stance on social media. Social Network Analysis and Mining, 12(1). https://doi.org/10.1007/s13278-022-00858-z 

Aljabri, M., Zagrouba, R., Shaahid, A., Alnasser, F., Saleh, A., & Alomari, D. M. (2023). Machine learning-based Social Media Bot Detection: A comprehensive literature review. Social Network Analysis and Mining, 13(1). https://doi.org/10.1007/s13278-022-01020-5 

Echeverrï£¡a, J., De Cristofaro, E., Kourtellis, N., Leontiadis, I., Stringhini, G., & Zhou, S. (2018). Lobo. Proceedings of the 34th Annual Computer Security Applications Conference. https://doi.org/10.1145/3274694.3274738 

Feng, S., Tan, Z., Wan, H., Wang, N., Chen, Z., Zhang, B., Zheng, Q., Zhang, W., Lei, Z., Yang, S., Feng, X., Zhang, Q., Wang, H., Liu, Y., Bai, Y., Wang, H., Cai, Z., Wang, Y., Zheng, L., … Luo, M. (2022). TwiBot-22: Towards Graph-Based Twitter Bot Detection. https://doi.org/https://arxiv.org/pdf/2206.04564.pdf 

Feng, S., Wan, H., Wang, N., Li, J., & Luo, M. (2021). TwiBot-20: A comprehensive twitter bot detection benchmark. Proceedings of the 30th ACM International Conference on Information & Knowledge Management. https://doi.org/10.1145/3459637.3482019 

Ferreira Dos Santos, E., Carvalho, D., Ruback, L., & Oliveira, J. (2019). Uncovering social media bots: A transparency-focused approach. Companion Proceedings of The 2019 World Wide Web Conference. https://doi.org/10.1145/3308560.3317599 

Heidari, M., Jones, J. H. J., & Uzuner, O. (2022). Online User Profiling to Detect Social Bots on Twitter. https://doi.org/10.48550/ARXIV.2203.05966 

Mohammad, S., Khan, M. U. S., Ali, M., Liu, L., Shardlow, M., & Nawaz, R. (2019). BOT detection using a single post on social media. 2019 Third World Conference on Smart Trends in Systems Security and Sustainablity (WorldS4). https://doi.org/10.1109/worlds4.2019.8903989 

Salminen, J., Kandpal, C., Kamel, A. M., Jung, S.-gyo, & Jansen, B. J. (2022). Creating and detecting fake reviews of online products. Journal of Retailing and Consumer Services, 64, 102771. https://doi.org/10.1016/j.jretconser.2021.102771 

Shao, C., Ciampaglia, G. L., Varol, O., Yang, K.-C., Flammini, A., & Menczer, F. (2018). The spread of low-credibility content by Social Bots. Nature Communications, 9(1). https://doi.org/10.1038/s41467-018-06930-7 

Varol, O., Ferrara, E., Davis, C., Menczer, F., & Flammini, A. (2017). Online human-bot interactions: Detection, estimation, and characterization. Proceedings of the International AAAI Conference on Web and Social Media, 11(1), 280–289. https://doi.org/10.1609/icwsm.v11i1.14871

Wilford's Algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

Pennington, J. (n.d.). GloVe. Glove: Global vectors for word representation. Retrieved April 22, 2023, from https://nlp.stanford.edu/projects/glove/ 
