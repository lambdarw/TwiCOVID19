# TwiCOVID19 Dataset

## Data Collection
We utilized a web crawler to climb Twitter to build a TwiCOVID19 dataset of the COVID-19 pandemic.

We replaced URLs and user handles (@user-name) with the symbols “\<url\>” and “\<user\>”

We label tweets with positive, neutral, or negative polarities using TextBlob

The timeframe of the TwiCOVID19 dataset is from 2021-01-01 00:00:00 to 2021-12-31 23:59:59.

## Data files
1) covid19_tweet.csv desipts information of tweets.
2) covid19_user.csv desipts information of users.

## Data Statistics
1) Event tags: #covid19, #covid-19, #coronaviruspandemic, #coronavirus, and others.
2) Tweet information: tweet ID, content, posting time, user ID, tag, number of retweets, number of favorited, number of replies, source tweet ID (unique field of retweets), and sentiment label.
3) User information: user ID, gender, nickname, number of followers, number of friends, number of favorites, and number of users’ tweets.

Here is the statistics of TwiCOVID19 dataset (“#” DENOTES “NUMBER OF”):

Statistic | TwiCOVID19
---- | -----
\# Tweets | 17,675 5
\# Positive tweets | 10,052
\# Neutral tweets | 3,999
\# Negative tweets | 3,624
avg. \# words per tweet | 23
\# Users | 7,489
\# Forwarding tweets of users | 15,350
Density of user forwarding network | 0.0274%

## Citation
Please cite our repository if you use METER in your work.
```bibtex
@article{zhang2024modeling,
  title={Modeling group-level public sentiment in social networks through topic and role enhancement},
  author={Zhang, Ruwen and Liu, Bo and Cao, Jiuxin and Zhao, Hantao and Sun, Xuheng and Liu, Yan and Sun, Xiangguo},
  journal={Knowledge-Based Systems},
  pages={112594},
  year={2024},
  publisher={Elsevier}
}
```
