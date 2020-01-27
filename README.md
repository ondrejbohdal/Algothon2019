# Algothon2019

# Deep Chain
Fund management strategy that utilizes various datasets, including supply chains, social media data, news data, stock market time series data and others.

## Inspiration
We were inspired by the many successes that neural networks achieved so far and also by the datasets offered by Quandl and Refinitiv.

## What it does
It constructs a pipeline of various signal indicators (based on social media data, news, stock prices, supply chains), and then summarizes them as a tick vector. The individual components are then weighted by an adjacency matrix to reflect the supply chain network of the specific company. Based on this we give a recommendation on how the investor should trade.

## How we built it
We initially did a lot of pre-processing of the data so that they are in a suitable format for our ML models. One of the key components was the use of Kernel-PCA that summarizes the incoming signals in a new & better way. On top of this, we have tried to combine this with LSTM network to capture temporal nature of data.

## Challenges we ran into
There were many challenges. Most importantly, we have decided to tackle a complex and ambitious project that was, as we found out, not easy to finish within 24 hours. We also had to do a large amount of data processing, which required many decisions. Further, combining various data sources together within one model turned out to be complicated.

## Accomplishments that we're proud of
We have a model that almost worked - and we have also managed to integrate together variable sources of data so that they could be used within one model. We also think that our ideas for a trading system were very innovative and unique.

## What we learned
We learned it may be difficult to successfully connect and utilize various source of data within one model, especially one based on deep learning. Another points include that it is better to start with a simpler project.

## What's next for Deep Chain
The next key point would be to successfully finish implementing our proposed approach, tuning it so that it works well. Parametrising the selection model to meet a desired risk-reward distribution.

Our project is also available at https://devpost.com/software/deep-chain.

The project won the 2nd overall place at Algothon 2019.

