# ANLY502-Project

## Team Members

Han Zhang (hz274), Jingjie Ma (jm3292), Jiamin Zhong ()

## Team Name 

Nirvana

## Executive Summary

In this project, we are going to predict the closing price of stocks traded on the Deutsche Borse public system. 

## Code Files
* [downloadData.ipynb](https://github.com/hzhang17/ANLY502-Project/blob/master/downloadData.ipynb): use script to download data from Deutsche Borse PDS's S3 bucket.
* [cleanData.ipynb](https://github.com/hzhang17/ANLY502-Project/blob/master/cleanData.ipynb): use Spark to clean data and separate data for each stock.
* [toPandas.ipynb](https://github.com/hzhang17/ANLY502-Project/blob/master/toPandas.ipynb): convert RDD to pandas dataframe and save files in csv format.
* [testing.ipynb](https://github.com/hzhang17/ANLY502-Project/blob/master/testing.ipynb): testing steps for previous notebooks, including failure message for copying data files directly to personal S3, and using Spark SQL to clean and separate data with intermediate data samples.


## Introduction
We are going to predict the close price of common stocks using the historical price of stocks. The dataset we are going to use is from the Deutsche Börse Public Dataset. The dataset consists of minute-by-minute data of common stocks, bonds, and other derivatives. We will use two models in our project, recurrent neural networks and XGB regressor. We will also compare and contrast between different models. 


## Methodology
### Dataset Collection and Cleaning
#### Dataset Collection

* We collect the data of all stocks trading in the Deutsche Börse Public Dataset from January 1, 2018, to December 31, 2019. 

* The original dataset lives on the AWS S3 bucket. The dataset was saved in the following format: each day has its own folder, named after the date, which contains 24 csv files, each storing minute-by-minute trading data in every hour.

* The official document contains the code for accessing the dataset of a specific date, which is using the AWS Command Line Interface with argument “no-sign-request”. However, data of multiple days couldn’t be directly copied to a personal S3 bucket. So we decided to use a script to first download the data of year 2018 and 2019 to our local machine and then upload those data files to our own S3 bucket. 
 

#### Data Cleaning
* After we successfully download and re-upload the data to our S3 bucket, we use Spark and SQL to clean the dataset. The original dataset contains the following 14 variables: ISIN, Mnemonic, SecurityDesc (description),  SecurityType, Currency, SecurityID, Date, Time, StartPrice, MaxPrice, MinPrice, EndPrice, TradedVolume, and NumberOfTrades.

* Since we are going to predict the close price based on historical prices of stocks, we first remove irrelevant variables, ISIN, Currency, Security ID, Traded Volume, and Number of Trades, from the dataset. We also cast the data type of the “Date” variable to remove the trailing “00:00:00” added by Spark for future use. We are mainly focusing on common stocks, so we select all securities with “common stock” as their security types.

* In order to get the open and close prices of each day, we would first extract the hour and minute parts from the Time variable. We then cast the hour and minute into int type. According to the official Xetra website, the trading of common stocks starts from 9:00am to 17:30pm. So we select the data within this time window.

* In order to understand how our models perform differently in different industries, we selected twenty stocks from five of the following industries based on the number trading record. We prefer to select stocks that have more trading records in order to get large samples for our models. 
	* Heavy industry: Airbus (AIR), Bayerische Motoren Werke (BMW), Volkswagen Group (VOW3)
	* High-tech and Internet: Amazon (AMZ), Ebay (EBA), Netflix (NFC), Microsoft (MSF), Twitter (TWR), Facebook (FB2A)
	* Manufacturing Industry: Daimler (DAI), Siemens (SIE), Koninklijke Philips NV (PHI1), Adidas (ADS), Continental (CON), 
	* Financial Industry: Deutsche Bank (DBK), Commerzbank (CBK), Allianz SE (ALV), 
	* Health and Chemical: BASF (BAS), Covestro (1COV), Bayer (BAYN)

* We will find the daily min and max prices for each stock by comparing the minimum and maximum price of each minute for all trading records in a day. The start price will be the start price of the first trading record of that day and the close price will be the close price of the last trading record of that day. 


#### Tools & Models Used
* Tools:
	* Spark: Manipulate data frames and feature engineering. 
	* SQL (Spark SQL): Find open, close min, max prices of stocks.
	* EMR: Make massive data process and manipulation possible and fast. 
	* S3: Store data

* Models:
	* Recurrent Neural Network: We used recurrent neural networks to predict close prices of stocks. Inputs of the model are open prices, min prices, and max prices of each stock We split the data into three parts: 80% as the training set, 10% as the validation set, and 10% as the test set. 
	* XGBoost Regressor: We use XBG regression to train the data. Inputs of the model are some financial indicators, including various moving averages and moving average convergence divergence signals. We split the data into three parts: 80% as the training set, 10% as the validation set, and 10% as the test set. 

#### Regressions Models
* Recurrent Neural Networks
	* Structure of the recurrent neural networks: We use 3 layers of neurons and each layer contains 300 neurons. We also normalized our data before we fit the model in order to improve convergence of the model [1]. 
	* Reasons for choosing Neural Networks as a general tool: Predicting stock price is not easy, especially for non-finance people like us. There are lots of factors that would affect the stock price, for example, dividends, interest rates, etc. Neural Networks are flexible and can reveal hidden patterns and relationships within the data. 
	* Reasons for choosing RNN: The fact that stock prices change sequentially determine the nature of our data. In financial analysis, each historical price point is useful in predicting the future price. Thus, we should build the neural network that is capable of carrying information from previous stages. RNN outperforms standard neural networks when every piece of previous information should be memorized. Each RNN cell can memorize information from all past states. 

 * XGB Regressor: 
	* Structure of the XGB regressor: We define different the number of trees, learning rate, and gammas. Then, we choose the best model among a set of models based on the mean square error. The best models for different stocks are not identical, which is what we expected. Instead of having fixed parameters for all stocks, like what we did for those RNN models, we think having a range of parameters would potentially improve the accuracy of models. 
	*Reasons for choosing XGB regressor: The first reason is based on the computational capacity of our local machines. XGB regressor runs parallely, thus it would be suitable for our relatively large dataset. The second reason relates to the nice statistical feature of the model. The XGB regressor uses boosting techniques, which is good for weak learners.  For stock price analysis, it is really hard to find strong learners and grow an entire tree based on those learners. Thus, one way we can think of is to improve the performances of weak learners in order to improve the accuracy of the data. 

#### Hypothesis (Goal of the Project)

	





## References:
[1] https://stackoverflow.com/questions/43467597/should-i-normalize-my-features-before-throwing-them-into-rnn

[2] https://www.investopedia.com/terms/c/closingprice.asp 





#### Table

|  Stock  | MSE (RNN) | MSE(XGB Regressor) |
|:---:|:---:|:---:|
|  1COV  |    0.019278     |   1412.233285697164   | 
| ADS |    0.053041      |    882.5451333652707     | 
|  AIR     |     0.027250         |    30.89318509379332     | 
|  ALV     |      0.031557       |      29.37601789103453     | 
|  AMZ     |      0.027568     |    1389.866055555021   | 
| BAS  |       0.011666    |    0.005018825792531937    | 
|  BAYN   |                     |    0.005276089780885209    | 
|   BMW    |      0.006598     |     0.006794532540526577   | 
|  CBK     |        0.005866  |    0.0047144974970246365   | 
|  CON  |      0.018633      |    1536.9848921344576 | 
|  DAI  |      0.014024      |    170.39798666388413    | 
|  DBK  |      0.004528      |    0.03472928725007138   | 
|  EBA  |      0.015661       |    0.15760988474111842   | 
| FB2A |      0.042094   |     286.2327992445581    | 
|  MSF |    0.015104       |   0.002160218918588729   | 
| NFC  |    0.018897      |   174.87718032840988     | 
|  PHI1 |                     |                     | 
|  SIE |     0.011042    |     53.82468243338227    | 
|   TWR    |      0.020783         |   70.6188395932242     | 
|  VOW3     |      0.006940     |    0.00016597378220988007  | 
