# ANLY502-Project

## Team Members

Han Zhang (hz274), Jingjie Ma (jm3292), Jiamin Zhong ()

## Team Name 

Nirvana

## Executive Summary

In this project, we are going to predict the closing price of stocks traded on the Deutsche Borse public system. 

## Code Files

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

* Since we are going to predict the close price based on historical prices of stocks, we first remove irrelevant variables, ISIN, Currency, Security ID, Traded Volume, and Number of Trades, from the dataset. We also cast the data type of the “Date” variable to remove the trailing “00:00:00” added by Spark for future use. We are mainly focusing on common stocks, so we select all securities with “common stock” as their security types. We will drop any observation with missing values before applying the model. The intermediate steps and data sample will be shown in the file named “cleanData.ipynb”.

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
