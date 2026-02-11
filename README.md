# CS506-Project

## Project Description 
Our project analyzes global energy consumption patterns and their relationship with economic indicators at the country level, aiming to study how energy consumption varies across countries and over time. We plan to use the World Energy Consumption dataset from Kaggle, which contains country-year data on energy usage, GDP, population, and energy mix composition. 

## Project Goals 
The primary goal of our project is to predict energy consumption per capita using economic and energy-related features such as GDP, population, and energy mix composition. Additionally, we want to analyze the relationship between GDP per capita and energy consumption per capita using GDP, population density, and fossil fuel share as features and examine trends in fossil fuel and renewable energy usage across countries over time using statistical analysis and visualizations.

## Data Collection Plan 
The data we plan to use is from the World Energy Consumption dataset available on Kaggle (https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption), which we will download using the Kaggle API. We noticed that all GDP from 2020-2022 are missing, and we will supplement this data from official sources like [World Bank Open Data's API](https://data360.worldbank.org/en/api) or by finding a site to scrape all relevant GDP information. After collection, we will clean the data by handling missing values, performing feature extraction, and filtering to focus on relevant years and countries. 

## Project Timeline 
In the first 1–2 weeks, we will collect the data using the Kaggle API and perform initial data cleaning and feature selection. In the following week, we will conduct exploratory data analysis and create preliminary visualizations for the first project check-in. Over the next few weeks, we will implement and evaluate a baseline model, likely linear regression, to predict energy consumption per capita. We will then explore additional modeling approaches, such as clustering, to provide complementary insights into global energy consumption patterns. In the final 1–2 weeks, we will evaluate all models, finalize visualizations, and complete the final report and presentation.
