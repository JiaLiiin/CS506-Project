# CS506-Project

## Project Description 
Our project analyzes global energy consumption patterns and their relationship with economic indicators at the country level, aiming to study how energy consumption varies across countries and over time. We plan to use the World Energy Consumption dataset from Kaggle, which contains country-year data on energy usage, GDP, population, and energy mix composition. 

## Project Goals 
The primary goal of our project is to predict energy consumption per capita using economic and energy-related features such as GDP, population, and energy mix composition. Additionally, we want to analyze the relationship between GDP per capita and energy consumption per capita using GDP, population density, and fossil fuel share as features and examine trends in fossil fuel and renewable energy usage across countries over time using statistical analysis and visualizations.

## Data Collection Plan 
The data we plan to use is from the World Energy Consumption dataset available on Kaggle (https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption), which we will download using the Kaggle API. We noticed that all GDP from 2020-2022 are missing, and we will supplement this data from official sources like [World Bank Open Data's API](https://data360.worldbank.org/en/api) or by finding a site to scrape all relevant GDP information. After collection, we will clean the data by handling missing values, performing feature extraction, and filtering to focus on relevant years and countries. 

## Modeling Methods
We will focus on supervised machine learning to predict our target variable - energy consumption per capita. We will begin by implementing a baseline Multiple Linear Regression model to quantify the relationships between our core economic and demographic predictors and energy demand. Following this, we plan to experiment with other ML models to see how far we can improve our prediction accuracy.

We will use the following variables for each country to prevent data leakage: 
- Target Variable: Energy Consumption per Capita
- Economic Indicator: GDP per Capita
- Demographic Indicator: Population Density
- Energy Mix: Fossil Fuel Share (% of total energy)

These features were selected because they represent important factors that define a country's energy profile. GDP per Capita is an important metric because wealth plays a huge role in energy usage. Higher income nation typically have greater energy generation and better transportation networks that separates them from developing nations. Population Density is an important factor for the infrastructure efficiency. More dense countries share public transit and have smaller living spaces which means two countries with the same GDP might cluster differently based on how their population is distributed over space. Fossil fuel share shows us how a country's energy grid impacts its primary energy consumption. Fossil fuel generates waste heat so including this percentage allows the model to account for the overall efficiency and modernization of a nation's energy generation. 

## Project Timeline 
In the first 1–2 weeks, we will collect the data using the Kaggle API and perform initial data cleaning and feature selection. In the following week, we will conduct exploratory data analysis and create preliminary visualizations for the first project check-in. Over the next few weeks, we will implement and evaluate a baseline model, likely linear regression, to predict energy consumption per capita. We will then explore additional modeling approaches, such as clustering, to provide complementary insights into global energy consumption patterns. In the final 1–2 weeks, we will evaluate all models, finalize visualizations, and complete the final report and presentation.
