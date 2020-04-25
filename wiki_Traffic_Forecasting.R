
###########################################



##########################################

# Clear all the variables in workspace
rm(list = ls())

# Load the packages
library(fpp2)

require(data.table)
require(TSA)
require(forecast)
require(xts)
require(tseries)
require(graphics)
require(dplyr)
require(ggplot2)

#require(prophet)

# Setting the working drectory
setwd('D:\\github\\dataSets\\kaggle\\web_traffic_forecasting\\')

# Load the data
train <- fread("train_1.csv")

#######################################
# Data Pre-Processing and Explorations
#######################################

head(train)

# From the data we can see there are lot of web Pages in the website.
# so lets'start picking an web page with maximum number of visit to analyze and build the model.

train$sum = rowSums(train[,2:551])

head(train %>% select(Page,sum) %>% arrange(desc(sum)))

top.Pages <- train %>% 
  select(Page,sum) %>% 
  arrange(desc(sum)) %>%
  #mutate(sum=log(sum)) %>%
  head() 

ggplot(data = top.Pages) + geom_bar(mapping = aes(x = Page,y=sum),stat = 'identity',fill='coral1') +
  xlab("Web Pages") +
  geom_label(aes(Page, sum, label = sum), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle("Overall Page Visits")

# We can see that Main Page is the one that getting maximum views/traffic.
# Let's Start modelling on that and make it generalize to every page

trainsep = train[which(train$sum == max(na.omit(train$sum))),]

f = t(trainsep[,-c(1,552)])
f = data.frame(f,row.names(f))
colnames(f) = c("f","d")
f$d = as.POSIXct(f$d, format="%Y-%m-%d")

t2_xts = xts(f$f, order.by = f$d)
t2_xts = t2_xts/1000

###############################
# Data Exploration
##############################

# Here I sampled the train data to 1/10 of all wiki Pages for faster runtime, 
# And will later iterate through each sample for the analysis. 
# This time we will analyze only the language specific projects and but drop the ones that belong to wiki media.




################################
# Splitting into train and test
################################

####################### Training set from 2015-07-01 to 2016-11-30 ############################

start_date = as.POSIXct("2015-07-01", format="%Y-%m-%d")
end_date = as.POSIXct("2016-11-30", format="%Y-%m-%d")
t2_tr = window(t2_xts, start = start_date, end = end_date)

#Converting the xts class to data frame format
t2_tr = data.frame(index(t2_tr),t2_tr)

#Changing the column names of t2_tr
colnames(t2_tr) = c("d","f")
rownames(t2_tr) = rownames("")

######################### Test set from 2016-12-01 till 2016-12-31 ############### 
start_date = as.POSIXct("2016-12-01", format="%Y-%m-%d")
end_date = as.POSIXct("2016-12-31", format="%Y-%m-%d")
t2_te = window(t2_xts, start = start_date, end = end_date)

t2_te = data.frame(index(t2_te),t2_te)

#Changing the column names of t2_te
colnames(t2_te) = c("d","f")
rownames(t2_te) = rownames("")


########################################
# Preliminary Analysis Before Modelling
########################################

########### Time Series PLot ########################
autoplot(t2_xts) + 
  ggtitle('Time Plot: Page Views Per Day') +
  ylab('Views (K)')

# Data doesnt has strong trend, this is due to less data(less than 2 years).
# But I can find seasonality, and outliers (peaks)


################### Test for stationarity ################

library(tseries)
adf.test(t2_tr$f) # p-valuse < 0.05

# test shows that our data is stationarity.

################## Investigating the seasonality ###############

# The data is from 2015-07-01 to 2016-12-31. there for it is less than two years and totally of 15 months of data. 
# we can apply weekly seasonality and monthly seasonality.  
# if the data is more than two years, we can also apply yealy seasonality.

# Theoretically there are only two seasonality 
y_2 <- msts(t2_tr$f, seasonal.periods=c(7,4.34*7))


autoplot(stl(y_2,s.window="periodic",robust=TRUE))
#plot clearly showed the seasonal and trend part.

######################################
# Modelling - 4 tpyes
# TBATS, BATS, ARIMA, STLM
#####################################

#####################
# 1.) TBATS 
# TBATS is an exponential smoothing model with Box-Cox transformation, ARMA errors, trend and seasonal components. 
# It tunes its parameters automatically. Very cool model, but it can't use external regressors
#####################

# fitting/training
tbats.model = tbats(y_2)
checkresiduals(tbats.model)

# forecasting the values
tbats.forecast = forecast(tbats.model,h=31)

# Visulization
autoplot(tbats.forecast) + 
  ggtitle('Forecasting the wikipedia Page for the next month(TBATS)') +
  ylab('Views (K)') +
  xlab("Days")
  

#Lets check on accuracy part 
accuracy(tbats.forecast$mean,t2_te$f) 
  
#              ME    RMSE     MAE      MPE     MAPE
#Test set 2997230 3423985 3045547 12.13825 12.39839

###################### Absolute Error ################################

#In real life absolute error rate may add more value to business.
tbatforecast = data.frame(t2_te$d,t2_te$f,tbats.forecast$mean)
colnames(tbatforecast) = c("d","actuals","tbats.forecast")

tbats.abs.error = abs(sum(tbatforecast$actuals)-sum(tbatforecast$tbats.forecast))/sum(tbatforecast$actuals)
tbats.abs.error

#0.1271615 - TBATS have performed approximatly 12% of error. 

############################# Prediction vs Actual Values ####################

#Lets have a look on the error
autoplot(cbind(tbatforecast$actuals, tbatforecast$tbats.forecast)) +
  ggtitle('TBATS Forecasted values vs Actual Values') +
  ylab('Views (in K)') +
  guides(fill=F) +
  labs(colour='Forecasting') +
  scale_colour_manual(labels=c('actual','forecasted'),values = c(1,2))

## From the graph we can see that the model was able to capture the seasonality but not the peaks. 
## Lets explore and confirm on residuals

tbatforecast$Residual = abs(tbatforecast$actuals - tbatforecast$tbats.forecast)

autoplot.zoo(tbatforecast$Residual) +
  ylab('Residuals')
## by residual plot we confirm that the model is not able to predict the peak values.
## Lets have the TBATS as the bench mark.

##########################
# 2.) BATS 
#########################

##################### fitting and forecasting #######################
bats.model = bats(y_2)
checkresiduals(bats.model)

# forecasting the values
bats.forecast = forecast(bats.model,h=31)

# Visulization
autoplot(bats.forecast) + 
  ggtitle('Forecasting the wikipedia Page for the next month(BATS)') +
  ylab('Views (K)') +
  xlab("Days")

#Lets check on accuracy part 
accuracy(bats.forecast$mean,t2_te$f)

#              ME    RMSE     MAE     MPE     MAPE
#Test set 3738600 4147897 3748747 15.2777 15.33233

###################### Absolute Error ################################

#in real life absolute error rate may add more value to business.
forecastValue = data.frame(t2_te$d,t2_te$f,tbats.forecast$mean,bats.forecast$mean)
colnames(forecastValue) = c("d","actuals","tbats.forecast","bats.forecast")

bats.abs.error = abs(sum(forecastValue$actuals)-sum(forecastValue$bats.forecast))/sum(forecastValue$actuals)
bats.abs.error

#0.1586151
#BATS have perform badly when compared to TBATS.

############################# Prediction vs Actual Values ####################

#Lets have a look on the error 

autoplot(cbind(forecastValue$actuals, forecastValue$bats.forecast)) +
  ggtitle('BATS Forecasted values vs Actual Values') +
  ylab('Views (in K)') +
  guides(fill=F) +
  labs(colour='Forecasting') +
  scale_colour_manual(labels=c('actual','forecasted'),values = c(1,2))

## from the graph we can see that the model does not able to capture anything. 
## Lets explore and confirm on residuals

batsResidual = abs(forecastValue$actuals - forecastValue$bats.forecast)

autoplot.zoo(tbatforecast$Residual) +
  ylab('Residuals')

#########################
# 3.) STLM with ARIMA 
#########################

####################### fitting and forecasting ##############

stlm.model = stlm(y_2,s.window="periodic")
checkresiduals(stlm.model)

# forecasting the values
stlm.forecast = forecast(stlm.model,h=31)

# Visulization
autoplot(stlm.forecast) + 
  ggtitle('Forecasting the wikipedia Page for the next month(STLM)') +
  ylab('Views (K)') +
  xlab("Days")


#Lets check on accuracy part 
accuracy(stlm.forecast$mean,t2_te$f)

#              ME    RMSE     MAE     MPE     MAPE
#Test set 4115535 4577968 4122950 16.7827 16.82066

#The model is really performing poor.

###################### Absolute Error ################################

#in real life absolute error rate may add more value to business.
forecastValue = data.frame(t2_te$d,t2_te$f,tbats.forecast$mean,stlm.forecast$mean,bats.forecast$mean)
colnames(forecastValue) = c("d","actuals","tbats.forecast","stlm.forecast","bats.forecast")

stlm.abs.error = abs(sum(forecastValue$actuals)-sum(forecastValue$stlm.forecast,na.rm = T))/sum(forecastValue$actuals)
stlm.abs.error

#0.1746071
#STLM have perform badly when compared to Tbats. 

############################# Prediction vs Actual Values ####################

#Lets have a look on the error 
plot.zoo(cbind(forecastValue$actuals, forecastValue$stlm.forecast), 
         plot.type = "single", 
         col = c("red", "blue"))

autoplot(cbind(forecastValue$actuals, forecastValue$stlm.forecast)) +
  ggtitle('STLM Forecasted values vs Actual Values') +
  ylab('Views (in K)') +
  guides(fill=F) +
  labs(colour='Forecasting') +
  scale_colour_manual(labels=c('actual','forecasted'),values = c(1,2))

## from the graph we can see that the model does not able to capture anything. 
## Lets explore and confirm on residuals

stlmResidual = abs(forecastValue$actuals - forecastValue$stlm.forecast)

autoplot.zoo(stlmResidual)+
  ylab('Residuals')

#####################
# 4.) ARIMA  
# The goal of this notebook is to show how to tune ARIMA model with additional regressors. 
# We will add some Fourier terms to capture multiple seasonality and compare the best model with TBATS model.
#####################

####################### Fitting and forecasting ####################3333

bestfit = list()
bestfit <- list(aicc=Inf)

for(i in 1:3) {
  for (j in 1:3){
    f1xreg <- fourier(ts(t2_tr$f, frequency=7), K=i)
    f2xreg <- fourier(ts(t2_tr$f, frequency=7*4.34), K=j)
    arima.model <- auto.arima(t2_tr$f, xreg=cbind(f1xreg, f2xreg), seasonal=F)
    if(arima.model$aicc < bestfit$aicc) {
      bestfit <- list(aicc=arima.model$aicc, i=i, j=j, fit=arima.model)
    }
  }
}

xregm=cbind(fourier(ts(t2_tr$f, frequency=7), K=bestfit$i, h=31),
            fourier(ts(t2_tr$f, frequency=7*4.34), K=bestfit$j, h=31))

checkresiduals(bestfit$fit)

# forecasting the values
arima.forecast <- forecast(bestfit$fit, xreg=xregm)

# Visulization
autoplot(tbats.forecast) + 
  ggtitle('Forecasting the wikipedia Page for the next month(TBATS)') +
  ylab('Views (K)') +
  xlab("Days")

#Lets check on accuracy part 
accuracy(arima.forecast$mean,t2_te$f)

#              ME    RMSE     MAE     MPE     MAPE
#Test set 4439896 4804145 4439896 18.27674 18.27674

#The model is really performing poor.

###################### Absolute Error ################################

#in real life absolute error rate may add more value to business.
forecastValue = data.frame(t2_te$d,t2_te$f,tbats.forecast$mean,stlm.forecast$mean,bats.forecast$mean,arima.forecast$mean)
colnames(forecastValue) = c("d","actuals","tbats.forecast","stlm.forecast","bats.forecast","arima.forecast")

arima.abs.error = abs(sum(forecastValue$actuals)-sum(forecastValue$arima.forecast))/sum(forecastValue$actuals)
arima.abs.error

#0.1883685
#ARIMA have perform badly whem compared to Tbats. 

############################# Prediction vs Actual Values ####################

#Lets have a look on the error 
autoplot(cbind(forecastValue$actuals, forecastValue$arima.forecast)) +
  ggtitle('ARIMA Forecasted values vs Actual Values') +
  ylab('Views (in K)') +
  guides(fill=F) +
  labs(colour='Forecasting') +
  scale_colour_manual(labels=c('actual','forecasted'),values = c(1,2))

## from the graph we can see that the model does not able to capture anything. 
## Lets explore and confirm on residuals

arimaResidual = abs(forecastValue$actuals - forecastValue$arima.forecast)
autoplot.zoo(arimaResidual)

#############################
# Model Evaluation
##############################

autoplot(cbind(ts(forecastValue$actuals),
               ts(forecastValue$tbats.forecast),
               ts(forecastValue$bats.forecast),
               ts(forecastValue$stlm.forecast),
               ts(forecastValue$arima.forecast))) +
  ggtitle('All 4 model Forecasted values vs Actual Values') +
  ylab('Views (in K)') +
  guides(fill=F) +
  labs(colour='Forecasting') +
  scale_colour_manual(labels=c('actual','TBATS','BATS','STLM','ARIMA'),values = c(1,2,3,4,5))

### from the graph we can conclude that TBATS have perform better than the other model.

################
# Auto Arima
################

####################### fitting and forecasting ##############

tbats.model = auto.arima(y_2,stepwise = F,approximation = F,trace = T)
print(summary(tbats.model))
checkresiduals(tbats.model)

# forecasting the values
tbats.forecast = forecast(tbats.model,h=31)

# Visulization
autoplot(tbats.forecast) + 
  ggtitle('Forecasting the wikipedia Page for the next month(TBATS)') +
  ylab('Views (K)') +
  xlab("Days")


#Lets check on accuracy part 
accuracy(tbats.forecast$mean,t2_te$f) 

#              ME    RMSE     MAE      MPE     MAPE
#Test set 2997230 3423985 3045547 12.13825 12.39839

###################### Absolute Error ################################

#In real life absolute error rate may add more value to business.
tbatforecast = data.frame(t2_te$d,t2_te$f,tbats.forecast$mean)
colnames(tbatforecast) = c("d","actuals","tbats.forecast")

tbats.abs.error = abs(sum(tbatforecast$actuals)-sum(tbatforecast$tbats.forecast))/sum(tbatforecast$actuals)
tbats.abs.error

#0.1271615 - TBATS have performed approximatly 12% of error. 

############################# Prediction vs Actual Values ####################

#Lets have a look on the error
autoplot(cbind(tbatforecast$actuals, tbatforecast$tbats.forecast)) +
  ggtitle('TBATS Forecasted values vs Actual Values') +
  ylab('Views (in K)') +
  guides(fill=F) +
  labs(colour='Forecasting') +
  scale_colour_manual(labels=c('actual','forecasted'),values = c(1,2))

## From the graph we can see that the model was able to capture the seasonality but not the peaks. 
## Lets explore and confirm on residuals

tbatforecast$Residual = abs(tbatforecast$actuals - tbatforecast$tbats.forecast)

autoplot.zoo(tbatforecast$Residual) +
  ylab('Residuals')
## by residual plot we confirm that the model is not able to predict the peak values.
## Lets have the TBATS as the bench mark.

