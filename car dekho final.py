#!/usr/bin/env python
# coding: utf-8

# # Import relevant libraries and modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# # import the data

# In[2]:


df=pd.read_csv('Cardekho Dataset.csv')
df.head()


# In[3]:


df1=df.copy()


# # Data cleaning and preprocessing

# # Exploring the descriptive statistics

# In[4]:


df1.info() #to see features,number of rows, non null count and datatype


# In[5]:


df1.describe() #gives us the discription of only the features that are numerical


# In[6]:


df1.describe(include='all') #describes all the features


# ## deal with missing or null values

# In[7]:


df1.isnull().sum() ## to check if there is any null values


# In[8]:


df1=df1.dropna(axis=0) ##no null value is present now
df1.isnull().sum()


# In[9]:


df1.shape


# In[10]:


type(df1['seats'][0])## number of seats in a car can't be a float value so we change it to int


# In[11]:


df1=df1.astype({'seats': int})


# In[12]:


type(df1['seats'][0])


# In[13]:


## we can remove name column as it has very little significance and thousand dummies can create a problem


# In[14]:


df1=df1.drop(['name'],axis=1)


# In[15]:


df1.head()


# In[16]:


df1['torque_n']=df1.torque.str.split(expand=True,)[0] ## we need to split teh torque column and fetch the torque values


# In[17]:


df1.head()


# In[18]:


## as the column has been split still there are values where units are different or some special character needs to
##be removed


# In[19]:


df1['torque_n']=df1.torque_n.str.replace('@','') 


# In[20]:


df1['torque_n']=df1.torque_n.str.replace('Nm','')


# In[21]:


df1['torque_n']=df1.torque_n.str.replace('nm','')


# In[22]:


df1['torque_n']=df1.torque_n.str.replace('kgm','')


# In[23]:


df1['torque_n']=df1.torque_n.str.replace('NM','')


# In[24]:


df1['torque_n']=df1.torque_n.str.split('(',expand=True,)[0]


# In[25]:


df1['torque_n']=df1.torque_n.str.replace('(','')


# In[26]:


df1.shape


# In[27]:


df1['torque_n'].unique()


# In[28]:


df1=df1.astype({'torque_n': float}) # to change the dtype of torque from str to float


# In[29]:


df1['torque_n'][0].dtype


# In[30]:


df1.head()


# In[31]:


##we see that some values are very low in magnitude but on googling a little we find that torque can't be less than 50 in Nm
##usually it is near 100 or above but if values is less than 50 it is probably in Kgm and needs to be converted to Nm by
## below function


# In[32]:


df1['final_torque']=df1.torque_n.apply(lambda x: x*9.8 if x<48 else x)


# In[33]:


df1.head()


# In[34]:


## as we have a final column that depicts torque well, we can remove the other columns for torque
df2=df1.drop(['torque','torque_n'],axis=1)
df2.head()


# In[35]:


df2['max_power'][0]


# In[36]:


## we see that max power has bhp unit attached to it...we shall seperate the numerical value


# In[37]:



df2.max_power.str.split(expand=True,)


# In[38]:


df2.max_power.str.split(expand=True,)[1].unique()


# In[39]:


df2['horsepower']=df2.max_power.str.split(expand=True,)[0]


# In[40]:


df2.head()


# In[41]:


df2=df2.astype({'horsepower':float})
df2.horsepower[1].dtype


# In[42]:


## we can drop the max power column as it is no longer needed
df3=df2.drop(['max_power'],axis=1)
df3.head()


# In[43]:


## similarily we will split the engine as well


# In[44]:


df3.engine.str.split(expand=True,)


# In[45]:


df3.engine.str.split(expand=True,)[1].unique()


# In[46]:


df3['engine_capacity']=df3.engine.str.split(expand=True,)[0]
df3.head()


# In[47]:


df4=df3.drop(['engine'],axis=1)
df4.head()


# In[48]:


type(df4['engine_capacity'][0])


# In[49]:


df4=df4.astype({'engine_capacity':int})


# In[50]:


df4['engine_capacity'][0].dtype


# In[51]:


## with mileage too we will do the same


# In[52]:


df4['mileage'][0]


# In[53]:


df4.mileage.unique()


# In[54]:


df4.mileage.str.split(expand=True,)[1].unique()


# In[55]:


df4['mileage_new']=df4.mileage.str.split(expand=True,)[0]
df4.head()


# In[56]:


df4=df4.astype({'mileage_new':float})
df4.mileage_new.dtype


# In[57]:


df5=df4.drop(['mileage'],axis=1)
df5.head()


# In[58]:


df.columns.values


# In[59]:


df5.columns.values


# In[60]:


columns_renamed=['year','selling_price','km_driven', 'fuel', 'seller_type',
       'transmission', 'owner', 'seats','torque','max_power','engine','mileage']


# In[61]:


## we shall rename the features as they were in the original dataset
df5.columns=columns_renamed
df5.head()


# In[62]:


cols=['selling_price','year','km_driven','torque','max_power','engine','mileage' ,'fuel', 'seller_type',
       'transmission', 'owner', 'seats']


# In[63]:


## re-arrange the columns for better imterpretation
df6=df5[cols]
df6.head()


# In[64]:


df6.corr()


# In[65]:


dataplot=sns.heatmap(df6.corr(),annot=True)
plt.show()


# # exploring the PDFs or probability distribution functions and dealing with outliers

# ### Year

# In[66]:


sns.displot(df6['year'])


# In[67]:


#years we are taking here is the near one as there are some vintage cars as well that can hamper our results
# we keep thhe values greater then top 1 percentile


# In[68]:


q = df6['year'].quantile(0.01)
df7 = df6[df6['year']>q]
df7.describe(include = 'all')


# In[69]:


sns.displot(df7['year'])


# In[70]:


df7.year.unique()


# In[71]:


df7.shape


# #### as year generally tells us the year when the car was purchased. We can find the age of the car by subtracting it from present year

# In[72]:


df7['age_of_car']=df7['year'].apply(lambda x: 2021-x)


# In[73]:


df7.head()


# In[74]:


np.sort(df7.age_of_car.unique())


# In[75]:


df7.columns.values


# In[76]:


cols=['selling_price', 'year', 'km_driven', 'torque', 'max_power',
       'engine', 'mileage',  'age_of_car','fuel', 'seller_type', 'transmission',
       'owner', 'seats']


# In[77]:


df8=df7[cols]
df8.head()


# In[78]:


## we can drop the year column
df8=df8.drop(['year'],axis=1)
df8.head()


# ### seats

# In[79]:


plt.boxplot(df8['seats'])
plt.show()


# In[80]:


## shows us that there are outliers as a car generally doesn't have 14 seats


# In[81]:


df8[df8.seats>10] ## as it is only one row we can drop it


# In[82]:


df8=df8.drop(4575)


# In[83]:


df8[df8.seats<4]


# In[84]:


df8=df8.drop(5900,axis=0)


# In[85]:


df8=df8.drop(6629,axis=0)


# In[86]:


df8.shape


# ### mileage

# In[87]:


sns.displot(df8.mileage)


# In[88]:


plt.boxplot(df8.mileage)


# In[89]:


df8[df8.mileage<9]


# In[90]:


## the mileage of a car can't be zero so we remove the values


# In[91]:


df8=df8.drop(df8[df8.mileage<9].index)


# In[92]:


plt.boxplot(df8.mileage)


# In[93]:


## there is one more outlier we can remove


# In[94]:


df8[df8.mileage>40]


# In[95]:


df8=df8.drop(df8[df8.mileage>40].index)


# In[96]:


df8.shape


# In[97]:


sns.displot(df8.mileage)


# ### engine

# In[98]:


sns.displot(df8.engine)


# In[99]:


plt.boxplot(df8.engine)


# In[100]:


df8[df8.engine>3000]


# In[101]:


df8=df8.drop(df8[df8['engine']>3000].index)


# In[102]:


df8.shape


# ### max power

# In[103]:


sns.displot(df8.max_power)


# In[104]:


plt.boxplot(df8.max_power)


# ### torque

# In[105]:


sns.displot(df8.torque)


# In[106]:


plt.boxplot(df8.torque)


# In[107]:


df8=df8.drop(df8[df8.torque>700].index) ## only one outlier that can be removed


# In[108]:


df8.shape


# In[109]:


df8.head()


# ### km driven

# In[110]:


## we can change the km driven in 1000 units


# In[111]:


df8['km_driven']=df8['km_driven'].apply(lambda x: x/1000)


# In[112]:


df8.head()


# In[113]:


sns.displot(df8.km_driven)


# In[114]:


df8.km_driven.max()


# In[115]:


plt.boxplot(df8.km_driven)


# In[116]:


## we remove the top 1% of our data to maintain effeciency and deal with the outliers
## for this we use quantile method


# In[117]:


q= df8['km_driven'].quantile(0.99)
df9 = df8[df8['km_driven']<q]
df9.shape


# In[118]:


sns.distplot(df9.km_driven)


# In[119]:


df9.describe(include='all')


# In[120]:


q= df9['torque'].quantile(0.99)
df10 = df9[df9['torque']<q]
df10.shape


# In[121]:


## we can do this with torque and max power as well


# In[122]:


sns.distplot(df10.torque)


# In[123]:


q= df10['max_power'].quantile(0.99)
df11 = df10[df10['max_power']<q]
df11.shape


# In[124]:


sns.distplot(df11.max_power)


# In[125]:


df12=df11.astype({'seats':str})
df12.describe(include='all')


# In[126]:


data_cleaned=df12.reset_index(drop=True)
data_cleaned


# In[127]:


data_cleaned.describe()


# # checking the OLS assumptions

# In[128]:


f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey = True,  figsize = (15,3))
ax1.scatter(data_cleaned['age_of_car'],data_cleaned['selling_price'])
ax1.set_title('selling_price and age of car')

ax2.scatter(data_cleaned['mileage'],data_cleaned['selling_price'])
ax2.set_title('Price and mileage')

ax3.scatter(data_cleaned['engine'],data_cleaned['selling_price'])
ax3.set_title('Price and engine')

plt.show()


# In[129]:


f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey = True,  figsize = (15,3))
ax1.scatter(data_cleaned['max_power'],data_cleaned['selling_price'])
ax1.set_title('selling_price and max_power')

ax2.scatter(data_cleaned['torque'],data_cleaned['selling_price'])
ax2.set_title('Price and torque')

ax3.scatter(data_cleaned['km_driven'],data_cleaned['selling_price'])
ax3.set_title('Price and kms')

plt.show()


# In[130]:


sns.distplot(data_cleaned['selling_price'])


# In[131]:


##we can transform our price variable with the help of Log Transformation
data_cleaned['log_price']=np.log(data_cleaned['selling_price'])
data_cleaned.head()


# In[132]:


f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey = True,  figsize = (15,3))
ax1.scatter(data_cleaned['age_of_car'],data_cleaned['log_price'])
ax1.set_title('selling_price and age of car')

ax2.scatter(data_cleaned['mileage'],data_cleaned['log_price'])
ax2.set_title('Price and mileage')

ax3.scatter(data_cleaned['engine'],data_cleaned['log_price'])
ax3.set_title('Price and engine')

plt.show()


# In[133]:


f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey = True,  figsize = (15,3))
ax1.scatter(data_cleaned['max_power'],data_cleaned['log_price'])
ax1.set_title('selling_price and max_power')

ax2.scatter(data_cleaned['torque'],data_cleaned['log_price'])
ax2.set_title('Price and torque')

ax3.scatter(data_cleaned['km_driven'],data_cleaned['log_price'])
ax3.set_title('Price and kms')

plt.show()


# In[134]:


data_cleaned=data_cleaned.drop(['selling_price'],axis=1)


# # multicollinearity
# 

# In[135]:


# for checking this we use VIF variation inflation factor


# In[136]:


data_cleaned.columns.values


# In[137]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['km_driven','torque','max_power','engine','mileage','age_of_car']]
variables.shape
vif = pd.DataFrame()
vif['VIF'] =[variance_inflation_factor(variables.values,i) for i in range (variables.shape[1])]
vif['Features'] = variables.columns


# In[138]:


vif


# In[139]:


## we see that max power has highest collinearity with other features so we can remove it


# In[140]:


data_cleaned=data_cleaned.drop(['max_power'],axis=1)


# In[141]:


variables = data_cleaned[['km_driven','torque','engine','mileage','age_of_car']]
variables.shape
vif = pd.DataFrame()
vif['VIF'] =[variance_inflation_factor(variables.values,i) for i in range (variables.shape[1])]
vif['Features'] = variables.columns


# In[142]:


vif


# In[143]:


## we can remove engine as well


# In[144]:


data_cleaned=data_cleaned.drop(['engine'],axis=1)


# In[145]:


variables = data_cleaned[['km_driven','torque','mileage','age_of_car']]
variables.shape
vif = pd.DataFrame()
vif['VIF'] =[variance_inflation_factor(variables.values,i) for i in range (variables.shape[1])]
vif['Features'] = variables.columns


# In[146]:


vif


# In[147]:


data_cleaned.head()


# In[148]:


data_cleaned.describe()


# # get dummies for categorical features

# In[149]:


data_with_dummies=pd.get_dummies(data_cleaned,drop_first=True)
data_with_dummies.head()


# In[150]:


data_with_dummies.columns.values


# In[151]:


cols=['log_price','km_driven', 'torque', 'mileage', 'age_of_car', 
       'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual',
       'seller_type_Trustmark Dealer', 'transmission_Manual',
       'owner_Fourth & Above Owner', 'owner_Second Owner',
       'owner_Test Drive Car', 'owner_Third Owner', 'seats_4', 'seats_5',
       'seats_6', 'seats_7', 'seats_8', 'seats_9']


# In[152]:


data_preprocessed=data_with_dummies[cols]


# In[153]:


data_preprocessed.head()


# # select inputs and targets

# In[154]:


targets=data_preprocessed['log_price']
unscaled_inputs=data_preprocessed.drop(['log_price'], axis=1)


# In[155]:


unscaled_inputs.columns.values


# ### select features to scale

# In[156]:


columns_to_scale=unscaled_inputs[['km_driven','torque','mileage','age_of_car']]
columns_to_scale.head()


# In[157]:


columns_to_scale.columns.values


# ## scale the features

# In[158]:


from sklearn.preprocessing import StandardScaler


# In[159]:


scaler=StandardScaler()


# In[160]:


scaled_inputs=scaler.fit_transform(columns_to_scale)


# In[161]:


scaled_inputs


# In[162]:


scaled_inputs_df=pd.DataFrame(columns=columns_to_scale.columns.values,data=scaled_inputs)


# In[163]:


scaled_inputs_df.head()


# In[164]:


scaled_inputs_df[['fuel_Diesel','fuel_LPG', 'fuel_Petrol', 'seller_type_Individual','seller_type_Trustmark Dealer', 'transmission_Manual','owner_Fourth & Above Owner', 'owner_Second Owner','owner_Test Drive Car', 'owner_Third Owner', 'seats_4', 'seats_5',
'seats_6', 'seats_7', 'seats_8', 'seats_9']]=unscaled_inputs[['fuel_Diesel',
       'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual',
       'seller_type_Trustmark Dealer', 'transmission_Manual',
       'owner_Fourth & Above Owner', 'owner_Second Owner',
       'owner_Test Drive Car', 'owner_Third Owner', 'seats_4', 'seats_5',
       'seats_6', 'seats_7', 'seats_8', 'seats_9']]


# In[165]:


scaled_inputs_df.head()


# In[166]:


inputs=scaled_inputs_df.copy()


# # linear regression model

# # train test split

# In[167]:


from sklearn.model_selection import train_test_split


# In[168]:


x_train,x_test,y_train,y_test=train_test_split(inputs,targets,test_size=0.2,random_state=500)


# ## import and fit the model

# In[169]:


from sklearn.linear_model import LinearRegression


# In[170]:


reg=LinearRegression()


# In[171]:


##note that this is not just a linear regression but a log-linear regression


# In[172]:


reg.fit(x_train,y_train)


# In[173]:


y_pred=reg.predict(x_train)


# ### plot the targets with predictions

# In[174]:


plt.scatter(y_train,y_pred)
plt.xlabel('Targets',size=18)
plt.ylabel('predictions',size=18)


# ### there is a 45 degree line. The closer our scatter plot is to the line the better the model.

# #### second check is the residual plot. Residuals are the differences between the target and predicted values

# In[175]:


sns.distplot(y_train - y_pred)
plt.title('Residual PDF', size = 18)


# In[176]:


## it looks normally distributed with tail at lift side


# #### let's check the accuracy of the model on train data

# In[177]:


reg.score(x_train,y_train)


# # finding weights and bias

# In[178]:


reg.intercept_


# In[179]:


reg.coef_


# In[180]:


reg_summary=pd.DataFrame(columns=['Features'],data=inputs.columns.values)
reg_summary['Weights']=reg.coef_
reg_summary


# In[181]:


## a positive weight shows that if the feature increases in value so does the log_price and hence price
## vice versa for negative weights, if negative increases the price decreases 


# ## testing

# In[182]:


y_pred_test = reg.predict(x_test)


# In[183]:


plt.scatter(y_test,y_pred_test)
plt.xlabel('y_test',size=18)
plt.ylabel('y_pred_test',size=18)


# In[184]:


df_pf=pd.DataFrame(columns=['Predictions'],data=y_pred_test)
df_pf.head()


# In[185]:


#as our predicted values are in log we will change them to exp


# In[186]:


df_pf = pd.DataFrame(columns = ['Predictions'],data=np.exp(y_pred_test))
df_pf.head()


# In[187]:


df_pf['Targets']=np.exp(y_test)
df_pf.head()


# In[188]:


#we see that there are a lot of NA values, as y_test preserves old indexing that needs to be reset


# In[189]:


y_test=y_test.reset_index(drop=True)


# In[190]:


df_pf['Targets']=np.exp(y_test)
df_pf.head()


# In[191]:


df_pf['Residual'] = df_pf['Targets'] - df_pf['Predictions']


# In[192]:


##as we know ols method is based on minimizing the SSE sum of squared errors(residuals). Hence it is very imp to study residuals


# In[193]:


df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Targets']*100)


# In[194]:


df_pf


# In[195]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by = ['Difference%'])


# In[196]:


df_pf.describe()


# In[197]:


## it shows that till 75% we had a good data prediction but after that seeing the max values things get fishy


# In[198]:


reg.score(x_test,y_test)


# # Save the model

# In[199]:


import pickle


# In[200]:


with open('model','wb') as file:
    pickle.dump(reg,file)


# In[201]:


with open('scaler','wb') as file:
    pickle.dump(scaler,file)


# In[ ]:




