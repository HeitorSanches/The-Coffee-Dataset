#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd


# In[96]:


coffee_domestic_consumption = pd.read_csv("Coffee_domestic_consumption.csv")
coffee_export = pd.read_csv("Coffee_export.csv")
coffee_import = pd.read_csv("Coffee_import.csv")
coffee_production = pd.read_csv("Coffee_production.csv")


# In[97]:


coffee_domestic_consumption.head()


# In[98]:


zero_coffee_domestic_consumption = coffee_domestic_consumption.query("Total_domestic_consumption == 0") #Is there any 0 value?


# In[99]:


zero_coffee_domestic_consumption.index # Yes there is.


# In[100]:


coffee_domestic_consumption.drop(28,inplace = True) #removing one


# In[101]:


coffee_domestic_consumption.drop(42 , inplace = True) # removing other


# In[102]:


#There are no more 0 values! We can start.


# In[103]:


# Sorting the values by the highest to the lowest
coffee_domestic_consumption.sort_values(by = "Total_domestic_consumption",ascending = False , inplace = True)


# In[104]:


# Splitting the most important columns
coffee_total_consumption = coffee_domestic_consumption[["Country","Total_domestic_consumption"]]


# In[105]:


coffee_total_consumption.head()


# In[106]:


#Changing the index, to start with 1
coffee_total_consumption.index = range(coffee_total_consumption.shape[0])


# In[107]:


coffee_total_consumption.index += 1


# In[108]:


cdc = coffee_domestic_consumption 


# In[109]:


brazil = cdc[cdc["Country"] == "Brazil"] #choosing brazil


# In[110]:


brazil


# In[111]:


brazil_nums = brazil.select_dtypes(include = "number") #we want just the numeric values!


# In[112]:


brazil_nums


# In[113]:


bnv = brazil_nums.values #selecting just the values (without the columns)


# In[114]:


pd.DataFrame(bnv.T) #transforming the numpyarray in a dataframe, setting the values in a vertical direction


# In[115]:


bnc = brazil_nums.columns #setting the columns as index


# In[116]:


df = pd.DataFrame(bnv.T,index = bnc , columns = ["Valores"] ) #creating a new dataframe!
df.index.name = "Anos"

df


# In[117]:


df.drop("Total_domestic_consumption", axis = 0 , inplace = True) #dropping the row about the sum of the values


# In[118]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure(figsize =( 15,8))
eixo = fig.add_axes([0,0,1,1])
eixo.plot(df.index,df["Valores"])
eixo.set_title("Coffee Consumption x Years",fontsize= 20)
eixo.set_xlabel("Passing Years (1990-2020)",fontsize = 15)
eixo.set_ylabel("Coffe Consumptions (Billions)",fontsize = 15)
eixo

#Time to plot!


# In[119]:


import plotly.express as px

fig = px.histogram(df,x=df.index , y="Valores")
fig


# In[120]:


# We can clearly see that coffee consumption has been increasing over the years.
# But what about coffee production? It must be increasing at the same time, right?


# In[121]:


####################################################################################################################################


# In[122]:


coffee_production.sort_values(by="Total_production",ascending = False ,inplace = True)
cp = coffee_production 


# In[123]:


cp.index = range(cp.shape[0])
cp.index += 1


# In[124]:


cp.head()


# In[125]:


brazil2 = cp[cp["Country"] == "Brazil"] #again, just BraziL!
brazil2


# In[126]:


brazil2= brazil2.select_dtypes(include="number") #just numbers


# In[127]:


bnv2 = brazil2.values #just the values


# In[128]:


bnc2 = brazil2.columns


# In[129]:


df2 = pd.DataFrame(bnv2.T , index = bnc2 , columns = ["Valores"])
df2.drop("Total_production", axis = 0, inplace = True)


# In[130]:


fig2 = px.histogram(df2, x =df.index , y="Valores")
fig2


# In[131]:


# Ok! The two variables are increasing together!
# Let's see about coffee exports


# In[132]:


################################################################################################################################


# In[133]:


ce = coffee_export
brazil3 = ce[ce["Country"] == "Brazil"]


# In[134]:


brazil3 = brazil3.select_dtypes(include = "number")


# In[135]:


bnv3 = brazil3.values


# In[136]:


bnc3 = brazil3.columns


# In[137]:


df3 = pd.DataFrame(bnv3.T , index = bnc2 , columns = ["Valores"])
df3.drop("Total_production", axis=0,inplace=True)
df3



# In[138]:


#Wait,, there are some negative values on the dataframe, let's fix this.


# In[139]:


df3 = df3.abs() #the absolute values :)
df3


# In[140]:


one = pd.concat([df["Valores"],df2["Valores"],df3["Valores"]],axis=1)
one.columns = ["Consumption","Production","Export"]


# In[141]:


#Creating a new DataFrame,setting  our different values at the same index!


# In[142]:


one


# In[143]:


################################################################################################################################


# In[144]:


#So far, we have noticed that there are significant values related to the coffee industry in Brazil, but who buys the most? 
#Let's see


# In[145]:


coffee_import.head()


# In[146]:


ci = coffee_import


# In[147]:


ci = ci.sort_values(by="Total_import",ascending = False)

ci.index = range(ci.shape[0])
ci.index += 1


# In[148]:


ci #Of course is the USA!


# In[149]:


usaimport = ci[ci["Country"]=="United States of America"] #splitting just the usa imports


# In[150]:


usaimport = usaimport.select_dtypes(include = "number") #just numbers


# In[151]:


usav = usaimport.values
usav #just the values


# In[152]:


usac = usaimport.columns
usac


# In[153]:


df4 = pd.DataFrame(usav.T,index=bnc2,columns=["Valores"])


# In[154]:


two = pd.concat([df["Valores"],df2["Valores"],df3["Valores"],df4["Valores"]],axis=1)
two.columns= ["Consumption","Production","Export","USA Import"]


# In[155]:


two


# In[156]:


#Thinking about it, it doesn't make much sense to compare domestic consumption with imports, does it?


# In[157]:


two.drop("Consumption",axis = 1,inplace = True)  #Let's drop it


# In[158]:


two.drop("Total_production",axis = 0,inplace = True) #Removing an irrelevant variable


# In[159]:


two


# In[160]:


two.corr() #Let's see if there is any correlation between the values


# In[161]:


#There is a nice correlation between production/export and USA imports.


# In[163]:


#Let's see that in a graph

import seaborn as sns
ax = sns.pairplot(two,x_vars = ["Production","Export"],y_vars="USA Import",kind="reg",height = 8)


# In[ ]:


#Both of them seem to have a linear crescent relation.
#Let's build a model.


# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[ ]:


y = two["USA Import"] # Creating a series for the dependent variable.


# In[ ]:


X = two[["Production","Export"]]#Creating a dataframe for the explanatory variables.


# In[ ]:


#Splitting in train and test data through the method train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 2811)


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(X_train,y_train) #training 


# In[ ]:


model.score(X_train,y_train).round(3) #verificando se o modelo foi bem ajustado


# In[ ]:


#Creating an interactive simulator for the predictions


def format_number(num):
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        formatted_number = f'{num/1_000_000_000:.2f} billion'
    elif abs_num >= 1_000_000:
        formatted_number = f'{num/1_000_000:.2f} million'
    elif abs_num >= 1_000:
        formatted_number = f'{num/1_000:.2f} thoousand'
    else:
        formatted_number = f'{num}'
    
    return formatted_number
    
production = float(input("Enter in billions the amount of coffee production in Brazil:"))
production = production*10**9

export = float(input("Enter in billions the amount of coffee exportation from Brazil:"))
export = export*10**9


aux = [[production,export]]
aux = pd.DataFrame(aux, columns = ["Production","Export"])

qty_import= model.predict(aux)[0]
qty_import = format_number(qty_import)

print(f"The quantity of Brazilian coffee importation by the USA will be: {qty_import}")


# In[164]:


import ipywidgets as widgets
from IPython.display import display



def format_number(num):
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        formatted_number = f'{num/1_000_000_000:.2f} billion'
    elif abs_num >= 1_000_000:
        formatted_number = f'{num/1_000_000:.2f} million'
    elif abs_num >= 1_000:
        formatted_number = f'{num/1_000:.2f} thousand'
    else:
        formatted_number = f'{num}'
    
    return formatted_number

input_production = widgets.FloatText(description='Production (billion):',layout=widgets.Layout(margin='0 0 0 0'))
input_export = widgets.FloatText(description='Export (billion):',layout=widgets.Layout(margin='0 0 0 0px'))

button_calculate = widgets.Button(description='Calculate')

label_result = widgets.Label()

def calculate(aux):
    production = input_production.value
    production = production*10**9

    export = input_export.value
    export = export*10**9

    aux = [[production,export]]
    aux = pd.DataFrame(aux, columns = ["Production","Export"])

    qty_import = model.predict(aux)[0]
    qty_import = format_number(qty_import)
    
    label_result.value = (f"The quantity of Brazilian coffee imports by the USA will be: {qty_import} ")
    
    return qty_import
    
button_calculate.on_click(calculate)


display(widgets.VBox([input_production, input_export, button_calculate, label_result]))




#In[ ]:
#There is a positive correlation between Brazilian coffee supply and US demand.



