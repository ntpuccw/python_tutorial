# read data from EXCEL file
import pandas as pd
file_dir = '../Data/'
data = pd.read_excel(file_dir + 'Iris.xls')
# data = pd.read_excel('TaiwanBank.xlsx') # need pip install openpyxl
data.head() #  list default 5 rows
# data.info() # information for columns
# data.describe() # descriptive stats

data.head(42)
data['Sepal Length'].mean()
ax = data.mean().plot.bar()

# add a column entitled 'Percentage'
data['Percentage']=data['Sepal Length']/data['Sepal Width']
data.loc[data['Percentage'].argmax()]

