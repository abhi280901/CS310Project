from bs4 import BeautifulSoup
import requests
import pandas as pd

url = 'https://www.dexerto.com/gaming/all-marvel-snap-characters-full-launch-roster-future-cards-1838195/'
page = requests.get(url)
soup = BeautifulSoup(page.text,'html.parser')
table1 = soup.find_all('table')[0]
table2 = soup.find_all('table')[1]
table3 = soup.find_all('table')[2]
table4 = soup.find_all('table')[3]
table5 = soup.find_all('table')[4]
table6 = soup.find_all('table')[5]
titles = ['Card', 'Cost', 'Power', 'Card Ability']
df = pd.DataFrame(columns = titles)
column_data = table1.find_all('tr')
column_data2 = table2.find_all('tr')
column_data3 = table3.find_all('tr')
column_data4 = table4.find_all('tr')
column_data5 = table5.find_all('tr')
column_data6 = table6.find_all('tr')
for row in column_data[1:]:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]
    length = len(df)
    df.loc[length]  = individual_row_data
for row in column_data2[1:]:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]
    length = len(df)
    df.loc[length]  = individual_row_data
for row in column_data3[1:]:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]
    length = len(df)
    df.loc[length]  = individual_row_data
for row in column_data4[1:]:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]
    length = len(df)
    df.loc[length]  = individual_row_data
for row in column_data5[1:]:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]
    length = len(df)
    df.loc[length]  = individual_row_data
for row in column_data6[1:]:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]
    length = len(df)
    df.loc[length]  = individual_row_data
df.to_csv(r'msnap_cards2.csv', index = False)
