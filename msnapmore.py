from bs4 import BeautifulSoup
import requests
import pandas as pd

url = 'https://dotesports.com/general/news/how-many-cards-are-in-marvel-snap'
page = requests.get(url)
soup = BeautifulSoup(page.text,'html.parser')
table1 = soup.find_all('table')[0]
titles = ['Card', 'Cost', 'Power', 'Card Ability']
df = pd.DataFrame(columns = titles)
column_data = table1.find_all('tr')
for row in column_data[1:]:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]
    length = len(df)
    df.loc[length]  = individual_row_data
df.to_csv(r'msnap_cards.csv', index = False)