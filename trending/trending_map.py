#!/usr/local/bin/python3

import folium as fo
import pandas as pd
import difflib

fake_news_data=pd.read_csv('trend_user.csv')

cols=['state','label']
fake_news_data.columns=cols

locations=pd.read_csv('worldcities.csv') # Need to look for a better dataset with more accurate lat and lng. (Found one but it's not free :( )

cities=list(locations['city_ascii'].dropna())
lat=list(locations['lat'].dropna())
lng=list(locations['lng'].dropna())
state=list(fake_news_data['state'].fillna('New York'))#smara data
label=list(fake_news_data['label'].fillna('true'))#smara data

def remove_null(list_items):
  while('' in list_items):
    list_items.remove('')
    list_items.remove(' ')
  return list_items

cities=remove_null(cities)
lat=remove_null(lat)
lng=remove_null(lng)
print(len(cities),len(lat),len(lng))

coordinates={}
for city,la,ln in zip(cities,lat,lng):
  coordinates[city]=[la,ln]

def find_city(city):
  # most_relative_city=difflib.get_close_matches(city,coordinates.keys(),n=1)
  return city

uniqe_state=list(set(state))

class city:
  location=[]
  def __init__(self,name):
    self.name=find_city(name)


  def find_location(self):
    self.location=coordinates[self.name]
    return self.location


  def sum_news(self):
    counter=0
    sums={'barely-true': 0,'false': 0, 'half-true': 0,'mostly-true': 0,'pants-fire': 0,'true': 0 }
    index=uniqe_state.index(self.name)
    for item in state:
      if item==self.name:
        counter=counter+1
    if len(label)==len(state):
      for i in range(len(label)):
        if state[i]==self.name:
          if label[i]=='barely-true':
            sums['barely-true']=sums['barely-true']+1
          elif label[i]=='false':
            sums['false']=sums['false']+1
          elif label[i]=='half-true':
            sums['half-true']=sums['half-true']+1
          elif label[i]=='mostly-true':
            sums['mostly-true']=sums['mostly-true']+1
          elif label[i]=='pants-fire':
            sums['pants-fire']=sums['pants-fire']+1
          elif label[i]=='true':
            sums['true']=sums['true']+1
      return [counter,sums]



  def calculate_rate(self):
    ratesum=0
    tools=self.sum_news()
    for key , value in tools[1].items():
      if key == 'true':
        ratesum=ratesum+(value*1)
      if key =='half-true':
        ratesum=ratesum+(value*0.6)
      if key =='mostly-true':
        ratesum=ratesum+(value*0.8)
      if key =='barely-true':
        ratesum=ratesum+(value*0.4)
      if key =='pants-fire':
        ratesum=ratesum+(value*0.2)
    return ratesum/tools[0]


  def set_color(self):
    rate=self.calculate_rate()*100
    if rate < 30.0:
      return 'red'
    elif  30.0 <= rate < 70.0:
      return 'yellow'
    else:
      return 'green'


  def calculate_radious(self):
    rate=self.calculate_rate()*100
    if rate < 10.0:
      return 3
    elif  10.0 <= rate < 30.0:
      return 7
    elif   30.0 <= rate < 50.0:
      return 11
    elif 50.0 <= rate < 70.0:
      return 15
    elif 70.0 <= rate <100.0:
      return 20

def create_base():
  all_locations=[]
  for item in uniqe_state:
    if item in cities:
      all_locations.append(city(item))
  return all_locations

all_locations=create_base()

map=fo.Map(tiles='OpenStreetMap')
fg=fo.FeatureGroup(name='truth')
for city in all_locations:
  fg.add_child(fo.CircleMarker(location=city.find_location(),popup=f'{str(round(city.calculate_rate()*100,2))}% of news published here are true',
                               radius=city.calculate_radious(),fill_color=city.set_color(),color=city.set_color(),fill_opacity=0.7))
map.add_child(fg)
map.save('../../webserver/public/trending_map.html')
