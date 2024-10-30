import pandas as pd

# Mes tables
uszip = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\uszips.csv'
sales = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Sales.csv'
factory = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Factories.csv'
product = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Products.csv'
target = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Targets.csv'

# Lire les fichiers CSV
df_uszip = pd.read_csv(uszip)
df_sales = pd.read_csv(sales)
df_factory = pd.read_csv(factory)
df_product = pd.read_csv(product)
df_target = pd.read_csv(target)

# Afficher les premières lignes des DataFrames
print(df_uszip.head(10))
print(df_sales.head(10))
print(df_factory.head(10))
print(df_product.head(10))
print(df_target.head(10))

# Convertir les colonnes 'Postal Code' et 'zip' en type string
df_sales['Postal Code'] = df_sales['Postal Code'].astype(str)
df_uszip['zip'] = df_uszip['zip'].astype(str)
#faire une jointure sales[postal code] et uszip[(lat,lng)] et fusionner les 2 tables

df_merged = pd.merge(df_sales, df_uszip, left_on='Postal Code', right_on='zip', how='left')

# Afficher les premières lignes du DataFrame fusionné
print(df_merged.head(10))

# Afficher les colonnes pertinentes
print(df_merged[['Customer ID', 'City', 'State/Province', 'Postal Code', 'lat', 'lng']].head(10))

# Sauvegarder le DataFrame fusionné en tant que nouveau fichier CSV
df_merged.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\sales_uszip.csv', index=False)

print("Table saved as 'sales_uszip.csv'")

import pandas as pd
import numpy as np

# Chemin vers tes fichiers CSV
sales_uszip = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\sales_uszip.csv'
factory = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Factories.csv'

# Lire les fichiers CSV
df_sales_uszip = pd.read_csv(sales_uszip)
df_factory = pd.read_csv(factory)

# Vérifie les noms des colonnes
print(df_sales_uszip.columns)

# Sélectionner les colonnes nécessaires pour les clients (ajuste les noms des colonnes si nécessaire)
df_customer_loc = df_sales_uszip[['Row ID', 'Customer ID', 'City', 'State/Province', 'lat', 'lng']]
df_customer_loc['Type'] = 'Customer'

# Sélectionner les colonnes nécessaires pour les usines
df_factory_loc = df_factory[['Factory', 'Latitude', 'Longitude']]
df_factory_loc = df_factory_loc.rename(columns={'Latitude': 'lat', 'Longitude': 'lng'})
df_factory_loc['Type'] = 'Factory'

# Ajouter une colonne Row ID à df_factory_loc avec des valeurs NaN
df_factory_loc['Row ID'] = np.nan

# Concaténer les deux tables en une seule
df_customer_factory_loc = pd.concat([df_customer_loc, df_factory_loc], axis=0, ignore_index=True)

# Sauvegarder la nouvelle table en tant que fichier CSV
df_customer_factory_loc.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\customer_factory_loc.csv', index=False)
print("Table saved as 'customer_factory_loc.csv'")


import pandas as pd
import folium

# Chemin vers ton fichier CSV
customer_factory_loc= r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\customer_factory_loc.csv'

# Lire le fichier CSV
df = pd.read_csv(customer_factory_loc)

# Vérifie les noms des colonnes
print(df.columns)

# Filtrer les lignes contenant des valeurs NaN dans les colonnes 'lat' et 'lng'
df = df.dropna(subset=['lat', 'lng'])

# Créer une carte centrée sur une position moyenne
map_center = [df['lat'].mean(), df['lng'].mean()]
mymap = folium.Map(location=map_center, zoom_start=5)

# Ajouter les positions des clients et des usines sur la carte
for _, row in df.iterrows():
    if row['Type'] == 'Customer':
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"Customer ID: {row['Customer ID']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(mymap)
    elif row['Type'] == 'Factory':
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"Factory: {row['Factory']}",
            icon=folium.Icon(color='red', icon='industry')
        ).add_to(mymap)

# Sauvegarder la carte en tant que fichier HTML
mymap.save(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\customer_factory_map.html')

print("Map saved as 'customer_factory_map.html'")

import pandas as pd
from geopy.distance import geodesic

# Chemin vers tes fichiers CSV
sales_uszip = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\sales_uszip.csv'
factory = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Factories.csv'

# Lire les fichiers CSV
df_sales_uszip = pd.read_csv(sales_uszip)
df_factory = pd.read_csv(factory)

# Sélectionner les colonnes nécessaires pour les clients
df_customer_loc = df_sales_uszip[['Row ID', 'Customer ID', 'City', 'State/Province', 'Country/Region', 'lat', 'lng']]

# Sélectionner les colonnes nécessaires pour les usines
df_factory_loc = df_factory[['Factory', 'Latitude', 'Longitude']]
df_factory_loc = df_factory_loc.rename(columns={'Latitude': 'lat_factory', 'Longitude': 'lng_factory'})

# Filtrer les lignes contenant des valeurs NaN dans les colonnes 'lat' et 'lng'
df_customer_loc = df_customer_loc.dropna(subset=['lat', 'lng'])
df_factory_loc = df_factory_loc.dropna(subset=['lat_factory', 'lng_factory'])

# Calculer les distances entre chaque client et chaque usine
distances = []
for _, customer in df_customer_loc.iterrows():
    for _, factory in df_factory_loc.iterrows():
        distance = geodesic((customer['lat'], customer['lng']), (factory['lat_factory'], factory['lng_factory'])).kilometers
        distances.append({
            'Customer ID': customer['Customer ID'],
            'Row ID': customer['Row ID'],
            'Country/Region': customer['Country/Region'],
            'City': customer['City'],
            'State/Province': customer['State/Province'],
            'Factory': factory['Factory'],
            'Distance': distance
        })

# Convertir la liste de distances en DataFrame
df_distances = pd.DataFrame(distances)

# Sauvegarder la nouvelle table en tant que fichier CSV
df_distances.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\customer_factory_distances.csv', index=False)

print("Table saved as 'customer_factory_distances.csv'")

import pandas as pd
from geopy.distance import geodesic
import folium

# Chemin vers tes fichiers CSV
customer_factory_loc_path = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\customer_factory_loc.csv'

# Lire le fichier CSV
df = pd.read_csv(customer_factory_loc_path)

# Filtrer les lignes contenant des valeurs NaN dans les colonnes 'lat' et 'lng'
df = df.dropna(subset=['lat', 'lng'])

# Créer une carte centrée sur une position moyenne
map_center = [df['lat'].mean(), df['lng'].mean()]
mymap = folium.Map(location=map_center, zoom_start=5)

# Ajouter les positions des clients et des usines sur la carte
for _, row in df.iterrows():
    if row['Type'] == 'Customer':
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"Customer ID: {row['Customer ID']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(mymap)
    elif row['Type'] == 'Factory':
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"Factory: {row['Factory']}",
            icon=folium.Icon(color='red', icon='industry')
        ).add_to(mymap)

# Calculer et ajouter les distances entre chaque client et chaque usine
for _, customer in df[df['Type'] == 'Customer'].iterrows():
    for _, factory in df[df['Type'] == 'Factory'].iterrows():
        distance = geodesic((customer['lat'], customer['lng']), (factory['lat'], factory['lng'])).kilometers
        folium.PolyLine(
            locations=[(customer['lat'], customer['lng']), (factory['lat'], factory['lng'])],
            color='green',
            weight=2.5,
            opacity=1
        ).add_to(mymap)
        folium.Marker(
            location=[(customer['lat'] + factory['lat']) / 2, (customer['lng'] + factory['lng']) / 2],
            popup=f"Distance: {distance:.2f} km",
            icon=folium.DivIcon(html=f'<div style="font-size: 12px; color: green;">{distance:.2f} km</div>')
        ).add_to(mymap)

# Sauvegarder la carte en tant que fichier HTML
mymap.save(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\customer_factory_distance_map.html')

print("Map saved as 'customer_factory_distance_map.html'")



# Chemin vers tes fichiers CSV
sales_uszip = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\sales_uszip.csv'
customer_factory_distances = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\customer_factory_distances.csv'

# Lire les fichiers CSV
df_sales_uszip = pd.read_csv(sales_uszip)
df_customer_factory_distances = pd.read_csv(customer_factory_distances)

# Sélectionner les colonnes nécessaires de la table sales_uszip
df_sales_uszip_selected = df_sales_uszip[['Row ID', 'Customer ID', 'Product Name', 'Division', 'Ship Mode', 'Sales', 'Units', 'Gross Profit', 'Cost', 'City', 'state_id', 'state_name', 'county_name', 'population', 'density', 'county_fips']]

# Fusionner les deux tables en utilisant Customer ID comme clé primaire
df_merged = pd.merge(df_sales_uszip_selected, df_customer_factory_distances[['Customer ID', 'Distance']], on='Customer ID', how='left')

# Sauvegarder la nouvelle table en tant que fichier CSV
df_merged.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\cost_ship_distances.csv', index=False)
print("Table saved as 'cost_ship_distances.csv'")

import pandas as pd

# Chemin vers tes fichiers CSV
customer_factory_loc = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\customer_factory_loc.csv'
cost_ship_distances = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\cost_ship_distances.csv'
candy_products = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Products.csv'

# Lire les fichiers CSV
df_customer_factory_loc = pd.read_csv(customer_factory_loc)
df_cost_ship_distances = pd.read_csv(cost_ship_distances)
df_candy_products = pd.read_csv(candy_products)

# Ajouter les colonnes Product ID, Unit Price et Unit Cost à la table cost_ship_distance
df_cost_ship_distance = pd.merge(df_cost_ship_distances, df_candy_products[['Factory','Product Name', 'Product ID', 'Unit Price', 'Unit Cost']], on='Product Name', how='left')

# Fusionner les tables customer_factory_loc et cost_ship_distance en utilisant Customer ID comme clé commune
df_merged = pd.merge(df_customer_factory_loc, df_cost_ship_distance, on='Customer ID', how='left')

# Sauvegarder la nouvelle table en tant que fichier CSV sous le nom de sales_profiling
df_merged.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\sales_profiling.csv', index=False)
print("Table saved as 'sales_profiling.csv'")
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Chemin vers tes fichiers CSV
customer_factory_loc = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\customer_factory_loc.csv'
cost_ship_distance = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\cost_ship_distances.csv'
candy_products = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Products.csv'

# Lire les fichiers CSV
df_customer_factory_loc = pd.read_csv(customer_factory_loc)
df_cost_ship_distance = pd.read_csv(cost_ship_distance)
df_candy_products = pd.read_csv(candy_products)

# Ajouter les colonnes Product ID, Unit Price et Unit Cost à la table cost_ship_distance
df_cost_ship_distance = pd.merge(df_cost_ship_distance, df_candy_products[['Product Name', 'Product ID', 'Unit Price', 'Unit Cost']], on='Product Name', how='left')

# Fusionner les tables customer_factory_loc et cost_ship_distance en utilisant Customer ID comme clé commune
df_sales_profiling = pd.merge(df_customer_factory_loc, df_cost_ship_distance, on='Customer ID', how='left')

# Sauvegarder la nouvelle table en tant que fichier CSV sous le nom de sales_profiling
df_sales_profiling.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\sales_profiling.csv', index=False)
print("Table saved as 'sales_profiling.csv'")

# Lire la table sales_profiling
df_sales_profiling = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\sales_profiling.csv')

# Convertir les valeurs de Product ID en valeurs numériques
df_sales_profiling['Product ID'] = df_sales_profiling['Product ID'].astype('category').cat.codes

# Sélectionner les caractéristiques pour le clustering
features = df_sales_profiling[['Product ID', 'Sales', 'Units', 'Gross Profit', 'Cost', 'Distance']]

# Remplacer les valeurs manquantes par la moyenne des colonnes correspondantes
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Créer une nouvelle table features_sales_profiling
df_features_sales_profiling = df_sales_profiling[['Customer ID', 'Product ID', 'Sales', 'Units', 'Gross Profit', 'Cost', 'Distance']]

# Appliquer l'algorithme K-means
kmeans = KMeans(n_clusters=3)
df_features_sales_profiling['Cluster'] = kmeans.fit_predict(features_imputed)

### Sauvegarder la nouvelle table en tant que fichier CSV sous le nom de features_sales_profiling
df_features_sales_profiling.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\features_sales_profiling.csv', index=False)
print("Table saved as 'features_sales_profiling.csv'")

import matplotlib.pyplot as plt

# Visualiser les clusters avec légende
plt.scatter(df_features_sales_profiling['Sales'], df_features_sales_profiling['Gross Profit'], c=df_features_sales_profiling['Cluster'], cmap='viridis')
plt.xlabel('Sales')
plt.ylabel('Gross Profit')
plt.title('Segmentation des clients selon les produits les plus achetés')

# Ajouter une légende
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = set(labels)
plt.legend(handles, unique_labels, title='Clusters')
plt.show()

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Chemin vers tes fichiers CSV
sales_profiling = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\sales_profiling.csv'
candy_products = r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Products.csv'

# Lire les fichiers CSV
df_sales_profiling = pd.read_csv(sales_profiling)
df_candy_products = pd.read_csv(candy_products)

# Ajouter une colonne Product Type à la table sales_profiling
df_sales_profiling = pd.merge(df_sales_profiling, df_candy_products[['Product ID', 'Division']], on='Product ID', how='left')
df_sales_profiling.rename(columns={'Division': 'Product Type'}, inplace=True)

# Convertir les valeurs de Product ID en valeurs numériques
df_sales_profiling['Product ID'] = df_sales_profiling['Product ID'].astype('category').cat.codes

# Sélectionner les caractéristiques pour le clustering
features = df_sales_profiling[['Product ID', 'Sales', 'Units', 'Gross Profit', 'Cost', 'Distance']]

# Remplacer les valeurs manquantes par la moyenne des colonnes correspondantes
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Créer une nouvelle table features_sales_profiling
df_features_sales_profiling = df_sales_profiling[['Customer ID', 'Product ID', 'Sales', 'Units', 'Gross Profit', 'Cost', 'Distance', 'Product Name']]

# Appliquer l'algorithme K-means
kmeans = KMeans(n_clusters=3)
df_features_sales_profiling['Cluster'] = kmeans.fit_predict(features_imputed)

# Sauvegarder la nouvelle table en tant que fichier CSV sous le nom de features_sales_profiling
df_features_sales_profiling.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\features_sales_profiling.csv', index=False)
print("Table saved as 'features_sales_profiling.csv'")

import matplotlib.pyplot as plt

# Visualiser les clusters avec légende
plt.scatter(df_features_sales_profiling['Sales'], df_features_sales_profiling['Gross Profit'], c=df_features_sales_profiling['Cluster'], cmap='viridis')
plt.xlabel('Sales')
plt.ylabel('Gross Profit')
plt.title('Segmentation des clients selon les produits les plus achetés')

# Ajouter une légende
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = set(labels)
plt.legend(handles, unique_labels, title='Clusters')
plt.show()

import pandas as pd

# Lire le fichier CSV
df = pd.read_csv('sales_profiling.csv')

# Afficher les colonnes et les types de données
print(df.dtypes)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Lire le fichier CSV
df = pd.read_csv('sales_profiling.csv')

# Afficher les premières lignes du dataframe
print(df.head())

# Afficher les statistiques descriptives
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum())

# Visualiser les corrélations entre les variables numériques
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
with open('Essai_projet_1.py', 'w') as file:
    file.write('# Ceci est un nouveau fichier Python\n')
