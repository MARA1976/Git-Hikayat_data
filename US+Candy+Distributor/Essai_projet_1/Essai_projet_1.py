import pandas as pd

# Lire le fichier CSV
df_u = pd.read_csv('uszips.csv')
df_cs= pd.read_csv('Candy_Sales.csv')
df_cp = pd.read_csv('Candy_Products.csv')
df_ct = pd.read_csv('Candy_Targets.csv')
df_cf = pd.read_csv('Candy_Factories.csv')
# Afficher les colonnes et les types de données
print(df_u.dtypes)
print(df_cs.dtypes)
print(df_cp.dtypes)
print(df_ct.dtypes)
print(df_cf.dtypes)
#Exploration des données
print(df_u.head())
print(df_cs.head())
print(df_cp.head())
print(df_ct.head())
print(df_cf.head())
# Compter (valeurs manquantes et valeurs abérantes)
print(df_u.isnull().sum())
print(df_cs.isnull().sum())
print(df_cp.isnull().sum())
print(df_ct.isnull().sum())
print(df_cf.isnull().sum())
# Convertir les colonnes de dates en format datetime
df_cs['Order Date'] = pd.to_datetime(df_cs['Order Date'])
df_cs['Ship Date'] = pd.to_datetime(df_cs['Ship Date'])
#Afficher les noms des colonnes df_cs 
print(df_cs.columns)
print(df_cs.dtypes)
# Convertir les colonnes 'Postal Code' et 'zip' en type string
df_cs['Postal Code'] = df_cs['Postal Code'].astype(str)
df_u['zip'] = df_u['zip'].astype(str)
#faire une jointure sales[postal code] et uszip[(lat,lng)] et fusionner les 2 tables

df_su = pd.merge(df_cs, df_u, left_on='Postal Code', right_on='zip', how='left')

# Afficher les premières lignes du DataFrame fusionné
print(df_su.head(10))

# Afficher les colonnes pertinentes
print(df_su[['Customer ID', 'City', 'State/Province', 'Postal Code', 'lat', 'lng']].head(10))

# Sauvegarder le DataFrame fusionné en tant que nouveau fichier CSV
df_su.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_su.csv', index=False)

print("Table saved as 'df_su.csv'")
print(df_su.columns)

#Supprimer les valeurs manquantes "Lat"

# Afficher le nombre de lignes avant suppression
print("Nombre de lignes avant suppression :", len(df_su))

# Supprimer les lignes où la colonne 'Lat' a des valeurs manquantes
df_su = df_su.dropna(subset=['lat'])

# Afficher le nombre de lignes après suppression
print("Nombre de lignes après suppression :", len(df_su))

# Sauvegarder le DataFrame nettoyé dans un nouveau fichier CSV
df_su.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_su_cleaned.csv', index=False)
# Charger les fichiers CSV dans des DataFrames
df_cs = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Sales.csv')
df_su_cleaned = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_su_cleaned.csv')

# Afficher les noms des colonnes et les types de données pour df_cs
print("Colonnes et types de données pour df_cs :")
print(df_cs.dtypes)

# Afficher les noms des colonnes et les types de données pour df_su_cleaned
print("Colonnes et types de données pour df_su_cleaned :")
print(df_su_cleaned.dtypes)
print(df_cp.dtypes)
print(df_cf.dtypes)



# Charger les fichiers CSV dans des DataFrames
import pandas as pd
df_su_cleaned = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_su_cleaned.csv')
df_cp = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Products.csv')
df_cf= pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Factories.csv')



print(df_su_cleaned.dtypes)
print(df_cp.dtypes)
print(df_cf.dtypes)


# Première jointure : ajouter la colonne "Factory" en utilisant "Product ID"
df_merged_1 = pd.merge(df_su_cleaned, df_cp[['Product ID', 'Factory']], on='Product ID', how='inner')

# Deuxième jointure : ajouter les colonnes "Latitude" et "Longitude" en utilisant "Factory"
df_merged_2 = pd.merge(df_merged_1, df_cf[['Factory', 'Latitude', 'Longitude']], on='Factory', how='inner')

# Sélectionner les colonnes spécifiques, y compris 'Country/Region'
columns_to_keep = ['Customer ID', 'State/Province', 'Postal Code', 'lat', 'lng', 'population', 'density', 'Product ID', 'Factory', 'Latitude', 'Longitude', 'Country/Region']
df_selected = df_merged_2[columns_to_keep]

# Filtrer les clients américains
df_us_customers = df_selected[df_selected['Country/Region'] == 'United States']

# Filtrer les clients non-américains
df_other_customers = df_selected[df_selected['Country/Region'] != 'United States']

# Sauvegarder les DataFrames filtrés dans des fichiers CSV séparés
df_us_customers.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers.csv', index=False)
df_other_customers.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_other_customers.csv', index=False)

print(df_other_customers.dtypes)

import pandas as pd
import numpy as np

# Fonction pour calculer la distance haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Rayon de la Terre en kilomètres
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Charger le fichier CSV dans un DataFrame
df_us_customers = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers.csv')

# Calculer la distance et ajouter une nouvelle colonne "distances"
df_us_customers['distances'] = df_us_customers.apply(lambda row: haversine(row['lat'], row['lng'], row['Latitude'], row['Longitude']), axis=1)

# Sauvegarder le DataFrame avec la nouvelle colonne dans un fichier CSV
df_us_customers.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers_with_distances.csv', index=False)
df_us_customers_with_distances = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers_with_distances.csv')
print(df_us_customers_with_distances.dtypes)

import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Charger le fichier CSV dans un DataFrame
df_us_customers = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers_with_distances.csv')

# Créer une carte de base centrée sur les États-Unis
m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)

# Ajouter un cluster de marqueurs pour les clients
marker_cluster = MarkerCluster().add_to(m)

# Ajouter des marqueurs pour chaque client
for idx, row in df_us_customers.iterrows():
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=f"Customer ID: {row['Customer ID']}<br>State/Province: {row['State/Province']}<br>Distance: {row['distances']} km",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# Ajouter des marqueurs pour chaque usine
for idx, row in df_us_customers.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Factory: {row['Factory']}<br>State/Province: {row['State/Province']}",
        icon=folium.Icon(color='red', icon='industry')
    ).add_to(marker_cluster)

# Sauvegarder la carte dans un fichier HTML
m.save(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\us_customers_map.html')

print("Carte géographique interactive créée avec succès.")

import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Charger le fichier CSV dans un DataFrame
df_us_customers = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers_with_distances.csv')

# Créer une carte de base centrée sur les États-Unis
m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)

# Ajouter un cluster de marqueurs pour les clients
marker_cluster = MarkerCluster().add_to(m)

# Ajouter des marqueurs pour chaque client
for idx, row in df_us_customers.iterrows():
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=f"Customer ID: {row['Customer ID']}<br>State/Province: {row['State/Province']}<br>Distance: {row['distances']} km",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# Ajouter des marqueurs pour chaque usine
for idx, row in df_us_customers.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Factory: {row['Factory']}<br>State/Province: {row['State/Province']}",
        icon=folium.Icon(color='red', icon='industry')
    ).add_to(marker_cluster)

# Ajouter une légende à la carte
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 90px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white; opacity: 0.8;">
 <b>Légende</b><br>
 <i class="fa fa-map-marker fa-2x" style="color:blue"></i> Clients<br>
 <i class="fa fa-industry fa-2x" style="color:red"></i> Usines
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Sauvegarder la carte dans un fichier HTML
m.save(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\us_customers_map2.html')

print("Carte géographique interactive créée avec succès.")
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Charger le fichier CSV dans un DataFrame
df_us_customers = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers_with_distances.csv')

# Créer une carte de base centrée sur les États-Unis
m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)

# Ajouter un cluster de marqueurs pour les clients
marker_cluster = MarkerCluster().add_to(m)

# Ajouter des marqueurs pour chaque client
for idx, row in df_us_customers.iterrows():
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=f"Customer ID: {row['Customer ID']}<br>State/Province: {row['State/Province']}<br>Distance: {row['distances']} km",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# Ajouter des marqueurs pour chaque usine
for idx, row in df_us_customers.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Factory: {row['Factory']}<br>State/Province: {row['State/Province']}",
        icon=folium.Icon(color='red', icon='industry')
    ).add_to(marker_cluster)

# Ajouter une légende à la carte
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 90px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white; opacity: 0.8;">
 <b>Légende</b><br>
 <i class="fa fa-map-marker fa-2x" style="color:blue"></i> Clients<br>
 <i class="fa fa-industry fa-2x" style="color:red"></i> Usines
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Sauvegarder la carte dans un fichier HTML
m.save(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\us_customers_map3.html')

print("Carte géographique interactive créée avec succès.")
#Statistiques Descriptives de Base
import pandas as pd
# Charger le fichier CSV dans un DataFrame
df_us_customers_with_distances = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers_with_distances.csv')

# Statistiques descriptives pour les distances
print("Statistiques descriptives pour les distances :")
print(df_us_customers_with_distances['distances'].describe())
# Comptabiliser le nombre de clients par usine
clients_par_usine = df_us_customers_with_distances.groupby('Factory')['Customer ID'].nunique().reset_index()
clients_par_usine.columns = ['Factory', 'Nombre de Clients']

# Afficher les résultats
print("\nNombre de clients par usine :")
print(clients_par_usine)

# Visualiser la répartition des clients par usine
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(x='Factory', y='Nombre de Clients', data=clients_par_usine)
plt.xlabel('Usine')
plt.ylabel('Nombre de Clients')
plt.title('Répartition des Clients par Usine')
plt.xticks(rotation=90)
plt.show()

# Calculer la densité des clients par usine
density_par_usine = df_us_customers_with_distances.groupby('Factory')['Customer ID'].count().reset_index()
density_par_usine.columns = ['Factory', 'Nombre de Clients']

# Afficher les résultats
print("\nDensité des clients par usine :")
print(density_par_usine)

# Visualiser la densité des clients par usine
plt.figure(figsize=(12, 6))
sns.barplot(x='Factory', y='Nombre de Clients', data=density_par_usine)
plt.xlabel('Usine')
plt.ylabel('Nombre de Clients')
plt.title('Densité des Clients par Usine')
plt.xticks(rotation=90)
plt.show()
# Analyser la distribution des distances avec un histogramme
plt.figure(figsize=(10, 6))
sns.histplot(df_us_customers_with_distances['distances'], bins=30, kde=True)
plt.xlabel('Distance (km)')
plt.ylabel('Nombre de Clients')
plt.title('Distribution des Distances Usine-Client')
plt.show()

# Analyser la distribution des distances avec un boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_us_customers_with_distances['distances'])
plt.xlabel('Distance (km)')
plt.title('Boxplot des Distances Usine-Client')
plt.show()
m.save(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\us_customers_map5.html')

# Charger le fichier CSV dans un DataFrame
df_us_customers = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers.csv')

# Comptabiliser le nombre de clients par State/Region
clients_par_region = df_us_customers.groupby('State/Province')['Customer ID'].nunique().reset_index()
clients_par_region.columns = ['State/Province', 'Nombre de Clients']

# Comptabiliser le nombre de clients par Postal Code
clients_par_postal_code = df_us_customers.groupby('Postal Code')['Customer ID'].nunique().reset_index()
clients_par_postal_code.columns = ['Postal Code', 'Nombre de Clients']

# Afficher les résultats
print("Nombre de clients par State/Region :")
print(clients_par_region)

print("\nNombre de clients par Postal Code :")
print(clients_par_postal_code)

import pandas as pd

# Charger le fichier CSV dans un DataFrame
df_us_customers = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers_with_distances.csv')

# Comptabiliser le nombre de clients par State/Region
clients_par_region = df_us_customers.groupby('State/Province')['Customer ID'].nunique().reset_index()
clients_par_region.columns = ['State/Province', 'Nombre de Clients']

# Comptabiliser le nombre de clients par Postal Code
clients_par_postal_code = df_us_customers.groupby('Postal Code')['Customer ID'].nunique().reset_index()
clients_par_postal_code.columns = ['Postal Code', 'Nombre de Clients']

# Sauvegarder les résultats dans des fichiers CSV
clients_par_region.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\clients_par_region.csv', index=False)
clients_par_postal_code.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\clients_par_postal_code.csv', index=False)

print("Fichiers CSV créés avec succès pour le nombre de clients par State/Region et par Postal Code.")



# Charger les fichiers CSV dans des DataFrames
df_cp = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Products.csv')

# Faire une jointure entre df_us_customers et df_cp sur 'Product ID'
df_merged = pd.merge(df_us_customers, df_cp[['Product ID', 'Factory']], on='Product ID', how='inner')

# Vérifier les colonnes après la jointure
print("Colonnes après la jointure :")
print(df_merged.columns)

# Utiliser la colonne correcte pour le groupement
df_merged['Factory'] = df_merged['Factory_y']

# Comptabiliser le nombre de clients par usine
clients_par_usine = df_merged.groupby('Factory')['Customer ID'].nunique().reset_index()
clients_par_usine.columns = ['Factory', 'Nombre de Clients']

# Sauvegarder les résultats dans un fichier CSV
clients_par_usine.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\clients_par_usine.csv', index=False)

print("Fichier CSV créé avec succès pour le nombre de clients par usine.")
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Charger le fichier CSV dans un DataFrame
df_us_customers_with_distances = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers_with_distances.csv')

# Créer une carte de base centrée sur les États-Unis
m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)

# Ajouter un cluster de marqueurs pour les clients
marker_cluster = MarkerCluster().add_to(m)

# Ajouter des marqueurs pour chaque client
for idx, row in df_us_customers_with_distances.iterrows():
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=f"Customer ID: {row['Customer ID']}<br>State/Province: {row['State/Province']}<br>Distance: {row['distances']} km",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# Ajouter des marqueurs pour chaque usine
factories = df_us_customers_with_distances[['Factory', 'Latitude', 'Longitude']].drop_duplicates()
for idx, row in factories.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Factory: {row['Factory']}",
        icon=folium.Icon(color='red', icon='industry')
    ).add_to(m)

# Ajouter une légende à la carte
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 120px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white; opacity: 0.8;">
 <b>Légende</b><br>
 <i class="fa fa-map-marker fa-2x" style="color:blue"></i> Clients<br>
 <i class="fa fa-industry fa-2x" style="color:red"></i> Usines<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Sauvegarder la carte dans un fichier HTML
m.save(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\us_customers_map4.html')

print("Carte géographique interactive créée avec succès.")
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Extraire les coordonnées des clients
coords = df_us_customers_with_distances[['lat', 'lng']]

# Appliquer K-means pour regrouper les clients
kmeans = KMeans(n_clusters=5, random_state=0).fit(coords)
df_us_customers_with_distances['Cluster'] = kmeans.labels_

# Visualiser les clusters
plt.scatter(df_us_customers_with_distances['lng'], df_us_customers_with_distances['lat'], c=df_us_customers_with_distances['Cluster'], cmap='viridis')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustering des Clients')
plt.show()
#Disribution distance clients_usine
plt.figure(figsize=(10, 6))
import seaborn as sns

sns.histplot(df_us_customers_with_distances['distances'], bins=30, kde=True)
plt.xlabel('Distance (km)')
plt.ylabel('Nombre de Clients')
plt.title('Distribution des Distances Usine-Client')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df_us_customers_with_distances['distances'])
plt.xlabel('Distance (km)')
plt.title('Boxplot des Distances Usine-Client')
plt.show()
