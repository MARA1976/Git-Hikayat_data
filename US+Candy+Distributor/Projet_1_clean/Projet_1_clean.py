import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster

# Lire les fichiers CSV
df_u = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\uszips.csv')
df_cs = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Sales.csv')
df_cp = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Products.csv')
df_ct = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Targets.csv')
df_cf = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\Candy_Factories.csv')

# Afficher les colonnes et les types de données
print("Types de données des fichiers CSV :")
print(df_u.dtypes)
print(df_cs.dtypes)
print(df_cp.dtypes)
print(df_ct.dtypes)
print(df_cf.dtypes)

# Exploration des données
print("Aperçu des données :")
print(df_u.head())
print(df_cs.head())
print(df_cp.head())
print(df_ct.head())
print(df_cf.head())

# Compter les valeurs manquantes
print("Valeurs manquantes :")
print(df_u.isnull().sum())
print(df_cs.isnull().sum())
print(df_cp.isnull().sum())
print(df_ct.isnull().sum())
print(df_cf.isnull().sum())

# Convertir les colonnes de dates en format datetime
df_cs['Order Date'] = pd.to_datetime(df_cs['Order Date'])
df_cs['Ship Date'] = pd.to_datetime(df_cs['Ship Date'])

# Afficher les noms des colonnes et les types de données pour df_cs
print("Colonnes et types de données pour df_cs :")
print(df_cs.columns)
print(df_cs.dtypes)

# Convertir les colonnes 'Postal Code' et 'zip' en type string
df_cs['Postal Code'] = df_cs['Postal Code'].astype(str)
df_u['zip'] = df_u['zip'].astype(str)

# Faire une jointure entre df_cs et df_u sur 'Postal Code' et 'zip'
df_su = pd.merge(df_cs, df_u, left_on='Postal Code', right_on='zip', how='left')

# Afficher les premières lignes du DataFrame fusionné
print("Aperçu du DataFrame fusionné :")
print(df_su.head(10))

# Afficher les colonnes pertinentes
print("Colonnes pertinentes :")
print(df_su[['Customer ID', 'City', 'State/Province', 'Postal Code', 'lat', 'lng']].head(10))

# Sauvegarder le DataFrame fusionné en tant que nouveau fichier CSV
df_su.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_su.csv', index=False)
print("Table saved as 'df_su.csv'")

# Supprimer les valeurs manquantes dans la colonne 'lat'
print("Nombre de lignes avant suppression :", len(df_su))
df_su = df_su.dropna(subset=['lat'])
print("Nombre de lignes après suppression :", len(df_su))

# Sauvegarder le DataFrame nettoyé dans un nouveau fichier CSV
df_su.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_su_cleaned.csv', index=False)

# Charger les fichiers CSV dans des DataFrames
df_su_cleaned = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_su_cleaned.csv')

# Afficher les noms des colonnes et les types de données pour df_su_cleaned
print("Colonnes et types de données pour df_su_cleaned :")
print(df_su_cleaned.dtypes)
print(df_cp.dtypes)
print(df_cf.dtypes)

# Première jointure : ajouter la colonne "Factory" en utilisant "Product ID"
df_merged_1 = pd.merge(df_su_cleaned, df_cp[['Product ID', 'Factory']], on='Product ID', how='inner')

# Deuxième jointure : ajouter les colonnes "Latitude" et "Longitude" en utilisant "Factory"
df_merged_2 = pd.merge(df_merged_1, df_cf[['Factory', 'Latitude', 'Longitude']], on='Factory', how='inner')

# Sélectionner les colonnes spécifiques
columns_to_keep = ['Customer ID', 'State/Province', 'Postal Code', 'lat', 'lng', 'population', 'density', 'Product ID', 'Factory', 'Latitude', 'Longitude', 'Country/Region']
df_selected = df_merged_2[columns_to_keep]

# Filtrer les clients américains
df_us_customers = df_selected[df_selected['Country/Region'] == 'United States']

# Filtrer les clients non-américains
df_other_customers = df_selected[df_selected['Country/Region'] != 'United States']

# Sauvegarder les DataFrames filtrés dans des fichiers CSV séparés
df_us_customers.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers.csv', index=False)
df_other_customers.to_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_other_customers.csv', index=False)

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

# Charger le fichier CSV dans un DataFrame
df_us_customers_with_distances = pd.read_csv(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\df_us_customers_with_distances.csv')
print(df_us_customers_with_distances.dtypes)
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
for idx, row in df_us_customers_with_distances.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Factory: {row['Factory']}<br>State/Province: {row['State/Province']}",
        icon=folium.Icon(color='red', icon='industry')
    ).add_to(marker_cluster)

# Ajouter une légende à la carte
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 120px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white; opacity: 0.8;">
&emsp;<b>Légende</b><br>
&emsp;<i class="fa fa-map-marker fa-2x" style="color:blue"></i>&emsp;Clients<br>
&emsp;<i class="fa fa-industry fa-2x" style="color:red"></i>&emsp;Usines<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Sauvegarder la carte dans un fichier HTML
m.save(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\us_customers_map.html')

print("Carte géographique interactive créée avec succès.")
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
&emsp;<b>Légende</b><br>
&emsp;<i class="fa fa-map-marker fa-2x" style="color:blue"></i>&emsp;Clients<br>
&emsp;<i class="fa fa-industry fa-2x" style="color:red"></i>&emsp;Usines<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Sauvegarder la carte dans un fichier HTML
m.save(r'C:\Users\drahl\Desktop\Projects\Data_Processing\My_First_Project\US+Candy+Distributor\us_customers_map1.html')

print("Carte géographique interactive créée avec succès.")