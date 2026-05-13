# Québec City Permit Intelligence Dashboard

## Présentation du projet

Québec City Permit Intelligence Dashboard est un projet de science des données et d’intelligence artificielle appliqué aux permis délivrés à la Ville de Québec.

Le projet vise à analyser les tendances territoriales et temporelles des permis, à visualiser les zones de concentration sur une carte interactive et à produire des indicateurs exploitables à partir de données ouvertes.

L’approche reprend une architecture simple et reproductible : Python prépare les données, entraîne les modèles et génère un fichier JSON ; le dashboard web lit ensuite ce JSON pour afficher les résultats.

## Objectif

Le projet permet de répondre à plusieurs questions analytiques :

- En quels arrondissements observe-t-on le plus grand nombre de permis délivrés ?
- Quels types de permis sont les plus fréquents ?
- Quels domaines d’intervention sont les plus représentés ?
- Existe-t-il des tendances mensuelles ou annuelles ?
- Observe-t-on des zones de concentration géographique ?
- Est-il possible d’estimer le volume futur de permis par période ?
- Est-il possible de classifier le type de permis à partir de variables comme le domaine, l’arrondissement et la période ?

## Source des données

Le jeu de données utilisé correspond aux permis délivrés à la Ville de Québec, disponible sur le portail Données Québec.

Il contient notamment les informations suivantes :

- numéro de permis ;
- date de délivrance ;
- adresse des travaux ;
- domaine de l’intervention ;
- lots impactés ;
- type de permis ;
- arrondissement ;
- raison principale du permis ;
- longitude ;
- latitude.

Le fichier utilisé par le projet est `vdq-permis.csv`.

Licence des données : Attribution Creative Commons 4.0 International (CC BY 4.0).

## Sensibilité et protection des données

Même si les données sont ouvertes, le projet applique une approche prudente dans le dashboard public.

Le fichier `dashboard_data.json` généré par Python n’inclut pas :

- le numéro de permis ;
- l’adresse exacte des travaux ;
- les lots impactés ;
- la raison détaillée du permis.

Les coordonnées géographiques sont arrondies afin de réduire la précision spatiale. L’objectif est de visualiser des zones de concentration, et non d’exposer des adresses ou des dossiers individuels.

## Architecture du projet

```text
quebec-city-permit-intelligence-dashboard/
│
├── index.html
├── build_dashboard_data.py
├── dashboard_data.json
├── requirements.txt
├── README.md
├── LICENSE
└── data/
    └── vdq-permis.csv
```

## Fonctionnement général

```text
CSV des permis
        ↓
Préparation des données avec Python
        ↓
Agrégation territoriale et temporelle
        ↓
Entraînement de modèles de classification et de prévision
        ↓
Génération de dashboard_data.json
        ↓
Lecture du JSON par le dashboard HTML
        ↓
Carte interactive avec filtres dynamiques
```

## Technologies utilisées

- Python ;
- Pandas ;
- NumPy ;
- Scikit-learn ;
- HTML5 ;
- CSS3 ;
- JavaScript ;
- Leaflet.js ;
- GitHub Pages.

## Modèles prévus

Le projet contient deux axes d’intelligence artificielle :

### Classification du type de permis

Le script peut entraîner plusieurs modèles de classification afin de prédire `TYPE_PERMIS` à partir de variables non sensibles comme :

- domaine ;
- arrondissement ;
- année ;
- mois ;
- trimestre ;
- localisation arrondie.

Modèles comparés :

- régression logistique équilibrée ;
- forêt aléatoire ;
- Extra Trees.

### Prévision du volume mensuel

Le script peut également entraîner des modèles de régression afin d’estimer le volume mensuel de permis délivrés.

Modèles comparés :

- régression linéaire ;
- forêt aléatoire ;
- Extra Trees.

## Dashboard web

Le dashboard met la carte au centre de l’expérience utilisateur.

Il comprend :

- une carte interactive Leaflet ;
- une visualisation des zones de concentration ;
- un panneau latéral de filtres ;
- des indicateurs dynamiques ;
- les principaux arrondissements ;
- les principaux domaines d’intervention ;
- les prévisions mensuelles lorsque l’historique est suffisant.

## Exécution locale

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

Sur Windows :

```bash
py -m pip install -r requirements.txt
```

### 2. Générer le JSON

```bash
python build_dashboard_data.py
```

Sur Windows :

```bash
py build_dashboard_data.py
```

Cette commande génère :

```text
dashboard_data.json
```

### 3. Lancer le serveur local

```bash
python -m http.server 8000
```

Sur Windows :

```bash
py -m http.server 8000
```

### 4. Ouvrir le dashboard

```text
http://localhost:8000
```

## Publication GitHub Pages

Le dashboard peut être publié avec GitHub Pages à partir de la branche `main`.

Configuration recommandée :

```text
Source : Deploy from a branch
Branch : main
Folder : /root
```

## Limites

Ce projet est une démonstration de science des données et d’intelligence artificielle appliquée.

Les résultats ne doivent pas être interprétés comme un outil officiel de planification urbaine ou de décision administrative.

Les principales limites sont :

- dépendance à la qualité des données ouvertes ;
- visualisation agrégée et non opérationnelle ;
- modèles dépendants du volume réel de données disponible ;
- absence d’API de prédiction dynamique dans cette première version.

## Améliorations possibles

- Ajouter une carte choroplèthe par arrondissement ;
- Ajouter une matrice de confusion pour la classification ;
- Afficher les métriques détaillées des modèles ;
- Ajouter une analyse textuelle simplifiée de la colonne `RAISON` ;
- Intégrer une récupération automatisée via l’API CKAN ;
- Optimiser la taille du fichier `dashboard_data.json` ;
- Ajouter des graphiques temporels interactifs ;
- Publier le dashboard avec GitHub Pages.

## Licence et source des données

Le code source du projet est distribué sous licence MIT.

Les données utilisées dans ce projet proviennent du portail Données Québec, la plateforme québécoise de diffusion de données ouvertes.

Jeu de données utilisé : `Permis délivrés à la Ville de Québec`  
Organisation diffusant les données : Ville de Québec  
Portail : [Données Québec](https://www.donneesquebec.ca/)  
Licence des données : Attribution Creative Commons 4.0 International (CC BY 4.0)

Les données demeurent soumises aux conditions de leur source d’origine. Le portail Données Québec indique que les jeux de données diffusés se voient attribuer une variante de la licence Creative Commons 4.0, la variante applicable étant précisée dans la fiche descriptive de chaque jeu de données. :contentReference[oaicite:0]{index=0}

Voir le fichier [LICENSE](LICENSE) pour plus de détails sur la licence applicable au code source du projet.
