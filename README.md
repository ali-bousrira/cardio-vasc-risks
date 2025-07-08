Veille régression logistique
La régression logistique est une méthode utilisée pour prédire une réponse binaire, c’est-à-dire une variable qui ne peut prendre que deux valeurs (par exemple : malade ou non, 0 ou 1, oui ou non).

Elle permet de modéliser la probabilité qu’un événement se produise en fonction de différentes caractéristiques (âge, poids, tension, etc.).

C’est un outil très utilisé en médecine pour estimer des risques de maladies ou de complications à partir de données patients. Elle est simple, rapide à entraîner, et offre des résultats souvent faciles à interpréter.

Dans notre cas, elle sert à prédire si un patient présente un risque cardio-vasculaire ou non.

Explication des données
AGE : age in number of days (integer)
HEIGHT : height in cm (integer)
WEIGHT : weight in kg (integer)
GENDER : gender, categorical (1: female, 2: male)
AP_HIGH : systolic blood pressure (integer)
AP_LOW : diastolic blood pressure (integer)
CHOLESTEROL : cholesterol level, categorical (1: normal, 2: above normal, 3: well above normal)
GLUCOSE : glucose level, categorical (1: normal, 2: above normal, 3: well above normal)
SMOKE : if patient smokes or not, categorical (0: no, 1: yes)
ALCOHOL : if patient drinks alcohol or not, categorical (0: no, 1: yes)
PHYSICAL_ACTIVITY : if patient is active or not, categorical (0: no, 1: yes)
-and target variable
CARDIO_DISEASE : if patient got the disease or not, categorical (0: no, 1: yes)
