import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


### EXERCICE 1 ###

## QUESTION 1 :

# Charge les données
data = pd.read_csv("spam.csv", encoding='latin-1')

# Gardee uniquement les colonnes utiles
data = data.iloc[:, :2]
data.columns = ['label', 'message']

# Encodage des labels (spam: 1, ham: 0)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Vectorisation des messages
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']


## QUESTION 2 :

# Divise les données en ensembles d’entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

'''
Pourquoi la stratification ?

- La stratification est importante pour maintenir les proportions des 
classes dans les ensembles d'entraînement et de test.

- Cela garantit une répartition équitable entre spam et ham.

'''


## QUESTION 3 :

# Entraîne un modèle SVM avec un noyau linéaire
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Génération de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)


## QUESTION 4 :

# Visualiser la matrice de confusion
plt.figure(figsize=(6, 5))
plt.title("Matrice de confusion SVM")
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Classe prédite')
plt.ylabel('Classe réelle')
plt.show()

'''
Quelles erreurs sont les plus fréquentes ?

Les erreurs les plus fréquentes se trouvent généralement dans les 
faux négatifs (spam prédit comme ham), ce qui peut être problématique 
dans un filtre anti-spam.

'''


## QUESTION 5 :

# Calcule la probabilité pour la classe positive (spam)
y_prob = svm_model.predict_proba(X_test)[:, 1]

# Génère la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Trace la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'r--', label='No Skill')
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC-AUC')
plt.legend(loc='lower right')
plt.show()

'''
Interprétation :

Une AUC proche de 1 indique une bonne performance du modèle.

'''




### EXERCICE 2 ###

## QUESTION 1 :

# Naïve Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# Régression Logistique
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Évaluation des performances individuelles
print("Naïve Bayes :")
print(classification_report(y_test, nb_pred))

print("Régression Logistique :")
print(classification_report(y_test, lr_pred))

print("SVM (noyau linéaire) :")
print(classification_report(y_test, y_pred))

'''
1. Naïve Bayes :

- Naïve Bayes montre une bonne performance globale avec une 
précision pondérée de 98%.

- Cependant, il est légèrement moins performant en termes de rappel 
pour la classe des spams (95%), ce qui signifie qu'il rate encore 
quelques spams.


2. Régression Logistique :

- La régression logistique est très précise pour prédire les spams 
(précision : 100%), mais elle a un rappel plus faible (82%). Cela 
signifie qu'elle classe certains spams comme des hams (faux négatifs). 

- Le F1-score pour la classe des spams est légèrement inférieur 
à celui de Naïve Bayes.


3. SVM (noyau linéaire) :

- Le modèle SVM offre une performance équilibrée avec un rappel (88%) 
supérieur à celui de la régression logistique et une précision (100%) 
identique pour les spams. 

- Le F1-score pour les spams (94%) est également le plus élevé des 
trois modèles, ce qui en fait un choix robuste pour cette tâche.

'''


## QUESTION 2 :

# Voting Classifier avec un vote 'hard'
hard_voting = VotingClassifier(
    estimators=[
        ('nb', nb_model),
        ('lr', lr_model),
        ('svm', svm_model)
    ],
    voting='hard'
)
hard_voting.fit(X_train, y_train)
hard_pred = hard_voting.predict(X_test)

# Évaluation des performances
print("Voting Classifier (Hard) :")
print(classification_report(y_test, hard_pred))

'''
1. Comparaison avec Naïve Bayes :

- Le Voting Classifier conserve une précision similaire (0.98), 
mais son rappel pour la classe spam diminue légèrement 
(0.87 contre 0.95 pour Naïve Bayes).

- Cependant, il améliore l'équilibre entre la précision et le rappel 
pour les spams, ce qui lui permet d'avoir un F1-score (0.93) similaire 
à celui de Naïve Bayes.


2. Comparaison avec Régression Logistique :

- Le Voting Classifier améliore nettement le rappel pour les spams 
(0.87 contre 0.82) tout en maintenant une précision élevée 
(1.00 pour la classe spam).

- Cela se traduit par un F1-score supérieur pour les spams 
(0.93 contre 0.90 pour la régression logistique).


3. Comparaison avec SVM :

- Les performances du Voting Classifier sont très proches de celles du 
SVM, avec une précision, un rappel, et un F1-score similaires.

- Cependant, en tant que modèle combiné, le Voting Classifier bénéficie 
d'une meilleure robustesse.

'''


## QUESTION 3 :

# Voting Classifier avec vote soft
soft_voting = VotingClassifier(
    estimators=[
        ('nb', nb_model),
        ('lr', lr_model),
        ('svm', svm_model)
    ],
    voting='soft'
)
soft_voting.fit(X_train, y_train)
soft_pred = soft_voting.predict(X_test)

# Évaluation des performances
print("Voting Classifier (Soft) :")
print(classification_report(y_test, soft_pred))

'''
- Rappel : Le vote soft est plus performant, atteignant 0.91 contre 
0.87 pour le vote hard. Cela signifie que le vote soft détecte plus 
efficacement les spams, réduisant les faux négatifs.

- F1-Score : Le F1-score du vote soft est également supérieur 
(0.95 contre 0.93), ce qui montre qu'il équilibre mieux précision 
et rappel.

- Accuracy : Le vote soft atteint 0.99, légèrement supérieur au 0.98 du 
vote hard, ce qui confirme sa supériorité générale.

Donc, dans ce cas, le vote soft est plus performant que le vote hard.

'''


## QUESTION 4 :

# Calcul des probabilités pour chaque modèle
nb_prob = nb_model.predict_proba(X_test)[:, 1]
lr_prob = lr_model.predict_proba(X_test)[:, 1]
svm_prob = svm_model.predict_proba(X_test)[:, 1]
soft_voting_prob = soft_voting.predict_proba(X_test)[:, 1]

# Naïve Bayes
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_prob)
auc_nb = auc(fpr_nb, tpr_nb)

# Régression Logistique
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
auc_lr = auc(fpr_lr, tpr_lr)

# SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)
auc_svm = auc(fpr_svm, tpr_svm)

# Voting Classifier (Soft)
fpr_voting, tpr_voting, _ = roc_curve(y_test, soft_voting_prob)
auc_voting = auc(fpr_voting, tpr_voting)

# Trace les courbes ROC
plt.figure(figsize=(10, 8))
plt.plot(fpr_nb, tpr_nb, label=f'Naïve Bayes (AUC = {auc_nb:.2f})')
plt.plot(fpr_lr, tpr_lr, label=f'Régression Logistique (AUC = {auc_lr:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})')
plt.plot(fpr_voting, tpr_voting, label=f'Voting Classifier Soft (AUC = {auc_voting:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')

plt.title('Courbes ROC des modèles')
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

'''
Comparaison individuelle :

- Naïve Bayes (AUC = 0.99) : La courbe est légèrement inférieure aux 
meilleurs modèles par endroits, mais montre tout de même une bonne 
performance globale. Cela montre que Naïve Bayes est un bon 
classificateur pour ce dataset, bien que parfois il est dépassé par 
d'autres modèles.

- Régression Logistique (AUC = 0.99) : Ce modèle atteint presque la même 
performance que le Voting Classifier Soft. Sa courbe ROC est très proche 
de l'idéal, ce qui en fait l'un des meilleurs modèles.

- SVM (AUC = 0.98) : Bien que performant, sa courbe est légèrement en 
dessous des deux autres modèles individuels dans certaines zones, 
indiquant qu'il peut parfois manquer de flexibilité par rapport à la 
régression logistique.

- Voting Classifier (Soft) (AUC = 0.99) : Sa courbe se situe parmi les 
meilleures, avec une performance globale comparable à celle de la 
régression logistique.


Comparaison générale :

- Les trois modèles et le Voting Classifier Soft atteignent presque un 
AUC parfait (0.99 ou 0.98), ce qui démontre leur excellente capacité à 
différencier les spams des hams.

- Le Voting Classifier Soft est légèrement meilleur.

'''


## QUESTION 5 :

'''
Le mélange de modèles permet de surpasser les performances des modèles 
individuels en exploitant leur diversité, en réduisant à la fois le 
biais et la variance, et en atténuant les erreurs spécifiques grâce à 
l'agrégation de leurs prédictions.

'''




### EXERCICE 3 ###

## QUESTION 1 :

'''
Les Gaussian Mixture Models (GMM) sont des modèles probabilistes qui 
supposent que les données proviennent d'une combinaison de 
plusieurs distributions gaussiennes. 

Chaque composant du mélange correspond à une distribution gaussienne, 
et le modèle est entraîné pour estimer les paramètres 
(moyenne, covariance) de chaque composant. 

Dans le cadre de la classification, chaque composant du modèle GMM peut 
être associé à une classe différente.

Le GMM est utilisé ici pour modéliser les distributions des données dans 
chaque classe (spam et ham).

'''


## QUESTION 2 :

# Initialisation et entraînement du modèle GMM (2 composants pour 2 classes : spam et ham)
#gmm_model = GaussianMixture(n_components=2, random_state=42)
#gmm_model.fit(X_train.toarray())  # Conversion de la matrice sparse en tableau dense

# Prédictions du modèle
#gmm_pred = gmm_model.predict(X_test.toarray())




# Réduction de la dimensionnalité pour GMM
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train.toarray())
X_test_pca = pca.transform(X_test.toarray())

# Initialisation et entraînement du modèle GMM
gmm_model = GaussianMixture(n_components=2, random_state=42)
gmm_model.fit(X_train_pca)

# Prédictions
gmm_pred = gmm_model.predict(X_test_pca)

# Évaluation des performances
print("Performance du GMM (après PCA) :")
print(classification_report(y_test, gmm_pred))



## QUESTION 3 :

# Évaluation des performances
print("GMM :")
print(classification_report(y_test, gmm_pred))

'''
- Précision : La précision globale est de 0.87, ce qui semble bon, mais 
cela est principalement dû à la performance élevée pour la classe ham 
(prédictions correctes pour la classe majoritaire).

- Rappel : Le rappel de 1.00 pour ham signifie que tous les messages ham 
sont correctement classifiés, mais le modèle ne parvient pas à détecter 
les messages spam (rappel de 0.00 pour spam).

- F1-score : Le F1-score global de 0.80 est également élevé, mais cela 
reflète davantage la bonne performance pour ham, sans prendre en compte 
les échecs pour spam.

Donc, bien que la précision globale semble acceptable, le modèle échoue 
à identifier les messages spam, ce qui en fait une solution non fiable 
pour cette tâche de classification.

'''


## QUESTION 4 :

# Comparer les performances du GMM avec celles des autres modèles
print("Comparaison des performances :")
print("SVM:")
print(classification_report(y_test, svm_model.predict(X_test)))

print("Voting Classifier (Hard) :")
print(classification_report(y_test, hard_voting.predict(X_test)))

print("Voting Classifier (Soft) :")
print(classification_report(y_test, soft_voting.predict(X_test)))

print("GMM :")
print(classification_report(y_test, gmm_pred))


## QUESTION 5 :

# Fonction pour afficher les frontières de décision
def plot_decision_boundary(X, y, model, title):
    h = .02  # Taille des pas pour le maillage
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=30)
    plt.title(title)

# Réduire la dimensionnalité à 2 pour la visualisation
#pca = PCA(n_components=2)

# Entraîner le PCA sur X_train
#X_train_2d = pca.fit_transform(X_train.toarray())

# Transformer X_test avec le même PCA déjà entraîné
#X_test_2d = pca.transform(X_test.toarray())


# Tracer les frontières de décision du GMM
plt.figure(figsize=(10, 8))
plot_decision_boundary(X_train_pca, y_train, gmm_model, "Frontière de décision - GMM")
plt.show()

'''
- SVM crée une frontière linéaire qui sépare les classes de manière 
optimale, mais qui peut être trop rigide pour des données complexes.

- GMM peut générer des frontières non linéaires, car il modélise chaque 
classe comme une distribution gaussienne, ce qui lui permet de s'adapter 
plus facilement à des distributions complexes dans les données.

- Ainsi, les frontières du GMM seront souvent plus flexibles et adaptées 
aux distributions des données, alors que celles de la SVM seront plus 
rigides et linéaires.

'''




### EXERCICE 4 ###

## QUESTION 1 :

'''
Les hyperparamètres principaux d'un SVM qu'on peut optimiser sont :
- C
- Kernel
- gamma
- degree
- coef0 

'''


## QUESTION 2 :

# Définir les hyperparamètres à tester
param_grid = {
    'C': [0.1, 1, 10, 100],  # Valeurs pour C
    'kernel': ['linear', 'rbf'],  # Types de noyaux à tester
}

# Initialiser le modèle SVM
svm_model = SVC(random_state=42)

# Effectuer une recherche par GridSearchCV
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres trouvés
print("Meilleurs paramètres SVM :")
print(grid_search.best_params_)

# Évaluer les performances du modèle optimisé
best_svm_model = grid_search.best_estimator_
y_pred = best_svm_model.predict(X_test)
print(classification_report(y_test, y_pred))


## QUESTION 3 :

# Initialiser le modèle SVM avec probability=True pour activer predict_proba
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# Voting Classifier avec vote soft
voting_model = VotingClassifier(
    estimators=[
        ('nb', nb_model),
        ('lr', lr_model),
        ('svm', svm_model)  # Utiliser le modèle SVM avec probability=True
    ],
    voting='soft',  # Vote basé sur la probabilité
    weights=[1, 1, 2]  # Poids attribués à chaque modèle
)

# Entraînement du modèle avec les poids
voting_model.fit(X_train, y_train)

# Prédiction et évaluation
y_pred = voting_model.predict(X_test)
print("Performance du Voting Classifier avec poids :")
print(classification_report(y_test, y_pred))


## QUESTION 4 :

# Définir les hyperparamètres à tester
param_grid_gmm = {
    'n_components': [2],  # Tester différents nombres de composantes
    'covariance_type': ['diag', 'spherical']  # Tester différents types de covariance
}

# Initialiser le modèle GMM
gmm_model = GaussianMixture(random_state=42)

# Utiliser GridSearchCV pour trouver les meilleurs paramètres
grid_search_gmm = GridSearchCV(
    estimator=gmm_model,
    param_grid=param_grid_gmm,
    cv=2,  # Validation croisée à 2 folds
    scoring='f1_weighted', #'accuracy'  # Indicateur de performance basé sur la précision
    n_jobs=-1  # Utiliser tous les cœurs disponibles
)

# Préparer les données : convertir les matrices sparse en matrices denses
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Lancer la recherche par grille
grid_search_gmm.fit(X_train_dense, y_train)

# Afficher les meilleurs paramètres trouvés
print("Meilleurs paramètres pour le GMM :")
print(grid_search_gmm.best_params_)

# Évaluer les performances du modèle optimisé sur l'ensemble de test
best_gmm_model = grid_search_gmm.best_estimator_
gmm_pred = best_gmm_model.predict(X_test_dense)

# Rapport de classification
print("Performances du modèle GMM optimisé :")
print(classification_report(y_test, gmm_pred))
