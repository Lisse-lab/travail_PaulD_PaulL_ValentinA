Le but principal des scripts de ce dossier est de trouver la position optimale d'un réseau d'hydrophones pour étudier la population de bélugas du Saint-Laurent. 

Pour ce faire, l'algorithme va maximiser une fonction de couverture grâce à un algorithme baysien utilisant des algos génétiques (pour trouver le prochain point à évaluer). J'ai choisi cet algorithme car il permet, en théorie, de converger avec peu d'évaluations de notre fonction de couverture qui est très longue à être calculée. De plus, il est censé éviter les maximums locaux car il explore soit les parties du domaine de définition peu explorées soit les parties qui sont très prometteuses (proches d'un maximum).

Cette partie purement algorithmique se trouve dans le dossier optimisation.

Concernant la fonction de couverture, le fleuve a été découpé en une multitude de petits cubes (avec une précision réglable), ces petits cubes constituent l'ensemble $A$ puis je me suis inspiré des fonctions classiques de couvertures qui sont $\sum_{i \in A} w_i \times \mathbb{1}_{O(i)}$,avec $w_i$ le poids de chaque zone et $\mathbb{1}_O(i)$ si la zone est observable par notre réseau ou non. Je me suis aussi inspiré des fonctions qui cherchent à minimer la somme des erreurs en supposant que toutes les zones sont observables.

Cela a donné cette fonction 
$$\sum_{i \in A} \frac{w_i}{1 + err(i)} \times \mathbb{1}_{err(i) < err\_max}$$
La fonction erreur prend en compte la géométrie du réseau, la perte de la transmission entre la zone qui nous intéresse et les différents hydrophones (pour le calcul de la perte de transmission, j'ai inversé le récepteur et l'émetteur pour des soucis de calcul), elle a été developpé dans la partie beluga_watch.

La définition des poids $w_i$ est réglable mais, par défaut, c'est le produit de trois poids : un lié à la densité de population des bélugas dans le Saint-Laurent, un deuxième lié à la proximité acoustique par rapport au chemin des traversiers et un dernier lié à notre capacité à prédire la trajectoire des bélugas dans les différents secteurs.

Le poids lié à la densité est obtenu grâce à la carte de densité du ministère des Pêches et des Océans, puis l'on divise la densité surfacique par le nombre de cubes que l'on a dans la colonne d'eau pour avoir les poids par petit cube.

Pour obtenir le poids lié au traversier, j'ai récupérer les trajectoires du traversier, trouver dans quel cubes il passait, notons cet ensemble $T$. Puis pour un cube appertenant à $A$, voici le poids que j'ai retenu pour une fréquence (le poids étant moyenné s'il y a plusieurs fréquences de bateau étudié):
$$w_i^{nav} = 1 + (g - 1) \times \frac{1}{|T|} \times \sum_{t \in T} exp_i^t$$
Avec $exp_i^t$, l'exposition du cube i lorsque le bateau est dans le cube t, et g un gain choisi.

Enfin, pour le poids lié à la capacité à prédire je vais expliquer ça après la présentation du dossier coords_belugas.

Dans le dossier to_optimise il y a notre fonction principale que l'on veut optimiser, un fichier pour calculer la vitesse du son et un dernier pour tous les calculs de topologie nécessaire pour calculer la perte de transmission.

Dans le dossier utils_scripts il y a toutes les définitions de classes liées à un réseau d'hydrophones (Il faut comprendre qu'un point correspond à un tétrahèdre et non à un hydrophone.), les différentes formules de conversion entre les différents systèmes de coordonnées et différentes fonctions utiles.

Pour essayer de prédire les trajectoires, j'ai d'abord utilisé le modèle d'Ornstein-Ulhenbeck, mais cela n'a pas donné de résultats concluants. Ce modèle s'inspire du mouvement d'une particule dans un champs de potentiel, en sachant que c'est la densité qui est transformée en potentiel. Les calculs liés à la densité sont dans le script calc_mu, et l'implémentation est dans le fichier ornstein_ulhenbeck.

Pour comparer les différents modèles j'ai utilisé la base de données du Gremm, que j'ai nettoyé grâce au notebook cleaning_coords. J'ai ensuite fait des régressions sur les trajectoires puis pris un point tous les X minutes pour avoir plusieurs trajectoires issues d'une seule.

J'ai ensuite comparé différents modèles : le modèle d'Ornstein-Uhlenbeck (OU), des modèles de random forest, des modèles gaussiens (il y en a plusieurs car j'ai fait différents tests sur les données d'entrée) et un modèle de persistence. Le modèle d'Ornstein-Uhlenbeck n'est pas bon car il avait de moins bons résultats que le modèle de persistence. Les autres modèles avaient de meilleurs résultats qui étaient très similaires. J'ai aussi comparé les différentes erreurs par zone pour savoir si elles étaient corrélées et ainsi en déduire que l'on pouvait supposer que plus l'erreur était faible et plus l'on arriverait à prévoir les trajectoires et inversement, y compris avec des modèles plus complexes qu'actuellement et peut-être avec des données résultats.

En choisissant le modèle de persistence pour avoir le plus de données j'ai pris l'erreur de prédicions de chaque zone pour le poids lié à notre capacité à prédire. En effet, plus l'erreur est importante plus il est important de pouvoir localiser les bélugas. Il y a cependant de nombreuses limites liées au nombre de trajectoires dans chaque zone et aux données plus générallement.

Concernant les différentes exécutions, si vous voulez avoir les poids dus à la qualité de prédiction il faut exécuter le script forecasting_errors, en ayant au préalable exécuté toutes les cellules du notebook cleaning_coords.

Pour la partie principale, il faut exécuter le fichier python ou le notebook optimization, puis on peut analyser la qualité de la convergence d'un modème avec visualisation_optimisation et comparer différents meilleurs réseaux avec comparison_modeles.

Cependant, je ne sais pas pourquoi les expected_improvements sont des matrix mais il faudrait modifier ça et il faudra supprimer les deux lignes à supprimer plus tard dans visualisation_optimisation, il est aussi possible que lorsque l'on affiche l'historique d'optimisation les expected improvements soient décalés d'une itération et il faudrait surement supprimer le premier 0.

De plus, l'erreur dépend beaucoup de la position géométrique du réseau. Ainsi, peut-être que l'on peut remplacer bellhop par des formules plus simples, beaucoup plus rapide à calculer, pour avoir un résultat qui reste correct.

Enfin, j'ai joint ma bibliographie (Stage UQO.rdf)

J'ai aussi été obligé de découper en deux le fichier substrat.csv, il n'existe pas sur le github et on peut utiliser le noteboo for_substrat pour le créer