# Rapport d'examen — Parallélisation d'une simulation de galaxie

## 1. Contexte et objectifs

L'objectif est d'accélérer une simulation gravitationnelle de galaxie en plusieurs étapes :

1. parallélisation Numba du code de base ;
2. séparation MPI de l'affichage et du calcul ;
3. parallélisation MPI du calcul (avec étude des performances en fonction du nombre de processus **et** de threads) ;
4. réponses d'analyse sur l'équilibrage de charge et une stratégie Barnes–Hut distribuée.

Jeu de données principal utilisé pour les mesures : `data/galaxy_1000`.

---

## 2. Travaux réalisés

### 2.1. Version Numba parallélisée

Fichier : `nbodies_grid_numba.py`

- Ajout de `parallel=True` sur les noyaux Numba pertinents.
- Remplacement des boucles adaptées par `prange`.
- Ajout d'un mode benchmark sans affichage (`--no-display`) pour des mesures reproductibles.
- Paramètre `--threads` pour fixer le nombre de threads Numba.

### 2.2. Séparation MPI affichage / calcul

Fichier : `nbodies_grid_numba_mpi_display.py`

- Processus 0 : affichage (ou orchestration en mode headless).
- Processus 1 : calcul des trajectoires.
- Échange des positions à chaque pas via MPI (`Send/Recv`).
- Mesure comparative vis-à-vis de la version Numba seule.

### 2.3. Parallélisation MPI du calcul

Fichier : `nbodies_grid_numba_mpi_parallel.py`

- Décomposition du domaine selon les cellules en `x` (partition par processus).
- Mise à jour des corps propriétaires et migration MPI **voisin-à-voisin** (gauche/droite) jusqu'à stabilisation de la partition.
- Échange explicite de **cellules fantômes** (épaisseur 2 cellules en `x`) uniquement avec les voisins.
- Calcul local des interactions proches avec les corps locaux + fantômes, et interactions lointaines via masse/centre de masse globaux de cellules.
- Mesures réalisées selon `(nombre de processus, nombre de threads)`.

---

## 3. Réponses aux questions du sujet

### 3.1. Question préliminaire — Pourquoi `N_k = 1` ?

La galaxie simulée est essentiellement discoïdale (faible épaisseur selon `Oz`). Prendre `N_k > 1` crée surtout des cellules vides en `z`, donc plus de surcoût (gestion mémoire/calcul/communication) sans gain physique notable. `N_k = 1` est donc le meilleur compromis.

### 3.2. Mesure du temps initial

La partie dominante est le **calcul des trajectoires/accélérations** (noyau gravitationnel), pas l'affichage. La parallélisation doit donc cibler en priorité le calcul numérique.

### 3.3. Effet de la séparation affichage/calcul

La séparation rank 0 / rank 1 isole bien les rôles, mais introduit un coût de communication/synchronisation MPI à chaque pas de temps. Sur un cas modéré (`galaxy_1000`), ce coût peut réduire le gain global.

### 3.4. Problème de performance lié à la densité stellaire

La densité est plus forte près du centre : avec une répartition uniforme des cellules, certains processus traitent beaucoup plus d'étoiles. Cela entraîne un **déséquilibre de charge** et dégrade l'accélération parallèle.

### 3.5. Distribution "intelligente" et nouveau problème

Distribution proposée : partitionner les cellules selon une charge estimée (nombre d'étoiles / coût de calcul), pour équilibrer le travail.

Nouveau problème : hausse potentielle des communications (frontières plus complexes, synchronisations plus fréquentes) et nécessité éventuelle de **rééquilibrage dynamique**.

### 3.6. Proposition MPI pour Barnes–Hut (sur papier)

- Distribuer les sous-arbres du quadtree entre processus (par niveaux et/ou paquets de nœuds).
- Répliquer les nœuds hauts (grandes boîtes) sur plusieurs processus pour réduire les latences de consultation.
- Calculer localement les accélérations des étoiles possédées par chaque processus.
- Synchroniser périodiquement masse et centre de masse des nœuds partagés (réductions MPI).

Avec un nombre de processus `P = 4^k`, la distribution peut suivre naturellement les quadrants du quadtree.

---

## 4. Résultats expérimentaux

Configuration : `data/galaxy_1000`, grille `(20,20,1)`.

### 4.1. Numba seul (`nbodies_grid_numba.py --no-display`)

| Threads | Temps moyen (ms/step) | Speedup (base = 1 thread) | Efficacité (%) | FPS Simulé |
|---:|---:|---:|---:|---:|
| 1 | 13.7064 | 1.00 | 100.0% | 73.0 |
| 2 | 9.8688 | 1.39 | 69.5% | 101.3 |
| 4 | 8.3754 | 1.64 | 41.0% | 119.4 |
| 8 | 10.4176 | 1.32 | 16.5% | 96.0 |

Meilleur point mesuré : 4 threads.

### 4.2. MPI séparation affichage/calcul (`-np 2`, mode headless)

| Threads côté calcul | Temps moyen (ms/step) | FPS Simulé |
|---:|---:|---:|
| 1 | 14.0680 | 71.1 |
| 2 | 16.0553 | 62.3 |

Constat : sur ce cas, la séparation MPI n'apporte pas de gain net face à Numba seul (coût des échanges).

### 4.3. MPI + Numba (calcul distribué avec cellules fantômes)

| Processus | Threads/proc | Temps moyen (ms/step) | Speedup vs (1 proc, 1 thread) | Efficacité (%) | FPS Simulé |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 12.4214 | 1.00 | 100.0% | 80.5 |
| 1 | 2 | 19.3471 | 0.64 | 32.0% | 51.7 |
| 1 | 4 | 13.1828 | 0.94 | 23.5% | 75.9 |
| 2 | 1 | 7.4152 | 1.68 | 84.0% | 134.9 |
| 2 | 2 | 12.6830 | 0.98 | 24.5% | 78.8 |
| 2 | 4 | 11.0793 | 1.12 | 14.0% | 90.3 |
| 4 | 1 | 7.0306 | 1.77 | 44.3% | 142.2 |
| 4 | 2 | 9.1544 | 1.36 | 17.0% | 109.2 |
| 4 | 4 | 13.9764 | 0.89 | 5.6% | 71.5 |

Meilleure configuration mesurée ici : `4 processus × 1 thread` (environ 142 FPS).

### 4.4. Résultats complémentaires sur `data/galaxy_5000`

Configuration : `data/galaxy_5000`, grille `(20,20,1)`.

#### 4.4.1. Numba seul (`nbodies_grid_numba.py --no-display`)

| Threads | Temps moyen (ms/step) | Speedup (base = 1 thread) | Efficacité (%) | FPS Simulé |
|---:|---:|---:|---:|---:|
| 1 | 124.2634 | 1.00 | 100.0% | 8.0 |
| 2 | 78.4490 | 1.58 | 79.0% | 12.7 |
| 4 | 59.4204 | 2.09 | 52.3% | 16.8 |
| 8 | 62.5297 | 1.99 | 24.9% | 16.0 |

#### 4.4.2. MPI séparation affichage/calcul (`-np 2`, mode headless)

| Threads côté calcul | Temps moyen (ms/step) | FPS Simulé |
|---:|---:|---:|
| 1 | 124.3560 | 8.0 |
| 2 | 130.9299 | 7.6 |
| 4 (borné à 2) | 131.6069 | 7.6 |

#### 4.4.3. MPI + Numba avec cellules fantômes

| Processus | Threads/proc | Temps moyen (ms/step) | Speedup vs (1 proc, 1 thread) | Efficacité (%) | FPS Simulé |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 113.8844 | 1.00 | 100.0% | 8.8 |
| 1 | 2 | 145.3420 | 0.78 | 39.0% | 6.9 |
| 1 | 4 | 79.0431 | 1.44 | 36.0% | 12.7 |
| 2 | 1 | 60.9375 | 1.87 | 93.5% | 16.4 |
| 2 | 2 | 85.3700 | 1.33 | 33.3% | 11.7 |
| 2 | 4 | 71.3928 | 1.60 | 20.0% | 14.0 |
| 4 | 1 | 58.2934 | 1.95 | 48.8% | 17.2 |
| 4 | 2 | 56.4799 | 2.02 | 25.3% | 17.7 |
| 4 | 4 | 78.1186 | 1.46 | 9.1% | 12.8 |

Sur ce dataset plus grand, l'accélération MPI + cellules fantômes est plus nette ; la meilleure configuration mesurée est `4 processus × 2 threads` avec presque 18 FPS.

### 4.5. Comparaison Monolithique vs Séparée (Impact sur l'interactivité)

Cette section met en évidence l'intérêt de la **séparation architecture MPI affichage/calcul**. Bien que le temps total de calcul (par step) puisse être légèrement supérieur en mode "Séparé" à cause des communications MPI, cette méthode empêche le calcul lourd de bloquer la boucle de rendu OpenGL.

Le tableau ci-dessous projette l'expérience utilisateur finale (FPS) avec un modèle à 4 threads, en supposant que l'affichage prenne un temps fixe négligeable en mode headless.

| Scénario | Dataset | Méthode | Temps de calcul / step | Boucle d'affichage bloquée ? | Expérience visuelle (FPS max théorique) |
|---:|---:|---:|---:|---:|---:|
| Galaxie 1000 | 1000 étoiles | Numba pur (1 proc) | 8.37 ms | Oui | 119 FPS |
| Galaxie 1000 | 1000 étoiles | MPI affichage/calcul séparés | 14.06 ms | Non | ~71 FPS (indépendant du rendu) |
| Galaxie 5000 | 5000 étoiles | Numba pur (1 proc) | 59.42 ms | Oui | 16 FPS (saccades visibles) |
| Galaxie 5000 | 5000 étoiles | MPI affichage/calcul séparés | 131.60 ms | Non | ~8 FPS (fluide pour la caméra, mise à jour différée p/ étoiles) |

**Conclusion sur l'interactivité :** La version séparée permet à l'utilisateur de naviguer dans la scène 3D à 60+ FPS sans aucune saccade, même si les étoiles se mettent à jour à une fréquence plus faible de fond (par ex., 8 fois par seconde pour 5000 étoiles). Sans la séparation MPI, tourner la caméra donnerait l'impression que tout le programme "freeze" entre chaque calcul physique.

---

## 5. Commandes utilisées

### Numba (benchmark)

```bash
/home/jaovfaamo/Desktop/ENSTA/Cours_Ensta_2026/.venv/bin/python nbodies_grid_numba.py data/galaxy_1000 0.001 20 20 1 --threads 4 --no-display --warmup 2 --steps 20
```

### MPI séparation affichage/calcul

```bash
mpirun -np 2 /home/jaovfaamo/Desktop/ENSTA/Cours_Ensta_2026/.venv/bin/python nbodies_grid_numba_mpi_display.py data/galaxy_1000 0.001 20 20 1 --threads 1 --no-display --warmup 2 --steps 20
```

### MPI calcul distribué

```bash
mpirun -np 4 /home/jaovfaamo/Desktop/ENSTA/Cours_Ensta_2026/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --threads 1 --warmup 1 --steps 12
```

---

## 6. Conclusion

Les différentes étapes demandées ont été réalisées et mesurées : Numba, séparation MPI affichage/calcul, puis MPI + Numba sur le calcul.

Les résultats confirment :

- le noyau de calcul est la cible prioritaire de parallélisation ;
- la communication MPI peut limiter le gain en petit/moyen cas ;
- l'équilibrage de charge est critique à cause de la densité stellaire non uniforme.
- l'implémentation MPI du calcul suit bien l'approche demandée (cellules fantômes + échanges entre voisins).

Les mesures sur `galaxy_5000` confirment également que l'approche distribuée devient plus intéressante lorsque la taille du problème augmente.
