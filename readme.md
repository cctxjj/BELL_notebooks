# Algorithmen der Computergrafik – Anwendungsorientierte Kurvenentwicklung mittels Neuronaler Netze

Dieses Repository stellt den praktischen Anteil der BELL mit oben aufgeführtem Titel dar.
Es ist in unmittelbarem Bezug zur schriftlichen Grundlage zu setzen.
Dem allgemeinen Sprachgebrauch der anwendungsbezogenen Informatik folgend ist ein überwiegender Großteil der Kommentare in Englisch;
der Code ist konsequent englisch. 
Den wichtigsten Dokumenten sind kurze deutsche Erklärungen hinzugefügt.

Ein besonderer Dank gilt A. Roschlau sowie Dr. L. Müller für die Betreuung des Arbeitsprozesses.

## Zielsetzung
Angestrebt wird 
    1. die Umsetzung ausgewählter Algorithmen 
    2. ggf. die Evaluation dieser Algorithmen hinsichtlich Platz- und Raumkomplexität
    3. ggf. die Veranschaulichung des Vorgehens bzw. der Resultate
    2. die Realisation der zweckbasierten Kurve

## Frameworks
Projekt umgesetzt in Python 3.11, z.T. Jupyter Notebook
Datenhandhabung: NumPy, pandas, PIL
Neuronale Netze: TensorFlow 3
Visualisierung: Matplotlib
Regression: Scikit-Learn
Aerodynamik: NeuralFoil, Aerosandbox
weiteres siehe requirements.txt

## Projektstruktur
Gemäß der schriftlichen Darstellung ist das Projekt in zwei zentrale Module zu gliedern:
- Umsetzung der Algorithmen:
  - Rasterung mit Bresenham, DDA, Midpoint-Circle (/rasterisation/rasterisation.py) mit Demonstration (/rasterisation/rasterisation_algs_comparison.py) 
  - Area Fill Algorithmen (/area_fill/fill_algs.py) mit Demonstration (/area_fill/demonstration.py) und Vergleich in Zeit- und Platzkomplexität/-bedarf (/area_fill/time_and_space_evaluation.py)
  - B-Splines + NURBs (curves/func_based/basis_spline.py), Bezier-Kurven (curves/func_based/bézier_curve.py); Beispiele/Vergleiche der Kurventypen (curves/func_based/examples/) 

- Umsetzung der zweckbasierten Kurve mittels Neuronaler Netze:
  - Realisation B-Splines und Bézierkurven mittels NN (/curves/reconstruction/)
  - Implementierung cw-Wert als valide Metrik für das Training durch Surrogate Model für NeuralFoil (/curves/neural/custom_metrics/)
  - Training der zweckorientierte Kurve für Minimierung des cw-Wertes (/curves/neural/min_drag_curve_training.py)
  - Analyse der entstehenden NN einzeln (/curves/neural/purposebased/individual_model_analyses.py) und vergleichend (/curves/neural/purposebased/crossmodel_analysis.py)
  - regressive Zusammenhangsbeschreibung ausgewählter Kenngrößen (/curves/neural/regressors/)

zusätzlich finden sich:
- /util/: Hilfsfunktionen zur Datensatzerstellung (/util/datasets/), Visualisierung (/util/graphics/) und Datenverarbeitung (/util/shape_modifier.py)
- /data/: diverse Daten, hervorzuheben sind:
  - gespeicherte TF-Modelle (/data/models/)
  - gespeicherte Vergleichsergebnisse der Algorithmen (/data/algs_comparison/)
  - Einzel- und Vergleichanalyse csv-files aller Modelle (/data/model_analysis/)
  - Datensätze für das Training des NF-Surrogates (/data/nf_training_data/)
  - weitere Airfoil-Daten (/data/airfoils/) aus vorangegangenen Tests + Entwicklungsprozess

## Lizenz
Copyright (c) 2025 Sebastian Dietrich, alle Rechte vorbehalten;
siehe LICENSE

## Transparenzhinweis 
Teile der Codebase wurden unter Zuhilfenahme künstlicher Intelligenz entwickelt. Es handelt sich um inhaltlich nicht primär
relevante Vorgänge wie bspw. Visualisierungsprozesse; alle Implementationen des Codes wurden nachvollzogen, geprüft, integriert
und als solche kenntlich gemacht.

## Abkürzungen/Synonyme
.. zur erleichterten Verständlichkeit des Bezuges zwischen Textgrundlage und Code
NN --> Neuronale Netze 
TF --> Tensorflow
NF --> NeuralFoil
drag --> cw-Wert
drag_improvement --> cwv
loss_bez --> MSE
bez_shift --> MSEz
cd --> cw
RFR --> Random Forest Regressor
ratio --> MSE/cw

## Sonstiges
z. T. sind Speicherpfade absolut angegeben und müssen für die Anwendung auf einem anderen System angepasst werden.



