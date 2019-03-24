### Code for project on Particle Track Reconstruction - trackml dataset

#### The repository has code for project done under Dr. Kinjal Banerjee

Current Progress:
- [x] Initial data exploration
- [x] Clustering
- [x] Neural Network - FC: 86%
- [x] Random Forest: 93%
- [x] Gradient Boosted Classifiers: 96%
- [x] XGBoost Classifier, 500 trees and (max_depth = 25), Trained on 1 event: 98.1%
- [ ] Exploration of different Neural Network architectures

Particle Physics and Quantum Mechanics:
- [x] Chapter 1 Griffiths
- [x] Chapter 2 Griffiths
- [x] Introductory Quantum Mechanics

Current Approach:
1. Classification of 2 hits as promising or not
2. Classification of a third promising hit
3. Reconstruction of the trajectory based on the three hits classified as promising

- The current models are trained 1st step(i.e., classification of 2 hits as promising or not), since the same model can be extended in the second step
- In the final step, the hits that are closest to the reconstructed trajectory will be selected

