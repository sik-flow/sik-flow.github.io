---
layout: project
title: On the Trustworthiness of Tree Ensemble Explainability Methods
description: Trustworthiness of Tree Ensemble Methods
category: Papers
---

Link to [Paper](https://arxiv.org/pdf/2110.00086v1.pdf)

## What did the authors try to accomplish?
The authors looked at the stability and accuracy of Tree Ensemble Explainability methods - Gain and SHAP values.  

## What are the key elemints of the approach?
The authors created several datasets and used a randomized set of coefficients to make the target variable, they used coefficients so they could see what the most important feature was (magnitude of the largest coefficient).  They performed input perturbation by adding small levels of noise to the data and model perturbation by using different random seeds or different hyperparameters.  They looked at the accuracy of Gain and SHAP by looking at how often these techniques produced the same most important features as the theoretical best features (magnitude of largest coefficient).  SHAP and Gain correctly ranked the #1 feature correct about 40% of the time.  To leak at stability, they looked at the correlation of the feature importances when either the input or model had perturbation performed.  They found that SHAP is more stable, but both techniques struggled.  

## What can you use yourself?
I find that when I am looking at the feature importance of tree ensemble methods I am typically using Gain because it is fast and does not require an additional package, both of the metrics the authors looked at (accuracy and stability) show that SHAP tree importance does better.  The authors also show that both of these techniques have issues and should not be trusted as a ground truth.  Going forward I will try using SHAP tree importance more, but also not weighing the feature importance too high as it may be incorrect.  

The authors also had an interesting description of the difference between SHAP and Gain.  The authors state: the difference lies on the fact that **gain measures the feature’s contribution to accuracy improvements or decreasing of uncertainty/variance** whereas **SHAP measures the feature’s contribution to the predicted output.** 

## What other references do you want to follow? 
The authors, in their Future Work, mention a new idea of using variable cloud importance [source](https://arxiv.org/pdf/1901.03209.pdf) and I would be interested in exploring that.  

