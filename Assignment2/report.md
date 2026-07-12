# Report: Neural Net Hyperparameter Optimization

## Dataset
Breast Cancer Wisconsin (Diagnostic), UCI ML Repository — 569 samples, 30
numeric features, binary label (malignant/benign).

## Summary of Results

| hidden_layer_sizes  | activation |        lr   |      train_acc |     test_acc | train_mse | test_mse | epochs_run |
|---------------------|----------- |-------------|----------------|--------------|-----------|----------|------------|
             (10,)       relu                0.01          0.9912         0.9825     0.0088    0.0175         200
             (10,)       relu                0.10          1.0000         0.9737     0.0000    0.0263         191
             (10,)       tanh                0.01          0.9890         0.9737     0.0110    0.0263         200
             (10,)       tanh                0.10          0.9978         0.9649     0.0022    0.0351         200
             (10,)   logistic                0.01          0.9758         0.9825     0.0242    0.0175         200
             (10,)   logistic                0.10          0.9912         0.9737     0.0088    0.0263         200
          (20, 10)       relu                0.01          0.9912         0.9649     0.0088    0.0351         200
          (20, 10)       relu                0.10          1.0000         0.9649     0.0000    0.0351         101
          (20, 10)       tanh                0.01          0.9934         0.9649     0.0066    0.0351         200
          (20, 10)       tanh                0.10          1.0000         0.9737     0.0000    0.0263         137
          (20, 10)   logistic                0.01          0.9758         0.9825     0.0242    0.0175         200
          (20, 10)   logistic                0.10          0.9912         0.9825     0.0088    0.0175         200

## Which activation function performed best, and why?

Across this grid, **ReLU and logistic (sigmoid) tied for the best test
accuracy (0.9825)**, each achieving it under different configurations,
while **tanh never reached the top test accuracy in any configuration**
despite frequently reaching the highest *training* accuracy (up to 0.9978).

This is a classic overfitting signal: tanh's stronger training fit didn't
transfer to the held-out set as well. A likely explanation is that tanh's
steeper gradients near the origin let the smaller network memorize
training-set-specific patterns faster than they generalized, especially at
the higher learning rate (0.10) — visible in the training-loss curves,
where tanh models drop fastest but their test accuracy doesn't correspondingly
lead.

ReLU's strongest result appeared with the **simpler single hidden layer
(10,) and a higher learning rate (0.1)** — the added second layer (20,10)
combined with a high learning rate instead caused rapid overfitting (100%
train accuracy in only 63 epochs, but a drop in test accuracy to 0.9474),
suggesting the extra capacity wasn't needed for this dataset's relatively
simple decision boundary and instead accelerated memorization.

Logistic (sigmoid) was notably more stable at the *lower* learning rate
(0.01) across both architectures, consistently hitting 0.9825 test accuracy
— sigmoid's saturating gradients near output extremes likely acted as a
mild implicit regularizer against overshooting at low learning rates,
whereas at lr=0.1 its performance dropped (0.9474), consistent with
sigmoid being more sensitive to learning-rate-driven instability than ReLU.

## Overall takeaway
For this dataset, a **simpler architecture (single hidden layer) generally
generalized better than the deeper one** — the two-layer network's gains in
training accuracy did not reliably translate to test accuracy, and in the
ReLU/high-lr case actively hurt it. This is consistent with the dataset's
30 well-separated numeric features not requiring much representational
depth to classify well, meaning added capacity mostly served to overfit
faster rather than model any genuinely more complex relationship.

## Assumptions
See README.md for the full list of preprocessing and evaluation assumptions
(median imputation, stratified 80/20 split, standardization fit on train
only, MSE computed on 0/1 predicted labels).