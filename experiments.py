




def run_tikho_experiment(X_train, X_test, y_train, y_test, scale=0.0, mtype = "linear"):
    model = LinearTikhonovClassifier(scale=scale)
    model = model.fit(X_train, y_train)
    # print(grad1)
    # print(model.gradient_check(X_train, y_train, model.coef_, model.intercept_).shape)
    predictions = model.predict(X_train)
    score = model.score(y_train, predictions)
    loss = model.loss(X_train, y_train)
    
    return score, loss
    
    
def run_data_experiment(X, y, scale=0.0, test_noise=0.0, train_noise=0.0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if train_noise >= 0.0:
        X_train = X_train + np.random.normal(loc=0, scale=train_noise, size = X_train.shape)
    else:
        X_train = X_train - np.random.normal(loc=0, scale=-train_noise, size = X_train.shape)
    if test_noise >= 0.0:
        X_test = X_test + np.random.normal(loc=0, scale=test_noise, size = X_test.shape)
    else:
        X_test = X_test - np.random.normal(loc=0, scale=-test_noise, size = X_test.shape)
    
    return run_tikho_experiment(X_train, X_test, y_train, y_test, scale=scale)

def run_feature_experiment(scale=0.0, train_noise = 0.0, test_noise = 0.0, **kwargs):
    X, y = make_classification(**kwargs)
    return run_data_experiment(X, y, scale=scale, train_noise=train_noise, test_noise=test_noise)





print("Experiment 2: Tikhonov regularization")
scale_scores = []
scale_losses = []
scales = list(np.logspace(-5, 5, 11) * -1)
scales.reverse()    
scales.append(0.0)
scales.extend(np.logspace(-5, 5, 11).tolist())
for scale in scales:
    score, loss = run_tikho_experiment(X_train, X_test, y_train, y_test, scale=scale)
    scale_scores.append(score)
    scale_losses.append(loss)
    print(f"Scale: {scale:.3e}, Score: {score}, Loss: {loss}")
print("#"*80)

print("Experiment 3: Training with Noise")
train_scores = []
train_losses = []
for train_noise in [-100, -10, -1, -.1, -.01, -.001, -.0001, 0, .0001, .001, .01, .1, 1, 10, 100, ]:
    score, loss = run_data_experiment(X, y, scale=0.0, test_noise=0.0, train_noise=train_noise)
    print(f"Train Noise: {train_noise:.3e}, Score: {score}, Loss: {loss}")
    train_scores.append(score)
    train_losses.append(loss)
print("#"*80)    

print("Experiment 4: Testing with Noise")
test_scores = []
test_losses = []
scale=scales[scale_losses.index(min(scale_losses))]
for test_noise in [-100, -10, -1, -.1, -.01, -.001, -.0001, 0, .0001, .001, .01, .1, 1, 10, 100, ]:
    score, loss = run_data_experiment(X, y, scale=scale, train_noise=0.0, test_noise=test_noise)
    print(f"Test Noise: {test_noise:.3e}, Score: {score}, Loss: {loss}")
    test_scores.append(score)
    test_losses.append(loss)
print("#"*80)

print("Experiment 5: Varying number of features")
data_scores = []
data_losses = []
features = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
n_informative = [int(round(x * .8))  for x in features]
for feat, inf in zip(features, n_informative):
    score, loss = run_feature_experiment(n_features=feat, n_informative=inf, n_redundant = feat-inf, n_clusters_per_class=1, scale=10)
    print(f"Features: {feat}, Score: {score}, Loss: {loss}")
    data_scores.append(score)
    data_losses.append(loss)

print("Experiment 6: Varying number of informative features")
info_scores = []
info_losses = []
features = [100] * 9
n_informative = [int(round(x * 100))  for x in [ .01, .1, .2, .3, .4, .5, .6, .7, .8]]
for feat, inf in zip(features, n_informative):
    score, loss = run_feature_experiment(n_features=feat, n_informative=inf, n_redundant = feat-inf, n_clusters_per_class=1, scale=-10)
    print(f"Number of Informative: {inf}, Score: {score}, Loss: {loss}")
    info_scores.append(score)
    info_losses.append(loss)
print("#"*80)

print("Experiment 7: Varying number of samples")
for samples in [100, 300, 500, 1000, 3000, 5000, 10000, 100000]:
    score, loss = run_feature_experiment(n_samples=samples, n_features=100, n_informative=80, n_redundant = 20, n_clusters_per_class=1, scale=10)
    print(f"Samples: {samples}, Score: {score}, Loss: {loss}")