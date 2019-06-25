from syn_dataset import SynDataset
from sklearn.metrics import accuracy_score
from sklearn import tree
from adl_features import get_all_feats
from sklearn import preprocessing
from adl_dataset import ADLDataset
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
if __name__ == '__main__':
    train_dataset = ADLDataset('dataset/adl', is_train=True, mini=True)

    train_dataset = SynDataset('samples/adl_crnn/06_25_09_43', None)
    train_data, train_labels = train_dataset.data, train_dataset.labels
    train_X = get_all_feats(train_data)
    test_dataset = ADLDataset('dataset/adl', is_train=False, mini=True)
    test_data, test_labels = test_dataset.data, test_dataset.labels
    test_X = get_all_feats(test_data)

    # Train classifier
    clf = SVC(kernel='rbf', C=1000, class_weight='balanced')
    #clf = tree.DecisionTreeClassifier()
    #clf = LogisticRegression()
    # clf = MLPClassifier(hidden_layer_sizes=(
    #    128), max_iter=1000, activation='tanh')
    scaler = preprocessing.StandardScaler().fit(train_X)

    train_X_scaled = scaler.transform(train_X)
    test_X_scaled = scaler.transform(test_X)
    clf.fit(train_X_scaled, train_labels)

    test_pred = clf.predict(test_X_scaled)

    test_acc = accuracy_score(test_labels, test_pred)
    print(test_acc)

    print(confusion_matrix(test_labels, test_pred))
