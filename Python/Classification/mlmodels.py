from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

def cross_validate_custom(X,y,estimator):
    metrics = ['accuracy','precision_macro','recall_macro','f1_macro','roc_auc']
    cv2 = cross_validate(estimator=estimator,X=X,y=y,cv=10,scoring=metrics,return_estimator=False)
    result ={}
    result['accuracy_mean'],result['accuracy_std'] = cv2['test_accuracy'].mean(),cv2['test_accuracy'].std()
    result['precision_mean'],result['precision_std'] = cv2['test_precision_macro'].mean(),cv2['test_precision_macro'].std()
    result['recall_mean'],result['recall_std'] = cv2['test_recall_macro'].mean(),cv2['test_recall_macro'].std()
    result['f1_mean'],result['f1_std'] = cv2['test_f1_macro'].mean(),cv2['test_f1_macro'].std()
    result['roc_auc_mean'],result['roc_auc_std'] = cv2['test_roc_auc'].mean(),cv2['test_roc_auc'].std()
    return result