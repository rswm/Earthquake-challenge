from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier



def get_score(n_estimators):
  my_pipeline = Pipeline(steps=[
      #('preprocessor', SimpleImputer()),
      ('model', RandomForestClassifier(n_estimators, random_state=0))
  ])
  scores =  -1 * cross_val_score(my_pipeline, X_train, y_train,
                                 cv=3,
                                 scoring='neg_mean_absolute_error')

  return scores.mean()


def find_best_score(start, stop, increment):
  
    results = {}
    # this is GridSearchCV
    # we are looking for the optimal number of estimators
    # using 3-fold cross validation
    for i in range(start, stop, increment):
      results[i] = get_score(i)


    # Finding the key with the lowest value
    min_key = min(results, key=results.get)
    min_value = results[min_key]

    return min_key, min_value, results


def plot_results(results):

    # Creating the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(results.keys(), results.values(), marker='o')
    plt.title('Line Plot of Results')
    plt.xlabel('Key')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()






