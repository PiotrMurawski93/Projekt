#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import clone
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, make_scorer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline


# ### Visualization Functions

# In[2]:


def annotate_boxplot(ax, boxplot_data):
    """The function is used to annotate the median values on a boxplot. 
    It iterates over the medians of the boxplot data 
    and adds text annotations at the corresponding positions."""
    
    medians = boxplot_data.median().values
    for i, m in enumerate(medians):
        ax.text(i, m, f'{m:.2f}', horizontalalignment='center', verticalalignment='baseline', color='black')

def plot_numerical_feature(feature, data,  bins='auto'):
    """Function used for exploratory data analysis (EDA) 
    of a numerical feature in relation to the churn status. 
    It generates histograms and boxplots, to compare the distribution 
    and central tendency of the feature between churned and non-churned customers. 
    """
    
    # Set up a 2x2 table of plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Visualizations of {feature} by Churn Status', fontsize=16)
    
    # 1. Histogram for the whole dataset
    sns.histplot(ax=axes[0, 0], data=data, x=feature, kde=True, bins=bins)
    axes[0, 0].set_title(f'Histogram of {feature}')
    
    # 2. Histogram for churn and non-churn on the same graph
    sns.histplot(ax=axes[0, 1], data=data, x=feature, hue='churn', kde=True, bins=bins, 
                 palette={0: '#99eabf', 1: '#FF6F61'})
    axes[0, 1].set_title(f'Histogram of {feature} by Churn Status')

    # 3. Boxplot for the whole dataset
    boxplot_whole = sns.boxplot(ax=axes[1, 0], y=data[feature])
    axes[1, 0].set_title(f'Boxplot of {feature}')
    axes[1, 0].set_ylabel(f'{feature}')
    annotate_boxplot(axes[1, 0], data[[feature]])
    
    # 4. Boxplots for churn and non-churn
    boxplot_churn = sns.boxplot(ax=axes[1, 1], x='churn', y=feature, data=data, palette={0: '#99eabf', 1: '#FF6F61'})
    axes[1, 1].set_title(f'Boxplot of {feature} by Churn Status')
    axes[1, 1].set_xlabel('Churn Status')
    axes[1, 1].set_ylabel(f'{feature}')
    axes[1, 1].set_xticklabels(['Non-churned', 'Churned'])
    annotate_boxplot(axes[1, 1], data.groupby('churn')[feature])
    
    # Show the plots
    plt.show()


# In[3]:


def plot_categorical_feature(feature, data):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    """Function used for exploratory data analysis (EDA)
    of a categorical feature in relation to the churn status.
    It generates two types of visualizations: pie chart that 
    shows the percentage of each category in the feature and 
    stacked bar plot that illustrates the count of churned and 
    non-churned customers for each category"""
    
    # 1. Pie chart showing the percentage of each category
    category_counts = data[feature].value_counts(normalize=True) * 100
    colors = ['#B7D8E8', '#92C6DF', '#70B4D5', '#4FA2CA']
    category_counts.plot(kind='pie', autopct='%.2f%%', ax=axs[0], colors=colors)
    axs[0].set_ylabel('')
    axs[0].set_title(f'Percentage of each category in {feature}')

    # 2. Stacked bar plot for churn and non-churn for each category
    churn_counts = data.groupby([feature, 'churn']).size().unstack(fill_value=0)
    churn_counts.plot(kind='bar', stacked=True, ax=axs[1], color=['#99eabf', '#FF6F61'])
    axs[1].set_xlabel(feature)
    axs[1].set_ylabel('Count')
    axs[1].set_title(f'Churn count by {feature}')

    # Add percentage labels on the stacked bar plot
    total_counts = churn_counts.sum(axis=1)
    for i, (_, row) in enumerate(churn_counts.iterrows()):
        churn_percentage = row[1] / total_counts[i] * 100
        non_churn_percentage = row[0] / total_counts[i] * 100
        axs[1].annotate(f'{churn_percentage:.2f}%', (i, row[0] + row[1] / 2), ha='center', va='center')
        axs[1].annotate(f'{non_churn_percentage:.2f}%', (i, row[0] / 2), ha='center', va='center')

    plt.tight_layout()
    plt.show()


# In[4]:


def plot_ordinal_feature(feature, data):
    
    """"Function designed for EDA ordinal categorical features
    in relation to the churn status. It generates bar plot for 
    the whole dataset. Stacked bar plot (percentage) which 
    shows churn percentages for each category. Box plot for 
    the whole dataset and two box plots for churn and non-churn"""
    
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

     # 1. Bar plot for whole data
    ax1 = sns.countplot(x=feature, data=data, ax=axs[0, 0], color='#B7D8E8')
    axs[0, 0].set_title(f'Count of {feature}')
    axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=90)
    for p in ax1.patches:
        ax1.annotate(format(p.get_height(), '.0f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 10), 
                     textcoords = 'offset points')

    # 2. Stacked bar plot (percentage)
    churn_counts = data.groupby([feature, 'churn']).size().unstack(fill_value=0)
    churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100
    churn_percentages.plot(kind='bar', stacked=True, ax=axs[0, 1], color=['#99eabf', '#FF6F61'])
    axs[0, 1].set_title(f'Churn percentage by {feature}')
    axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=90)

    for p, index in zip(axs[0, 1].patches, churn_percentages.index):
        non_churn_pct = churn_percentages.loc[index, 0]
        churn_pct = churn_percentages.loc[index, 1]
        x = p.get_x() + p.get_width() / 2
        if p.get_height() > 0:
            axs[0, 1].annotate(f"{non_churn_pct:.0f}%", (x, non_churn_pct / 2), ha='center', va='center', color='black', fontsize=10)
        if p.get_height() + churn_pct > 0:
            axs[0, 1].annotate(f"{churn_pct:.0f}%", (x, p.get_height() + churn_pct / 2), ha='center', va='center', color='black', fontsize=10)


    # 3. Box plot for whole data
    sns.boxplot(y=data[feature], ax=axs[1, 0], orient='v')
    axs[1, 0].set_title(f'Box plot of {feature} for whole data')
    annotate_boxplot(axs[1, 0], data[[feature]])

    # 4. Two box plots for churn and non-churn
    sns.boxplot(x='churn', y=feature, data=data, ax=axs[1, 1], palette={0: '#99eabf', 1: '#FF6F61'})
    axs[1, 1].set_title(f'Box plot of {feature} for churn and non-churn')
    annotate_boxplot(axs[1, 1], data.groupby('churn')[feature])

    plt.tight_layout()
    plt.show()


# ### Experiment Functions

# In[5]:


def run_experiments(preprocessor, classifiers, pipeline, experiment_name, X, y, cv=5):
    results = []
    
    """
    This function conducts machine learning experiments by accepting a preprocessor, 
    classifiers, a pipeline, and an experiment name as inputs. It records the performance 
    of each classifier and creates a DataFrame with key performance metrics.
    """
    
    results = []

    for classifier in classifiers:
        pipeline.steps.append(('clf', classifier))
        pipeline.fit(X, y)

        classifier_name = classifier.__class__.__name__

        # Perform cross-validation
        cv_recall = cross_val_score(pipeline, X, y, scoring='recall', cv=cv).mean()
        cv_precision = cross_val_score(pipeline, X, y, scoring='precision', cv=cv).mean()
        cv_f1 = cross_val_score(pipeline, X, y, scoring='f1', cv=cv).mean()
        cv_roc_auc = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv).mean()

        # Append results
        results.append([classifier_name, experiment_name, cv_f1, cv_recall, cv_precision, cv_roc_auc])

        # Remove the classifier step from the pipeline
        pipeline.steps.pop()

    # Create a DataFrame with the results
    results_df = pd.DataFrame(results, columns=['Classifier', 'Experiment', 'Cross-Validation F1', 'Cross-Validation Recall', 
                                                'Cross-Validation Precision',  'Cross-Validation ROC AUC'])

    return results_df


# In[6]:


def run_experiments_hyperparameter(preprocessor, classifiers, pipeline, experiment_name, X_train, y_train, scoring_metric='f1', 
                                   rs_n_iter=100 ):
    
    """This function conducts machine learning experiments by accepting a preprocessor, 
    classifiers, a pipeline, and an experiment name as inputs. It applies GridSearchCV 
    or RandomizedSearchCV based on the classifier to optimize hyperparameters. 
    It records the best performing model for each classifier and creates a DataFrame with 
    key performance metrics."""
    
    
    
    randomized_search_classifiers = [XGBClassifier, LGBMClassifier]
    
    best_classifiers = {}
    
    scoring = {'f1': make_scorer(f1_score), 'recall': make_scorer(recall_score), 
               'precision': make_scorer(precision_score), 'roc_auc': make_scorer(roc_auc_score)}
    
    results = []
    
    for classifier, param_grid in classifiers:
        pipeline_copy = clone(pipeline) 
        pipeline_copy.steps.append(('clf', classifier))
        
        if isinstance(classifier, tuple(randomized_search_classifiers)):
            search = RandomizedSearchCV(pipeline_copy, param_grid, scoring=scoring, refit=scoring_metric, 
                                        cv=5, n_iter=rs_n_iter, n_jobs=-1)
        else:
            search = GridSearchCV(pipeline_copy, param_grid, scoring=scoring, refit=scoring_metric, 
                                  cv=5, n_jobs=-1)

        search.fit(X_train, y_train)
        classifier_name = classifier.__class__.__name__
        best_classifiers[classifier_name] = search.best_estimator_

    
        recall_train = search.cv_results_['mean_test_recall'][search.best_index_]
        precision_train = search.cv_results_['mean_test_precision'][search.best_index_]
        f1_train = search.cv_results_['mean_test_f1'][search.best_index_]
        roc_auc_train = search.cv_results_['mean_test_roc_auc'][search.best_index_]

        results.append([experiment_name, classifier_name, f1_train, recall_train, precision_train, roc_auc_train])

    results_df = pd.DataFrame(results, columns=['Experiment', 'Classifier', 'Train F1', 'Train Recall', 
                                                'Train Precision', 'Train ROC AUC'])

    mean_results = results_df.mean(numeric_only=True)  
    mean_results['Experiment'] = experiment_name  
    mean_results['Classifier'] = 'Mean' 

    mean_results_df = pd.DataFrame(mean_results).transpose()
    results_df = pd.concat([results_df, mean_results_df], ignore_index=True) 

    
    return best_classifiers, results_df


# In[7]:


def evaluate_best_classifiers(best_classifiers, X_test, y_test):
    results = []
    
    for classifier_name, model in best_classifiers.items():
        y_pred_test = model.predict(X_test)
        f1_test = f1_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)
        y_prob_test = model.predict_proba(X_test)[:,1]
        roc_auc_test = roc_auc_score(y_test, y_prob_test)

        results.append([classifier_name, f1_test, recall_test, precision_test, roc_auc_test])

    results_df_final = pd.DataFrame(results, columns=['Classifier', 'Test F1', 'Test Recall', 'Test Precision',  
                                                      'Test ROC AUC'])

    return results_df_final


# In[8]:


def append_dataframe( new_df, existing_df):
    """
    Appends dataframe of each experiment.
    """
    concatenated_df = pd.concat([existing_df, new_df], ignore_index=True)
    concatenated_df = concatenated_df.sort_values(by=['Classifier', 'Cross-Validation F1'],
                                                  ascending=False).reset_index(drop=True)

    return concatenated_df


# In[9]:


def append_dataframes_list(df_list):
    """
    Appends multiple dataframes.
    """
    # Concatenate the DataFrames
    concatenated_df = pd.concat(df_list, ignore_index=True)

    # Sort the columns in alphabetical order
    concatenated_df = concatenated_df.sort_values(by=['Classifier', 'Cross-Validation F1'],
                                                  ascending=False).reset_index(drop=True)

    return concatenated_df


# In[ ]:




