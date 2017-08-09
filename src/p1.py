#!/usr/bin/env python

import pandas
from sklearn import preprocessing

def to_feature(trxfile, cardfile, custfile):
    trx_df = pandas.read_csv(trxfile)
    card_df = pandas.read_csv(cardfile)
    cust_df = pandas.read_csv(custfile)

    feature_df = pandas.merge(card_df, cust_df, on='cst_id').set_index('card_no').loc[:, ['cr_lmt_amt', 'prev_cr_lmt_amt', 'incm_amt', 'age', 'main_zip_cd', 'cr_line_amt']]

    # compute txn_amount_total/mean/median/max/min/count
    trx_amt_df = trx_df.loc[:, ['card_no', 'txn_amount']]
    feature_df['amount_total'] = trx_amt_df.groupby(by='card_no').sum()
    feature_df['amount_mean'] = trx_amt_df.groupby(by='card_no').mean()
    feature_df['amount_median'] = trx_amt_df.groupby(by='card_no').median()
    feature_df['amount_min'] = trx_amt_df.groupby(by='card_no').max()
    feature_df['amount_max'] = trx_amt_df.groupby(by='card_no').min()
    feature_df['count'] = trx_amt_df.groupby(by='card_no').count()

    # create another attribute to help the learner
    feature_df['amount_by_income'] = feature_df['incm_amt'] / feature_df['amount_mean']
    feature_df.fillna(0, inplace=True)

    return feature_df

def to_learn(trxfile, cardfile, custfile, trainfile, testfile):
    """Return data in the format usable with sklearn
    """
    feature_df = to_feature(trxfile, cardfile, custfile)
    feature_df.loc[:] = preprocessing.scale(feature_df)
    #feature_df.loc[:] = preprocessing.normalize(feature_df, norm='l2')
    
    # card_no, label
    train_df = pandas.read_csv(trainfile, header=None)
    # card_no
    test_df = pandas.read_csv(testfile, header=None)

    train_data = feature_df.loc[train_df.loc[:, 0]]
    train_label = train_df.loc[:, 1]
    test_data = feature_df.loc[test_df.loc[:, 0]]

    return (train_data.values, train_label.values, test_data.values)

if __name__ == '__main__':
    from sys import argv
    if len(argv) < 6:
        print("Usage: %s trxfile cardfile custfile trainfile testfile" % argv[0])
        exit(1)
    (X1, y1, T1) = to_learn(argv[1], argv[2], argv[3], argv[4], argv[5])
    from sklearn import svm
    svm1 = svm.SVC()
    #svm1 = svm.LinearSVC(loss='squared_hinge', penalty='l1', dual=False)
    svm1.fit(X1, y1)
    P1 = svm1.predict(T1)
    for label in P1:
        print(label)
