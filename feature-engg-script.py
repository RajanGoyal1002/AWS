import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='Placement.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    parser.add_argument('--categorical_features', type=str, default='gender,ssc_b,hsc_b,hsc_s,degree_t,workex,specialisation,status')

    return parser.parse_known_args()

if __name__=="__main__":
    args, _ = _parse_args()
    
    df.drop(['sl_no'],axis=1,inplace=True)
    df.isnull().sum()
    df["gender"].value_counts()
    dic={"M":0,"F":1}
    df["gender"]=df["gender"].map(dic)

    df["ssc_b"].value_counts()
    dic={"Others":0,"Central":1}
    df["ssc_b"]=df["ssc_b"].map(dic)

    df["hsc_b"].value_counts()
    dic={"Others":0,"Central":1}
    df["hsc_b"]=df["hsc_b"].map(dic)

    df["workex"].value_counts()
    dic={"No":0,"Yes":1}
    df["workex"]=df["workex"].map(dic)

    df["specialisation"].value_counts()
    dic={"Mkt&Fin":0,"Mkt&HR":1}
    df["specialisation"]=df["specialisation"].map(dic)


    df["status"].value_counts()
    dic={"Placed":1,"Not Placed":0}
    df["status"]=df["status"].map(dic)

    df["degree_t"]=df["degree_t"].astype("object")

    df=pd.get_dummies(df,drop_first=True)
    # Train, test, validation split
    train_data, validation_data, test_data = np.split(df.sample(frac=1, random_state=42), [int(0.7 * len(df)), int(0.9 * len(df))])   # Randomly sort the data then split out first 70%, second 20%, and last 10%
    # Local store
    train_data.to_csv(os.path.join(args.outputpath, 'train/train_script.csv'), index=False, header=False)
    validation_data.to_csv(os.path.join(args.outputpath, 'validation/validation_script.csv'), index=False, header=False)
    test_data['status'].to_csv(os.path.join(args.outputpath, 'test/test_script_y.csv'), index=False, header=False)
    test_data.drop(['status'], axis=1).to_csv(os.path.join(args.outputpath, 'test/test_script_x.csv'), index=False, header=False)
    print("## Processing completed. Exiting.")