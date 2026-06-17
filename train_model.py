# -*- coding: utf-8 -*-
"""
train_model.py
Training script to build the Random Forest model for Late Delivery Risk prediction.
Loads local dataset, performs preprocessing, trains the classifier, and saves the artifacts.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def main():
    print("--- MEMULAI PROSES TRAINING MODEL ---")
    
    # Path dataset lokal
    dataset_path = 'DataCoSupplyChainDataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_path}' tidak ditemukan di folder lokal. Silakan salin terlebih dahulu.")
        
    print("1. Loading dataset...")
    dt = pd.read_csv(dataset_path, encoding='latin1')
    print(f"Dataset berhasil dimuat: {dt.shape[0]} baris, {dt.shape[1]} kolom")
    
    # Definisi 24 Fitur yang digunakan oleh model
    fitur = [
        'Type',
        'Days for shipping (real)',
        'Benefit per order',
        'Sales per customer',
        'Category Name',
        'Customer Segment',
        'Department Name',
        'Market',
        'Order City',
        'Order Country',
        'Order Region',
        'Order State',
        'Order Status',
        'Order Item Discount',
        'Order Item Discount Rate',
        'Order Item Product Price',
        'Order Item Profit Ratio',
        'Order Item Quantity',
        'Sales',
        'Order Item Total',
        'Order Profit Per Order',
        'Product Name',
        'Product Price',
        'Shipping Mode'
    ]
    target = 'Late_delivery_risk'
    
    # Memisahkan Fitur (X) dan Target (y)
    X = dt[fitur].copy()
    y = dt[target].astype(int)
    
    # Identifikasi tipe kolom untuk preprocessing
    kolom_numerik = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    kolom_kategorikal = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Jumlah kolom numerik: {len(kolom_numerik)}")
    print(f"Jumlah kolom kategorikal: {len(kolom_kategorikal)}")
    
    # Split data training dan testing (80/20 stratified split)
    print("2. Melakukan Split Train-Test Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Pipeline Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, kolom_numerik),
            ('cat', categorical_transformer, kolom_kategorikal)
        ]
    )
    
    # Model Random Forest Classifier (max_depth=2, n_estimators=100)
    print("3. Membuat Model Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=2,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rf_model)
    ])
    
    # Training Model
    print("4. Melakukan Training Pipeline Model...")
    model_pipeline.fit(X_train, y_train)
    print("Training model selesai.")
    
    # Evaluasi pada Data Test
    print("5. Mengevaluasi Performa Model...")
    y_pred = model_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n=== HASIL EVALUASI MODEL RANDOM FOREST ===")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%")
    
    # Menyimpan Artifact Model
    print("\n6. Menyimpan model_rf.pkl dan fitur_final.pkl...")
    joblib.dump(model_pipeline, 'model_rf.pkl')
    joblib.dump(fitur, 'fitur_final.pkl')
    print("Artifact berhasil disimpan di folder lokal.")
    print("Proses Selesai!")

if __name__ == '__main__':
    main()
