#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 15:16:58 2025

@author: cristiantobar
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.utils import resample
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.tree import export_graphviz
import graphviz

def sample_size_finite_population(N, Z=1.96, p=0.5, e=0.05):
    """
    Calcula el tamaño de muestra necesario para una población finita.

    Parámetros:
    N -- tamaño de la población (entero)
    Z -- valor Z para el nivel de confianza deseado (por defecto 1.96 para 95%)
    p -- proporción esperada (por defecto 0.5 para máxima variabilidad)
    e -- margen de error tolerado (por defecto 0.05 para ±5%)

    Retorna:
    n -- tamaño de la muestra necesario (entero)
    """
    numerator = (Z**2) * p * (1 - p) * N
    denominator = (e**2) * (N - 1) + (Z**2) * p * (1 - p)
    n = numerator / denominator
    return int(round(n))


def preparar_datos(namefile, num_merges):

    df = pd.read_csv(namefile, delimiter=';')
    
    if num_merges == 1:
        df.columns = ['ORIGEN', 'NUMPR', 'MATRI', 'ISIC4111', 'ISIC7111', 'ISIC7112',
                      'ISIC4220', 'ISIC6810', 'ISIC4290', 'ISIC4112', 'ISIC4390', 'ISIC4210',
                      'NUMISIC', 'ISICING', 'MIPYME', 'FALL', 'MALL', 'CF1']
    
    if num_merges == 2:
        df.columns = ['ORIGEN', 'NUMPR', 'MATRI', 'ISIC4111', 'ISIC7111', 'ISIC7112',
                      'ISIC4220', 'ISIC6810', 'ISIC4290', 'ISIC4112', 'ISIC4390', 'ISIC4210',
                      'NUMISIC', 'ISICING', 'MIPYME', 'FALL', 'MALL', 'CF1', 'CF2']
    
    if num_merges == 3:
        df.columns = ['ORIGEN', 'NUMPR', 'MATRI', 'ISIC4111', 'ISIC7111', 'ISIC7112',
                      'ISIC4220', 'ISIC6810', 'ISIC4290', 'ISIC4112', 'ISIC4390', 'ISIC4210',
                      'NUMISIC', 'ISICING', 'MIPYME', 'FALL', 'MALL', 'CF1', 'CF2', 'CF3']
    
    if num_merges == 4:
        df.columns = ['ORIGEN', 'NUMPR', 'MATRI', 'ISIC4111', 'ISIC7111', 'ISIC7112',
                      'ISIC4220', 'ISIC6810', 'ISIC4290', 'ISIC4112', 'ISIC4390', 'ISIC4210',
                      'NUMISIC', 'ISICING', 'MIPYME', 'FALL', 'MALL', 'CF1', 'CF2', 'CF3', 'CF4']
    
    df['CLASE_BINARIA'] = (df['MALL'] > 2).astype(int)

    # Ver distribución original
    original_distribution = df['CLASE_BINARIA'].value_counts(normalize=True)
    conteo_original_dist = df['CLASE_BINARIA'].value_counts()

    N = 141
    total_target = sample_size_finite_population(N)
        
    # Objetivo: 300 muestras con misma proporción
    #total_target = 360
    target_counts = (original_distribution * total_target).round().astype(int)
    
    # Separar las clases
    df_0 = df[df['CLASE_BINARIA'] == 0]
    df_1 = df[df['CLASE_BINARIA'] == 1]
    
    # Re-muestrear manteniendo la proporción
    df_0_resampled = resample(df_0, replace=True, n_samples=target_counts[0], random_state=42)
    df_1_resampled = resample(df_1, replace=True, n_samples=target_counts[1], random_state=42)
    
    # Concatenar el nuevo dataset balanceado proporcionalmente
    df_resampled = pd.concat([df_0_resampled, df_1_resampled], ignore_index=True)
    
    df_balanced = df_resampled
    
    # Verificación
    resampled_distribution = df_resampled['CLASE_BINARIA'].value_counts(normalize=True)
    resampled_counts = df_resampled['CLASE_BINARIA'].value_counts()
        
    # Asegúrate de que ambos tengan las mismas columnas (variables)
    #variables_a_comparar = df[['NUMPR','MATRI','ORIGEN']]  # puedes personalizar esta lista si quieres comparar solo algunas
    variables_a_comparar = df
    # Diccionario para guardar los resultados
    resultados_all = []

    # Recorremos todas las variables para hacer la prueba
    for var in variables_a_comparar:
        # Asegúrate de que sean variables numéricas
        if pd.api.types.is_numeric_dtype(df[var]):
            stat, p_value = mannwhitneyu(df[var], df_balanced[var], alternative='two-sided')
            resultados_all.append({
                'Variable': var,
                'U-Statistic': stat,
                'p-value': p_value,
                'Significativo (α=0.05)': 'Sí' if p_value < 0.05 else 'No'
            })
    
    # Mostrar resultados como DataFrame
    df_resultados_all = pd.DataFrame(resultados_all)
    df_resultados_all = df_resultados_all.sort_values(by='p-value')
    
    resultados = []
    
    for var in variables_a_comparar:
        # Original
        group0_orig = df[df['CLASE_BINARIA'] == 0][var]
        group1_orig = df[df['CLASE_BINARIA'] == 1][var]
    
        # Aumentado
        group0_aug = df_balanced[df_balanced['CLASE_BINARIA'] == 0][var]
        group1_aug = df_balanced[df_balanced['CLASE_BINARIA'] == 1][var]
        
        #Pruebas de hipótesis
        stat_orig, p_grupo_0 = mannwhitneyu(group0_orig, group0_aug, alternative='two-sided')
        stat_aug,  p_grupo_1 = mannwhitneyu(group1_orig, group1_aug, alternative='two-sided')
    
        resultados.append({
            'Variable': var,
            'p-value Grupo 0 (Or. vs Aug.)': p_grupo_0,
            'Significativo Original (α=0.05)': 'Sí' if p_grupo_0 < 0.05 else 'No',
            'p-value Grupo 1 (0r. vs Aug.)': p_grupo_1,
            'Significativo Aumentado (α=0.05)': 'Sí' if p_grupo_1 < 0.05 else 'No'
        })
    
    # Mostrar resultados como DataFrame
    df_resultados = pd.DataFrame(resultados)
        
    if num_merges == 1:
        # === 3. Preparar datos ===
        X = df_balanced.drop(['MALL','ISIC4111','ISIC7111','ISIC7112','ISIC4220',
                              'ISIC6810','ISIC4290','ISIC4112','ISIC4390','ISIC4210',
                              'MIPYME','ISICING','FALL','CLASE_BINARIA'], axis=1).copy()
        y = df_balanced['CLASE_BINARIA']

        # Codificación one-hot
        X_encoded = pd.get_dummies(X, columns=['ORIGEN', 'CF1'])
    
    if num_merges == 2:
        # === 3. Preparar datos ===
        X = df_balanced.drop(['MALL','ISIC4111','ISIC7111','ISIC7112','ISIC4220',
                              'ISIC6810','ISIC4290','ISIC4112','ISIC4390','ISIC4210',
                              'MIPYME','ISICING','FALL','CLASE_BINARIA'], axis=1).copy()
        y = df_balanced['CLASE_BINARIA']

        # Codificación one-hot
        X_encoded = pd.get_dummies(X, columns=['ORIGEN', 'CF1', 'CF2'])
    
    if num_merges == 3:
        # === 3. Preparar datos ===
        X = df_balanced.drop(['MALL','ISIC4111','ISIC7111','ISIC7112','ISIC4220',
                              'ISIC6810','ISIC4290','ISIC4112','ISIC4390','ISIC4210',
                              'MIPYME','ISICING','FALL','CLASE_BINARIA'], axis=1).copy()
        y = df_balanced['CLASE_BINARIA']

        # Codificación one-hot
        X_encoded = pd.get_dummies(X, columns=['ORIGEN', 'CF1', 'CF2', 'CF3'])
    
    if num_merges == 4:
        # === 3. Preparar datos ===
        X = df_balanced.drop(['MALL','ISIC4111','ISIC7111','ISIC7112','ISIC4220',
                              'ISIC6810','ISIC4290','ISIC4112','ISIC4390','ISIC4210',
                              'MIPYME','ISICING','FALL','CLASE_BINARIA'], axis=1).copy()
        y = df_balanced['CLASE_BINARIA']

        # Codificación one-hot
        X_encoded = pd.get_dummies(X, columns=['ORIGEN', 'CF1', 'CF2', 'CF3', 'CF4'])
    
    return X_encoded, y, df_resultados, resampled_counts, conteo_original_dist

def change_labels(uncert_source):
    origen = ['(Origin) Cauca',
            '(Origin) Nariño',
            '(Origin) Valle del Cauca',
            '(Origin) Huila',
            '(Other)']
    
    org_1 = ['(S) Project complexity',
            '(S) Implementation uniqueness',
            '(S) Both']

    org_2 = ['(S) Contractor selection',
            '(S) Subproject delegation',
            '(S) Bidding applications',
            '(S) Selection rubric creation']

    org_3 = ['(S) Subjective expert information',
            '(S) Linguistic variability among experts',
            '(S) Data variability among experts',
            '(S) Expert performance']

    org_4 = ['(S) Decision perception',
            '(S) Risk aversion',
            '(S) Risk-taking attitude']
    
    ad =   ['(S) Site handover timing',
            '(S) Project surface opening rate',
            '(S) Key supplier delivery times',
            '(S) Leader decision timing',
            '(S) Equipment logistics times']

    resou = ['(S) Inflexible cost est.',
            '(S) Inflexible resource est.',
            '(S) Inflexible cash flow es.']

    require = ['(S) Scope changes',
                '(S) Equipment changes',
                '(S) Design changes',
                '(S) Technical spec. changes',
                '(S) Unnecessary client interferences',
                '(S) Unclear client requirements',
                '(S) Design error',
                '(S) Design misalig. with client expect.']

    availa = ['(S) Inflexible renewable res. avail.',
            '(S) Inflexible non-renewable res. avail.',
            '(S) Limited worker avail.',
            '(S) Resource scarcity']

    log_1 = ['(S) Travel to site',
            '(S) Safety violations by construction staff',
            '(S) Others']

    log_2 = ['(S) Diffi. site access for machinery',
             '(S) Diffi. site access for contractors']

    log_3 = ['(S) Supply chain structure',
            '(S) Inventory management',
            '(S) Material acquisition',
            '(S) Timely resource availa.']

    env_1 = ['(S) Heavy rains',
            '(S) Thunderstorms',
            '(S) Extreme cold']

    env_2 = ['(S) Earthquakes',
             '(S) Geological hazards',
             '(S) Challenging terrain',]

    socio_1 = ['(S) Env. requirements',
            '(S) Non-env. friendly equip.',
            '(S) Others',
            '(S) None']

    socio_2 = ['(S) Worker social discon.',
                '(S) Political unrest',
                '(S) Labor conflicts',
                '(S) Cultural issues',
                '(S) Non-work. days granted']

    market_1 = ['(S) Contractor wages',
                '(S) Supply prices',
                '(S) Inflation',
                '(S) Interest',
                '(S) Exchange rates',
                '(S) Fuel prices',
                '(S) Transportation costs',
                '(S) Global recession',
                '(S) Credit access',
                '(S) Taxes',
                '(S) Product sales volume']

    market_2 = ['(S) Requirement changes',
                '(S) Contract disputes',
                '(S) Regulatory issues',
                '(S) Unclear specifications',
                '(S) Communication issues',
                '(S) Supplier capacity']

    tech = ['(S) Equipment reliability',
            '(S) Construction program',
            '(S) Design risks',
            '(S) Renewable resource effic.']
    
    if uncert_source == 'Organizational':
        manage_names = {
                    'NUMPR' : 'Num. projects',
                    'MATRI' : 'Months in service',
                    'NUMISIC': 'Num. activities',
                    'FALL' : 'Perceived Frequency',
                      'ORIGEN_1': origen[0], 
                      'ORIGEN_2': origen[1],
                      'ORIGEN_3': origen[2],
                      'ORIGEN_4': origen[3],
                      'ORIGEN_5': origen[4],
                      'CF1_1': org_1[0],
                      'CF1_2': org_1[1],
                      'CF1_3': org_1[2],
                      'CF2_1': org_2[0],
                      'CF2_2': org_2[1],
                      'CF2_3': org_2[2],
                      'CF2_4': org_2[3],
                      'CF3_1': org_3[0],
                      'CF3_2': org_3[1],
                      'CF3_3': org_3[2],
                      'CF3_4': org_3[3],
                      'CF4_1': org_4[0],
                      'CF4_2': org_4[1],
                      'CF4_3': org_4[2],
                      }
    
    if uncert_source == 'Activity Durations':
        manage_names = {
                                        'NUMPR' : 'Num. projects',
                                        'MATRI' : 'Months in service',
                                        'NUMISIC': 'Num. activities',
                                        'FALL' : 'Perceived Frequency',
                                          'ORIGEN_1': origen[0], 
                                          'ORIGEN_2': origen[1],
                                          'ORIGEN_3': origen[2],
                                          'ORIGEN_4': origen[3],
                                          'ORIGEN_5': origen[4],
                                          'CF1_1': ad[0],
                                          'CF1_2': ad[1],
                                          'CF1_3': ad[2],
                                          'CF1_4': ad[3],
                                          'CF1_5': ad[4]
                                          }
        
        
    if uncert_source == 'Resources Use':
        
        manage_names = {
                                        'NUMPR' : 'Num. projects',
                                        'MATRI' : 'Months in service',
                                        'NUMISIC': 'Num. activities',
                                        'FALL' : 'Perceived Frequency',
                                          'ORIGEN_1': origen[0], 
                                          'ORIGEN_2': origen[1],
                                          'ORIGEN_3': origen[2],
                                          'ORIGEN_4': origen[3],
                                          'ORIGEN_5': origen[4],
                                          'CF1_1': resou[0],
                                          'CF1_2': resou[1],
                                          'CF1_3': resou[2],
                                          }
        
        
    if uncert_source == 'Changes in Req. & Qual.':
        
        manage_names = {
                                        'NUMPR' : 'Num. projects',
                                        'MATRI' : 'Months in service',
                                        'NUMISIC': 'Num. activities',
                                        'FALL' : 'Perceived Frequency',
                                          'ORIGEN_1': origen[0], 
                                          'ORIGEN_2': origen[1],
                                          'ORIGEN_3': origen[2],
                                          'ORIGEN_4': origen[3],
                                          'ORIGEN_5': origen[4],
                                          'CF1_1': require[0],
                                          'CF1_2': require[1],
                                          'CF1_3': require[2],
                                          'CF1_4': require[3],
                                          'CF1_5': require[4],
                                          'CF1_6': require[5],
                                          'CF1_7': require[6],
                                          'CF1_8': require[7],
                                          }
        
    if uncert_source == 'Resource Availability':

        manage_names = {
                                        'NUMPR' : 'Num. projects',
                                        'MATRI' : 'Months in service',
                                        'NUMISIC': 'Num. activities',
                                        'FALL' : 'Perceived Frequency',
                                          'ORIGEN_1': origen[0], 
                                          'ORIGEN_2': origen[1],
                                          'ORIGEN_3': origen[2],
                                          'ORIGEN_4': origen[3],
                                          'ORIGEN_5': origen[4],
                                          'CF1_1': availa[0],
                                          'CF1_2': availa[1],
                                          'CF1_3': availa[2],
                                          'CF1_4': availa[3],
                                          }
    
    
    
    if uncert_source == 'Logistics':

        manage_names = {
                                        'NUMPR' : 'Num. projects',
                                        'MATRI' : 'Months in service',
                                        'NUMISIC': 'Num. activities',
                                        'FALL' : 'Perceived Frequency',
                                          'ORIGEN_1': origen[0], 
                                          'ORIGEN_2': origen[1],
                                          'ORIGEN_3': origen[2],
                                          'ORIGEN_4': origen[3],
                                          'ORIGEN_5': origen[4],
                                          'CF1_1': log_1[0],
                                          'CF1_2': log_1[1],
                                          'CF1_3': log_1[2],
                                          'CF2_1': log_2[0],
                                          'CF2_2': log_2[1],
                                          'CF3_1': log_3[0],
                                          'CF3_2': log_3[1],
                                          'CF3_3': log_3[2],
                                          'CF3_4': log_3[3],
                                          }       
        
        
    if uncert_source == 'Environmental':

        manage_names = {
                                        'NUMPR' : 'Num. projects',
                                        'MATRI' : 'Months in service',
                                        'NUMISIC': 'Num. activities',
                                        'FALL' : 'Perceived Frequency',
                                          'ORIGEN_1': origen[0], 
                                          'ORIGEN_2': origen[1],
                                          'ORIGEN_3': origen[2],
                                          'ORIGEN_4': origen[3],
                                          'ORIGEN_5': origen[4],
                                          'CF1_1': env_1[0],
                                          'CF1_2': env_1[1],
                                          'CF2_1': env_2[0],
                                          'CF2_2': env_2[1],
                                          'CF2_3': env_2[2],
                                          }       
        
        
    if uncert_source == 'Sociopolitical':

        manage_names = {
                                        'NUMPR' : 'Num. projects',
                                        'MATRI' : 'Months in service',
                                        'NUMISIC': 'Num. activities',
                                        'FALL' : 'Perceived Frequency',
                                          'ORIGEN_1': origen[0], 
                                          'ORIGEN_2': origen[1],
                                          'ORIGEN_3': origen[2],
                                          'ORIGEN_4': origen[3],
                                          'ORIGEN_5': origen[4],
                                          'CF1_1': socio_1[0],
                                          'CF1_2': socio_1[1],
                                          'CF1_3': socio_1[2],
                                          'CF1_4': socio_1[3],
                                          'CF2_1': socio_2[0],
                                          'CF2_2': socio_2[1],
                                          'CF2_3': socio_2[2],
                                          'CF2_4': socio_2[3],
                                          'CF2_5': socio_2[4],
                                          }      
        
    if uncert_source == 'Market':

        manage_names = {
                                        'NUMPR' : 'Num. projects',
                                        'MATRI' : 'Months in service',
                                        'NUMISIC': 'Num. activities',
                                        'FALL' : 'Perceived Frequency',
                                          'ORIGEN_1': origen[0], 
                                          'ORIGEN_2': origen[1],
                                          'ORIGEN_3': origen[2],
                                          'ORIGEN_4': origen[3],
                                          'ORIGEN_5': origen[4],
                                          'CF1_1': market_1[0],
                                          'CF1_2': market_1[1],
                                          'CF1_3': market_1[2],
                                          'CF1_4': market_1[3],
                                          'CF1_5': market_1[4],
                                          'CF1_6': market_1[5],
                                          'CF1_7': market_1[6],
                                          'CF1_8': market_1[7],
                                          'CF1_9': market_1[8],
                                          'CF1_10': market_1[9],
                                          'CF1_11': market_1[10],
                                          }     
            
    if uncert_source == 'Technological':

        manage_names = {
                                        'NUMPR' : 'Num. projects',
                                        'MATRI' : 'Months in service',
                                        'NUMISIC': 'Num. activities',
                                        'FALL' : 'Perceived Frequency',
                                          'ORIGEN_1': origen[0], 
                                          'ORIGEN_2': origen[1],
                                          'ORIGEN_3': origen[2],
                                          'ORIGEN_4': origen[3],
                                          'ORIGEN_5': origen[4],
                                          'CF1_1': tech[0],
                                          'CF1_2': tech[1],
                                          'CF1_3': tech[2],
                                          'CF1_4': tech[3],
                                          }
    return manage_names

def bootstrap_rf_ct(X_encoded, y, uncert_source, lit):
    
    # === 4. Random Forest para feature importance + bootstrap ===
    feature_counts_rf = defaultdict(int)
    feature_importance_values = defaultdict(list)

    n_iterations = 1000

    for i in range(n_iterations):
        X_resampled, y_resampled = resample(X_encoded, y, replace=True, random_state=42+i)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_resampled, y_resampled)
        
        for idx, col in enumerate(X_encoded.columns):
            imp = rf.feature_importances_[idx]
            if imp > 0:
                feature_counts_rf[col] += 1
            feature_importance_values[col].append(imp)


    # Consolidar en DataFrame
    df_importancia_rf = pd.DataFrame({
        'Feature': list(feature_counts_rf.keys()),
        'Frequency': [feature_counts_rf[k] for k in feature_counts_rf.keys()],
        'MeanImportance': [np.mean(feature_importance_values[k]) for k in feature_counts_rf.keys()],
        'StdImportance': [np.std(feature_importance_values[k]) for k in feature_counts_rf.keys()]
    }).sort_values(by='Frequency', ascending=False)
    
    # Ordenar por importancia
    df_plot = df_importancia_rf.sort_values(by='MeanImportance', ascending=True)
        
    # Cambio de nombres tecnicos a nombres management
    manage_names = change_labels(uncert_source)
    df_plot['Feature'] = df_plot['Feature'].replace(manage_names)

    # === 5. Bootstrap para encontrar el mejor árbol podado en cada muestra ===
    tree_rule_counts = defaultdict(int)
    tree_rules_list = []
    trees_saved = {}

    for i in range(n_iterations):
        X_boot, y_boot = resample(X_encoded, y, replace=True, random_state=1000+i)
        X_train, X_test, y_train, y_test = train_test_split(X_boot, y_boot, test_size=0.2, random_state=42)

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Poda por costo de complejidad
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas[:-1]
                
        best_score = 0
        best_tree = None
        for alpha in ccp_alphas:
            clf_alpha = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
            scores = cross_val_score(clf_alpha, X_train, y_train, cv=5)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_tree = clf_alpha

        if best_tree:
            best_tree.fit(X_train, y_train)
            rule_text = export_text(best_tree, feature_names=list(X_encoded.columns), show_weights=False)
            tree_rules_list.append(rule_text)
            trees_saved[rule_text] = best_tree

    # Encontrar el árbol más frecuente
    rule_counts = Counter(tree_rules_list)
    most_common_rule, freq = rule_counts.most_common(1)[0]
    best_model = trees_saved[most_common_rule]
        
    letra = 30
    letra_ct =  25
    # === 6. Graficar ambos en un solo plot ===
    fig, axs = plt.subplots(1, 2, figsize=(35, 18), gridspec_kw={'width_ratios': [1, 2]})

    # Subplot 1: Importancia de características
    cmap = plt.cm.Greens
    norm = plt.Normalize(0, n_iterations)
    colors = [cmap(norm(f)) for f in df_plot['Frequency']]
    bars = axs[0].barh(df_plot['Feature'], df_plot['MeanImportance'], xerr=df_plot['StdImportance'], color=colors, edgecolor='black', capsize=5)
    axs[0].set_xlabel('Average Feature Importance', fontsize=letra)
    axs[0].set_ylabel(f"Feature of {uncert_source} Uncertainty", fontsize=letra)
    axs[0].set_title('a) Feature Importance with Bootstrap Frequency', fontsize=letra+2)
    axs[0].tick_params(axis='both', labelsize=letra)  # Aumenta tamaño de ticks en eje x e y
    axs[0].set_xlim(0, 0.5)  # Limitar eje x de 0 a 0.4
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs[0])
    cbar.set_label(f"Frequency in {n_iterations} bootstraps", fontsize=letra)
    cbar.ax.tick_params(labelsize=letra-5)
    
    # Parámetros del árbol de clasificación y la gráfica
    num_leaves = best_model.tree_.node_count  # Número de nodos del árbol
    fontsize = letra_ct - (num_leaves // 2) if num_leaves > 10 else letra_ct  # Ajusta el tamaño de fuente
    
    X_encoded = X_encoded.rename(columns = manage_names)
    
    # Subplot 2: Árbol de decisión
    plot_tree(best_model,
              filled=True,
              rounded=True,
              proportion=True,
              class_names=['Lower Uncer.', 'Higher Uncer.'],
              fontsize=fontsize,
              feature_names=X_encoded.columns,
              ax=axs[1])
    axs[1].set_title(f"b) Most Frequent Classification Tree (frecuency: {freq}/{n_iterations})",fontsize=letra+2)
    
    # Exportar a formato DOT
    dot_data = export_graphviz(
        best_model,
        out_file=None,
        feature_names=X_encoded.columns,
        class_names=['Lower Uncer.', 'Higher Uncer.'],
        filled=True,
        rounded=True,
        proportion=True,
        precision=2
    )
    
    # Parámetros de tamaño de la figura
    fig_width = 5      # pulgadas
    fig_height = 5     # pulgadas
    dpi = 100
    title_ratio = 0.06  # 6% del alto
    
    # Calcular el fontsize del título
    title_fontsize = int(dpi * fig_height * title_ratio)

    
    titulo = f'Most Frequent CT for {uncert_source} uncer. (frequency: {freq}/{n_iterations})'

        
    # Insertar atributos globales para tamaño cuadrado y calidad
    #custom_header = 'graph Tree {\nsize="5,5!"; ratio=fill; dpi=300;\n'
    custom_header = (
    'graph Tree {\n'
    f'label="{titulo}";\n'
    'labelloc=top;\n'
    f'fontsize={title_fontsize};\n'
    f'size="{fig_width},{fig_height}!";\n'
    #'ratio=fill;\n'
    f'dpi={dpi};\n'
    'margin=0.05;\n'
    # 'nodesep=0.6;\n'
    # 'ranksep=0.8;\n'
    )
    dot_data_custom = dot_data.replace('graph Tree {', custom_header, 1)
    
    # Crear objeto Graphviz
    graph = graphviz.Source(dot_data_custom)

    # Guardar como PDF
    graph.render(f"tree_graphviz_{uncert_source}", format="pdf", cleanup=True)
    
    # Convertir cm a pulgadas
    cm = 1 / 2.54
    figsize = (25 * cm, 20 * cm)  # 6x6 cm
    
    # Crear figura cuadrada
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colormap y normalización con tope en n_iterations
    cmap = plt.cm.Greens
    norm = plt.Normalize(vmin=0, vmax=n_iterations)
    colors = [cmap(norm(f)) for f in df_plot['Frequency']]
    
    # Dibujar gráfico de barras horizontales
    bars = ax.barh(
        df_plot['Feature'],
        df_plot['MeanImportance'],
        xerr=df_plot['StdImportance'],
        color=colors,
        edgecolor='black',
        capsize=2
    )
    
    # Títulos y etiquetas con tamaño compacto
    fontsize = 16  # Pequeño pero legible en 6x6 cm
    ax.set_xlabel('Avg. Importance', fontsize=fontsize)
    ax.set_ylabel(f"{uncert_source} Features", fontsize=fontsize)
    ax.set_title(f"{lit}) Feature Importance for {uncert_source} Unc.", fontsize=fontsize+6, pad=15)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_xlim(0, 0.5)
    
    # Colorbar del mismo alto que el eje
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # # Añadir colorbar ajustada al tamaño reducido
    # cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.01, aspect=10)
    # cbar.set_label(f"Freq. ({n_iterations})", fontsize=fontsize)
    # cbar.ax.tick_params(labelsize=fontsize)
    
    # Colorbar
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f"Frequency in {n_iterations} bootstraps", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    
    # Ajustar espaciado para que todo quepa
    plt.tight_layout(pad=0.5)
    plt.savefig(f'feature_importance_{uncert_source}_6x6cm.pdf', bbox_inches='tight')
    plt.show()
    
    plt.tight_layout()
    plt.savefig('bootstrap' + uncert_source + '_tree' + '.pdf', bbox_inches='tight')
    plt.show()

    print(f"Uncertainty source: {uncert_source} processed")
    return df_plot

# === 1. Cargar CSV y preparar columnas ===
data_v2 = [ '5_test_v4_organ_fac_all.csv',
            '6_test_v4_ad.csv',
            '7_test_v4_resources.csv',
            '8_test_v4_requirements.csv',
            '9_test_v4_availability.csv',
            '10_test_v4_logistics.csv',
            '11_test_v4_environment.csv',
            '12_test_v4_socio.csv',
            '13_test_v4_market.csv',
            '14_test_v4_tech.csv'
            ]

ex_feat              = [4,
                        1,
                        1,
                        1,
                        1,
                        3,
                        2,
                        2,
                        2,
                        1]

uncert_source = ['Organizational',
                 'Activity Durations',
                 'Resources Use',
                 'Changes in Req. & Qual.',
                 'Resource Availability',
                 'Logistics',
                 'Environmental',
                 'Sociopolitical',
                 'Market',
                 'Technological'
                 ]


X_encoded, y, org_dif_sig, org_aug, org_dis_or = preparar_datos('5_test_v4_organ_fac_all.csv', 4)
impor_org = bootstrap_rf_ct(X_encoded, y, 'Organizational', 'a')

X_encoded, y, ad_dif_sig, ad_aug, ad_dis_or = preparar_datos('6_test_v4_ad.csv', 1)
impor_ad = bootstrap_rf_ct(X_encoded, y, 'Activity Durations', 'b')

X_encoded, y, res_dif_sig, res_aug, res_dis_or = preparar_datos('7_test_v4_resources.csv', 1)
impor_res = bootstrap_rf_ct(X_encoded, y, 'Resources Use', 'c')

X_encoded, y, requi_dif_sig, requi_aug, requi_dis_or = preparar_datos('8_test_v4_requirements.csv', 1)
impor_requi = bootstrap_rf_ct(X_encoded, y, 'Changes in Req. & Qual.', 'd')

X_encoded, y, avail_dif_sig, avail_aug, avail_dis_or = preparar_datos('9_test_v4_availability.csv', 1)
impor_avai = bootstrap_rf_ct(X_encoded, y, 'Resource Availability','e')

X_encoded, y, log_dif_sig, log_aug, log_dis_or = preparar_datos('10_test_v4_logistics.csv', 3)
impor_log = bootstrap_rf_ct(X_encoded, y, 'Logistics','a')

X_encoded, y, env_dif_sig, env_aug, env_dis_or = preparar_datos('11_test_v4_environment.csv', 2)
impor_env = bootstrap_rf_ct(X_encoded, y, 'Environmental','b')

X_encoded, y, socio_dif_sig, socio_aug, socio_dis_or = preparar_datos('12_test_v4_socio.csv', 2)
impor_socio = bootstrap_rf_ct(X_encoded, y, 'Sociopolitical','c')

X_encoded, y, mark_dif_sig, mark_aug, mark_dis_or = preparar_datos('13_test_v4_market.csv', 1)
impor_mark = bootstrap_rf_ct(X_encoded, y, 'Market','d')

X_encoded, y, tech_dif_sig, tech_aug, tech_dis_or = preparar_datos('14_test_v4_tech.csv', 1)
impor_tech = bootstrap_rf_ct(X_encoded, y, 'Technological','e')


# reporte_aumento = pd.DataFrame(np.array([
#     ['Organizational',          int(org_dis_or.loc[0]), int(org_aug.loc[0]), float(org_dif_sig['p-value Grupo 0 (Or. vs Aug.)'].min()), int(org_dis_or.loc[1]), int(org_aug.loc[1]), float(org_dif_sig['p-value Grupo 1 (0r. vs Aug.)'].min())],
#     ['Activity Durations',      int(ad_dis_or.loc[0]), int(ad_aug.loc[0]), float(ad_dif_sig['p-value Grupo 0 (Or. vs Aug.)'].min()), int(ad_dis_or.loc[1]), int(ad_aug.loc[1]), float(ad_dif_sig['p-value Grupo 1 (0r. vs Aug.)'].min())],
#     ['Resources Use',           int(res_dis_or.loc[0]), int(res_aug.loc[0]), float(res_dif_sig['p-value Grupo 0 (Or. vs Aug.)'].min()), int(res_dis_or.loc[1]), int(res_aug.loc[1]), float(res_dif_sig['p-value Grupo 1 (0r. vs Aug.)'].min())],
#     ['Changes in Req. & Qual.', int(requi_dis_or.loc[0]), int(requi_aug.loc[0]), float(requi_dif_sig['p-value Grupo 0 (Or. vs Aug.)'].min()), int(requi_dis_or.loc[1]), int(requi_aug.loc[1]), float(requi_dif_sig['p-value Grupo 1 (0r. vs Aug.)'].min())],
#     ['Resource Availability',   int(avail_dis_or.loc[0]), int(avail_aug.loc[0]), float(avail_dif_sig['p-value Grupo 0 (Or. vs Aug.)'].min()), int(avail_dis_or.loc[1]), int(avail_aug.loc[1]), float(avail_dif_sig['p-value Grupo 1 (0r. vs Aug.)'].min())],
#     ['Logistics',               int(log_dis_or.loc[0]), int(log_aug.loc[0]), float(log_dif_sig['p-value Grupo 0 (Or. vs Aug.)'].min()), int(log_dis_or.loc[1]), int(log_aug.loc[1]), float(log_dif_sig['p-value Grupo 1 (0r. vs Aug.)'].min())],
#     ['Environmental',           int(env_dis_or.loc[0]), int(env_aug.loc[0]), float(env_dif_sig['p-value Grupo 0 (Or. vs Aug.)'].min()), int(env_dis_or.loc[1]), int(env_aug.loc[1]), float(env_dif_sig['p-value Grupo 1 (0r. vs Aug.)'].min())],
#     ['Sociopolitical',          int(socio_dis_or.loc[0]), int(socio_aug.loc[0]), float(socio_dif_sig['p-value Grupo 0 (Or. vs Aug.)'].min()), int(socio_dis_or.loc[1]), int(socio_aug.loc[1]), float(socio_dif_sig['p-value Grupo 1 (0r. vs Aug.)'].min())],
#     ['Market',                  int(mark_dis_or.loc[0]), int(mark_aug.loc[0]), float(mark_dif_sig['p-value Grupo 0 (Or. vs Aug.)'].min()), int(mark_dis_or.loc[1]), int(mark_aug.loc[1]), float(mark_dif_sig['p-value Grupo 1 (0r. vs Aug.)'].min())],
#     ['Technological',           int(tech_dis_or.loc[0]), int(tech_aug.loc[0]), float(tech_dif_sig['p-value Grupo 0 (Or. vs Aug.)'].min()), int(tech_dis_or.loc[1]), int(tech_aug.loc[1]), float(tech_dif_sig['p-value Grupo 1 (0r. vs Aug.)'].min())],
#     ]),
#     columns= ['Unc.Source','Low_Or','Low_Aug','Low_Low_p','High_or','High_Aug','High_Low_p']
#     ) 

# import xlsxwriter

# writer = pd.ExcelWriter("importances_df.xlsx")

# List_dfs = [impor_org, impor_ad, impor_res, impor_requi, impor_avai, impor_log, impor_env, impor_socio, impor_mark, impor_tech]

# names = ["org", "ad", "res", "requi", "avai", "log", "env", "socio", "mark", "tech"]

# for i, frame in enumerate(List_dfs):
#    frame.to_excel("{names[i]}+.xlsx", sheet_name = names[i], index=False)