import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
import pandas as pd
import numpy as np


class Mimic3(Dataset):
    def __init__(self, data_path, cohorts=None, apply_vocab=True, to_tensor=True):
        super().__init__()
        
        if cohorts is None:
            cohorts = ['Atherosclerosis',
                       'Heart Failure',
                       'Kidney Failure',
                       'Intestinal Diseases',
                       'Liver Diseases',
                       'Pneumonia',
                       'Septicemia',
                       'Respiratory Failure',
                       'Gastriti']
        self.cohorts = cohorts
        self.to_tensor = to_tensor
        
        self.load_tables(data_path)
        self.data = self.build_medical_concepts()
        self.labels = self.find_cohorts()
        
        self.vocab = None
        if apply_vocab:
            self.vocab = self.build_vocab()
        
        
    def __getitem__(self, index):
        codes = self.data.iloc[index]
        cohorts = self.labels[index]
        
        if self.vocab:
            codes = self.vocab(codes)
        
        if self.to_tensor:
            codes = torch.tensor(codes, dtype=torch.long)
        
        return codes, cohorts
    
    
    def __len__(self):
        return len(self.data)
    
    
    def build_vocab(self):
        return build_vocab_from_iterator(self.data)
    
    
    def num_codes(self):
        if self.vocab is None:
            return None
        return len(self.vocab)
    
        
    def load_tables(self, data_path): 
        df = pd.read_csv(data_path + 'ADMISSIONS.csv')
        diag_icd = pd.read_csv(data_path + 'DIAGNOSES_ICD.csv')
        drgcodes = pd.read_csv(data_path + 'PRESCRIPTIONS.csv')
        proc_icd = pd.read_csv(data_path + 'PROCEDURES_ICD.csv')
        diag_desc = pd.read_csv(data_path + 'D_ICD_DIAGNOSES.csv')

        df['ADMITTIME'] = pd.to_datetime(df.ADMITTIME, format='%Y-%m-%d %H:%M:%S')
        df['ADMITTIME'] = pd.to_datetime(df.ADMITTIME, format='%Y-%m-%d')
        df['ADMITTIME'] = df['ADMITTIME'].dt.strftime('%Y-%m-%d')

        df['DISCHTIME'] = pd.to_datetime(df.DISCHTIME, format='%Y-%m-%d %H:%M:%S')
        df['DISCHTIME'] = pd.to_datetime(df.DISCHTIME, format='%Y-%m-%d')
        df['DISCHTIME'] = df['DISCHTIME'].dt.strftime('%Y-%m-%d')

        df['EDREGTIME'] = pd.to_datetime(df.EDREGTIME, format='%Y-%m-%d %H:%M:%S')
        df['EDREGTIME'] = pd.to_datetime(df.EDREGTIME, format='%Y-%m-%d')
        df['EDREGTIME'] = df['EDREGTIME'].dt.strftime('%Y-%m-%d')

        df['EDOUTTIME'] = pd.to_datetime(df.EDOUTTIME, format='%Y-%m-%d %H:%M:%S')
        df['EDOUTTIME'] = pd.to_datetime(df.EDOUTTIME, format='%Y-%m-%d')
        df['EDOUTTIME'] = df['EDOUTTIME'].dt.strftime('%Y-%m-%d')

        diag_icd = diag_icd.merge(diag_desc, on='ICD9_CODE', how='inner')
        
        self.df = df 
        self.diag_icd = diag_icd
        self.drgcodes = drgcodes
        self.proc_icd = proc_icd
        self.diag_desc = diag_desc
        
        
    def filter_subjects(self):
        remove_ids = set()
        
        # (1) We remove the patients with missing data on admission date and discharge date;
        remove_ids.update(self.df[self.df['ADMITTIME'].isna() | self.df['DISCHTIME'].isna()]['SUBJECT_ID'])

        # (2) We keep the patients which consist of at least thirty medical codes;
        # size = diag_icd.groupby(['SUBJECT_ID']).size()
        # remove_ids.update(size[size < 30].index)


        # (3) We remove the patients which have the discharge date after 2200/1/1; 
        remove_ids.update(self.df[self.df['DISCHTIME'] > '2200-1-1']['SUBJECT_ID'])

        # (4) We remove the patients who have the missing data on diagnosis.
        remove_ids.update(self.diag_icd[self.diag_icd['ICD9_CODE'].isna()]['SUBJECT_ID'])

        # (5) Atherosclerosis, Heart Failure, Kidney Failure, Intestinal Diseases, Liver Diseases, Pneumonia, Septicemia, Respiratory Failure and Gastriti
        is_in_cohort = self.diag_icd['LONG_TITLE'].str.lower().str.contains('|'.join(self.cohorts).lower()).rename('IN_COHORT')
        not_in_cohort = pd.concat((self.diag_icd['SUBJECT_ID'], ~is_in_cohort), axis=1).groupby('SUBJECT_ID')['IN_COHORT'].aggregate(np.all)
        remove_ids.update(not_in_cohort[not_in_cohort].index)

        # (6) we remove the medical concepts that are co-occurring less than three times
        codes = self.diag_icd.groupby('ICD9_CODE')['SUBJECT_ID'].nunique()
        remove_codes = codes[codes < 3].index
        self.diag_icd = self.diag_icd[~ self.diag_icd['ICD9_CODE'].isin(remove_codes)]

        codes = self.drgcodes.groupby('NDC')['SUBJECT_ID'].nunique()
        remove_codes = codes[codes < 3].index
        self.drgcodes = self.drgcodes[~ self.drgcodes['NDC'].isin(remove_codes)]


        codes = self.proc_icd.groupby('ICD9_CODE')['SUBJECT_ID'].nunique()
        remove_codes = codes[codes < 3].index
        self.proc_icd = self.proc_icd[~ self.proc_icd['ICD9_CODE'].isin(remove_codes)]
        
        return remove_ids
    
    
    def build_medical_concepts(self):
        remove_ids = self.filter_subjects()
        
        proc = (self.proc_icd.groupby('SUBJECT_ID')['ICD9_CODE']
                .aggregate(list)
                .rename('PROC')
                .apply(lambda codes: ['PROC_'+str(c) for c in codes])
        )
        
        drug = (self.drgcodes.groupby('SUBJECT_ID')['NDC']
                .aggregate(list)
                .rename('DRUG')
                .apply(lambda codes: ['DRUG_'+str(c) for c in codes])
        )
        
        diag = (self.diag_icd.groupby('SUBJECT_ID')['ICD9_CODE']
                .aggregate(list)
                .rename('DIAG')
                .apply(lambda codes: ['DIAG_'+str(c) for c in codes])
                )
        
        med_concepts = pd.concat([diag, drug, proc], axis=1, join='outer')
        med_concepts.drop(index=remove_ids, errors='ignore', inplace=True)
        
        # replace NAN with [] on every column
        for c in med_concepts.columns:
            med_concepts[c] = med_concepts[c].apply(lambda x: x if isinstance(x, list) else [])
        
        med_concepts = med_concepts['DIAG'] + med_concepts['PROC'] + med_concepts['DRUG']
        
        # remove subjects with less than 30 codes #TODO
        med_concepts = med_concepts[med_concepts.apply(len) > 30]

        return med_concepts

    def find_cohorts(self):
        """finds cohorts of each subject"""
        labels = []
        for sid in self.data.index:
            diags = self.diag_icd[self.diag_icd['SUBJECT_ID'] == sid]['LONG_TITLE']
            diags = ' '.join(diags)
        
            label = []
            for cid, cohort in enumerate(self.cohorts):
                if cohort.lower() in diags.lower():
                    label.append(cid)
                
            labels.append(label)    
        
        return labels