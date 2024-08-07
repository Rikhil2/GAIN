import pandas as pd

df = pd.read_csv('data/adjusted/imputed.csv',
                 names=['GENDER_SRC_DESC', 'MARITAL_STATUS_AT_DX_DESC', 'ETHNICITY_SRC_DESC',
                        'RACE_CR_SRC_DESC_1', 'HISTOLOGY_CD', 'PRIMARY_SITE_CD',
                        'PRIMARY_SITE_GROUP_DESC', 'PRIMARY_SITE_REGION_DESC',
                        'METS_AT_DX_DISTANT_LYMPH_NODES_DESC', 'METS_AT_DX_OTHER_DESC',
                        'SUMMARY_OF_RX_1ST_COURSE_DESC', 'SURGERY_RADIATION_SEQ_NUM',
                        'SYSTEMIC_RX_SURGERY_SEQ_NUM', 'DIAGNOSTIC_CONFIRMATION_METHOD_DESC',
                        'TUMOR_BEHAVIOR_DESC', 'VITAL_STATUS', 'Cessation Meds',
                        'Cessation Meds Past 90 Days', 'CRA_SMOKE_5PK_EVER',
                        'CRA_SMOKE_HOME_EXPOSED', 'CRA_SMOKE_WORK_EXPOSED', 'CRA_TOBACCO_SNUFF',
                        'AGE_AT_DIAGNOSIS_NUM', 'REGIONAL_NODES_EXAMINED',
                        'REGIONAL_NODES_POSITIVE', 'SURVIVAL_TIME_IN_MONTHS',
                        'PreTreatment_Smoke_Status', 'Tumor_size', 'Quit_Status_Any_time',
                        'CRA_SMOKE_AGE', 'CRA_SMOKE_TOTALYRS'])
df = df.astype('int64')

df.to_csv('data/adjusted/imputed.csv', index=False)
print(df.head(2))
