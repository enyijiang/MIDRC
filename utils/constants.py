import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = '/shared/rsaas/enyij2/'

DATA_DIR = os.path.join(SHARED_DIR, 'midrc', 'data')
META_DATA_DIR = 'metadata'
STATE_DATA_DIR = os.path.join(META_DATA_DIR, 'midrc', 'states')
CHEXPERT_DATA_DIR = os.path.join(META_DATA_DIR,'chexpert')
MIMIC_DATA_DIR = os.path.join(META_DATA_DIR, 'mimic', 'mimic_small')
# DEMO_DATA_DIR = os.path.join(SHARED_DIR, 'CXR', 'data_demo')
MSDA_BASE_DIR = os.path.join(SHARED_DIR, 'msda')

SOURCE_TRAIN_CSV_PATH = os.path.join(META_DATA_DIR, 'MIDRC_cr_table_all_train.csv')
TARGET_TRAIN_CSV_PATH = os.path.join(META_DATA_DIR, 'MIDRC_dx_table_all_train.csv')
SOURCE_TEST_CSV_PATH = os.path.join(META_DATA_DIR, 'MIDRC_cr_table_all_test.csv')
TARGET_TEST_CSV_PATH = os.path.join(META_DATA_DIR, 'MIDRC_dx_table_all_test.csv')

IL_TRAIN_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_IL_train.csv')
IL_TEST_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_IL_test.csv')
NC_TRAIN_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_NC_train.csv')
NC_TEST_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_NC_test.csv')
CA_TRAIN_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_CA_train.csv')
CA_TEST_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_CA_test.csv')
IN_TRAIN_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_IN_train.csv')
IN_TEST_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_IN_test.csv')
TX_TRAIN_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_TX_train.csv')
TX_TEST_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_TX_test.csv')

# RESULTS_DIR = os.path.join(SHARED_DIR, 'domain_adapt_cr2dx')

# FINE_TUNING_RESULTS_DIR = os.path.join(RESULTS_DIR, 'fine_tuning')

# SHARED_DIR = '/shared/rsaas/nschiou2/'

# DATA_DIR = os.path.join(SHARED_DIR, 'CXR', 'data')
# DEMO_DATA_DIR = os.path.join(SHARED_DIR, 'CXR', 'data_demo')

# REAL_MIMIC_TRAIN_PATH = os.path.join(DATA_DIR, 'train', 'mimic')
# REAL_CHEXPERT_TRAIN_PATH = os.path.join(DATA_DIR, 'train', 'chexpert')
# REAL_MIMIC_TEST_PATH = os.path.join(DATA_DIR, 'test', 'mimic')
# REAL_CHEXPERT_TEST_PATH = os.path.join(DATA_DIR, 'test', 'chexpert')

# XCAR_TRAIN_CSV_PATH = os.path.join(META_DATA_DIR, 'chexpert_table_train.csv')
# XCAR_TEST_CSV_PATH = os.path.join(META_DATA_DIR, 'chexpert_table_test.csv')
EDEMA_TRAIN_CSV_PATH = os.path.join(CHEXPERT_DATA_DIR, 'chexpert_table_Edema_train.csv')
EDEMA_TEST_CSV_PATH = os.path.join(CHEXPERT_DATA_DIR, 'chexpert_table_Edema_test.csv')
PNEU_TRAIN_CSV_PATH = os.path.join(CHEXPERT_DATA_DIR, 'chexpert_table_Pneumonia_train.csv')
PNEU_TEST_CSV_PATH = os.path.join(CHEXPERT_DATA_DIR, 'chexpert_table_Pneumonia_test.csv')
ATEL_TRAIN_CSV_PATH = os.path.join(CHEXPERT_DATA_DIR, 'chexpert_table_Atelectasis_train.csv')
ATEL_TEST_CSV_PATH = os.path.join(CHEXPERT_DATA_DIR, 'chexpert_table_Atelectasis_test.csv')
LUNGL_TRAIN_CSV_PATH = os.path.join(CHEXPERT_DATA_DIR, 'chexpert_table_LungLesion_train.csv')
LUNGL_TEST_CSV_PATH = os.path.join(CHEXPERT_DATA_DIR, 'chexpert_table_LungLesion_test.csv')
CONS_TRAIN_CSV_PATH = os.path.join(CHEXPERT_DATA_DIR, 'chexpert_table_Consolidation_train.csv')
CONS_TEST_CSV_PATH = os.path.join(CHEXPERT_DATA_DIR, 'chexpert_table_Consolidation_test.csv')


WHITE_TRAIN_CSV_PATH = os.path.join(MIMIC_DATA_DIR, 'mimic_table_White_train.csv')
WHITE_TEST_CSV_PATH = os.path.join(MIMIC_DATA_DIR, 'mimic_table_White_test.csv')
ASIAN_TRAIN_CSV_PATH = os.path.join(MIMIC_DATA_DIR, 'mimic_table_Asian_train.csv')
ASIAN_TEST_CSV_PATH = os.path.join(MIMIC_DATA_DIR, 'mimic_table_Asian_test.csv')