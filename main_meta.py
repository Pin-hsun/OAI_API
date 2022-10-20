import numpy as np
import pandas as pd
import os
from functools import reduce


def oai_extract_data(path_oai_root, key, ver, var_list=None):
    """Return a dataframe given a specific category of OAI data and the version number
    Args:
        path_oai_root: Path to the root of OAI database

        key: category of OAI files
            CLI: Clinical
                ex: 'AllClinical_SAS/AllClinical00.sas7bdat' for clinical baseline

            ENR: Enrollment
                ex: 'General_SAS/enrollees.sas7bdat', there is no version number

            KXR_SQ: Semi-Quant X-Ray reading
                ex: X-Ray Image Assessments_SAS/Semi-Quant Scoring_SAS/kxr_sq_bu00.sas7bdat' for baseline

            MOAKS: MRI moaks score
                ex: 'MR Image Assessment_SAS/Semi-Quant Scoring_SAS/kmri_sq_moaks_bicl00.sas7bdat' for baseline

            dicom00: path to the dicom files by imaging sequences of baseline dataset (not included in original file)
                ex: (OAI_dicom_path_V00.xlsx')

        ver: version number of time points:
            00: baseline
            01: 12m
            02: 18m (interim, no images)
            03: 24m
            04: 30m (interim, no images)
            05: 36m
            06: 48m
            07: 60m (phone, no images)
            08: 72m
            09: 84m (phone, no images)
            10: 96m
            11: 108m (phone, no images)
            99: outcomes

    Returns:
        x (pandas dataframe):

    """
    #path_dict = {'CLI': 'AllClinical_SAS/allclinical',
    #             'ENR': 'General_SAS/enrollees',
    #             'outcome': 'General_SAS/outcomes99',
    #             'KXR_SQ': 'X-Ray Image Assessments_SAS/Semi-Quant Scoring_SAS/kxr_sq_bu',
    #             'KXR_QJSW_Duryea': 'X-Ray Image Assessments_SAS/Quant JSW_SAS/kxr_qjsw_duryea',
    #             'MOAKS': 'MR Image Assessment_SAS/Semi-Quant Scoring_SAS/kmri_sq_moaks_bicl'}

    if key == 'ENR' and ver != '':
        print('ERROR oai_extract_data: enrollment should not have version number')
        return 0

    x = pd.read_sas(os.path.join(path_oai_root, key + ver + '.sas7bdat'))

    # decode all bytes columns
    for col, dtype in x.dtypes.items():
        if dtype == np.object:  # Only process byte object columns.
            x[col] = x[col].str.decode("utf-8")

    # select variables, if there is $$ sign then replace by version number
    if var_list:
        for i, var in enumerate(var_list):
            if '$$' in var:
                var_list[i] = var_list[i].replace('$$', ver)
        x = x.loc[:, var_list]

    return x


def MOAKS_get_vars(categories, ver):
    moaks_summary = pd.read_excel(os.path.join(os.path.expanduser('~'), 'Dropbox',
                                               'TheSource/OAIDataBase/OAI_Labels/MOAKS/KMRI_SQ_MOAKS_variables_summary.xlsx'))
    moaks_variables = moaks_summary.loc[moaks_summary['CATEGORY'].isin(categories), 'VARIABLE']
    l = list(moaks_variables.values)
    return [x.replace('$$', ver) for x in l]


def merge_multiple_data(data_list, how, on):
    data = data_list[0]
    for i in range(1, len(data_list)):
        data = pd.merge(data, data_list[i], how=how, on=on)
    data = sort_columns(data, on)
    return data


def merge_prjs(y, prjs, keep):
    y['READPRJ'] = pd.Categorical(y['READPRJ'], prjs)
    y = y.sort_values(by=keep + ['READPRJ'])
    # drop duplication after sorted by project number
    y = y.drop_duplicates(subset=['ID', 'SIDE'], keep='first')
    return y


def read_some(filename, path_root, var_list, ver_list, prj_list, keep):
    data = dict()
    for v in ver_list:
        temp = oai_extract_data(path_root, filename, v)
        temp.columns = map(lambda x: str(x).upper(), temp.columns)
        if prj_list is not None:
            temp = merge_prjs(y=temp, prjs=prj_list, keep=keep)
        data[v] = temp[keep + [x.replace('$$', v) for x in var_list]]
    to_merge = [data[v] for v in list(data.keys())]
    df = reduce(lambda left, right: pd.merge(left, right, on=keep, how='left'), to_merge)
    return df


def main():
    oai = dict()
    ver_list = ['00', '01', '03', '05', '06', '08', '10']
    # PATHS
    path_oai_root = os.path.join(os.path.join(os.path.expanduser('~'), 'Dropbox'), 'TheSource/OAIDataBase')

    # ENROLLMENT
    enr = oai_extract_data(path_oai_root, 'General_SAS/enrollees',
                           '', var_list=['ID', 'V00SITE', 'P02SEX', 'V00COHORT'])
    oai['ENR'] = enr

    # X-RAYS readings: KL, JSM, JSL
    XR = read_some(filename='X-Ray Image Assessments_SAS/Semi-Quant Scoring_SAS/kxr_sq_bu',
                   path_root=path_oai_root,
                   var_list=['V$$XRKL', 'V$$XRJSM', 'V$$XRJSL'], ver_list=ver_list,
                   prj_list=['15', '37', '42'], keep=['ID', 'SIDE'])
    oai['XR'] = XR

    # CLINICAL: ID, BMI, AGE, WOMAC disability (V$$WOMADL#), pain (V$$WOMKP@)
    # baseline: age, frequent pain...
    cli00 = read_some(filename='AllClinical_SAS/allclinical',
                      path_root=path_oai_root,
                      var_list=['V00AGE', 'P01KPNREV', 'P01KPNLEV', 'V$$WOMADLR', 'V$$WOMADLL', 'V$$WOMKPR', 'V$$WOMKPL'],
                      ver_list=ver_list[:1],
                      prj_list=None, keep=['ID'])
    # follow-up: bmi, womac.....
    cliXX = read_some(filename='AllClinical_SAS/allclinical',
                      path_root=path_oai_root,
                      var_list=['V$$BMI', 'V$$WOMADLR', 'V$$WOMADLL', 'V$$WOMKPR', 'V$$WOMKPL'],
                      ver_list=ver_list[1:],
                      prj_list=None, keep=['ID'])
    clinical = pd.merge(cli00, cliXX, how='left', on=['ID'])
    oai['CLINICAL'] = clinical

    # MRI: MOAKS
    moaks = []
    for ver in ['00', '01', '03', '05', '06']:
        found = oai_extract_data(path_oai_root, 'MR Image Assessment_SAS/Semi-Quant Scoring_SAS/kmri_sq_moaks_bicl'
                                 , ver=ver, var_list=['ID', 'SIDE', 'READPRJ']
                                                     + MOAKS_get_vars(['Cartilage Morphology', 'BML Size', 'BML #', 'BML (Edema %)', 'Whole Knee Effusion'], ver=ver))

        found['VER'] = ver
        prjs = list(found['READPRJ'].value_counts().keys())
        moaks.append(merge_prjs(found, prjs=prjs, keep=['ID', 'SIDE']))
    moaks = merge_multiple_data(moaks, on=['ID', 'SIDE', 'VER'], how='outer')
    pj = moaks[['READPRJ', 'READPRJ_x', 'READPRJ_x', 'READPRJ_y','READPRJ_y']]
    pj = pj.fillna(method='bfill', axis=1).iloc[:, 0]
    moaks['READPRJ'] = pj
    oai['MOAKS'] = moaks

    # outcomes
    outcome = oai_extract_data(path_oai_root, 'General_SAS/outcomes99','')
    outcome.rename(columns={'id': 'ID'}, inplace=True)
    oai['outcome'] = outcome

    return oai


def sort_columns(x, first):
    left_over = sorted(list(set(x.columns) - set(first)))
    x = x[first + left_over]
    x = x.sort_values(first, ascending=[True] * len(first))
    return x


def copy_left_right(x):
    xL = x.copy()
    xR = x.copy()
    xL['SIDE'] = 'LEFT'
    xR['SIDE'] = 'RIGHT'
    x = pd.concat([xL, xR], 0)
    return x


def ver_to_months(x):
    month = {'00': '00',
             '01': '12',
             '03': '24',
             '05': '36',
             '06': '48',
             '08': '72',
             '10': '96'}
    return month[x]


def load_path_files():
    path = dict()
    for VER in ['00', '01', '03', '05', '06', '08', '10']:
        path[VER] = pd.read_csv('meta/path' + ver_to_months(VER) + 'm.csv')
    return path


def find_mri(x):
    x['folders'] = None
    for i in range(x.shape[0]):
        VER = x.iloc[i]['VER']
        path = path_all[VER]
        ID = x.iloc[i]['ID']
        SIDE = x.iloc[i]['SIDE']
        sequences = x.iloc[i]['sequences']
        found = path.loc[(path['ID'] == int(ID)) & (path['sequences'] == (sequences + SIDE))]['folders']
        if found.shape[0] > 0:
            x['folders'].iloc[i] = found.values[0]
    return x


def left_right_have_mri(x):
    xl = x.loc[(x['SIDE'] == 'RIGHT') & (~x['folders'].isna()), ['VER', 'ID']]
    xr = x.loc[(x['SIDE'] == 'LEFT') & (~x['folders'].isna()), ['VER', 'ID']]
    y = pd.merge(xl, xr, how='inner')
    x = pd.merge(x, y, how='inner', on=['VER', 'ID'])
    return x


def has_moaks(moaks, id, side, ver):
    return (moaks.loc[(moaks['ID']==id) & (moaks['SIDE']==side) & (moaks['VER']==ver)].shape[0]) > 0


if __name__ == '__main__':
    oai = main()
    path_all = load_path_files()

    do_thing = 'X'

    if do_thing == 'ver=00':
        x = []
        threshold = 5
        for VER in ['00']:
            var0 = 'V' + VER + 'WOMKPR'
            var1 = 'V' + VER + 'WOMKPL'
            y = (lambda x: x.loc[((x[var0]-x[var1]).abs() >= threshold)])(oai['CLINICAL']).loc[:, ['ID', var0, var1]]
            y['VER'] = VER
            x.append(y)

        x = pd.DataFrame({'ID': oai['CLINICAL']['ID']})
        x['VER'] = '00'
        x['sequences'] = 'SAG_3D_DESS_'
        x = copy_left_right(x)
        x = find_mri(x)
        x = x.loc[~x['folders'].isna()]
        x = sort_columns(x, ['VER', 'ID', 'SIDE', 'sequences', 'folders'])
        x.to_csv('meta/allver0.csv')

    if do_thing == 'unilateral frequent pain with womac pain difference >= 3':
        # unilateral frequency pain with womac pain difference >= 3 between left and right knees
        y = (lambda x: x.loc[((x['P01KPNREV'] + x['P01KPNLEV']) == 1) &
                             ((x['V00WOMKPR'] - x['V00WOMKPL']).abs() >= 3)])(oai['CLINICAL'])
        using = np.load('meta/subjects_unipain_womac3.npy')
        y = y.loc[y['ID'].isin(using)]
        y = y.loc[:, ['ID', 'P01KPNREV', 'P01KPNLEV', 'V00WOMADLR', 'V00WOMADLL', 'V00WOMKPR', 'V00WOMKPL']]

        y_right = y.loc[:, ['ID', 'P01KPNREV',  'V00WOMADLR', 'V00WOMKPR']]
        y_left = y.loc[:, ['ID', 'P01KPNLEV',  'V00WOMADLL', 'V00WOMKPL']]

        y_right['SIDE'] = 1
        y_left['SIDE'] = 2

        y_right = y_right.rename(columns={'P01KPNREV': 'P01KPN#EV', 'V00WOMADLR': 'V00WOMADL#', 'V00WOMKPR': 'V00WOMKP#'})
        y_left = y_left.rename(columns={'P01KPNLEV': 'P01KPN#EV', 'V00WOMADLL': 'V00WOMADL#', 'V00WOMKPL': 'V00WOMKP#'})

        y = pd.concat([y_right, y_left], 0)

        xr = oai['XR'].loc[oai['XR']['ID'].isin(using)][['ID', 'SIDE', 'V00XRJSM', 'V00XRJSL']]

        y = pd.merge(left=y, right=xr, how='outer', on=['ID', 'SIDE'])

        y = pd.merge(left=y, right=oai['MOAKS'].loc[oai['MOAKS']['VER']=='00'], how='left', on=['ID', 'SIDE'])[list(y.columns) + ['READPRJ']]

        #painful_knee_has_moaks = [has_moaks(oai['MOAKS'], y.iloc[i]['ID'],
        #                                    np.argmax([y.iloc[i]['P01KPNREV'], y.iloc[i]['P01KPNLEV']]) + 1, '00') for i in range(y.shape[0])]

        # tkr > to be a function
        tkr = oai['outcome']
        tkrleft = pd.DataFrame(tkr[['ID', 'V99ELKTLPR']])
        tkrleft['SIDE'] = 2
        tkrleft.rename(columns={'V99ELKTLPR': 'V99E#KTLPR'}, inplace=True)
        tkr = oai['outcome']
        tkrright = pd.DataFrame(tkr[['ID', 'V99ERKTLPR']])
        tkrright['SIDE'] = 1
        tkrright.rename(columns={'V99ERKTLPR': 'V99E#KTLPR'}, inplace=True)
        tkr = pd.concat([tkrright, tkrleft], 0)

        y = pd.merge(left=y, right=tkr, how='left', on=['ID', 'SIDE'])

        y = y.sort_values(by=['ID', 'SIDE'], axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

        y.to_csv('meta/womac3new.csv', index=False)

    if do_thing == 'womac pain difference >= 5 between knees':
        x = []
        threshold = 5
        for VER in ['00', '01', '03', '05', '06', '08', '10']:
            var0 = 'V' + VER + 'WOMKPR'
            var1 = 'V' + VER + 'WOMKPL'
            y = (lambda x: x.loc[((x[var0]-x[var1]).abs() >= threshold)])(oai['CLINICAL']).loc[:, ['ID', var0, var1]]
            y['VER'] = VER
            x.append(y)

        x = merge_multiple_data(x, how='outer', on=['ID', 'VER'])
        x['sequences'] = 'SAG_IW_TSE_'
        x = copy_left_right(x)
        x = find_mri(x)
        x = left_right_have_mri(x)
        x = sort_columns(x, ['VER', 'ID', 'SIDE', 'sequences', 'folders'])

    if do_thing == 'womac pain difference >= 5 between knees, min(womacp) == 0':
        x = []
        threshold = 5
        for VER in ['00', '01', '03', '05', '06', '08', '10']:
            var0 = 'V' + VER + 'WOMKPR'
            var1 = 'V' + VER + 'WOMKPL'
            y = (lambda x: x.loc[((x[var0]-x[var1]).abs() >= threshold) & (x[[var0, var1]].min(1) == 0)])(oai['CLINICAL']).loc[:, ['ID', var0, var1]]
            y['VER'] = VER
            x.append(y)

        x = merge_multiple_data(x, how='outer', on=['ID', 'VER'])
        x['sequences'] = 'SAG_IW_TSE_'
        x = copy_left_right(x)
        x = find_mri(x)
        x = left_right_have_mri(x)
        x = sort_columns(x, ['VER', 'ID', 'SIDE', 'sequences', 'folders'])
        x.to_csv('meta/womac5min0.csv')

    if do_thing == 'single side whole knee effusion':
        x = []
        threshold = 5
        right = oai['MOAKS'].loc[oai['MOAKS']['SIDE'] == 1]
        left = oai['MOAKS'].loc[oai['MOAKS']['SIDE'] == 2]
        for VER in ['00', '01', '03', '05', '06']:
            var = 'V' + VER + 'MEFFWK'

            l1 = left.loc[left[var] > 0, ['ID', 'SIDE', 'VER', var]]
            r0 = right.loc[right[var] == 0, ['ID', 'SIDE', 'VER', var]]
            l1r0 = pd.merge(l1, r0, how='inner', on=['ID', 'VER'])

            l0 = left.loc[left[var] == 0, ['ID', 'SIDE', 'VER', var]]
            r1 = right.loc[right[var] > 0, ['ID', 'SIDE', 'VER', var]]
            l0r1 = pd.merge(l0, r1, how='inner', on=['ID', 'VER'])
            print(pd.concat([l1r0, l0r1]).shape)
            x.append(pd.concat([l1r0, l0r1]))
