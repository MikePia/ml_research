# melanocytic
{False, True}

Contingency matrix axis 0 got better results
{'ARI': 0.6307721451411598, 'NMI': 0.4583613012163206, 
'Purity': AXIS_1
0    0.765306
1    0.950693
dtype: float64, 'Purity0': AXIS_0
0    0.931343
1    0.822532
dtype: float64, 'Contingency': col_0     0    1
row_0           
0       253  825
1      3432  178}

Cluster: 0
Counts: {0: 253, 1: 825}
Max value: 825
Max label: 1 True
76.53% exclusive
Total count: 1078

----------------
Cluster: 1
Counts: {0: 3432, 1: 178}
Max value: 3432
Max label: 0 False
95.07% exclusive
Total count: 3610

----------------
Total Counts for Each Cluster: {0: 3685, 1: 1003, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
Maximum Value and Label for Each Cluster: {0: {'label': 0, 'value': 3432}, 1: {'label': 1, 'value': 825}, 2: {'label': None, 'value': 0}, 3: {'label': None, 'value': 0}, 4: {'label': None, 'value': 0}, 5: {'label': None, 'value': 0}, 6: {'label': None, 'value': 0}, 7: {'label': None, 'value': 0}}


# benign_malignant
{'benign', 'malignant'}

Contingency matrix axis 0 got better results
{'ARI': 0.6146870427007544, 'NMI': 0.4133957364039698, 'Purity': AXIS_1
0    0.965206
1    0.686160
dtype: float64, 'Purity0': AXIS_0
0    0.949002
1    0.765217
dtype: float64, 'Contingency': col_0     0    1
row_0           
0      2996  108
1       161  352}

Evaluation Results: {'ARI': 0.6131410841125133, 'NMI': 0.41177858774121495, 'Purity': row_0
0    0.965206
1    0.684211
dtype: float64}
Cluster: 0
Counts: {0: 2996, 1: 108}
Max value: 2996
Max label: 0 benign
96.52% exclusive
Total count: 3104

----------------
Cluster: 1
Counts: {0: 162, 1: 351}
Max value: 351
Max label: 1 malignant
68.42% exclusive
Total count: 513

----------------
Total Counts for Each Cluster: {0: 3158, 1: 459, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
Maximum Value and Label for Each Cluster: {0: {'label': 0, 'value': 2996}, 1: {'label': 1, 'value': 351}, 2: {'label': None, 'value': 0}, 3: {'label': None, 'value': 0}, 4: {'label': None, 'value': 0}, 5: {'label': None, 'value': 0}, 6: {'label': None, 'value': 0}, 7: {'label': None, 'value': 0}}

# diagnosis
{'nevus', 'squamous cell carcinoma', 'actinic keratosis', 'pigmented benign keratosis', 'dermatofibroma', 'vascular lesion', 'basal cell carcinoma', 'melanoma'}

Contingency matrix axis 0 got better results
{'ARI': 0.6692806058953018, 'NMI': 0.4623288273543443, 'Purity': AXIS_1
0    0.500000
1    0.683794
2    0.451613
3    0.612595
4    0.932923
5    0.663636
6    0.448718
7    0.773333
dtype: float64, 'Purity0': AXIS_0
0    0.708738
1    0.907057
2    0.633136
3    0.748918
4    0.402299
5    0.595745
6    0.852941
7    0.508475
dtype: float64, 'Contingency': col_0    0     1    2    3   4   5   6   7
row_0                                     
0       12     0    2    4  11   1   0  30
1       21    31   10  173   7   3   4   4
2        6    16    0    5   6  28   0   1
3       34   141  321    6  11   4   0   7
4       63  2879  108   19   6   6   4   1
5      365    91   56    9  11   5   2  11
6       11     8    7   12  35   0   0   5
7        3     8    3    3   0   0  58   0}

Cluster: 0
Counts: {0: 12, 2: 2, 3: 4, 4: 11, 5: 1, 7: 30}
Max value: 30
Max label: 7 vascular lesion
50.0% exclusive
Total count: 60

----------------
Cluster: 1
Counts: {0: 21, 1: 31, 2: 10, 3: 173, 4: 7, 5: 3, 6: 4, 7: 4}
Max value: 173
Max label: 3 melanoma
68.38% exclusive
Total count: 253

----------------
Cluster: 2
Counts: {0: 6, 1: 16, 3: 5, 4: 6, 5: 28, 7: 1}
Max value: 28
Max label: 5 pigmented benign keratosis
45.16% exclusive
Total count: 62

----------------
Cluster: 3
Counts: {0: 34, 1: 141, 2: 321, 3: 6, 4: 11, 5: 4, 7: 7}
Max value: 321
Max label: 2 dermatofibroma
61.26% exclusive
Total count: 524

----------------
Cluster: 4
Counts: {0: 63, 1: 2879, 2: 108, 3: 19, 4: 6, 5: 6, 6: 4, 7: 1}
Max value: 2879
Max label: 1 basal cell carcinoma
93.29% exclusive
Total count: 3086

----------------
Cluster: 5
Counts: {0: 365, 1: 91, 2: 56, 3: 9, 4: 11, 5: 5, 6: 2, 7: 11}
Max value: 365
Max label: 0 actinic keratosis
66.36% exclusive
Total count: 550

----------------
Cluster: 6
Counts: {0: 11, 1: 8, 2: 7, 3: 12, 4: 35, 7: 5}
Max value: 35
Max label: 4 nevus
44.87% exclusive
Total count: 78

----------------
Cluster: 7
Counts: {0: 3, 1: 8, 2: 3, 3: 3, 6: 58}
Max value: 58
Max label: 6 squamous cell carcinoma
77.33% exclusive
Total count: 75

----------------
Total Counts for Each Cluster: {0: 515, 1: 3174, 2: 507, 3: 231, 4: 87, 5: 47, 6: 68, 7: 59}
Maximum Value and Label for Each Cluster: {0: {'label': 0, 'value': 365}, 1: {'label': 1, 'value': 2879}, 2: {'label': 2, 'value': 321}, 3: {'label': 3, 'value': 173}, 4: {'label': 4, 'value': 35}, 5: {'label': 5, 'value': 28}, 6: {'label': 6, 'value': 58}, 7: {'label': 7, 'value': 30}}
No max diagnoses were clustered together.







