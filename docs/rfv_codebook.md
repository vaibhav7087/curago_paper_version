# RFV codebook — model features mapped to official NCHS labels

Source: CDC NHAMCS 2019 ED SAS format file
`https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/sas/ed19for.txt`
(`VALUE RFVF` block, 816 labels; retrieved 2026-10-07). Labels below are verbatim,
including the `...` truncations, which are in the official SAS format text itself.

These are the 20 most frequent `RFV1` (reason-for-visit) codes used as dummy
features in `ds/train_triage.py` (`rfv_*` in `triage_features.json`).

| Feature | NCHS label |
|---|---|
| rfv_10100 | Fever |
| rfv_10501 | Chest pain |
| rfv_10552 | Side pain, flank pain |
| rfv_11650 | Oth symptoms/problems relat to psycho... |
| rfv_12100 | Headache, pain in head |
| rfv_12250 | Vertigo - dizziness |
| rfv_13551 | Earache, pain |
| rfv_14150 | Shortness of breath |
| rfv_14400 | Cough |
| rfv_14551 | Throat soreness |
| rfv_15250 | Nausea |
| rfv_15300 | Vomiting |
| rfv_15451 | Abdominal pain, cramps, spasms, NOS |
| rfv_18600 | Skin rash |
| rfv_19051 | Back pain, ache, soreness, discomfort |
| rfv_19201 | Leg pain, ache, soreness, discomfort |
| rfv_19251 | Knee pain, ache, soreness, discomfort |
| rfv_52250 | Laceration/cut of upper extremity |
| rfv_55050 | Injury, other and unspecified of head... |
| rfv_58100 | Accident, NOS |

Top-5 RFV features by model gain importance (as reported in the
`feature_contributions` field of `predict_severity`): psych-related symptoms
(rfv_11650), chest pain, shortness of breath, back pain, skin rash. For
direction, read the `shap_summary.png` beeswarm (red = code present): e.g. the
rfv_11650 mass sits on the High-severity side while skin-rash presence leans
Low — both clinically legible.
