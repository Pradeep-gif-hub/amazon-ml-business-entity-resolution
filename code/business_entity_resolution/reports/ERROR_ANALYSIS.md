# Error Analysis & Model Diagnostics

## 1. Summary of Error Distributions

- Total Evaluated Validation Queries: 19,998
- Exact Predictions: 16,434 (82.18%)
- False Positive Errors: 662 (3.31%)
- False Negative Errors: 2,999 (15.00%)

## 2. Representative False Positive Inspections (False Merges)

False positives occur primarily when different businesses share very common generic brand words in similar localities.

| S1 ID | S1 Name | S1 Address | False Positive Target ID | Reason / Diagnostic |
| --- | --- | --- | --- | --- |
| `S1-302162136` | Commission on Parks and Recreation | OH, Dayton, 4153 Tulip Tree Drive | `S3-272007585` | High lexical similarity on generic terms |
| `S1-709501346` | Tri-State Coalition | 1300 -1400 Sharlo Avenue, Baton Rouge City, LA | `S2-876177042` | High lexical similarity on generic terms |
| `S1-809491632` | Laramee, Josephine, L.C.S.W. | 4734 Kenneth Avenue, Chicago, IL | `S3-893459074` | High lexical similarity on generic terms |
| `S1-93710667` | Dental Smart Specialists | 55240 Greenbank Avenue, Bridgeport, OH | `S2-392726546` | High lexical similarity on generic terms |
| `S1-93322959` | Wenonah L. Egge, P.A., P.C. | 4883 Briar Ridge Court, Gunbarrel, CO | `S3-520524679` | High lexical similarity on generic terms |
| `S1-513160487` | Phelan Jab | 1000 Concordia Drive, Loch Raven, MD | `S3-380751868` | High lexical similarity on generic terms |
| `S1-196137389` | Summit Metro Arcelormittal Inc. | 1152 Quinby Avenue, Wooster, OH | `S3-102727766` | High lexical similarity on generic terms |
| `S1-514385841` | Office of Motor Vehicles | 2052 Lincoln Avenue, St. Albans, WV | `S3-5749154` | High lexical similarity on generic terms |
| `S1-398675570` | Om Business Private Limited | C/O Sangeeta W/O Sri S. Sinha, 301, Parvati Aptt West Lohanipur, Kadamkuan, Patna, Bihar | `S3-502068039` | High lexical similarity on generic terms |
| `S1-334489603` | NC Singh Limited | Bhopal, Madhya Pradesh, Huzur, H.No. 35 Abhinav Homes Phase-Ii Ayodhya By Pass Road | `S2-29049414` | High lexical similarity on generic terms |

## 3. Representative False Negative Inspections (Missed Matches)

False negatives occur primarily when entities have severe name corruptions combined with missing address numbers.

| S1 ID | S1 Name | S1 Address | Missed Target ID | Reason / Diagnostic |
| --- | --- | --- | --- | --- |
| `S1-62190294` | Porter and Stuart Regional LP | 1834 Northpoint Street, WI, City Of Oshkosh | `S2-218407723` | Extreme noise / transliteration discrepancy |
| `S1-668464584` | Timber Tavia | 8475 Rippled Creek Court, Fairfax County, VA | `S2-820607939` | Extreme noise / transliteration discrepancy |
| `S1-920787432` | Ace Avalanche Inc | ME, Benedicta Twp, 200 W Shore Road | `S3-890980750` | Extreme noise / transliteration discrepancy |
| `S1-316907323` | Kingsbury Oncology Center | 25 Wagoner Road, Patriot, OH | `S2-776724671` | Extreme noise / transliteration discrepancy |
| `S1-122567555` | Savera Group | Plot No. 1, Sector - 5, Main Road, Ashok Vihar, Phase - Iii, Gurgaon, Haryana | `S2-677699520` | Extreme noise / transliteration discrepancy |
| `S1-464396946` | Dermatology Care Associates LLC | 603 Smallwood Drive, Durham, NC | `S2-113950117` | Extreme noise / transliteration discrepancy |
| `S1-302162136` | Commission on Parks and Recreation | OH, Dayton, 4153 Tulip Tree Drive | `S3-79477143` | Extreme noise / transliteration discrepancy |
| `S1-245332060` | Quality Desert Otg LLC | MD, 6514 Woodbridge Circle, Catonsville | `S3-663045964` | Extreme noise / transliteration discrepancy |
| `S1-469733686` | Hunger Center, Inc. | 117 Patterson, Unit 47, Youngsville, NC | `S2-765368845` | Extreme noise / transliteration discrepancy |
| `S1-861054` | Dream Infrastructure Private Limited | P No 128, Road No 10 Jubilee Hills, Hyderabad, Telangana | `S3-830328831` | Extreme noise / transliteration discrepancy |
