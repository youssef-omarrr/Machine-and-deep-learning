## White blood cells
- Basophil 
- Eosinophil 
- Lymphocyte 
- Monocyte 
- Neutrophil -> General term for **any neutrophil**, including immature and mature forms
	- Segmented Neutrophil (SNE) -> **Mature** *neutrophil* (segmented nucleus; normal WBC)
	- Neutrophilic Granulocyte Segmented (NGS) -> same as above just a different name
---
- Band Neutrophil (BNE) -> **Immature** *neutrophil*, non-segmented nucleus
- Myeloblast -> **Immature** *WBC*
- Immature Granulocyte (ig) -> **Generic** term for *immature granulocytes*:
	- Metamyelocytes (MMY) -> **Late-stage** immature granulocyte
	- Myelocytes (MY) -> **Earlier-stage** immature granulocyte
	- Promyelocytes (PMY) -> **Even earlier-stage** granulocyte precursor
---
## Red blood cells
- Erythroblast -> **Immature** Red Blood Cell (RBC precursor)
## Platelets

---
## For small data
- We will use **duplication + augmentation** instead of debiasing-VAE.

	| Class Name           | Count |
	| -------------------- | ----- |
	| Immature Granulocyte | 151   |
	| Promyelocytes        | 592   |
	| Myeloblast           | 1,000 |
	| Metamyelocytes       | 1,015 |
	| Myelocytes           | 1,137 |
	| Erythroblast         | 1,551 |
	| Band Neutrophil      | 1,634 |
	| Basophil             | 1,653 |
	| Platlets             | 2,348 |
	| Segmented Neutrophil | 2,646 |
	| Monocyte             | 5,046 |
	| Neutrophil           | 6,779 |
	| Eosinophil           | 7,141 |
	| Lymphocyte           | 8,685 |

- Target 4,500 samples per class
- Duplicate + augment (with TrivialAugmentWide) only for classes below that
- Save augmented images back to their original class folder with new unique filenames