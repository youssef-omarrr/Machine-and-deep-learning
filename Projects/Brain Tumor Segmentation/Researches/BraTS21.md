## **What is BraTS 2021 (BraTS21)?**

**BraTS** stands for **Brain Tumor Segmentation Challenge**: an annual international competition (part of MICCAI) where researchers develop algorithms to segment brain tumors from MRI scans.

- **BraTS 2021** specifically was the **2021 edition** of this challenge.
- Dataset contains **multi-institutional, preoperative, multi-modal 3D MRI scans** of patients with **gliomas** (both low- and high-grade).
- **All images are skull-stripped** (only the brain remains) and co-registered to the same anatomical space, in **NIfTI** format.
- Each case has **manual expert annotations** on the **tumor subregions** for:
    - **ET** – Enhancing Tumor
    - **NET/NCR** – Non-Enhancing Tumor / Necrotic Core
    - **ED** – Peritumoral Edema
    
- Data split: Training set (with labels), validation set (labels hidden for leaderboard scoring), and test set (for final challenge evaluation).
- Typical size: ~ 1,200 labeled 3D volumes in the training split.

**Reference**:  
Bakas et al., “The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)”, _IEEE Transactions on Medical Imaging_, 2018.  
[https://www.med.upenn.edu/cbica/brats2021/data.html](https://www.med.upenn.edu/cbica/brats2021/data.html)

---
The three labels (**ET**, **NET/NCR**, **ED**) are the **tumor subregions** that BraTS provides as ground truth for segmentation.
Here’s what each means in the context of a brain MRI:

#### **1. ET – Enhancing Tumor**

- The part of the tumor that **takes up contrast agent** (in T1ce scans it appears bright).
- Indicates areas where the **blood–brain barrier is broken** and the tumor is actively growing or highly vascularized.
- Clinically important: Often targeted in surgery and radiation.
#### **2. NET/NCR – Non-Enhancing Tumor / Necrotic Core**

- **Non-Enhancing Tumor (NET)**: Tumor tissue that does **not** enhance with contrast (T1ce) — may represent infiltrative tumor that hasn’t broken the blood–brain barrier.
- **Necrotic Core (NCR)**: Dead tumor tissue in the center due to lack of blood supply — usually appears **very dark** in T1ce.
- In BraTS, these are combined into **one label**, because on *MRI it’s hard to separate them cleanly*.
#### **3. ED – Peritumoral Edema**

- Swelling in brain tissue **around** the tumor.
- Caused by fluid leakage from tumor vessels.
- Appears **bright on T2 and FLAIR** scans.
- Not tumor cells themselves, but clinically relevant because edema increases intracranial pressure.

### How BraTS Labels Are Encoded in the Data

In the ground truth `.nii.gz` label files:

- **Label 0** → Background (normal brain / no abnormality)
- **Label 1** → NET/NCR
- **Label 2** → ED
- **Label 4** → ET  
    _(There is no label 3 — they skip it to avoid confusion with old dataset versions.)_

### Why They Matter in Segmentation

When training, nnU-Net (or your custom model) learns to predict these regions separately.  
For example:

- **Whole Tumor (WT)** = ET + ED + NET/NCR
- **Tumor Core (TC)** = ET + NET/NCR
- **Enhancing Tumor (ET)** = ET only

The challenge evaluates Dice scores for WT, TC, and ET, not just each label individually.


---

## **What are T1, T1ce (a.k.a. T1Gd), T2, and FLAIR?**

These are **different MRI sequences**, same patient, same brain, but imaged in different ways to highlight different tissues or fluids.  
Think of them like different "filters" on the same scene.

### **1. T1-weighted MRI (T1)**

- Anatomical detail: CSF (cerebrospinal fluid) appears dark, white matter appears bright.
- Good for structural information.
- Tumors are often **not well visible** unless contrast agent is used.

### **2. T1ce / T1Gd / T1c** (T1-weighted with contrast enhancement)

- The "ce" means **contrast-enhanced** — usually **Gadolinium** is injected.
- Tumor areas with leaky blood–brain barriers **light up brightly**.
- Essential for identifying **enhancing tumor regions**.

### **3. T2-weighted MRI (T2)**

- Fluid (like CSF or edema) appears **bright**.
- Tumor-associated edema is very visible.
- White matter is darker compared to T1.

### **4. FLAIR (Fluid Attenuated Inversion Recovery)**

- Similar to T2, but **suppresses the signal from free fluid** like CSF in ventricles.
- This makes **edema around the tumor** much more distinct without interference from bright CSF.

---
### **Visual Summary**

|Sequence|CSF signal|Tumor core|Edema visibility|Key use|
|---|---|---|---|---|
|**T1**|Dark|Poor contrast|Low|Anatomy reference|
|**T1ce**|Dark|Bright (enhancing tumor)|Low|Detecting active tumor|
|**T2**|Bright|Variable|Bright|Edema detection|
|**FLAIR**|Dark|Variable|Bright & clear|Edema without CSF interference|

---

If you use MONAI or nnU-Net with BraTS data, you’ll feed **all four modalities as four input channels** (like RGB in natural images, but here it’s T1/T1ce/T2/FLAIR instead of R/G/B).

---
