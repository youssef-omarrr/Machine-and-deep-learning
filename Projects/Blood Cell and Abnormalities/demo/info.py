class_names = ['Band Neutrophil', 'Basophil', 'Eosinophil', 'Erythroblast', 
            'Immature Granulocyte', 'Lymphocyte', 'Metamyelocytes', 'Monocyte', 
            'Myeloblast', 'Myelocytes', 'Neutrophil', 'Platelets', 'Promyelocytes', 'Segmented Neutrophil']

diagnosis_map = {
    "Band Neutrophil": (
        "Band neutrophils are immature neutrophils released during an active immune response. "
        "Their presence in high numbers (a 'left shift') may indicate **acute bacterial infection**, sepsis, or significant inflammation. "
        "They can also appear during bone marrow stimulation or recovery after chemotherapy or bone marrow suppression."
    ),
    "Basophil": (
        "Basophils are rare white blood cells involved in inflammatory and allergic reactions. "
        "Elevated basophils may suggest **chronic allergic conditions** (e.g., asthma, atopic dermatitis), **chronic myelogenous leukemia (CML)**, or **hypothyroidism**. "
        "They contain histamine and heparin, playing a role in anaphylaxis and hypersensitivity responses."
    ),
    "Eosinophil": (
        "Eosinophils are primarily involved in fighting **parasitic infections** and mediating **allergic responses**. "
        "Elevated eosinophils (eosinophilia) may point to conditions such as **asthma**, **eczema**, **hay fever**, **drug hypersensitivity**, or **parasitic infections** like schistosomiasis. "
        "Marked eosinophilia may also be seen in certain malignancies or autoimmune diseases."
    ),
    "Erythroblast": (
        "Erythroblasts are immature red blood cell precursors normally confined to bone marrow. "
        "Their appearance in peripheral blood (erythroblastemia) can be associated with **severe anemia**, **bone marrow stress**, **hypoxia**, or **leukemias** like **acute erythroid leukemia**. "
        "May also appear after trauma or in myelophthisic processes."
    ),
    "Immature Granulocyte": (
        "Immature granulocytes (IGs) include promyelocytes, myelocytes, and metamyelocytes. "
        "Their presence in peripheral blood may signal **severe infection**, **sepsis**, **acute inflammation**, or **bone marrow infiltration** (e.g., leukemia). "
        "They typically reflect early release of granulocytes before full maturation, indicating marrow stress or dysregulation."
    ),
    "Lymphocyte": (
        "Lymphocytes are central to **adaptive immunity**, including T-cells, B-cells, and NK cells. "
        "Elevated lymphocytes (lymphocytosis) can suggest **viral infections** (e.g., Epstein-Barr virus, CMV), **chronic lymphocytic leukemia (CLL)**, or **autoimmune diseases**. "
        "Low lymphocyte counts (lymphopenia) may indicate **HIV**, **malnutrition**, or **immunodeficiency disorders**."
    ),
    "Metamyelocytes": (
        "Metamyelocytes are precursors to neutrophils and should typically remain in the bone marrow. "
        "Their presence in blood can indicate **acute bacterial infection**, **myeloproliferative disorders**, or **stress hematopoiesis**. "
        "May also be seen in response to significant blood loss, hemolysis, or marrow recovery."
    ),
    "Monocyte": (
        "Monocytes are phagocytic cells that mature into macrophages and dendritic cells in tissues. "
        "Elevated monocyte counts (monocytosis) are often linked to **chronic infections** (e.g., tuberculosis, endocarditis), **inflammatory diseases** (e.g., IBD), or **hematologic malignancies** (e.g., monocytic leukemia)."
    ),
    "Myeloblast": (
        "Myeloblasts are early-stage precursors of granulocytes. Their appearance in peripheral blood is **abnormal** and often indicative of **acute myeloid leukemia (AML)**. "
        "A high percentage of myeloblasts (>20% in marrow or blood) is a key diagnostic criterion for AML. "
        "Urgent clinical evaluation is warranted if myeloblasts are detected."
    ),
    "Myelocytes": (
        "Myelocytes are intermediate precursors in the granulocytic lineage. "
        "Their presence outside the marrow may signal **severe infection**, **bone marrow disorders**, or **leukemoid reactions**. "
        "They may also appear in **chronic myeloid leukemia (CML)** or during marrow regeneration."
    ),
    "Neutrophil": (
        "Neutrophils are first responders to **bacterial infections** and play a major role in inflammation. "
        "Elevated neutrophils (neutrophilia) are commonly seen in **acute infections**, **trauma**, **stress**, or **corticosteroid use**. "
        "Reduced neutrophils (neutropenia) may occur in **aplastic anemia**, **chemotherapy**, or certain viral infections."
    ),
    "Platelets": (
        "Platelets are crucial for **blood clotting** and wound healing. "
        "Thrombocytosis (high platelets) may occur in **inflammatory states**, **myeloproliferative disorders**, or **iron deficiency anemia**. "
        "Thrombocytopenia (low platelets) can be due to **immune thrombocytopenic purpura (ITP)**, **leukemia**, **DIC**, or **drug-induced suppression**."
    ),
    "Promyelocytes": (
        "Promyelocytes are early precursors to granulocytes. Their presence in peripheral blood is **abnormal** and may indicate **acute promyelocytic leukemia (APL)**—a subtype of AML requiring urgent treatment. "
        "They may also appear during severe infection or marrow stress but warrant further investigation when seen in blood."
    ),
    "Segmented Neutrophil": (
        "Segmented neutrophils are mature neutrophils, central to the innate immune response. "
        "Elevated levels typically reflect **acute bacterial infections**, **stress**, or **inflammatory diseases** (e.g., rheumatoid arthritis). "
        "A 'right shift' (hypersegmented neutrophils) can be seen in **megaloblastic anemia** due to B12 or folate deficiency."
    )
}

prefix_to_label = {
    "BA": "Basophil",
    "BNE": "Band Neutrophil",
    "EO": "Eosinophil",
    "ERB": "Erythroblast",
    "IG": "Immature Granulocyte",
    "LY": "Lymphocyte",
    "MMY": "Metamyelocyte",
    "MO": "Monocyte",
    "MY": "Myelocyte",
    "MYO": "Myeloblast",
    "NEUTROPHIL": "Neutrophil",
    "NGS": "Segmented Neutrophil",
    "PLATELET": "Platelet",
    "PMY": "Promyelocyte"
}
