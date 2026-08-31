"""Generate synthetic cardiac (dataset-0001) training data.

Produces realistic-looking 3-D cardiac CT volumes with anatomically
consistent binary organ masks, then writes all four task flavours:

    cap     ->  M3D_Cap_npy/  (images + text reports)
    vqa     ->  M3D-VQA/      (presence / attribute / count CSVs)
    seg     ->  M3D_Seg_npy/0001/  (per-organ binary masks)
    refseg  ->  M3D_RefSeg_npy/   (free-text Q&A + mask)

Run from the repo root:
    python scripts/generate_cardiac_synthetic_data.py
"""

from __future__ import annotations
import csv, json, random
from pathlib import Path
import numpy as np

SEED = 42
DATA_ROOT = Path(__file__).resolve().parents[1] / "Data" / "data"
VOL_SHAPE = (1, 32, 256, 256)
N_TRAIN, N_VAL, N_TEST = 40, 8, 8

CLASSES = [
    (0,  "myocardium of LV (Myo)",                   "Myo"),
    (1,  "left atrium (LA)",                          "LA"),
    (2,  "left ventricle (LV)",                       "LV"),
    (3,  "right atrium (RA)",                         "RA"),
    (4,  "right ventricle (RV)",                      "RV"),
    (5,  "ascending aorta (AO)",                      "AO"),
    (6,  "pulmonary artery (PA)",                     "PA"),
    (7,  "descending aorta (DO)",                     "DO"),
    (8,  "right coronary artery (RCA)",               "RCA"),
    (9,  "right posterior descending artery (RPDA)",  "RPDA"),
    (10, "left anterior descending artery (LAD)",     "LAD"),
    (11, "first diagonal branch (D1)",                "D1"),
    (12, "second diagonal branch (D2)",               "D2"),
    (13, "left circumflex artery (LCX)",              "LCX"),
    (14, "first obtuse marginal branch (OM1)",        "OM1"),
    (15, "second obtuse marginal branch (OM2)",       "OM2"),
    (16, "left posterior descending artery (LPDA)",   "LPDA"),
    (17, "right posterolateral branch (RPLB)",        "RPLB"),
    (18, "ramus intermedius (RAMUS)",                 "RAMUS"),
    (19, "left posterolateral branch (LPLB)",         "LPLB"),
]

ANATOMY_PRIORS = {
    0:  (0.50, 0.52, 0.50,  0.12, 0.10, 0.10),
    1:  (0.45, 0.42, 0.48,  0.10, 0.09, 0.09),
    2:  (0.52, 0.50, 0.50,  0.11, 0.10, 0.09),
    3:  (0.45, 0.42, 0.54,  0.10, 0.09, 0.09),
    4:  (0.50, 0.48, 0.56,  0.10, 0.10, 0.09),
    5:  (0.35, 0.38, 0.50,  0.16, 0.05, 0.05),
    6:  (0.35, 0.40, 0.55,  0.14, 0.05, 0.05),
    7:  (0.50, 0.58, 0.50,  0.22, 0.04, 0.04),
    8:  (0.50, 0.50, 0.58,  0.12, 0.03, 0.03),
    9:  (0.58, 0.55, 0.58,  0.06, 0.03, 0.03),
    10: (0.50, 0.55, 0.45,  0.14, 0.03, 0.03),
    11: (0.48, 0.53, 0.44,  0.06, 0.03, 0.03),
    12: (0.52, 0.54, 0.44,  0.06, 0.03, 0.03),
    13: (0.50, 0.48, 0.44,  0.12, 0.03, 0.03),
    14: (0.48, 0.46, 0.43,  0.05, 0.03, 0.03),
    15: (0.53, 0.47, 0.43,  0.05, 0.03, 0.03),
    16: (0.58, 0.55, 0.45,  0.06, 0.03, 0.03),
    17: (0.55, 0.52, 0.58,  0.05, 0.03, 0.03),
    18: (0.48, 0.49, 0.46,  0.04, 0.03, 0.03),
    19: (0.55, 0.50, 0.44,  0.05, 0.03, 0.03),
}

CAPTION_TEMPLATES = [
    ("Cardiac CT demonstrates normal morphology of the {chambers}. "
     "The {vessel} is within normal limits. "
     "Coronary arteries including {coronaries} are patent without significant stenosis. "
     "No pericardial effusion is identified."),
    ("CT angiography of the heart reveals normal size and function of the {chambers}. "
     "The {vessel} shows no aneurysmal dilatation. "
     "The {coronaries} are well-visualised and appear normal. "
     "No evidence of coronary artery disease."),
    ("Cardiac-gated CT: The four cardiac chambers are normal in size. "
     "The myocardium of the left ventricle (Myo) shows homogeneous attenuation. "
     "The {vessel} and {vessel2} are unremarkable. "
     "Coronary calcification is absent. {coronaries} appear unremarkable."),
    ("CT cardiac imaging demonstrates normal biventricular size and function. "
     "The {chambers} and great vessels are unremarkable. "
     "The {coronaries} show no significant disease. "
     "Impression: Normal cardiac CT."),
]
CHAMBERS = [
    "left ventricle (LV), right ventricle (RV), left atrium (LA), and right atrium (RA)",
    "all four cardiac chambers",
    "left and right ventricles and atria",
]
VESSELS = ["ascending aorta (AO)", "descending aorta (DO)", "pulmonary artery (PA)"]
CORONARIES = [
    "LAD, LCX, and RCA",
    "left anterior descending artery (LAD), left circumflex artery (LCX), and right coronary artery (RCA)",
    "LAD with diagonal branches (D1, D2), LCX with obtuse marginals (OM1, OM2), and RCA",
]

VQA_TRAIN_POOL = [
    ("Is the left ventricle (LV) visible in this cardiac CT?",
     "Yes","Yes","No","Partially","Cannot determine","A","presence"),
    ("Is the right ventricle (RV) present in this scan?",
     "Yes","Yes","No","Partially","Cannot determine","A","presence"),
    ("Can the ascending aorta (AO) be identified?",
     "Yes","Yes","No","Partially","Cannot determine","A","presence"),
    ("Is the left anterior descending artery (LAD) visualised?",
     "Yes","Yes","No","Partially","Cannot determine","A","presence"),
    ("Is the myocardium of the left ventricle (Myo) intact?",
     "Yes","Yes","No","Partially","Cannot determine","A","presence"),
    ("Is the pulmonary artery (PA) present in this cardiac CT?",
     "Yes","Yes","No","Partially","Cannot determine","A","presence"),
    ("Is the right coronary artery (RCA) visible?",
     "Yes","Yes","No","Partially","Cannot determine","A","presence"),
    ("Is the left circumflex artery (LCX) identifiable?",
     "Yes","Yes","No","Partially","Cannot determine","A","presence"),
    ("Which chamber pumps oxygenated blood to the systemic circulation?",
     "Left ventricle (LV)","Left ventricle (LV)","Right ventricle (RV)","Left atrium (LA)","Right atrium (RA)","A","attribute"),
    ("Which vessel carries deoxygenated blood from the right ventricle?",
     "Pulmonary artery (PA)","Pulmonary artery (PA)","Ascending aorta (AO)","Descending aorta (DO)","Left coronary artery","A","attribute"),
    ("What is the abbreviation for the left anterior descending artery?",
     "LAD","LAD","LCX","RCA","RPDA","A","attribute"),
    ("Which coronary artery is known as the widow-maker?",
     "Left anterior descending artery (LAD)","Left anterior descending artery (LAD)","Right coronary artery (RCA)","Left circumflex artery (LCX)","Ramus intermedius (RAMUS)","A","attribute"),
    ("What does the abbreviation Myo refer to?",
     "Myocardium of LV (Myo)","Myocardium of LV (Myo)","Left ventricle (LV)","Left atrium (LA)","Right ventricle (RV)","A","attribute"),
    ("Which artery supplies the posterior septum in right-dominant anatomy?",
     "Right posterior descending artery (RPDA)","Right posterior descending artery (RPDA)","Left posterior descending artery (LPDA)","Right coronary artery (RCA)","Left circumflex artery (LCX)","A","attribute"),
    ("What is the primary function of the left atrium (LA)?",
     "Receive oxygenated blood from pulmonary veins","Receive oxygenated blood from pulmonary veins","Pump blood to the body","Pump blood to the lungs","Receive venous blood from the body","A","attribute"),
    ("Which structure gives rise to the first diagonal branch (D1)?",
     "Left anterior descending artery (LAD)","Left anterior descending artery (LAD)","Left circumflex artery (LCX)","Right coronary artery (RCA)","Left main coronary artery","A","attribute"),
    ("How many cardiac chambers are typically present in a normal heart?",
     "4","4","2","3","6","A","count"),
    ("How many major coronary vessels are labelled in this dataset?",
     "3 (LAD, LCX, RCA)","3 (LAD, LCX, RCA)","2","4","5","A","count"),
    ("How many obtuse marginal branches are annotated in this dataset?",
     "2 (OM1 and OM2)","2 (OM1 and OM2)","1","3","4","A","count"),
    ("How many diagonal branches are annotated for the LAD?",
     "2 (D1 and D2)","2 (D1 and D2)","1","3","4","A","count"),
]

VQA_YN_POOL = [
    ("Is the left ventricle (LV) the most muscular cardiac chamber?",
     "Yes","Yes","No","Maybe","Cannot determine","A","yes_no"),
    ("Does the pulmonary artery (PA) carry oxygenated blood?",
     "No","No","Yes","Partially","Cannot determine","B","yes_no"),
    ("Is the ascending aorta (AO) connected to the left ventricle?",
     "Yes","Yes","No","Partially","Cannot determine","A","yes_no"),
    ("Is the right coronary artery (RCA) part of the left coronary system?",
     "No","No","Yes","Partially","Cannot determine","B","yes_no"),
    ("Is the myocardium of the LV (Myo) responsible for cardiac output?",
     "Yes","Yes","No","Partially","Cannot determine","A","yes_no"),
    ("Does the left circumflex artery (LCX) arise from the right coronary ostium?",
     "No","No","Yes","Sometimes","Cannot determine","B","yes_no"),
    ("Is the descending aorta (DO) located posterior to the heart?",
     "Yes","Yes","No","Partially","Cannot determine","A","yes_no"),
    ("Is the ramus intermedius (RAMUS) always present in every patient?",
     "No","No","Yes","Sometimes","Cannot determine","B","yes_no"),
]

REFSEG_QA = [
    (0,  "Segment the myocardium of the left ventricle (Myo).","The myocardium of LV (Myo) is segmented. [SEG]"),
    (1,  "Please delineate the left atrium (LA) in this cardiac CT.","The left atrium (LA) is segmented. [SEG]"),
    (2,  "Identify and segment the left ventricle (LV).","The left ventricle (LV) is segmented. [SEG]"),
    (3,  "Please segment the right atrium (RA) in this scan.","The right atrium (RA) is segmented. [SEG]"),
    (4,  "Delineate the right ventricle (RV) in this cardiac CT.","The right ventricle (RV) is segmented. [SEG]"),
    (5,  "Segment the ascending aorta (AO).","The ascending aorta (AO) is segmented. [SEG]"),
    (6,  "Please identify the pulmonary artery (PA) and generate its mask.","The pulmonary artery (PA) is segmented. [SEG]"),
    (7,  "Segment the descending aorta (DO) in this CT.","The descending aorta (DO) is segmented. [SEG]"),
    (8,  "Identify and segment the right coronary artery (RCA).","The right coronary artery (RCA) is segmented. [SEG]"),
    (9,  "Please segment the right posterior descending artery (RPDA).","The right posterior descending artery (RPDA) is segmented. [SEG]"),
    (10, "Segment the left anterior descending artery (LAD), the widow-maker.","The left anterior descending artery (LAD) is segmented. [SEG]"),
    (11, "Delineate the first diagonal branch (D1) of the LAD.","The first diagonal branch (D1) is segmented. [SEG]"),
    (12, "Please segment the second diagonal branch (D2).","The second diagonal branch (D2) is segmented. [SEG]"),
    (13, "Identify and segment the left circumflex artery (LCX).","The left circumflex artery (LCX) is segmented. [SEG]"),
    (14, "Segment the first obtuse marginal branch (OM1) of the LCX.","The first obtuse marginal branch (OM1) is segmented. [SEG]"),
    (15, "Please delineate the second obtuse marginal branch (OM2).","The second obtuse marginal branch (OM2) is segmented. [SEG]"),
    (16, "Segment the left posterior descending artery (LPDA).","The left posterior descending artery (LPDA) is segmented. [SEG]"),
    (17, "Identify the right posterolateral branch (RPLB) and segment it.","The right posterolateral branch (RPLB) is segmented. [SEG]"),
    (18, "Please segment the ramus intermedius (RAMUS).","The ramus intermedius (RAMUS) is segmented. [SEG]"),
    (19, "Delineate the left posterolateral branch (LPLB) in this cardiac CT.","The left posterolateral branch (LPLB) is segmented. [SEG]"),
]


def _ellipsoid_mask(rng, class_id):
    _, D, H, W = VOL_SHAPE
    dc, hc, wc, dr, hr, wr = ANATOMY_PRIORS[class_id]
    j = 0.10
    dc += rng.uniform(-j*dr, j*dr)
    hc += rng.uniform(-j*hr, j*hr)
    wc += rng.uniform(-j*wr, j*wr)
    dv = int(np.clip(dc*D, 1, D-2))
    hv = int(np.clip(hc*H, 1, H-2))
    wv = int(np.clip(wc*W, 1, W-2))
    rdv = max(1, int(dr*D)); rhv = max(1, int(hr*H)); rwv = max(1, int(wr*W))
    zz,yy,xx = np.ogrid[:D,:H,:W]
    return (((zz-dv)/rdv)**2 + ((yy-hv)/rhv)**2 + ((xx-wv)/rwv)**2) <= 1.0


def gen_vol_and_mask(rng, class_id):
    _, D, H, W = VOL_SHAPE
    vol = rng.uniform(0.02, 0.08, (D,H,W)).astype(np.float32)
    zz,yy,xx = np.ogrid[:D,:H,:W]
    body = (((zz/D-0.5)/0.40)**2 + ((yy/H-0.5)/0.38)**2 + ((xx/W-0.5)/0.40)**2) <= 1.0
    vol[body] = rng.uniform(0.30, 0.45, body.sum()).astype(np.float32)
    for cid,_,_ in CLASSES:
        m = _ellipsoid_mask(rng, cid)
        vol[m] = np.float32(rng.uniform(0.75, 0.95))
    vol = np.clip(vol, 0.0, 1.0)[np.newaxis]
    mask = _ellipsoid_mask(rng, class_id)[np.newaxis]
    return vol, mask


def _append_csv(path, fieldnames, new_rows):
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    seen = {r.get("Image Path") or r.get("Image") for r in existing}
    for r in new_rows:
        k = r.get("Image Path") or r.get("Image")
        if k not in seen:
            existing.append(r); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(existing)


def write_cap(rng):
    print("\n=== CAP ===")
    img_dir = DATA_ROOT/"M3D_Cap_npy"/"images"; img_dir.mkdir(parents=True,exist_ok=True)
    txt_dir = DATA_ROOT/"M3D_Cap_npy"/"texts";  txt_dir.mkdir(parents=True,exist_ok=True)
    jpath   = DATA_ROOT/"M3D_Cap_npy"/"M3D_Cap.json"
    manifest = json.loads(jpath.read_text()) if jpath.exists() else {}
    for split, n in [("train",N_TRAIN),("validation",N_VAL),("test",N_TEST)]:
        manifest.setdefault(split,[])
        seen = {e["image"] for e in manifest[split]}
        for i in range(n):
            img_f = f"cardiac_{i:03d}_{split}.npy"
            txt_f = f"cardiac_{i:03d}_{split}.txt"
            key   = f"M3D_Cap_npy/images/{img_f}"
            if key in seen: continue
            vol,_ = gen_vol_and_mask(rng, 0)
            np.save(img_dir/img_f, vol)
            tmpl = rng.choice(CAPTION_TEMPLATES)
            text = tmpl.format(chambers=rng.choice(CHAMBERS), vessel=rng.choice(VESSELS),
                               vessel2=rng.choice(VESSELS), coronaries=rng.choice(CORONARIES))
            (txt_dir/txt_f).write_text(text, encoding="utf-8")
            manifest[split].append({"image":key,"text":f"M3D_Cap_npy/texts/{txt_f}"})
            seen.add(key)
            print(f"  [{split}] {img_f}")
    jpath.write_text(json.dumps(manifest,indent=2), encoding="utf-8")


def write_vqa(rng):
    print("\n=== VQA ===")
    img_dir = DATA_ROOT/"M3D-VQA"/"images"; img_dir.mkdir(parents=True,exist_ok=True)
    fn = ["Image Path","Question","Answer","Choice A","Choice B","Choice C","Choice D","Answer Choice","Question Type"]
    for split, n, csv_name, yn_name in [
        ("train", N_TRAIN, "M3D_VQA_train.csv", "M3D_VQA_yn_train.csv"),
        ("val",   N_VAL,   "M3D_VQA_val.csv",   None),
        ("test",  N_TEST,  "M3D_VQA_test.csv",  None),
    ]:
        rows=[]; yn_rows=[]
        for i in range(n):
            img_f = f"cardiac_{i:03d}_{split}.npy"
            vol,_ = gen_vol_and_mask(rng, 0)
            np.save(img_dir/img_f, vol)
            q = VQA_TRAIN_POOL[i % len(VQA_TRAIN_POOL)]
            rows.append({"Image Path":f"M3D-VQA/images/{img_f}","Question":q[0],"Answer":q[1],
                         "Choice A":q[2],"Choice B":q[3],"Choice C":q[4],"Choice D":q[5],
                         "Answer Choice":q[6],"Question Type":q[7]})
            if yn_name:
                yq = VQA_YN_POOL[i % len(VQA_YN_POOL)]
                yn_rows.append({"Image Path":f"M3D-VQA/images/{img_f}","Question":yq[0],"Answer":yq[1],
                                "Choice A":yq[2],"Choice B":yq[3],"Choice C":yq[4],"Choice D":yq[5],
                                "Answer Choice":yq[6],"Question Type":yq[7]})
            print(f"  [{split}] {img_f}")
        _append_csv(DATA_ROOT/"M3D-VQA"/csv_name, fn, rows)
        if yn_name and yn_rows:
            _append_csv(DATA_ROOT/"M3D-VQA"/yn_name, fn, yn_rows)


def write_seg(rng):
    print("\n=== SEG ===")
    base = DATA_ROOT/"M3D_Seg_npy"/"0001"
    img_tr = base/"imagesTr"; img_tr.mkdir(parents=True,exist_ok=True)
    img_ts = base/"imagesTs"; img_ts.mkdir(parents=True,exist_ok=True)
    lbl_tr = base/"labelsTr"; lbl_tr.mkdir(parents=True,exist_ok=True)
    lbl_ts = base/"labelsTs"; lbl_ts.mkdir(parents=True,exist_ok=True)
    jpath  = base/"0001.json"
    existing = json.loads(jpath.read_text()) if jpath.exists() else {}
    existing.setdefault("train",[]); existing.setdefault("test",[])

    for split, n, img_root, lbl_root, key in [
        ("train",N_TRAIN,img_tr,lbl_tr,"train"),
        ("test", N_TEST, img_ts,lbl_ts,"test"),
    ]:
        seen = {e["image"] for e in existing[key]}
        for i in range(n):
            img_f = f"cardiac_{i:03d}_{split}.npy"
            img_rel = f"0001/{img_root.name}/{img_f}"
            if img_rel in seen: continue
            vol,_ = gen_vol_and_mask(rng, 0)
            np.save(img_root/img_f, vol)
            for cid,_,_ in CLASSES:
                _,mask = gen_vol_and_mask(rng, cid)
                lbl_f = f"cardiac_{i:03d}_{split}_{cid}.npy"
                np.save(lbl_root/lbl_f, mask)
                existing[key].append({"image":img_rel,"label":f"0001/{lbl_root.name}/{lbl_f}"})
            seen.add(img_rel)
            print(f"  [{split}] {img_f} x {len(CLASSES)} classes")
    jpath.write_text(json.dumps(existing,indent=2), encoding="utf-8")


def write_refseg(rng):
    print("\n=== REFSEG ===")
    img_dir  = DATA_ROOT/"M3D_RefSeg_npy"/"images"; img_dir.mkdir(parents=True,exist_ok=True)
    mask_dir = DATA_ROOT/"M3D_RefSeg_npy"/"masks";  mask_dir.mkdir(parents=True,exist_ok=True)
    fn = ["Image","Mask","Mask_ID","Question","Answer"]
    qa = list(REFSEG_QA)
    rng.shuffle(qa)
    for split, n, csv_name in [
        ("train", N_TRAIN, "M3D_RefSeg.csv"),
        ("test",  N_TEST,  "M3D_RefSeg_test.csv"),
    ]:
        rows=[]
        for i in range(n):
            cid, question, answer = qa[i % len(qa)]
            img_f  = f"cardiac_refseg_{i:03d}_{split}.npy"
            mask_f = f"cardiac_refseg_{i:03d}_{split}_{cid}.npy"
            vol,mask = gen_vol_and_mask(rng, cid)
            np.save(img_dir/img_f, vol)
            np.save(mask_dir/mask_f, mask)
            rows.append({"Image":f"M3D_RefSeg_npy/images/{img_f}",
                         "Mask":f"M3D_RefSeg_npy/masks/{mask_f}",
                         "Mask_ID":cid,"Question":question,"Answer":answer})
            print(f"  [{split}] {img_f}  cls={cid}")
        _append_csv(DATA_ROOT/"M3D_RefSeg_npy"/csv_name, fn, rows)


def main():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    print(f"Data root : {DATA_ROOT}")
    print(f"Cases: {N_TRAIN} train / {N_VAL} val / {N_TEST} test")
    write_cap(rng)
    write_vqa(rng)
    write_seg(rng)
    write_refseg(rng)
    print("\n✅ Done.")
    print(f"  Cap    : {N_TRAIN}+{N_VAL}+{N_TEST}")
    print(f"  VQA    : {N_TRAIN}+{N_VAL}+{N_TEST}")
    print(f"  Seg    : {N_TRAIN}x{len(CLASSES)} train + {N_TEST}x{len(CLASSES)} test = {(N_TRAIN+N_TEST)*len(CLASSES)} records")
    print(f"  RefSeg : {N_TRAIN} train + {N_TEST} test")

if __name__ == "__main__":
    main()
