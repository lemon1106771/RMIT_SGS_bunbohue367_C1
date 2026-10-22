# 5-fold CV fine-tune of one competition model with a fresh 3-class head. Metrics only, no submission.
# MODEL_KEY picks the attached model ("deberta" or "wolf"). Train logic follows the team notebook's train.py.
import glob, json, os, re, subprocess, sys, time
import numpy as np, pandas as pd

MODEL_KEY = os.environ.get("MODEL_KEY", "deberta")
EPOCHS = int(os.environ.get("EPOCHS", 3))
MAX_LEN = int(os.environ.get("MAX_LEN", 384))
OUT = os.environ.get("OUT", "/kaggle/working")
INP = os.environ.get("INP", "/kaggle/input")  # root holding train.csv + model dirs
FOLDS = os.environ.get("FOLDS", "0,1,2,3,4")
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
L2I = {l: i for i, l in enumerate(LABELS)}

def find(name):
    return sorted(glob.glob(f"{INP}/**/{name}", recursive=True))[0]

def norm(s):
    return re.sub(r"\W+", " ", str(s).lower()).strip()

def model_dir():
    c = [os.path.dirname(p) for p in glob.glob(f"{INP}/**/model.safetensors", recursive=True)
         if MODEL_KEY in p.lower() and "onnx" not in p.lower()]
    if not c:
        c = [os.path.dirname(p) for p in glob.glob(f"{INP}/**/config.json", recursive=True)
             if MODEL_KEY in p.lower() and "onnx" not in p.lower()]
    assert c, "model not attached: " + MODEL_KEY
    return min(c, key=len)

# ---------------- worker: train given folds on one GPU ----------------
def worker(folds):
    import gc, random, torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    BS = int(os.environ.get("BS", 16)); HEAD, LR, HEAD_LR, LLRD = 128, 2e-5, 1e-4, 0.9
    dev = "cuda"
    MD = model_dir(); print("MODEL_DIR", MD, flush=True)
    tok = AutoTokenizer.from_pretrained(MD)
    train = pd.read_csv(find("train.csv"))
    fold_of = pd.read_csv(f"{OUT}/folds.csv").set_index("id").loc[train.id, "fold"].values
    y_all = train.label.map(L2I).values

    def encode(texts):  # head+tail truncation, raw text
        raw = tok([str(t) for t in texts], add_special_tokens=False, truncation=False)["input_ids"]
        sp = tok("", add_special_tokens=True)["input_ids"]  # e.g. [CLS, SEP]; build_inputs_* is gone in transformers v5
        pre, suf = sp[:1], sp[1:]
        L = MAX_LEN - len(sp); head = min(HEAD, L // 2); out = []
        for ids in raw:
            if len(ids) > L: ids = ids[:head] + ids[len(ids) - (L - head):]
            out.append(pre + ids + suf)
        return out
    seqs_all = encode(train.text.values)

    def collate(s):
        m = max(len(x) for x in s)
        ii = torch.full((len(s), m), tok.pad_token_id, dtype=torch.long); am = torch.zeros_like(ii)
        for j, x in enumerate(s): ii[j, :len(x)] = torch.tensor(x); am[j, :len(x)] = 1
        return ii.to(dev), am.to(dev)

    @torch.no_grad()
    def predict(model, seqs):
        model.eval(); order = np.argsort([len(s) for s in seqs]); out = np.zeros((len(seqs), 3), np.float32)
        for b in range(0, len(order), 64):
            ix = order[b:b + 64]; ii, am = collate([seqs[i] for i in ix])
            with torch.autocast("cuda", dtype=torch.float16):
                out[ix] = model(input_ids=ii, attention_mask=am).logits.float().cpu().numpy()
        return out

    def groups(model):
        nl = model.config.num_hidden_layers; g = []
        for n, p in model.named_parameters():
            m = re.search(r"\.layers?\.(\d+)\.", n)
            if n.startswith(("classifier", "pooler", "head")): lr = HEAD_LR
            elif m: lr = LR * LLRD ** (nl - 1 - int(m.group(1)))
            elif "embed" in n: lr = LR * LLRD ** nl
            else: lr = LR
            g.append({"params": [p], "lr": lr, "weight_decay": 0.0 if (n.endswith(".bias") or "norm" in n.lower()) else 0.01})
        return g

    from sklearn.metrics import f1_score
    for k in folds:
        t0 = time.time()
        tr = np.where(fold_of != k)[0]; va = np.where(fold_of == k)[0]
        trs = [seqs_all[i] for i in tr]; vas = [seqs_all[i] for i in va]; ytr = y_all[tr]
        w = (1 / (np.bincount(ytr, minlength=3) / len(ytr))) ** 0.5
        w = torch.tensor(w / w.mean(), dtype=torch.float, device=dev)
        seed = 42 + k; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rng = np.random.RandomState(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            MD, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=L2I, ignore_mismatched_sizes=True).to(dev)
        opt = torch.optim.AdamW(groups(model), lr=LR)
        nb = (len(trs) + BS - 1) // BS; total = nb * EPOCHS
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        scaler = torch.cuda.amp.GradScaler()
        lossf = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
        for ep in range(EPOCHS):
            model.train(); idx = rng.permutation(len(trs)); batches = []
            for c in range(0, len(idx), BS * 50):  # length-bucketed batches
                part = sorted(idx[c:c + BS * 50], key=lambda i: len(trs[i]))
                batches += [part[b:b + BS] for b in range(0, len(part), BS)]
            rng.shuffle(batches)
            for s, bix in enumerate(batches):
                ii, am = collate([trs[i] for i in bix]); y = torch.tensor(ytr[bix], device=dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=ii, attention_mask=am).logits
                loss = lossf(logits.float(), y)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step()
                if s % 200 == 0: print(f"f{k} ep{ep} {s}/{len(batches)} loss {loss.item():.4f}", flush=True)
            f1 = f1_score(y_all[va], predict(model, vas).argmax(1), average="macro")
            print(f"f{k} epoch {ep + 1} val macro-F1 {f1:.4f} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(f"{OUT}/oof_f{k}.npz", idx=va, logits=predict(model, vas))
        del model, opt; gc.collect(); torch.cuda.empty_cache()

# ---------------- report from OOF ----------------
def mf1(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3); tp = np.diag(cm).astype(float); d = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)))

def fit_bias(lp, y):
    best, bb = mf1(y, lp.argmax(1)), np.zeros(3)
    for bj in np.arange(-2, 2.01, 0.1):
        for bp in np.arange(-2, 2.01, 0.1):
            b = np.array([0.0, bj, bp]); f = mf1(y, (lp + b).argmax(1))
            if f > best + 1e-9: best, bb = f, b
    return bb

def report():
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    train = pd.read_csv(find("train.csv")); y = train.label.map(L2I).values
    fold_of = pd.read_csv(f"{OUT}/folds.csv").set_index("id").loc[train.id, "fold"].values
    oof = np.full((len(train), 3), np.nan); done = []
    for k in range(5):
        if os.path.exists(f"{OUT}/oof_f{k}.npz"):
            d = np.load(f"{OUT}/oof_f{k}.npz"); oof[d["idx"]] = d["logits"]; done.append(k)
    m = ~np.isnan(oof[:, 0]); lg = oof[m]; lp = lg - np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True)) - lg.max(1, keepdims=True)
    yo, fo, lang = y[m], fold_of[m], train.language.values[m]; pred = lp.argmax(1)
    npred = pred.copy()
    if len(done) >= 2:
        for k in done:
            b = fit_bias(lp[fo != k], yo[fo != k]); npred[fo == k] = (lp[fo == k] + b).argmax(1)
    r = dict(model=MODEL_KEY, epochs=EPOCHS, max_len=MAX_LEN, folds=done, n=int(m.sum()),
             oof_macro_f1=mf1(yo, pred), nested_bias_macro_f1=mf1(yo, npred), full_bias=fit_bias(lp, yo).tolist(),
             per_fold={int(k): mf1(yo[fo == k], pred[fo == k]) for k in done},
             per_class_f1=dict(zip(LABELS, f1_score(yo, pred, average=None, labels=[0, 1, 2]).round(4).tolist())),
             per_lang_f1={l: mf1(yo[lang == l], pred[lang == l]) for l in sorted(set(lang))},
             confusion=confusion_matrix(yo, pred, labels=[0, 1, 2]).tolist())
    print(classification_report(yo, pred, target_names=LABELS, digits=4))
    print(json.dumps(r, indent=1)); json.dump(r, open(f"{OUT}/report_{MODEL_KEY}.json", "w"), indent=1)
    np.save(f"{OUT}/oof_{MODEL_KEY}.npy", oof)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker([int(x) for x in sys.argv[1].split(",")]); sys.exit()
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    train = pd.read_csv(find("train.csv"))
    folds = np.full(len(train), -1)  # same split as the team notebook
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (_, va) in enumerate(sg.split(train, train.label.astype(str) + "_" + train.language.astype(str), train.text.map(norm))):
        folds[va] = k
    pd.DataFrame({"id": train.id, "fold": folds}).to_csv(f"{OUT}/folds.csv", index=False)
    os.makedirs(OUT, exist_ok=True); os.system("nvidia-smi")
    n = torch.cuda.device_count(); print("GPUs", n, torch.__version__, torch.version.cuda, flush=True)
    assert n > 0, "no GPU visible"

    fl = FOLDS.split(",")
    jobs = [(",".join(fl[0::2]), "0"), (",".join(fl[1::2]), "1")] if n >= 2 and len(fl) > 1 else [(FOLDS, "0")]
    ps = [subprocess.Popen([sys.executable, __file__, f], env={**os.environ, "CUDA_VISIBLE_DEVICES": g, "MODEL_KEY": MODEL_KEY})
          for f, g in jobs]
    for p in ps: p.wait()
    report()
