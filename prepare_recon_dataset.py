"""
Recon 결과(fake_B)와 원본 valA를 묶어 classification 입력용 recon_dataset 생성.
실행: Inference 폴더에서  python prepare_recon_dataset.py [RUN_ID]
"""
import os
import sys
import shutil

def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "A"
    base = os.path.dirname(os.path.abspath(__file__))
    results = os.path.join(base, "results", run_id)
    recon_fake_b = os.path.join(results, "recon")
    recon_dataset = os.path.join(results, "recon_dataset")
    vala_dst = os.path.join(recon_dataset, "valA")
    predb_dst = os.path.join(recon_dataset, "predB")

    for data_root in [os.path.join(base, "..", "datasets", "BCI_HER2"), os.path.join(base, "datasets", "BCI_HER2")]:
        data_root = os.path.abspath(data_root)
        if os.path.isdir(data_root):
            break
    else:
        raise FileNotFoundError("datasets/BCI_HER2 not found under Inference/.. or Inference/")

    vala_src = os.path.join(data_root, "valA")
    if not os.path.isdir(vala_src):
        vala_src = os.path.join(data_root, "trainA")
    if not os.path.isdir(vala_src):
        raise FileNotFoundError(f"valA/trainA not found under {data_root}")
    if not os.path.isdir(recon_fake_b):
        raise FileNotFoundError(f"Recon output not found: {recon_fake_b}. Run recon first.")

    os.makedirs(vala_dst, exist_ok=True)
    os.makedirs(predb_dst, exist_ok=True)
    for f in os.listdir(vala_src):
        if f.endswith(".png"):
            src = os.path.join(vala_src, f)
            dst = os.path.join(vala_dst, f)
            if not os.path.lexists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)
    for f in sorted(os.listdir(recon_fake_b)):
        if f.endswith(".png"):
            src = os.path.join(recon_fake_b, f)
            dst = os.path.join(predb_dst, f)
            if not os.path.lexists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)
    print(f"Prepared {recon_dataset} (valA + predB)")

if __name__ == "__main__":
    main()
