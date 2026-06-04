import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, ".")
from ascr.generators.mmada_native import MMaDANativeEngine
from ascr.evaluators.mmada_self import MMaDASelfEvaluator

clean = sys.argv[1]
grid = sys.argv[2]
prompt = "A red cube on the left and a blue sphere on the right"
eng = MMaDANativeEngine().load()
ev = MMaDASelfEvaluator(grid_size=32, max_selected_cells=64)
ev.attach_engine(eng)

for label, img in [("CLEAN", clean), ("GRID", grid)]:
    print("\n##########", label, img)
    loc_q = ev._localization_question(prompt, "the blue object is centered; red cube and right-side blue sphere are missing")
    for mnt in (128, 256):
        out = eng.answer_image(loc_q, img, max_new_tokens=mnt)
        print(f"-- localization max_new={mnt}:")
        print("   ", repr(out))
    # also a simpler spatial question
    out = eng.answer_image(prompt + ". In the image, where is the main object located (left, center, right, top, bottom)? Answer briefly.", img, max_new_tokens=64)
    print("-- spatial:", repr(out))
