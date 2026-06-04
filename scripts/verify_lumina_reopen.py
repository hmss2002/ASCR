"""Direct verification of Lumina reopen (forced mask). Run in .venv-lumina on a GPU."""
import os, sys
sys.path.insert(0, ".")
from ascr.generators.lumina_native import LuminaNativeEngine

eng = LuminaNativeEngine(generation_timesteps=int(os.environ.get("STEPS", "64")))
prompt = "a red cube on top of a blue sphere, plain background"
base = eng.generate(prompt, seed=1234)
print("baseline len", len(base), "uniq", len(set(base)))
eng.decode_to(base, "outputs/lumina_reopen_test/baseline.png")

# Force-mask a 16x16 center block of the 64x64 grid (one coarse 4x4 cell).
g = eng.token_grid_size
idx = [(r, c) for r in range(24, 40) for c in range(24, 40)]
rev = eng.reopen(base, idx, prompt, seed=99)
print("reopen len", len(rev))
eng.decode_to(rev, "outputs/lumina_reopen_test/revised.png")

changed_inside = sum(1 for (r, c) in idx if base[r * g + c] != rev[r * g + c])
changed_outside = sum(1 for r in range(g) for c in range(g)
                      if (r, c) not in set(idx) and base[r * g + c] != rev[r * g + c])
print(f"changed_inside={changed_inside}/{len(idx)} changed_outside={changed_outside}")
b = open("outputs/lumina_reopen_test/baseline.png", "rb").read()
v = open("outputs/lumina_reopen_test/revised.png", "rb").read()
print("images differ:", b != v)
print("OK_REOPEN" if changed_inside > 0 and changed_outside == 0 and b != v else "FAIL_REOPEN")
