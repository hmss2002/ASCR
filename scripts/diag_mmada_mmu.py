import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, ".")
from ascr.generators.mmada_native import MMaDANativeEngine

img = sys.argv[1] if len(sys.argv) > 1 else None
eng = MMaDANativeEngine().load()
print("loaded; mask_token_id", eng.mask_token_id, "offset", eng.token_offset)

# sanity: open-ended description
for q in [
    "Describe this image in one sentence.",
    "Original text-to-image prompt: A red cube on the left and a blue sphere on the right. Does the image fully satisfy the prompt? Answer yes or no, then explain briefly.",
]:
    out = eng.answer_image(q, img, max_new_tokens=128)
    print("\n=== Q:", q[:60])
    print("A:", repr(out))
