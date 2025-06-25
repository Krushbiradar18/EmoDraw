import numpy as np, cv2, os, argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--class_name", required=True,
                    help="e.g. star, sun, tree, heart")
parser.add_argument("--max_images", type=int, default=500,
                    help="How many images to extract per class")
args = parser.parse_args()

npy_path   = f"numpy_bitmap_{args.class_name}.npy"
out_folder = f"dataset/{args.class_name}"
os.makedirs(out_folder, exist_ok=True)

data = np.load(npy_path, allow_pickle=True)

for i, flat in tqdm(enumerate(data[:args.max_images]), total=args.max_images):
    img = flat.reshape(28, 28).astype(np.uint8)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{out_folder}/{args.class_name}_{i}.png", img)