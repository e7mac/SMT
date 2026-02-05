"""Local test script for SMT model with MusicXML output."""

import torch
import cv2
from music21 import converter
from data_augmentation.data_augmentation import convert_img_to_tensor
from smt_model import SMTModelForCausalLM


def preprocess_image(img, max_height=256, max_width=3056):
    h, w = img.shape[:2]
    scale = min(max_height / h, max_width / w, 1.0)

    if scale < 1.0:
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)

    return img


def kern_to_musicxml(kern_string: str, output_path: str) -> None:
    # Convert ekern_1.0 headers to standard kern for music21 compatibility
    kern_string = kern_string.replace("**ekern_1.0", "**kern")
    score = converter.parse(kern_string, format='humdrum')
    score.write('musicxml', fp=output_path)


def main():
    image_path = "/Users/mayank10/Developer/SMT/input3.png"

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    print(f"Original image size: {img.shape}")
    img = preprocess_image(img)
    print(f"Preprocessed image size: {img.shape}")

    # Convert to tensor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_tensor = convert_img_to_tensor(img).unsqueeze(0).to(device)
    print(f"Tensor shape: {img_tensor.shape}")

    # Load model
    print("Loading model...")
    model = SMTModelForCausalLM.from_pretrained("antoniorv6/smt-grandstaff").to(device)
    model.eval()

    # Run prediction
    print("Running prediction...")
    with torch.no_grad():
        predictions, _ = model.predict(img_tensor, convert_to_str=True)

    # Convert to kern format
    kern_result = "".join(predictions)
    kern_result = kern_result.replace("<b>", "\n").replace("<s>", " ").replace("<t>", "\t")

    print("\n=== KERN OUTPUT ===")
    print(kern_result)

    # Save kern file
    kern_path = "/Users/mayank10/Developer/SMT/test_output.krn"
    with open(kern_path, "w") as f:
        f.write(kern_result)
    print(f"\nKern saved to: {kern_path}")

    # Convert to MusicXML
    print("\nConverting to MusicXML...")
    musicxml_path = "/Users/mayank10/Developer/SMT/test_output.musicxml"
    try:
        kern_to_musicxml(kern_result, musicxml_path)
        print(f"MusicXML saved to: {musicxml_path}")
    except Exception as e:
        print(f"MusicXML conversion failed: {e}")


if __name__ == "__main__":
    main()
