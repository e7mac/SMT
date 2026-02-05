"""Test script loading a sample from the HuggingFace grandstaff dataset."""

import torch
from datasets import load_dataset
from music21 import converter
from data_augmentation.data_augmentation import convert_img_to_tensor
from smt_model import SMTModelForCausalLM


def kern_to_musicxml(kern_string: str, output_path: str) -> None:
    kern_string = kern_string.replace("**ekern_1.0", "**kern")
    score = converter.parse(kern_string, format='humdrum')
    score.write('musicxml', fp=output_path)


def main():
    # Load a sample from the grandstaff dataset
    print("Loading dataset sample...")
    dataset = load_dataset("antoniorv6/grandstaff", split="test", streaming=True)
    sample = next(iter(dataset))

    image = sample["image"]
    ground_truth = sample["transcription"]

    print(f"Image size: {image.size}")
    print(f"Ground truth:\n{ground_truth[:200]}...")

    # Save the input image for comparison
    image.save("dataset_test_input.png")
    print("Input image saved to: dataset_test_input.png")

    # Convert PIL image to tensor
    import numpy as np
    img_array = np.array(image)
    img_tensor = convert_img_to_tensor(img_array).unsqueeze(0)
    print(f"Tensor shape: {img_tensor.shape}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = SMTModelForCausalLM.from_pretrained("antoniorv6/smt-grandstaff").to(device)
    model.eval()
    img_tensor = img_tensor.to(device)

    # Run prediction
    print("Running prediction...")
    with torch.no_grad():
        predictions, _ = model.predict(img_tensor, convert_to_str=True)

    # Convert to kern format
    kern_result = "".join(predictions)
    kern_result = kern_result.replace("<b>", "\n").replace("<s>", " ").replace("<t>", "\t")

    print("\n=== PREDICTION ===")
    print(kern_result[:500])

    print("\n=== GROUND TRUTH ===")
    print(ground_truth[:500])

    # Save outputs
    with open("dataset_test_output.krn", "w") as f:
        f.write(kern_result)
    print("\nKern saved to: dataset_test_output.krn")

    # Convert to MusicXML
    print("\nConverting to MusicXML...")
    try:
        kern_to_musicxml(kern_result, "dataset_test_output.musicxml")
        print("MusicXML saved to: dataset_test_output.musicxml")
    except Exception as e:
        print(f"MusicXML conversion failed: {e}")


if __name__ == "__main__":
    main()
