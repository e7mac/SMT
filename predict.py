"""Replicate prediction interface for Sheet Music Transformer (SMT)."""

from cog import BasePredictor, Input, Path
import torch
import cv2
import tempfile
import verovio
from data_augmentation.data_augmentation import convert_img_to_tensor

from smt_model import SMTModelForCausalLM


MODEL_CHOICES = {
    "grandstaff": "antoniorv6/smt-grandstaff",
    "camera-grandstaff": "antoniorv6/smt-camera-grandstaff",
}


def preprocess_image(img, max_height=256, max_width=3056):
    """
    Preprocess image to ensure dimensions are compatible with the model.
    The grandstaff model was trained with maxh=256, maxw=3056.
    Images must be resized to fit within these bounds.
    """
    h, w = img.shape[:2]

    # Calculate scale to fit within max dimensions while preserving aspect ratio
    scale = min(max_height / h, max_width / w, 1.0)

    if scale < 1.0:
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Ensure dimensions are divisible by 16 (the model's reduction factor)
    h, w = img.shape[:2]
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    if pad_h > 0 or pad_w > 0:
        # Pad with white (255 for grayscale sheet music)
        img = cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=255
        )

    return img


def kern_to_musicxml(kern_string: str) -> str:
    """Convert Humdrum kern notation to MusicXML using verovio."""
    tk = verovio.toolkit()
    tk.loadData(kern_string)
    return tk.renderToMusicXml()


class Predictor(BasePredictor):
    def setup(self):
        """Load all models into memory for fast switching."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}

        # Pre-load the default model
        print("Loading grandstaff model...")
        self.models["grandstaff"] = SMTModelForCausalLM.from_pretrained(
            MODEL_CHOICES["grandstaff"]
        ).to(self.device)
        self.models["grandstaff"].eval()

    def _get_model(self, model_name: str):
        """Get or load a model by name."""
        if model_name not in self.models:
            print(f"Loading {model_name} model...")
            self.models[model_name] = SMTModelForCausalLM.from_pretrained(
                MODEL_CHOICES[model_name]
            ).to(self.device)
            self.models[model_name].eval()
        return self.models[model_name]

    def predict(
        self,
        image: Path = Input(description="Sheet music image to transcribe"),
        model: str = Input(
            description="Model variant to use. 'grandstaff' for clean scanned sheet music, 'camera-grandstaff' for smartphone photos.",
            default="grandstaff",
            choices=["grandstaff", "camera-grandstaff"],
        ),
        output_format: str = Input(
            description="Output format: 'musicxml' for MusicXML file, 'kern' for Humdrum kern text",
            default="musicxml",
            choices=["musicxml", "kern"],
        ),
    ) -> Path:
        """Transcribe sheet music image to symbolic notation."""
        # Load image
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Could not load image from {image}")

        # Preprocess image (resize and pad for model compatibility)
        img = preprocess_image(img)

        # Convert to tensor
        img_tensor = convert_img_to_tensor(img).unsqueeze(0).to(self.device)

        # Get model
        model_instance = self._get_model(model)

        # Run prediction
        with torch.no_grad():
            predictions, _ = model_instance.predict(img_tensor, convert_to_str=True)

        # Convert tokens to kern format
        kern_result = "".join(predictions)
        kern_result = kern_result.replace("<b>", "\n").replace("<s>", " ").replace("<t>", "\t")

        # Output based on format
        if output_format == "musicxml":
            musicxml_content = kern_to_musicxml(kern_result)
            output_path = Path(tempfile.mktemp(suffix=".musicxml"))
            with open(output_path, "w") as f:
                f.write(musicxml_content)
            return output_path
        else:
            output_path = Path(tempfile.mktemp(suffix=".krn"))
            with open(output_path, "w") as f:
                f.write(kern_result)
            return output_path
