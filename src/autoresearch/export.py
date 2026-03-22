"""ONNX model export for deployed inference."""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.onnx

from .logging_config import get_logger

logger = get_logger(__name__)


class ONNXExporter:
    """Export PyTorch CNN models to ONNX format."""

    def __init__(self):
        self.logger = get_logger("export")

    def export_model(
        self,
        model: torch.nn.Module,
        checkpoint_path: Path,
        output_path: Path,
        input_shape: tuple = (1, 3, 224, 224),  # Standard CNN input
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Export checkpoint to ONNX.
        
        Args:
            model: PyTorch model instance
            checkpoint_path: Path to .pt checkpoint
            output_path: Where to save .onnx
            input_shape: Expected input tensor shape
            metadata: Optional metadata to attach
        
        Returns:
            True if successful
        """
        self.logger.info("Starting ONNX export", checkpoint=str(checkpoint_path))

        try:
            # Load checkpoint
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)

            # Prepare dummy input
            dummy_input = torch.randn(*input_shape)

            # Export to ONNX
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["image"],
                output_names=["steering"],
                dynamic_axes={"image": {0: "batch_size"}},
                verbose=False,
            )

            self.logger.info("ONNX export successful", output=str(output_path))
            return True

        except Exception as e:
            self.logger.error("ONNX export failed", error=str(e))
            return False

    def validate_export(self, onnx_path: Path) -> bool:
        """Check if exported model is valid."""
        try:
            import onnx
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            self.logger.info("ONNX validation passed", path=str(onnx_path))
            return True
        except Exception as e:
            self.logger.error("ONNX validation failed", error=str(e))
            return False
