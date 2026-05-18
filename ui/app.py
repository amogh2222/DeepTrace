import gradio as gr
import tempfile
from pathlib import Path

from inference.pipeline import InferencePipeline


pipeline = InferencePipeline(
    checkpoint_path="checkpoints/best_model.pth"
)


def analyze_image(image):

    if image is None:
        return "No image uploaded", None

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    ) as tmp:

        image.save(tmp.name)

        result = pipeline.predict_image(
            tmp.name
        )

    pred_str = (
        "🚨 FAKE"
        if result["prediction"] == "FAKE"
        else "✅ REAL"
    )

    summary = (

        f"# {pred_str}\n\n"

        f"### Detection Analysis\n\n"

        f"Confidence: "
        f"{result.get('confidence', 0) * 100:.2f}%\n\n"

        f"Fake Probability: "
        f"{result.get('fake_probability', 0) * 100:.2f}%\n\n"

        f"Manipulation Type: "
        f"{result.get('manipulation_type', 'N/A')}\n\n"

        f"### Compression Analysis\n\n"

        f"Compression Severity: "
        f"{result.get('compression_label', 'unknown')}\n\n"

        f"Compression Quality Score: "
        f"{result.get('compression_quality', 0):.3f}\n\n"

        f"Adaptive Threshold Used: "
        f"{result.get('threshold_used', 0):.3f}\n\n"

        f"### Forensic Report\n\n"

        f"{result.get('forensic_report', '')}"
    )

    heatmap = result.get(
        "heatmap_path",
        None
    )

    return summary, heatmap


demo = gr.Interface(

    fn=analyze_image,

    inputs=gr.Image(
        type="pil",
        label="Upload Image"
    ),

    outputs=[

        gr.Markdown(
            label="Detection Report"
        ),

        gr.Image(
            label="GradCAM Heatmap"
        )
    ],

    title="DeepTrace",

    description=(
        "Compression-Aware Multimodal "
        "Deepfake Detection System"
    )
)


if __name__ == "__main__":

    demo.launch(
        share=True
    )
