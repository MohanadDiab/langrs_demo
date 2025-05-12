import gradio as gr
from langrs import LangRS
from langrs.common import apply_nms
from PIL import Image
import tempfile
import os
from io import BytesIO

# Global state for storing intermediate results
STATE_BOXES = {}  # method -> boxes
STATE_IMAGE = None
STATE_SEGMENTER = None
STATE_SELECTED_METHOD = None
STATE_NMS_BOXES = []

def generate_masks_ui():
    if not STATE_SEGMENTER or not STATE_NMS_BOXES:
        raise RuntimeError("Boxes or model not ready")

    mask = STATE_SEGMENTER.generate_masks(boxes=STATE_NMS_BOXES)
    mask_img_path = STATE_SEGMENTER.output_path_image_masks
    return Image.open(mask_img_path)


def apply_nms_ui(method, iou_thresh):
    global STATE_NMS_BOXES, STATE_SELECTED_METHOD

    STATE_SELECTED_METHOD = method
    if method not in STATE_BOXES:
        raise ValueError("Selected method not found.")
    
    raw_boxes = STATE_BOXES[method]
    nms_boxes = apply_nms(raw_boxes, iou_threshold=iou_thresh)
    STATE_NMS_BOXES = nms_boxes

    # Draw NMS result
    image = STATE_IMAGE.copy()
    draw_boxes_on_image(image, nms_boxes)
    return image

def generate_boxes(image, prompt, window_size, box_thresh, text_thresh):
    global STATE_BOXES, STATE_IMAGE, STATE_SEGMENTER

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize LangRS model
        model = LangRS(image=image, prompt=prompt, output_path=tmpdir)

        # Run detection and outlier rejection
        model.generate_boxes(
            window_size=window_size,
            overlap=int(0.3 * window_size),
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            text_prompt=prompt
        )
        rejection_boxes = model.outlier_rejection()

        # Store state for next steps
        STATE_BOXES = rejection_boxes
        STATE_IMAGE = model.pil_image
        STATE_SEGMENTER = model

        # Render and save overlay previews
        previews = {}
        for method, boxes in rejection_boxes.items():
            img = draw_boxes_on_image(model.pil_image, boxes)
            path = os.path.join(tmpdir, f"{method}.jpg")
            img.save(path)
            previews[method] = path

        return list(previews.values()), list(previews.keys())

def draw_boxes_on_image(image, boxes, color='red'):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots()
    ax.imshow(image)
    for x1, y1, x2, y2 in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def filter_and_display(methods_selected):
    images = []
    for method in methods_selected:
        if method in STATE_BOXES:
            img = STATE_IMAGE.copy()
            draw_boxes_on_image(img, STATE_BOXES[method])
            images.append(img)
    return images

# Build the interface
with gr.Blocks() as demo:
    gr.Markdown("# LangRS: Step 1 — Bounding Box Generation")
    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Image")
        text_input = gr.Textbox(label="Prompt", placeholder="e.g. roads, buildings")
    
    with gr.Row():
        win_slider = gr.Slider(100, 1000, value=500, step=50, label="Window Size")
        box_slider = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Box Threshold")
        txt_slider = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Text Threshold")

    generate_btn = gr.Button("Generate Boxes")
    preview_gallery = gr.Gallery(label="Preview Overlays").style(grid=[2])
    method_selector = gr.CheckboxGroup(label="Select Overlays to Display")
    output_gallery = gr.Gallery(label="Selected Overlays")

    generate_btn.click(
        fn=generate_boxes,
        inputs=[img_input, text_input, win_slider, box_slider, txt_slider],
        outputs=[preview_gallery, method_selector]
    )

    method_selector.change(
        fn=filter_and_display,
        inputs=method_selector,
        outputs=output_gallery
    )

    gr.Markdown("## Step 2 — NMS Filtering")

    nms_method = gr.Dropdown(label="Select method for NMS", choices=[], interactive=True)
    iou_slider = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="IoU Threshold")
    nms_btn = gr.Button("Apply NMS")
    nms_result = gr.Image(label="NMS Result")

    # Populate dropdown on box generation
    generate_btn.click(
        fn=generate_boxes,
        inputs=[img_input, text_input, win_slider, box_slider, txt_slider],
        outputs=[preview_gallery, method_selector]
    ).then(
        lambda keys: gr.update(choices=keys),
        inputs=None,
        outputs=nms_method
    )

    nms_btn.click(
        fn=apply_nms_ui,
        inputs=[nms_method, iou_slider],
        outputs=nms_result
    )

    gr.Markdown("## Step 3 — Generate Masks")

    mask_btn = gr.Button("Generate Masks")
    mask_output = gr.Image(label="Final Segmentation Mask")

    mask_btn.click(
        fn=generate_masks_ui,
        inputs=[],
        outputs=mask_output
    )


if __name__ == "__main__":
    demo.launch()
