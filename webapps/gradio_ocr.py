import gradio as gr
from sinapsis.webapp.agent_gradio_helper import add_logo_and_title, css_header, init_image_inference
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP

DEFAULT_CONFIG = "packages/sinapsis_easyocr/src/sinapsis_easyocr/configs/easyocr_demo.yaml"
CONFIG_FILE = AGENT_CONFIG_PATH or DEFAULT_CONFIG


def create_demo() -> gr.Blocks:
    """Creates and returns the Gradio Blocks demo interface.

    Returns:
        gr.Blocks: The Gradio Blocks object containing the entire interface.
    """
    with gr.Blocks(title="Sinapsis OCR Demo", css=css_header()) as demo_interface:
        add_logo_and_title("Sinapsis OCR Demo")
        init_image_inference(CONFIG_FILE)
    return demo_interface


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=GRADIO_SHARE_APP)
