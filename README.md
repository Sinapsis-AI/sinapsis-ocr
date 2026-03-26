<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis OCR
<br>
</h1>

<h4 align="center">Templates for Optical Character Recognition (OCR) in images or PDFs</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#transformers-compatibility">⚠️ Compatibility</a> •
<a href="#packages">📦 Packages</a> •
<a href="#features">🚀 Features</a> •
<a href="#usage">📚 Usage example</a> •
<a href="#webapp">🌐 Webapp</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

**Sinapsis OCR** provides powerful and flexible implementations for extracting text from images using different OCR engines. It enables users to easily configure and run OCR tasks with minimal setup.

<h2 id="installation">🐍 Installation</h2>

This mono repo consists of different packages for OCR:

* <code>sinapsis-deepseek-ocr</code>
* <code>sinapsis-doctr</code>
* <code>sinapsis-easyocr</code>
* <code>sinapsis-glm-ocr</code>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-doctr --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-doctr --extra-index-url https://pypi.sinapsis.tech
```

**Change the name of the package for the one you want to install**.

> [!IMPORTANT]
> Templates in each package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

with <code>uv</code>:

```bash
  uv pip install sinapsis-doctr[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-doctr[all] --extra-index-url https://pypi.sinapsis.tech
```

> [!TIP]
> You can also install all the packages within this project:
>
```bash
  uv pip install sinapsis-ocr[all] --extra-index-url https://pypi.sinapsis.tech
```

<h3 id="transformers-compatibility">⚠️ Transformers Version Compatibility</h3>

**DeepSeek OCR and GLM OCR have different transformers version requirements.** They cannot be used together in the same environment:

| Package | Transformers Version | Notes |
|---------|---------------------|-------|
| `sinapsis-deepseek-ocr` | `==4.46.3` (pinned) | DeepSeek models require this exact version |
| `sinapsis-glm-ocr` | `>=4.46.3` (flexible) | GLM-OCR works with `>=5.1.0` |

**When installing from PyPI:**
```bash
# DeepSeek OCR - installs transformers==4.46.3
uv pip install sinapsis-deepseek-ocr[all] --extra-index-url https://pypi.sinapsis.tech

# GLM OCR - installs latest transformers (5.x)
uv pip install sinapsis-glm-ocr[all] --extra-index-url https://pypi.sinapsis.tech
```

**Important:** Installing both `sinapsis-deepseek-ocr` and `sinapsis-glm-ocr` in the same environment may force transformers==4.46.3, which will cause GLM OCR to fail. Use separate virtual environments if you need both.

<h2 id="packages">📦 Packages</h2>

<details>
<summary><strong><span style="font-size: 1.0em;">Packages summary</span></strong></summary>

- **Sinapsis DeepSeek OCR**
  - Uses the DeepSeek OCR model for high-quality OCR
  - Supports optional grounding for bounding box extraction
  - Multiple inference modes (tiny, small, base, large, gundam)

- **Sinapsis DocTR**
  - Uses the DocTR library for high-quality OCR with modern deep learning models
  - Supports multiple detection and recognition architectures
  - Provides detailed text extraction with bounding boxes and confidence scores

- **Sinapsis EasyOCR**
  - Leverages the EasyOCR library for simple yet effective OCR
  - Supports multiple languages
  - Extracts text with bounding boxes and confidence scores

- **Sinapsis GLM OCR**
  - Uses Zhipu AI's GLM-OCR model for high-quality OCR
  - Supports document parsing (text, formula, table) and structured information extraction via JSON schema
  - Batch inference support for faster processing of multiple images
</details>

> [!TIP]
> Use CLI command ```sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis OCR.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***DocTROCRPrediction*** use ```sinapsis info --example-template-config DocTROCRPrediction``` to produce an example config.



<h2 id="usage">📚 Usage example</h2>

<details>
<summary><strong><span style="font-size: 1.4em;">DocTR Example</span></strong></summary>

```yaml
agent:
  name: doctr_prediction
  description: agent to run inference with DocTR, performs on images read, recognition and save

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: FolderImageDatasetCV2
    class_name: FolderImageDatasetCV2
    template_input: InputTemplate
    attributes:
      data_dir: dataset/input

  - template_name: DocTROCRPrediction
    class_name: DocTROCRPrediction
    template_input: FolderImageDatasetCV2
    attributes:
      recognized_characters_as_labels: True

  - template_name: BBoxDrawer
    class_name: BBoxDrawer
    template_input: DocTROCRPrediction
    attributes:
      draw_confidence: True
      draw_extra_labels: True

  - template_name: ImageSaver
    class_name: ImageSaver
    template_input: BBoxDrawer
    attributes:
      save_dir: output
      root_dir: dataset
```
</details>

<details>
<summary><strong><span style="font-size: 1.4em;">EasyOCR Example</span></strong></summary>

```yaml
agent:
  name: easyocr_inference
  description: agent to run inference with EasyOCR, performs on images read, recognition and save

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: FolderImageDatasetCV2
    class_name: FolderImageDatasetCV2
    template_input: InputTemplate
    attributes:
      data_dir: dataset/input

  - template_name: EasyOCR
    class_name: EasyOCR
    template_input: FolderImageDatasetCV2
    attributes: {}

  - template_name: BBoxDrawer
    class_name: BBoxDrawer
    template_input: EasyOCR
    attributes:
      draw_confidence: True
      draw_extra_labels: True

  - template_name: ImageSaver
    class_name: ImageSaver
    template_input: BBoxDrawer
    attributes:
      save_dir: output
      root_dir: dataset
```
</details>

<details>
<summary><strong><span style="font-size: 1.4em;">DeepSeek OCR Example</span></strong></summary>

```yaml
agent:
  name: deepseek_ocr_inference
  description: agent to run inference with DeepSeek OCR

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: FolderImageDatasetCV2
    class_name: FolderImageDatasetCV2
    template_input: InputTemplate
    attributes:
      data_dir: dataset/input

  - template_name: DeepSeekOCRInference
    class_name: DeepSeekOCRInference
    template_input: FolderImageDatasetCV2
    attributes:
      prompt: "Convert the document to markdown."
      enable_grounding: true
      mode: base

  - template_name: BBoxDrawer
    class_name: BBoxDrawer
    template_input: DeepSeekOCRInference
    attributes:
      draw_confidence: True
      draw_extra_labels: True

  - template_name: ImageSaver
    class_name: ImageSaver
    template_input: BBoxDrawer
    attributes:
      save_dir: output
      root_dir: dataset
```
</details>

<details>
<summary><strong><span style="font-size: 1.4em;">GLM OCR Example</span></strong></summary>

```yaml
agent:
  name: glm_ocr_table_agent
  description: "Agent to read images, perform GLM OCR for table recognition."

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: FolderImageDatasetCV2
    class_name: FolderImageDatasetCV2
    template_input: InputTemplate
    attributes:
      load_on_init: True
      root_dir: "."
      data_dir: "artifacts"
      pattern: "expense.jpg"

  - template_name: GLMOCRInference
    class_name: GLMOCRInference
    template_input: FolderImageDatasetCV2
    attributes:
      prompt: "Table Recognition:"
      init_args:
        pretrained_model_name_or_path: zai-org/GLM-OCR
        torch_dtype: auto
        attn_implementation: kernels-community/flash-attn2
        device_map: auto
      generation_config:
        max_new_tokens: 8192
        do_sample: false
```
</details>

To run, simply use:

```bash
sinapsis run name_of_the_config.yml
```

<h2 id="webapp">🌐 Webapp</h2>

The webapp provides a simple interface to extract text from images using OCR. Upload your image, and the app will process it and display the detected text with bounding boxes.

> [!IMPORTANT]
> To run the app you first need to clone this repository:

```bash
git clone https://github.com/Sinapsis-ai/sinapsis-ocr.git
cd sinapsis-ocr
```

> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

> [!TIP]
> The agent configuration can be updated using the AGENT_CONFIG_PATH environment var.
For default uses the config for easy ocr but this can be chaged with:
`AGENT_CONFIG_PATH=/app/packages/sinapsis_doctr/src/sinapsis_doctr/configs/doctr_demo.yaml`

<details>
<summary id="docker"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT** This docker image depends on the sinapsis:base image. Please refer to the official [sinapsis](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker) instructions to Build with Docker.

1. **Build the sinapsis-ocr image**:

```bash
docker compose -f docker/compose.yaml build
```

2. **Start the app container**:

```bash
docker compose -f docker/compose_app.yaml up
```

3. **Check the status**:

```bash
docker logs -f sinapsis-ocr-app
```

4. The logs will display the URL to access the webapp, e.g.:

**NOTE**: The url can be different, check the output of logs

```bash
Running on local URL:  http://127.0.0.1:7860
```

5. To stop the app:

```bash
docker compose -f docker/compose_app.yaml down
```

</details>

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">💻 UV</span></strong></summary>

To run the webapp using the <code>uv</code> package manager, please:

1. **Create the virtual environment and sync the dependencies**:

```bash
uv sync --frozen
```

2. **Install packages**:
```bash
uv pip install sinapsis-ocr[all] --extra-index-url https://pypi.sinapsis.tech
```
3. **Run the webapp**:

```bash
uv run webapps/gradio_ocr.py
```

4. **The terminal will display the URL to access the webapp, e.g.**:

```bash
Running on local URL:  http://127.0.0.1:7860
```
NOTE: The url can be different, check the output of the terminal

5. To stop the app press `Control + C` on the terminal

</details>

<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)

<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.
