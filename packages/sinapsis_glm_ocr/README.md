<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis GLM OCR
<br>
</h1>

<h4 align="center">GLM-OCR-based Optical Character Recognition (OCR) for images</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">🚀 Features</a> •
<a href="#usage">📚 Usage example</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

**Sinapsis GLM OCR** provides a powerful implementation for extracting text from images using Zhipu AI's GLM-OCR model. Built on the GLM-V encoder-decoder architecture, it supports document parsing (text, formula, table recognition) and structured information extraction via JSON schema prompts. It also supports batch inference for faster processing of multiple images.

<h2 id="installation">🐍 Installation</h2>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-glm-ocr --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-glm-ocr --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

with <code>uv</code>:

```bash
  uv pip install sinapsis-glm-ocr[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-glm-ocr[all] --extra-index-url https://pypi.sinapsis.tech
```

> [!TIP]
> Use CLI command ```sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis OCR.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config GLMOCRInference``` to produce an example Agent config for the GLMOCRInference template.

<h2 id="features">🚀 Features</h2>

<h3>Templates Supported</h3>

This module includes templates tailored for the GLM-OCR engine:

- **GLMOCRInference**: Uses GLM-OCR model to extract text from images. Supports document parsing (text, formula, table) and structured information extraction.
- **GLMOCRBatchInference**: Batch inference version for processing multiple images efficiently.

<details>
<summary><strong><span style="font-size: 1.25em;">GLMOCRInference Attributes</span></strong></summary>

- **`prompt`** (str): The prompt to send to the model. Defaults to `"Text Recognition:"`. Other options include `"Formula Recognition:"` and `"Table Recognition:"`.
- **`init_args`** (GLMOCRInitArgs): Initialization arguments for the model including:
  - `pretrained_model_name_or_path`: Model identifier. Defaults to `"zai-org/GLM-OCR"`.
  - `torch_dtype`: Model precision (`"float16"`, `"bfloat16"`, `"auto"`). Defaults to `"auto"`.
  - `attn_implementation`: Attention implementation (`"kernels-community/flash-attn2"`, `"kernels-community/paged-attention"`). Defaults to `"kernels-community/flash-attn2"`.
  - `device_map`: Device mapping (`"auto"`, `"balanced"`, `"balanced_low_0"`, `"sequential"`, or specific device like `"cuda:0"`). Defaults to `"auto"`.
  - **Note**: This model requires CUDA. CPU inference is not supported.
- **`generation_config`** (GLMOCRGenerationConfig): Generation configuration including:
  - `max_new_tokens`: Maximum tokens to generate. Defaults to `8192`.
  - `min_new_tokens`: Minimum tokens to generate. Defaults to `1`.
  - `do_sample`: Whether to use sampling. Defaults to `False`.
  - `repetition_penalty`: Penalty for repeating tokens. Defaults to `1.0`.
  - `length_penalty`: Penalty for sequence length. Defaults to `1.0`.

</details>

<h2 id="usage">📚 Usage example</h2>

<details>
<summary><strong><span style="font-size: 1.4em;">Text Recognition</span></strong></summary>

```yaml
agent:
  name: glm_ocr_agent
  description: Agent to run inference with GLM OCR for text recognition

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: FolderImageDatasetCV2
  class_name: FolderImageDatasetCV2
  template_input: InputTemplate
  attributes:
    data_dir: dataset/input

- template_name: GLMOCRInference
  class_name: GLMOCRInference
  template_input: FolderImageDatasetCV2
  attributes:
    prompt: "Text Recognition:"
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

<details>
<summary><strong><span style="font-size: 1.4em;">Table Recognition</span></strong></summary>

```yaml
agent:
  name: glm_ocr_table_agent
  description: Agent to run inference with GLM OCR for table recognition

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: FolderImageDatasetCV2
  class_name: FolderImageDatasetCV2
  template_input: InputTemplate
  attributes:
    data_dir: dataset/input

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

<details>
<summary><strong><span style="font-size: 1.4em;">Information Extraction (JSON Schema)</span></strong></summary>

```yaml
agent:
  name: glm_ocr_json_agent
  description: Agent to extract structured information using JSON schema

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: FolderImageDatasetCV2
  class_name: FolderImageDatasetCV2
  template_input: InputTemplate
  attributes:
    data_dir: dataset/input

- template_name: GLMOCRInference
  class_name: GLMOCRInference
  template_input: FolderImageDatasetCV2
  attributes:
    prompt: |
      Extract the information from the provided image and output
      it strictly as a valid JSON object matching the following schema.
      {
        "name": "string",
        "date": "string",
        "amount": "number"
      }
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

<details>
<summary><strong><span style="font-size: 1.4em;">Batch Inference</span></strong></summary>

```yaml
agent:
  name: glm_ocr_batch_agent
  description: Agent to run batch inference with GLM OCR for faster processing

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: FolderImageDatasetCV2
  class_name: FolderImageDatasetCV2
  template_input: InputTemplate
  attributes:
    data_dir: dataset/input
    batch_size: 4

- template_name: GLMOCRBatchInference
  class_name: GLMOCRBatchInference
  template_input: FolderImageDatasetCV2
  attributes:
    prompt: "Text Recognition:"
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

<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)

<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.
