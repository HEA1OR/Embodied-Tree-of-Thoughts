## VLM Task Planning

### Installation
VLM task planning only requires the installation of OpenAI's Python library, which can be directly installed in the `simworld` environment:
```bash
conda activate simworld
pip install openai
```

### Usage

We organize the planning module of **EToT** into two notebooks: `Priori Branching.ipynb` and `Reflective Branching.ipynb`.

The prompt used by **Rekep** is located at: `realworld/rekep/vlm_query/prompt_template.txt`

The planning implementations for **Rekep w/ CoT** and **Reflect** are provided in `rekep_cot.ipynb`.

The `rekep_results` and `examples` directories contain sample images and videos captured during our real-world experiments.
```
vlm/
├── examples/
├── rekep_results/
├── priori_branching.ipynb
├── reflective_branching.ipynb
├── rekep_cot.ipynb
└── vlm.py
```


Our code uses the OpenAI API to access **GPT-4o**. The file `vlm.py` provides a multimodal (image-and-text) interface for invoking the model. At the beginning of each notebook (.ipynb), we provide placeholders where users can fill in their `api_key, base_url`, and other required configuration fields. **After completing these configurations, you can freely run the notebooks.**
