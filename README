# Medical Assistant Bot Assignment

The goal of this assignment is to develop a medical question-answering system utilizing the provided dataset containing medical information. More specifically, create a model that can effectively answer user queries related to medical diseases.

Since the dataset contains Question and Answer pairs, I will use this information to fine-tune a pre-trained language model. More specifically I will use instruction tuning to adapt the model to the specific task of medical question answering.

## Sample Data

**Question**: What is (are) Common variable immunodeficiency ?

**Answer**: Common variable immunodeficiency (CVID) is a group of disorders in which the immune system cannot make antibodies against agents that cause infection (such as bacteria). CVID is characterized by low levels of most or all of the immunoglobulin (Ig) classes. This causes affected people to get frequent infections, particularly in the sinuses, lungs, and digestive tract. Symptoms most commonly begin in early adulthood but have been found in children as young as age two. While in most cases the cause of CVID is unknown, it has been associated with changes (mutations) in at least 10 genes. About 10% of cases are due to mutations in the TNFRSF13B gene. Treatment for CVID includes Ig replacement therapy, which stops the cycle of recurrent infections.

## Data Processing

I split the dataset into:
- Training set: Used for training the model (75%)
- Validation set: Used for tuning the model (12.5%)
- Test set: Used for evaluating the model (12.5%)

Finally I used Huggingface `DatasetDict` to store all splits, allowing for easy access and manipulation during training and evaluation.

During the data processing, I had to follow the same tokenization and chat formatting as the original dataset to ensure consistency. I used `tokenizer.apply_chat_template` to format the questions and answers appropriately.

## Model Selection

In this assignment, I used one of my favorite models: `MediPhi-Instruct` developed and open sourced by Microsoft. The MediPhi Model Collection comprises 7 small language models of 3.8B parameters from the base model Phi-3.5-mini-instruct specialized in the medical and clinical domains.

I chose this model because it is specifically designed for medical applications, making it well-suited for understanding and generating medical text.

Here is the model Architecture:

```python
Phi3ForCausalLM(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(
          (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
          (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
        )
        (mlp): Phi3MLP(
          (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
          (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
          (activation_fn): SiLU()
        )
        (input_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
        (post_attention_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
        (resid_attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (norm): Phi3RMSNorm((3072,), eps=1e-05)
    (rotary_emb): Phi3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=32064, bias=False)
)
```

Visit this [link](https://huggingface.co/microsoft/MediPhi-Instruct) for more details.

## Parameter Efficient Training

I used: 1 x RTX 5090 16 vCPU 141 GB RAM to train the model

The model is not super big, but I still used `BitsAndBytesConfig` to reduce the model size for convenience. I also use LoRA (Low-Rank Adaptation) to fine-tune the model with fewer parameters, where the rank is set to 8 (this parameter controls the rank of the adaptation matrices and we can increase it for more capacity).

For more details of the training see the notebook.

## Evaluation

To evaluate the model performance, I wanted to use BLEU and ROUGE, but for some reason, GPU was out of memory when evaluating the model. So I stuck to the default cross-entropy loss for monitoring. Below is the first few steps performance:

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 20   | 1.179900      | 0.830046        |
| 40   | 0.783400      | 0.716754        |
| 60   | 0.718500      | 0.689175        |
| 80   | 0.683800      | 0.670964        |
| 100  | 0.664100      | 0.659222        |
| 120  | 0.663800      | 0.651528        |
| 140  | 0.644100      | 0.644649        |

Around 140 steps, the performance became stable and training and validation became plateaued. So I decided to stop training and evaluate the model, that's why you see `KeyboardInterrupt` in the notebook.

You can download the trained adapter here: https://huggingface.co/sabber/medphi-medical-qa-adapter

For rigorous evaluation, I tried to use llm-eval to assess the model's performance on various datasets, but it produced errors and I did not have much time to complete this. But we should consider evaluating the model on various datasets. The MediPhi original paper mentioned that they have created some evaluation datasets that should be tried.

## Test Results

### Question 1
**Question**: What is (are) Dry Eye?

**Original Answer**: National Eye Institute National Institutes of Health 2020 Vision Place Bethesda, MD 20892-3655 301-496-5248 E-mail: 2020@nei.nih.gov www.nei.nih.gov

**Fine-tuned Model**: Dry eye is a condition where there is insufficient lubrication on the surface of the eye, leading to discomfort, irritation, and potential damage to the eye. It can be caused by various factors, including environmental conditions, aging, certain medications, or underlying health issues. Symptoms often include a gritty or burning sensation, redness, blurred vision, and increased sensitivity to light. Treatment typically involves artificial tears, lifestyle changes, and in some cases, medications or surgery to address the underlying cause.

### Question 2
**Question**: What is (are) Genetic Disorders?

**Original Answer**: Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to another, build structures, break down toxins, and do many other maintenance jobs. Sometimes there is a mutation, a change in a gene or genes. The mutation changes the gene's instructions for making a protein, so the protein does not work properly or is missing entirely. This can cause a medical condition called a genetic disorder. You can inherit a gene mutation from one or both parents. A mutation can also happen during your lifetime. There are three types of genetic disorders:
- Single-gene disorders, where a mutation affects one gene. Sickle cell anemia is an example.
- Chromosomal disorders, where chromosomes (or parts of chromosomes) are missing or changed. Chromosomes are the structures that hold our genes. Down syndrome is a chromosomal disorder.
- Complex disorders, where there are mutations in two or more genes. Often your lifestyle and environment also play a role. Colon cancer is an example.

Genetic tests on blood and other tissue can identify genetic disorders. NIH: National Library of Medicine

**Fine-tuned Model**: Genetic disorders are diseases that result from changes or mutations in an individual's DNA. These mutations can be inherited from one or both parents or can occur spontaneously. Genetic disorders can affect various parts of the body and can lead to a wide range of health issues.

Here are some key points about genetic disorders:

1. **Inheritance Patterns**: Genetic disorders can be inherited in several ways, including autosomal dominant, autosomal recessive, X-linked, or mitochondrial inheritance.

2. **Examples**: Some well-known genetic disorders include cystic fibrosis, sickle cell anemia, Huntington's disease, and Down syndrome.

3. **Diagnosis**: Genetic disorders are often diagnosed through genetic testing, which can identify specific mutations in an individual's DNA.

4. **Management**: While some genetic disorders have no cure, management strategies can include medication, therapy, lifestyle changes, and in some cases, surgical interventions.

5. **Prevention**: Genetic counseling can help individuals and families understand their risks and make informed decisions about family planning and preventive measures.

In summary, genetic disorders are conditions caused by alterations in the genetic material, and they can have significant impacts on an individual's health and well-being.

See more test results in the evaluation file: `model_evaluation.ipynb`

## Future Improvements

- Use higher rank in the LoRA
- Possibly use full precision model rather than quantized version of the model
- Explore additional data augmentation techniques to improve model robustness
- Extensively evaluate the model where model is failing and then focus on that specific area