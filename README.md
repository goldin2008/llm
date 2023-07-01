# openai

```commandline
poetry new --src openai
cd openai
```


```commandline
poetry env list
poetry env use python3
poetry config --list
poetry show


poetry lock
poetry install
poetry update
poetry run pytest
poetry export --output requirements.txt
```

```commandline
git commit -m 'add pre-commit examples' --no-verify
```

#### (Install a Package With Poetry)
```commandline
poetry add requests
poetry add black --group dev
```

ChatGPT, developed by OpenAI, is a powerful tool used for various applications, including chatbots, content generation, and customer service. Its strength lies in generating human-like text based on the prompts it receives. In this tutorial, we will delve into the art and science of Prompt Engineering - crafting precise and effective prompts to get the best responses from ChatGPT.
We decided to focus on ChatGPT Prompt Engineering because it's a crucial skill when working with language models. Understanding how to create effective prompts leads to more accurate, focused, and useful responses.

### Setup API
> https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety

### Prompt Engineering
> https://github.com/openai/openai-cookbook/blob/main/how_to_work_with_large_language_models.md

### Rate Limits
> https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
> https://github.com/openai/openai-cookbook/blob/main/examples/Unit_test_writing_using_a_multi-step_prompt.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/Unit_test_writing_using_a_multi-step_prompt_with_older_completions_API.ipynb

### Q&A (56)
> https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_a_search_API.ipynb

### Summarization (56)

### Content Generation (53)

### Semantic Search (26)
> https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/Code_search.ipynb

### Content Extraction (21)
> https://github.com/openai/openai-cookbook/blob/main/examples/Entity_extraction_for_long_documents.ipynb

### Embeddings
> https://github.com/openai/openai-cookbook/blob/main/examples/Customizing_embeddings.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_Wikipedia_articles_for_search.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/Regression_using_embeddings.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/User_and_product_embeddings.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/Zero-shot_classification_with_embeddings.ipynb

### Clustering
> https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/Clustering_for_transaction_classification.ipynb

### Others
> https://github.com/openai/openai-cookbook/blob/main/examples/Fine-tuned_classification.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/How_to_build_a_tool-using_agent_with_Langchain.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_for_knowledge_retrieval.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
> https://github.com/openai/openai-cookbook/blob/main/examples/Multiclass_classification_for_transactions.ipynb
> https://github.com/openai/openai-python

### What are embeddings?
OpenAI’s text embeddings measure the relatedness of text strings. Embeddings are commonly used for:
- Search (where results are ranked by relevance to a query string)
- Clustering (where text strings are grouped by similarity)
- Recommendations (where items with related text strings are recommended)
- Anomaly detection (where outliers with little relatedness are identified)
- Diversity measurement (where similarity distributions are analyzed)
- Classification (where text strings are classified by their most similar label)
An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.

## References
> https://github.com/dair-ai/Prompt-Engineering-Guide

> https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

> https://github.com/openai/openai-cookbook/blob/main/examples/vector_databases/Using_vector_databases_for_embeddings_search.ipynb

> https://www.datacamp.com/tutorial/a-beginners-guide-to-chatgpt-prompt-engineering

> https://learn.microsoft.com/en-us/azure/app-service/# llm

> https://github.com/khuyentran1401/data-science-template/blob/dvc-poetry/README.md

> https://towardsdatascience.com/how-to-structure-a-data-science-project-for-readability-and-transparency-360c6716800

> https://towardsdatascience.com/4-pre-commit-plugins-to-automate-code-reviewing-and-formatting-in-python-c80c6d2e9f5