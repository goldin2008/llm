# openai

### ML system design
Frame the problem as an ML task
- Defining the ML objective
- Specifying the system's input and output
- Choosing the right ML category
Data Preparation
- Data engineering
- Feature engineering


### USE CASES
In the context of using a Large Language Model (LLM) like GPT-3.5 for access and identity management, there are several
use cases that can provide valuable insights, enhance efficiency, and improve security. Here are some specific use
cases:

Access Rights Verification:
Associates can query the LLM to verify their current access rights and entitlements. For example:
  - "What systems can I access?"
  - "Do I have permission to view this document?"
  - "Am I authorized to modify this record?"

Access Request Guidance:
When associates need to request additional access rights or permissions, the LLM can provide guidance:
  - "What steps do I need to follow to request access to the sales database?"
  - "Can you help me draft an access request for the financial reports?"

Policy Interpretation:
Associates often need help understanding company policies related to access and entitlements:
  - "What is the policy for accessing confidential customer data?"
  - "Can you explain the access control policy for the HR portal?"

Policy Violation Alerts:
The LLM can monitor access requests and identify potential policy violations, triggering alerts:
  - "Notify me if any access requests violate the principle of least privilege."
  - "Alert me if an employee tries to access a restricted resource."

User Training and Onboarding:
During employee onboarding, the LLM can provide information about access and entitlements:
  - "Can you explain the different user roles and their associated permissions?"
  - "What training materials are available for understanding access management?"

Access Audit Trail:
Associates can query the LLM to retrieve information about their past access history:
  - "Can you provide a log of my access activities for the last month?"
  - "Show me a summary of the resources I accessed last quarter."

Access Recommendations:
The LLM can recommend access rights based on historical data and patterns:
  - "Based on your role, here are the typical access rights you should have."
  - "Would you like me to suggest the access permissions for this project?"

Regulatory Compliance:
The LLM can assist in understanding and applying access controls to ensure compliance with industry regulations:
  - "What access controls are required for handling healthcare data under HIPAA?"
  - "Can you help me configure access rights to comply with GDPR?"

These use cases demonstrate how an LLM can play a role in enhancing access and identity management processes by
providing quick and accurate information, assisting with policy compliance, and helping associates make informed
decisions regarding access requests and entitlements. However, it's crucial to consider the security, privacy, and
ethical implications of implementing such a system, and to ensure that it aligns with your organization's policies and
regulatory requirements.


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
pdoc --http HOST:PORT <filename.py>
pdoc --http localhost:8080 src/llm/test.py
pdoc --html <filename.py>
pdoc --html src/llm/test.py
```

```commandline
git commit -m 'add pre-commit examples' --no-verify
```
In this template, we use five different plugins that are specified in .pre-commit-config.yaml . They are:
- black — formats Python code
- flake8 — checks the style and quality of your Python code
- isort — automatically sorts imported libraries alphabetically and separates them into sections and types.
- mypy — checks static type 
- nbstripout — strips output from Jupyter notebooks

To check, I use the command 
```commandline
cat -e -t -v makefile_name
```
It shows the presence of tabs with ^I and line endings with $.
Both are vital to ensure that dependencies end properly and tabs mark the action for 
the rules so that they are easily identifiable to the make utility.

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

> https://stanford-cs324.github.io/winter2022/lectures/ 

> https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs/tree/main

> https://courses.d2l.ai/zh-v2/

> https://zh-v2.d2l.ai/chapter_introduction/index.html

> https://www.coursera.org/instructor/htlin

> 李宏毅2023春机器学习课程 https://www.bilibili.com/video/BV1TD4y137mP/?spm_id_from=333.337.search-card.all.click&vd_source=dac84efc7251c2dbebbcdf001659dc53

> https://huggingface.co/getvector/earnings-transcript-summary

> https://huggingface.co/soleimanian/financial-roberta-large-sentiment

> https://huggingface.co/nickmuchi/quantized-optimum-finbert-tone

> https://huggingface.co/sentence-transformers/all-mpnet-base-v2

> https://huggingface.co/philschmid/flan-t5-base-samsum