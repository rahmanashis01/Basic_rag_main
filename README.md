# Basic_rag_main
A simple RAG based model build that capitalize the holding mini file size content and read &amp;amp; deliver it to the trained model for optimized searching content

# Basic Retrieval-Augmented Generation (RAG)
This is a basic RAG system that can be used to ask questions to GPT on custom data (PDF).

## How to run
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Set the environment variables
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    export OPENAI_BASE_URI="your_openai_base_uri"
    ```

3. Store the embeddings:
    ```bash
    python store.py
    ```

4. Run the app:
    ```bash
    python app.py

##Some git command

5. echo "# Basic_rag_main" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/rahmanashis01/Basic_rag_main.git
git push -u origin main

6.git remote add origin https://github.com/rahmanashis01/Basic_rag_main.git
git branch -M main
git push -u origin main
