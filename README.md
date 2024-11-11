# D01IssueProject
Issue link: https://github.com/langchain-ai/langchain/issues/26518

## How to recreate the issue

1. Make a new venv with the `requirements.txt`

2. Get the MongoDB connection string, and add your database name. e.g.
```"mongodb+srv://<db_user>:<db_password>@ai-cluster.ffx00.mongodb.net/<db_name>?retryWrites=true&w=majority&appName=AI-Cluster```

3. Get an API key from ZhipuAI and add it as an environment variable in a `.env` file along with your mongoDB credentials
```bash
ZHIPU_API_KEY='your_provider_api_key'
MONGO_USER='your_mongo_user'
MONGO_PASS='your_mongo_password'
```
4. Download a PDF file with at least approximately 3000 words (6 pages of full dense text). 

5. Run the code in the `test.py` file. It should fail. Notably, it doesn't fail with smaller pdfs.
   The error message is:
   ```
   Error: Error code: 400, with error text {"error":{"code":"1210","message":"API 调用参数有误，请检查文档。"}}
   ```
   where code 1210 refers to the error message "incorrect API call parameters".

6. Check the MongoDB collection for the vectors to see if it suceeded.

### Remarks

- We tried using Cohere as its embedding, and it worked fine with large files. It appears that the issue is with ZhipuAI's API only.
- It seems unlikely that the issues is therefore with the large 

### Plan

- [ ] Figure out where (which file and line) the error is coming from.
- [x] Set up a test pipeline to run queries on embeddings of a document to test if the fix is working.
- [ ] Determine if the error is with the way MongoDB calls the embeddings or with the implementation of the embeddings interface for ZhipuAI.
- [ ] Read the equivalent source code with Cohere and try to determine where the error is.
- [ ] Test fixes 