"""
Test if ZhipuAI alone is the problem. Doesn't seem like it. 
"""

from zhipuai import ZhipuAI
from dotenv import dotenv_values

config = dotenv_values(".env")

ZHIPU_API_KEY = config["ZHIPU_API_KEY"]
client = ZhipuAI(api_key=ZHIPU_API_KEY)
response = client.embeddings.create(
    model="embedding-3",  # Fill in the model code to be called
    input=[
        "The food was very delicious, and the service was friendly.",
        "This movie is both thrilling and exciting.",
        "Reading books is a great way to expand knowledge."
    ],
)
print(response)
