import shutil
import os
import json
from voyager.agents import SkillManager
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

ckpt_dir = "./ckpt"
vectordb_path = f"{ckpt_dir}/skill/vectordb"

if os.path.exists(vectordb_path):
    #shutil.rmtree(vectordb_path)
    print(f"please remove {vectordb_path}")
else:
    print(f"vectordb output to {vectordb_path}")


skills_json_path = f"{ckpt_dir}/skill/skills.json"
with open(skills_json_path, 'r', encoding='utf-8') as f:
    skills = json.load(f)


skill_manager = SkillManager(
    model_name="gpt-3.5-turbo",
    temperature=0,
    retrieval_top_k=5,
    request_timout=120,
    ckpt_dir=ckpt_dir,
    resume=False,  
    embedding_model="./embedding/paraphrase-multilingual-MiniLM-L12-v2",
    useOllama=False,
    ollama_model_name="ollama_model_name",
    llm="None"
)

# 
skill_manager.skills = skills
