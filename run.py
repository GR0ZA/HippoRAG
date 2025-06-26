from datetime import datetime
import json
import csv
import argparse
import os
from src.hipporag import HippoRAG

def run(dataset_name: str):
    docs = []
    corpus_path = f"/ukp-storage-1/rolka1/thesis/data/{dataset_name}/corpus_sentence_256.jsonl"
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj["contents"])

    hipporag = HippoRAG(
        save_dir=f"/ukp-storage-1/rolka1/thesis/data/{dataset_name}/indexes/hipporag",
        llm_model_name='/storage/ukp/shared/shared_model_weights/models--Qwen3-8B',
        embedding_model_name='Qwen/Qwen3-Embedding-8B',
        llm_base_url="http://localhost:8000/v1"
    )

    # Run indexing
    hipporag.index(docs=docs)
    
    ids = []
    queries = []
    gold_answers = []
    meta = []
    gold_docs = []
    dev_path = f"/ukp-storage-1/rolka1/thesis/data/{dataset_name}/dev.jsonl"
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
            queries.append(obj["question"])
            meta.append(obj.get("metadata", {}))
            gold_answers.append(obj.get("golden_answers", []))
            # metadata.context.sentences is List[List[str]], one sub-list per supporting doc
            ctx_lists = obj["metadata"]["context"]["sentences"]
            # join each sub-list of sentences into one string
            gold_docs.append([" ".join(sent_list) for sent_list in ctx_lists])

    solutions, _, _, _, _ = hipporag.rag_qa(
        queries=queries, gold_docs=gold_docs, gold_answers=gold_answers)

    # build the intermediate JSON
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_dir = f"/ukp-storage-1/rolka1/thesis/output/{dataset_name}_{timestamp}_hipporag2"
    os.makedirs(out_dir, exist_ok=True)
    intermediate = []
    for i, sol in enumerate(solutions):
        # original QA entry
        qa = {
            "id": ids[i],
            "question": queries[i],
            "metadata": meta[i],
            "golden_answers": gold_answers[i]
        }
        # parse sol.docs entries of form "Title\nContents"
        retrieval_result = []
        for d in sol.docs[:5]:
            title, _, body = d.partition("\n")
            retrieval_result.append({
                "title":    title,
                "contents": d
            })

        intermediate.append({
            "id":              qa["id"],
            "question":        qa["question"],
            "metadata":        qa.get("metadata", {}),
            "golden_answers":  qa.get("golden_answers", []),
            "output": {
                "retrieval_result": retrieval_result,
                "pred":             sol.answer
            }
        })

    intermediate_path = os.path.join(out_dir, "intermediate_data.json")
    with open(intermediate_path, "w", encoding="utf-8") as f:
        json.dump(intermediate, f, indent=4, ensure_ascii=True)

    print(f"✅ Wrote intermediate data to {intermediate_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    args = parser.parse_args()
    run(args.dataset_name)